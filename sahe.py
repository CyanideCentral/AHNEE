import numpy as np
import scipy.sparse as sp
import time
import resource
import config
from sklearn.preprocessing import normalize
import faiss
from sparse_dot_mkl import dot_product_mkl
from pts import pts_tlog

def extend_hypergraph(ahg, knn = 10, index_type = "IVF512,PQ10"):
    attributes = ahg.attributes.toarray()
    n, d = attributes.shape
    faiss.normalize_L2(attributes)
    if n > 1000000: # Use IVF for large datasets
        index = faiss.index_factory(d, index_type, faiss.METRIC_INNER_PRODUCT)
        index.train(attributes)
    else: # exact search for small datasets
        index = faiss.IndexFlatIP(d)
    index.add(attributes)
    D, I = index.search(attributes, knn+1)
    # Construct the incidence matrix H_K
    H_K = sp.csr_matrix((D.flatten(), I.flatten(), np.arange(0, n*(knn+1)+1, knn+1)), shape=(n, n))
    H_K.setdiag(1.0)

    # Extended hypergraph; Balance via hyperedge weighting
    H_K = config.beta * H_K
    extended_H = sp.vstack([ahg.hypergraph, H_K], format='csr')
    H_vol = (ahg.hypergraph.sum(1).A1).sum() # Volume of H_0
    A_vol = (H_K.sum(1).A1).sum() # Volume of H_K
    attr_weight = H_vol / (H_vol + A_vol)
    # Hyperedge weights
    he_weights = np.concatenate([np.ones(ahg.m), np.full(H_K.shape[0], attr_weight)])
    return extended_H, he_weights

def sahe(ahg):
    start_time = time.perf_counter()
    H, hyperedge_weights= extend_hypergraph(ahg, knn=config.knn)
    vol = float(H.sum())
    m, n = H.shape
    deg_vec = (H.T@hyperedge_weights[:,np.newaxis]).flatten()
    deg_vec[deg_vec==0] = 1
    deg_alpha = deg_vec**(-0.5)
    D_v = sp.diags(deg_alpha, format='csr')
    deg_edge = H.sum(axis=1).A1
    deg_edge[deg_edge==0] = 1
    D_e = sp.diags(deg_edge**(-0.5))
    D_e = sp.diags(hyperedge_weights**0.5) @ D_e

    # Common computations for nodes and hyperedges
    norm_H = D_e @ H @ D_v # \tilde{H}
    # Truncated SVD
    def linear_operator(x):
        return dot_product_mkl(norm_H, x, cast=True)
    def linear_operator1(x):
        return dot_product_mkl(norm_H.T, x, cast=True)
    H_LO = sp.linalg.LinearOperator((m, n),matvec = linear_operator,rmatvec = linear_operator1)
    U, s, VT = sp.linalg.svds(H_LO, min(config.svd_rank, n//2), tol=config.svd_tol, random_state=config.rng)
    sigma_sq = s**2
    sigma_T = np.full_like(sigma_sq, config.alpha)
    for i in range(1, config.window+1):
        sigma_T += config.alpha * (1- config.alpha)**i * sigma_sq**i

    # Approximating HMS-N node similarity
    F_v = np.sqrt(vol) * sp.diags(np.sqrt(sigma_T)).dot(VT@D_v).T
    Y_v, Th_v = pts_tlog(F_v) # Polynomial tensor sketch
    YTh_v = Y_v @ Th_v
    def laplacian_operator(v):
        return Y_v @ (YTh_v.T @ v) # L(v) = Y (\Theta Y (v))
    Gamma_v_LO = sp.linalg.LinearOperator((n, n), matvec=laplacian_operator)
    _, Z_v = sp.linalg.eigsh(Gamma_v_LO, min(config.embedding_dim, n//2), tol=config.svd_tol,which='LM')
    Z_v = normalize(Z_v, norm='l2', axis=1)

    # Approximating HMS-E hyperedge similarity
    F_e = np.sqrt(vol)*sp.diags(np.sqrt(sigma_T)).dot(D_e.dot(U)[:m-n].T).T
    Y_e, Th_e = pts_tlog(F_e) # Polynomial tensor sketch
    YTh_e = Y_e @ Th_e
    def laplacian_operator(v):
        return Y_e @ (YTh_e.T @ v)  # L(v) = Y (\Theta Y (v))
    Gamma_e_LO = sp.linalg.LinearOperator((m-n, m-n), matvec=laplacian_operator)
    _, Z_e = sp.linalg.eigsh(Gamma_e_LO, min(config.embedding_dim, (m-n)//2), tol=config.svd_tol,which='LM')
    Z_e = normalize(Z_e, norm='l2', axis=1)

    return Z_v, Z_e, (time.perf_counter() - start_time, resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 / 1024)

