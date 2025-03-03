import numpy as np
import scipy.sparse as sp
import time
import resource
import config
from sklearn.preprocessing import normalize
import pickle as pkl
import faiss
from sparse_dot_mkl import dot_product_mkl
from pts import pts_tlog

def extend_hypergraph(ahg, knn = 10, weighted = True, index_type = "IVF512,PQ10", return_index = False):
    attributes = ahg.attributes.toarray()
    n, d = attributes.shape
    faiss.normalize_L2(attributes)
    if n > 100000:
        index = faiss.index_factory(d, index_type, faiss.METRIC_INNER_PRODUCT)
        index.train(attributes)
    else:
        index = faiss.IndexFlatIP(d)
    index.add(attributes)
    D, I = index.search(attributes, knn+1)
    H = sp.csr_matrix((D.flatten(), I.flatten(), np.arange(0, n*(knn+1)+1, knn+1)), shape=(n, n))
    self_weight = 1.0
    H.setdiag(self_weight)

    aug_hg = config.beta* H
    athg = sp.vstack([ahg.hypergraph, aug_hg], format='csr')
    H_vol = (ahg.hypergraph.sum(1).A1).sum()
    A_vol = (aug_hg.sum(1).A1).sum()
    attr_weight = H_vol / (H_vol + A_vol)
    he_weights = np.concatenate([np.ones(ahg.m), np.full(aug_hg.shape[0], attr_weight)])
    return athg, he_weights

def sahe(ahg):
    start_time = time.perf_counter()
    H, hyperedge_weights= extend_hypergraph(ahg, knn=config.knn, weighted=True)
    config.rng = np.random.default_rng(0)
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
    
    norm_H = D_e @ H @ D_v # \tilde{H}
    # Truncated SVD
    def linear_operator(x):
        return dot_product_mkl(norm_H, x, cast=True)
    def linear_operator1(x):
        return dot_product_mkl(norm_H.T, x, cast=True)
    lap_lo = sp.linalg.LinearOperator((m, n),matvec = linear_operator,rmatvec = linear_operator1)
    U, s, VT = sp.linalg.svds(lap_lo, min(config.svd_rank, n//2), tol=config.svd_tol, random_state=config.rng)
    sigma_sq = s**2
    sigma_T = np.full_like(sigma_sq, config.alpha)
    for i in range(1, config.window+1):
        sigma_T += config.alpha * (1- config.alpha)**i * sigma_sq**i
    
    F_v = np.sqrt(vol) * sp.diags(np.sqrt(sigma_T)).dot(VT@D_v).T
    Y_v, Th_v = pts_tlog(F_v)
    YTh_v = Y_v @ Th_v
    def laplacian_operator(x):
        return Y_v @ (YTh_v.T @ x)
    Gamma_v_LO = sp.linalg.LinearOperator((n, n), matvec=laplacian_operator)
    _, Z_v = sp.linalg.eigsh(Gamma_v_LO, min(config.embedding_dim, n//2), tol=config.svd_tol,which='LM')
    Z_v = normalize(Z_v, norm='l2', axis=1)

    D_rt_invU = D_e.dot(U)[:m-n]
    F_e = np.sqrt(vol)*sp.diags(np.sqrt(sigma_T)).dot(D_e.dot(U)[:m-n].T).T
    Y_e, Th_e = pts_tlog(F_e)
    YTh_e = Y_e @ Th_e
    def laplacian_operator(x):
        return Y_e @ (YTh_e.T @ x)
    Gamma_e_LO = sp.linalg.LinearOperator((m-n, m-n), matvec=laplacian_operator)
    _, Z_e = sp.linalg.eigsh(Gamma_e_LO, min(config.embedding_dim, (m-n)//2), tol=config.svd_tol,which='LM')
    Z_e = normalize(Z_e, norm='l2', axis=1)

    return Z_v, Z_e, (time.perf_counter() - start_time, resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 / 1024)

