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

def rwr_filter(evals):
    filtered_evals = np.full_like(evals, config.rwr_alpha)
    for i in range(1, config.rwr_window+1):
        filtered_evals += config.rwr_alpha * (1- config.rwr_alpha)**i * evals**i
    return filtered_evals

def extend_hypergraph(ahg, knn = 10, weighted = True, index_type = "IVF512,PQ10", return_index = False):
    features = ahg.features.toarray()
    n, d = features.shape
    faiss.normalize_L2(features)
    if n > 100000:
        index = faiss.index_factory(d, index_type, faiss.METRIC_INNER_PRODUCT)
        index.train(features)
    else:
        index = faiss.IndexFlatIP(d)
    index.add(features)
    D, I = index.search(features, knn+1)
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

def hgmf_full_sym_pts(H, dim=256, hyperedge_weights=None, rank=32, window=10, negative=1.0):
    config.rng = np.random.default_rng(0)
    if config.verbose:
        print(f"parameter test: rank {config.svd_rank} window {config.window} svd_tol {config.svd_tol} negative {config.negative}")
    from sklearn.utils.extmath import randomized_svd
    start_time = time.time()
    vol = float(H.sum())
    m = H.shape[0]
    n = H.shape[1]
    if hyperedge_weights is None:
        deg_vec = H.sum(axis=0).A1
    else:
        deg_vec = (H.T@hyperedge_weights[:,np.newaxis]).flatten()
    deg_vec[deg_vec==0] = 1
    deg_alpha = deg_vec**(-0.5)
    D_v = sp.diags(deg_alpha, format='csr')
    deg_edge = H.sum(axis=1).A1
    deg_edge[deg_edge==0] = 1
    D_e = sp.diags(deg_edge**(-0.5))
    if hyperedge_weights is not None:
        D_e = sp.diags(hyperedge_weights**0.5) @ D_e
    L = D_e @ H @ D_v
    lap_time = time.time()
    if config.verbose:
        print("Laplacian computation time:", lap_time-start_time)
    print("SVD decomposition...")

    def linear_operator(x):
        return dot_product_mkl(L, x, cast=True)
    def linear_operator1(x):
        return dot_product_mkl(L.T, x, cast=True)
    lap_lo = sp.linalg.LinearOperator((m, n),matvec = linear_operator,rmatvec = linear_operator1)
    U, s, VT = sp.linalg.svds(lap_lo, min(config.svd_rank, n//2), tol=config.svd_tol, random_state=config.rng)
    svd_time = time.time()

    if config.test_time:
        print("SVD computation time:", svd_time-lap_time)
    V = VT.T
    evals = s**2
    print("Maximum eigenvalue %f, minimum eigenvalue %f", np.max(s), np.min(s))
    D_rt_invU = D_v.dot(V)
    evals = rwr_filter(evals)
    U = sp.diags(np.sqrt(evals)).dot(D_rt_invU.T).T
    # U = sp.diags(evals**2).dot(D_rt_invU.T).T
    sample_x = None
    U*=np.sqrt(vol/negative)

    TU, C = pts_tlog(U)
    TV = TU @ C
    #to test with dot_product_mkl
    def laplacian_operator(x):
        return TU @ (TV.T @ x)
    sketch_lo = sp.linalg.LinearOperator((n, n), matvec=laplacian_operator)
    try:
        evals, X = sp.linalg.eigsh(sketch_lo, min(dim, n//2), tol=config.svd_tol,which='LM')
    except sp.linalg.ArpackError as e:
        print(e)
        evals, X = e.eigenvalues, e.eigenvectors
    X = normalize(X, norm='l2', axis=1)
    embed_time = time.time()
    if config.verbose:
        print("Embedding computation time:", embed_time-svd_time)
    return X

def hgmf_edge_full_sym_pts(H, dim=256, hyperedge_weights=None, rank=32, window=10, negative=1.0):
    if config.verbose:
        print(f"parameter test: rank {config.svd_rank} window {config.window} svd_tol {config.svd_tol} negative {config.negative}")
    from sklearn.utils.extmath import randomized_svd
    start_time = time.time()
    vol = float(H.sum())
    m = H.shape[0]
    n = H.shape[1]
    if hyperedge_weights is None:
        deg_vec = H.sum(axis=0).A1
    else:
        deg_vec = (H.T@hyperedge_weights[:,np.newaxis]).flatten()
    deg_vec[deg_vec==0] = 1
    deg_alpha = deg_vec**(-0.5)
    D_v = sp.diags(deg_alpha, format='csr')
    deg_edge = H.sum(axis=1).A1
    deg_edge[deg_edge==0] = 1
    D_e = sp.diags(deg_edge**(-0.5))
    if hyperedge_weights is not None:
        D_e = sp.diags(hyperedge_weights**0.5) @ D_e
    L = D_e @ H @ D_v
    lap_time = time.time()
    if config.verbose:
        print("Laplacian computation time:", lap_time-start_time)
    print("SVD decomposition...")

    spmv_count, spmv_count1 = 0, 0
    def linear_operator(x):
        return dot_product_mkl(L, x, cast=True)
    def linear_operator1(x):
        return dot_product_mkl(L.T, x, cast=True)
    lap_lo = sp.linalg.LinearOperator((m, n),matvec = linear_operator,rmatvec = linear_operator1)
    U, s, VT = sp.linalg.svds(lap_lo, min(config.svd_rank, n//2), tol=config.svd_tol)
    # V = VT.T
    evals = s**2
    print("Maximum eigenvalue %f, minimum eigenvalue %f", np.max(s), np.min(s))
    # D_rt_invU = D_v.dot(V)
    D_rt_invU = D_e.dot(U)[:m-n]
    evals = rwr_filter(evals)
    U = sp.diags(np.sqrt(evals)).dot(D_rt_invU.T).T
    # U = sp.diags(evals**2).dot(D_rt_invU.T).T
    U*=np.sqrt(vol/negative)
    TU, C = pts_tlog(U)
    TV = TU @ C
    #to test with dot_product_mkl
    def laplacian_operator(x):
        return TU @ (TV.T @ x)
    sketch_lo = sp.linalg.LinearOperator((m-n, m-n), matvec=laplacian_operator)
    try:
        evals, X = sp.linalg.eigsh(sketch_lo, min(dim, (m-n)//2), tol=config.svd_tol,which='LM')
    except sp.linalg.ArpackError as e:
        print(e)
        evals, X = e.eigenvalues, e.eigenvectors
    X = normalize(X, norm='l2', axis=1)
    return X

def sahe(ahg, dim, hyperedge=False):
    start_time = time.perf_counter()
    athg, he_weights= extend_hypergraph(ahg, knn=config.knn, weighted=True)
    if not hyperedge:
        embedding = hgmf_full_sym_pts(athg, dim, hyperedge_weights=he_weights)
    else:
        embedding = hgmf_edge_full_sym_pts(athg, dim, hyperedge_weights=he_weights)
    return embedding, (time.perf_counter() - start_time, resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 / 1024)

