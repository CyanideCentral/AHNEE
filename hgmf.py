import scipy.sparse as sparse
from scipy.sparse import csgraph
from sklearn.preprocessing import normalize
import numpy as np
import logging
import warnings
import time
import config
from sparse_dot_mkl import dot_product_mkl

logger = logging.getLogger(__name__)
warnings.simplefilter("ignore", sparse.SparseEfficiencyWarning)

def deepwalk_filter(evals, window=10):
    for i in range(len(evals)):
        x = evals[i]
        evals[i] = 1. if x >= 1 else x*(1-x**window) / (1-x) / window
        # evals[i] = x
    evals = np.maximum(evals, 0)
    # logger.info("After filtering, max eigenvalue=%f, min eigenvalue=%f", np.max(evals), np.min(evals))
    return evals

def multi_deepwalk_filter(evals, min_hop=3, max_hop=10):
    for i in range(len(evals)):
        x = evals[i]
        window = max_hop - min_hop
        evals[i] = 1. if x >= 1 else x*(1-x**window) / (1-x) / window
        evals[i] *= x**min_hop
        # evals[i] = x
    evals = np.maximum(evals, 0)
    # logger.info("After filtering, max eigenvalue=%f, min eigenvalue=%f", np.max(evals), np.min(evals))
    return evals

def rwr_filter(evals):
    filtered_evals = np.full_like(evals, config.rwr_alpha)
    for i in range(1, config.rwr_window+1):
        filtered_evals += config.rwr_alpha * (1- config.rwr_alpha)**i * evals**i
    return filtered_evals

def approximate_normalized_graph_laplacian(H, rank, hyperedge_weights=None, which="LA"):
    m = H.shape[0]
    n = H.shape[1]
    if hyperedge_weights is None:
        deg_vec = H.sum(axis=0).A1
    else:
        deg_vec = (H.T@hyperedge_weights[:,np.newaxis]).flatten()
    deg_vec[deg_vec==0] = 1
    deg_alpha = deg_vec**(-0.5)
    # L, d_rt = csgraph.laplacian(H, normed=True, return_diag=True)
    D_v = sparse.diags(deg_alpha, format='csr')
    deg_edge = H.sum(axis=1).A1
    deg_edge[deg_edge==0] = 1
    D_e = sparse.diags(deg_edge**(-0.5))
    if hyperedge_weights is not None:
        D_e = sparse.diags(hyperedge_weights**0.5) @ D_e
    L = D_e @ H @ D_v
    logger.info("SVD decomposition...")
    U, s, VT = sparse.linalg.svds(L, min(rank, n//2))
    V = VT.T
    s = s**2
    logger.info("Maximum eigenvalue %f, minimum eigenvalue %f", np.max(s), np.min(s))
    D_rt_invU = D_v.dot(V)
    return s, D_rt_invU

    if hyperedge_weights is not None:
        D_e = sparse.diags(hyperedge_weights) @ D_e
    L = D_v @ H.T @ D_e @ H @ D_v
    logger.info("Eigen decomposition...")
    if which == 'LA':
        evals, evecs = sparse.linalg.eigsh(L, min(rank, n//2), which='LA')
    else:
        L = sparse.identity(n) - L
        evals, evecs = sparse.linalg.eigsh(L, min(rank, n//2), which='SA')
        evals = 1-evals
    # try:
    #     evals, evecs = sparse.linalg.eigsh(L, min(rank, n//2), which=which)
    # except sparse.linalg.ArpackError as e:
    #     print(e)
    #     evals, evecs = sparse.linalg.eigsh(L, rank, which=which, ncv=min(n, 1000))
    logger.info("Maximum eigenvalue %f, minimum eigenvalue %f", np.max(evals), np.min(evals))
    logger.info("Computing D^{-1/2}U..")
    # D_rt_inv = sparse.diags(d_rt ** -1)
    D_rt_invU = D_v.dot(evecs)
    return evals, D_rt_invU

def approximate_deepwalk_matrix(evals, D_rt_invU, window, vol, b):
    evals = deepwalk_filter(evals, window=window)
    X = sparse.diags(np.sqrt(evals)).dot(D_rt_invU.T).T
    Y = X @ X.T * vol / b
    # from pts import poly_fit_lsq
    # coeffs = poly_fit_lsq(lambda x: np.log(x) if x > 1 else 0, deg=10, domain=(-1, 3), nonnegative=False, visualize=False)
    # coeffs = poly_fit_lsq(lambda x: np.log(x) if x > 1 else 0, deg=10, domain=(-10, 30), nonnegative=False, visualize=False)
    # coeffs = poly_fit_lsq(lambda x: np.log(x) if x > 1 else 0, deg=10, sample=np.sort(Y[:, :].flatten()), nonnegative=False, visualize=True)
    # logY = sum(c * Y**i for i, c in enumerate(coeffs))
    logY = np.log(np.maximum(Y, 1.)) # trunc_log(x)
    # logY = (np.maximum(Y-1., 0.)) # relu(x)-1a
    # logY = np.tanh(Y-1.0) # sigmoid(x-1)
    # logger.info("Computed DeepWalk matrix with %d non-zero elements", np.count_nonzero(logY))
    return sparse.csr_matrix(logY)

def svd_deepwalk_matrix(X, dim):
    from sklearn.utils.extmath import randomized_svd
    u, s, v = randomized_svd(X,dim,n_oversamples=1,n_iter=1)
    # u, s, v = sparse.linalg.svds(X, dim, return_singular_vectors="u")
    return sparse.diags(np.sqrt(s)).dot(u.T).T

def deepwalk_no_log(evals, D_rt_invU, window, vol, b):
    evals = deepwalk_filter(evals, window=window)
    X = sparse.diags(np.sqrt(evals)).dot(D_rt_invU.T).T
    # Y = np.log(np.maximum((X @ X.T) * vol / b, 1.))
    # Y = (X @ X.T) * vol / b
    # logger.info("Computed DeepWalk matrix with %d non-zero elements", np.count_nonzero(Y))
    return X

def hgmf_svdx(H, dim=256, hyperedge_weights=None, rank=32, window=10, negative=1.0):
    from sklearn.utils.extmath import randomized_svd
    if hyperedge_weights is None:
        vol = float(H.sum())
    else:
        vol = (H.T @ hyperedge_weights[:, np.newaxis]).sum()
    m = H.shape[0]
    n = H.shape[1]
    if hyperedge_weights is None:
        deg_vec = H.sum(axis=0).A1
    else:
        deg_vec = (H.T@hyperedge_weights[:,np.newaxis]).flatten()
    deg_vec[deg_vec==0] = 1
    deg_alpha = deg_vec**(-0.5)
    # L, d_rt = csgraph.laplacian(H, normed=True, return_diag=True)
    D_v = sparse.diags(deg_alpha, format='csr')
    deg_edge = H.sum(axis=1).A1
    deg_edge[deg_edge==0] = 1
    D_e = sparse.diags(deg_edge**(-0.5))
    if hyperedge_weights is not None:
        D_e = sparse.diags(hyperedge_weights**0.5) @ D_e
    L = D_e @ H @ D_v
    logger.info("SVD decomposition...")
    U, s, VT = sparse.linalg.svds(L, min(rank, n//2))
    V = VT.T
    evals = s**2
    logger.info("Maximum eigenvalue %f, minimum eigenvalue %f", np.max(s), np.min(s))
    D_rt_invU = D_v.dot(V)
    # approximate deepwalk matrix
    # deepwalk_matrix = approximate_deepwalk_matrix(evals, D_rt_invU, window=window, vol=vol, b=negative)
    evals = deepwalk_filter(evals, window=window)
    X = sparse.diags(np.sqrt(evals)).dot(D_rt_invU.T).T
    # Y = np.log(np.maximum((X @ X.T) * vol / b, 1.))
    # Y = (X @ X.T) * vol / negative
    # factorize deepwalk matrix with SVD
    # deepwalk_embedding = svd_deepwalk_matrix(deepwalk_matrix, dim=dim)
    # scalar = np.sqrt(vol/negative)
    u, s, v = randomized_svd(X*np.sqrt(vol/negative),dim,n_oversamples=1,n_iter=1)
    deepwalk_embedding = sparse.diags(np.sqrt(s)).dot(u.T).T
    return deepwalk_embedding

def hg_svd(H, dim=256, hyperedge_weights=None, rank=32, window=10, negative=1.0):
    if config.verbose:
        print(f"parameter test: rank {config.svd_rank} window {config.window} svd_tol {config.svd_tol} negative {config.negative} beta {config.beta}")
    from sklearn.utils.extmath import randomized_svd
    start_time = time.time()
    vol = float(H.sum())
    m = H.shape[0]
    n = H.shape[1]
    # if hyperedge_weights is None:
    #     deg_vec = H.sum(axis=0).A1
    # else:
    #     deg_vec = (H.T@hyperedge_weights[:,np.newaxis]).flatten()
    # deg_vec[deg_vec==0] = 1
    # deg_alpha = deg_vec**(-0.5)
    # D_v = sparse.diags(deg_alpha, format='csr')
    # deg_edge = H.sum(axis=1).A1
    # deg_edge[deg_edge==0] = 1
    # D_e = sparse.diags(deg_edge**(-0.5))
    # if hyperedge_weights is not None:
    #     D_e = sparse.diags(hyperedge_weights**0.5) @ D_e
    # L = D_e @ H @ D_v
    L = sparse.diags(hyperedge_weights**0.5) @ H
    lap_time = time.time()
    if config.verbose:
        print("Laplacian computation time:", lap_time-start_time)
    logger.info("SVD decomposition...")
    def linear_operator(x):
        return dot_product_mkl(L, x, cast=True)
    def linear_operator1(x):
        return dot_product_mkl(L.T, x, cast=True)
    lap_lo = sparse.linalg.LinearOperator((m, n),matvec = linear_operator,rmatvec = linear_operator1)
    U, s, VT = sparse.linalg.svds(lap_lo, min(config.svd_rank, n//2), tol=config.svd_tol)
    # U, s, VT = sparse.linalg.svds(L, min(config.svd_rank, n//2), tol=config.svd_tol)
    svd_time = time.time()
    if config.verbose:
        print("SVD computation time:", svd_time-lap_time)
    # V = VT.T
    evals = s**2
    logger.info("Maximum eigenvalue %f, minimum eigenvalue %f", np.max(s), np.min(s))
    # D_rt_invU = D_v.dot(V)
    # evals = deepwalk_filter(evals, window=config.window)
    # evals = multi_deepwalk_filter(evals, min_hop=3, max_hop=config.window)
    X = VT.T
    # X = sparse.diags(np.sqrt(evals)).dot(D_rt_invU.T).T
    # X = sparse.diags(evals**2).dot(D_rt_invU.T).T # sqev
    # scalar = np.sqrt(vol/config.negative)
    # scalar = 1.
    # X = X * scalar
    X = normalize(X, norm='l2', axis=1)
    if X.shape[1] < dim:
        padding = np.zeros((X.shape[0], dim-X.shape[1]))
        X = np.hstack((X, padding))
    elif X.shape[1] > dim:
        X = X[:, :dim]
    embed_time = time.time()
    if config.verbose:
        print("Embedding computation time:", embed_time-svd_time)
    return X

def hgmf_1svd(H, dim=256, hyperedge_weights=None, rank=32, window=10, negative=1.0):
    if config.verbose:
        print(f"parameter test: rank {config.svd_rank} window {config.window} svd_tol {config.svd_tol} negative {config.negative} beta {config.beta}")
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
    D_v = sparse.diags(deg_alpha, format='csr')
    deg_edge = H.sum(axis=1).A1
    deg_edge[deg_edge==0] = 1
    D_e = sparse.diags(deg_edge**(-0.5))
    if hyperedge_weights is not None:
        D_e = sparse.diags(hyperedge_weights**0.5) @ D_e
    L = D_e @ H @ D_v
    lap_time = time.time()
    if config.verbose:
        print("Laplacian computation time:", lap_time-start_time)
    logger.info("SVD decomposition...")
    def linear_operator(x):
        return dot_product_mkl(L, x, cast=True)
    def linear_operator1(x):
        return dot_product_mkl(L.T, x, cast=True)
    lap_lo = sparse.linalg.LinearOperator((m, n),matvec = linear_operator,rmatvec = linear_operator1)
    U, s, VT = sparse.linalg.svds(lap_lo, min(config.svd_rank, n//2), tol=config.svd_tol)
    # U, s, VT = sparse.linalg.svds(L, min(config.svd_rank, n//2), tol=config.svd_tol)
    svd_time = time.time()
    if config.verbose:
        print("SVD computation time:", svd_time-lap_time)
    V = VT.T
    evals = s**2
    logger.info("Maximum eigenvalue %f, minimum eigenvalue %f", np.max(s), np.min(s))
    D_rt_invU = D_v.dot(V)
    evals = rwr_filter(evals)
    # evals = deepwalk_filter(evals, window=config.window)
    # evals = multi_deepwalk_filter(evals, min_hop=3, max_hop=config.window)
    X = sparse.diags(np.sqrt(evals)).dot(D_rt_invU.T).T
    # X = sparse.diags(evals**2).dot(D_rt_invU.T).T # sqev
    scalar = np.sqrt(vol/config.negative)
    # scalar = 1.
    X = X * scalar
    X = normalize(X, norm='l2', axis=1)
    if X.shape[1] < dim:
        padding = np.zeros((X.shape[0], dim-X.shape[1]))
        X = np.hstack((X, padding))
    elif X.shape[1] > dim:
        X = X[:, :dim]
    embed_time = time.time()
    if config.verbose:
        print("Embedding computation time:", embed_time-svd_time)
    return X

def hgmf_teigs(H, dim=256, hyperedge_weights=None, rank=32, window=10, negative=1.0):
    if config.verbose:
        print(f"parameter test: rank {config.svd_rank} window {config.window} svd_tol {config.svd_tol} negative {config.negative} beta {config.beta}")
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
    D_v = sparse.diags(deg_alpha, format='csr')
    deg_edge = H.sum(axis=1).A1
    deg_edge[deg_edge==0] = 1
    D_e = sparse.diags(deg_edge**(-0.5))
    if hyperedge_weights is not None:
        D_e = sparse.diags(hyperedge_weights**0.5) @ D_e
    L = D_e @ H @ D_v
    lap_time = time.time()
    if config.verbose:
        print("Laplacian computation time:", lap_time-start_time)
    logger.info("SVD decomposition...")
    def linear_operator(x):
        y = dot_product_mkl(L.T, dot_product_mkl(L, x, cast=True), cast=True)
        x = y
        for i in range(window-1):
            x = dot_product_mkl(L.T, dot_product_mkl(L, x, cast=True), cast=True)
            x += y
        return x
    lap_lo = sparse.linalg.LinearOperator((n, n),matvec = linear_operator)
    eigvs, U = sparse.linalg.eigsh(lap_lo, min(config.svd_rank, n//2), which='LA', tol=config.svd_tol)
    eigsh_time = time.time()
    if config.verbose:
        print("EIGSH computation time:", eigsh_time-lap_time)
    logger.info("Maximum eigenvalue %f, minimum eigenvalue %f", np.max(eigvs), np.min(eigvs))
    # evals = deepwalk_filter(evals, window=config.window)
    # evals = multi_deepwalk_filter(evals, min_hop=3, max_hop=config.window)
    X = np.sqrt(vol/window) * (D_v @ U)
    X = normalize(X, norm='l2', axis=1)
    if X.shape[1] < dim:
        padding = np.zeros((X.shape[0], dim-X.shape[1]))
        X = np.hstack((X, padding))
    elif X.shape[1] > dim:
        X = X[:, :dim]
    embed_time = time.time()
    if config.verbose:
        print("Embedding computation time:", embed_time-eigsh_time)
    return X

def hgmf_rwr_base(H, dim=256, hyperedge_weights=None, rank=32, window=10, negative=1.0):
    if config.verbose:
        print(f"parameter test: rank {config.svd_rank} window {config.window} svd_tol {config.svd_tol} negative {config.negative} beta {config.beta}")
    start_time = time.time()
    vol = float(H.sum())
    m = H.shape[0]
    n = H.shape[1]
    if hyperedge_weights is None:
        deg_vec = H.sum(axis=0).A1
    else:
        deg_vec = (H.T@hyperedge_weights[:,np.newaxis]).flatten()
    deg_vec[deg_vec==0] = 1
    deg_alpha = deg_vec**(-1)
    D_v = sparse.diags(deg_alpha, format='csr')
    deg_edge = H.sum(axis=1).A1
    deg_edge[deg_edge==0] = 1
    D_e = sparse.diags(deg_edge**(-1))
    if hyperedge_weights is not None:
        D_e = sparse.diags(hyperedge_weights) @ D_e
    H = H.toarray()
    D_e = D_e.toarray()
    D_v = D_v.toarray()
    P = (D_v @ H.T @ D_e @ H)
    L = np.eye(n)
    # deg_vec1 = (H.T@hyperedge_weights[:,np.newaxis]).flatten()
    # WDerH = np.diag(hyperedge_weights)@np.diag(H.sum(1)**-1)@H
    for i in range(config.rwr_window):
        L = config.rwr_alpha * L + (1-config.rwr_alpha) * P @ L
    L = (D_v * vol) @ L
    # L = L + L.T #remove now
    lap_time = time.time()
    if config.verbose:
        print("Laplacian computation time:", lap_time-start_time)
    Y = np.log(np.maximum(L, 1.))
    log_time = time.time()
    if config.verbose:
        print("Log computation time:", log_time-lap_time)
    eigvs, U = sparse.linalg.eigsh(Y, min(dim, n//2), which='LA')
    eigsh_time = time.time()
    if config.verbose:
        print("EIGSH computation time:", eigsh_time-log_time)
    logger.info("Maximum eigenvalue %f, minimum eigenvalue %f", np.max(eigvs), np.min(eigvs))
    # evals = deepwalk_filter(evals, window=config.window)
    # evals = multi_deepwalk_filter(evals, min_hop=3, max_hop=config.window)
    X = U @ np.diag(eigvs ** 0.5)
    X = normalize(X, norm='l2', axis=1)
    # if X.shape[1] < dim:
    #     padding = np.zeros((X.shape[0], dim-X.shape[1]))
    #     X = np.hstack((X, padding))
    # elif X.shape[1] > dim:
    #     X = X[:, :dim]
    embed_time = time.time()
    if config.verbose:
        print("Embedding computation time:", embed_time-eigsh_time)
    return X

def hgmf_rwr_base_hyperedge(H, dim=256, hyperedge_weights=None, rank=32, window=10, negative=1.0):
    if config.verbose:
        print(f"parameter test: rank {config.svd_rank} window {config.window} svd_tol {config.svd_tol} negative {config.negative} beta {config.beta}")
    start_time = time.time()
    if hyperedge_weights is not None:
        H = H.multiply(hyperedge_weights[:, np.newaxis])
    vol = float(H.sum())
    m = H.shape[0]
    n = H.shape[1]
    deg_vec = H.sum(axis=0).A1
    deg_vec[deg_vec==0] = 1
    deg_alpha = deg_vec**(-1)
    D_v = sparse.diags(deg_alpha, format='csr')
    deg_edge = H.sum(axis=1).A1
    deg_edge[deg_edge==0] = 1
    D_e = sparse.diags(deg_edge**(-1))
    # if hyperedge_weights is not None:
    #     D_e = sparse.diags(hyperedge_weights) @ D_e
    H = H.toarray()
    D_e = D_e.toarray()
    D_v = D_v.toarray()
    P = (D_e @ H @ D_v @ H.T)
    L = np.eye(m)
    # deg_vec1 = (H.T@hyperedge_weights[:,np.newaxis]).flatten()
    # WDerH = np.diag(hyperedge_weights)@np.diag(H.sum(1)**-1)@H
    for i in range(config.rwr_window):
        L = config.rwr_alpha * L + (1-config.rwr_alpha) * P @ L
    L = (D_e * vol) @ L
    # L = L + L.T
    L = L[:m-n,:m-n]
    lap_time = time.time()
    if config.verbose:
        print("Laplacian computation time:", lap_time-start_time)
    Y = np.log(np.maximum(L, 1.))
    log_time = time.time()
    if config.verbose:
        print("Log computation time:", log_time-lap_time)
    eigvs, U = sparse.linalg.eigsh(Y, min(dim, (m-n)//2), which='LA')
    eigsh_time = time.time()
    if config.verbose:
        print("EIGSH computation time:", eigsh_time-log_time)
    logger.info("Maximum eigenvalue %f, minimum eigenvalue %f", np.max(eigvs), np.min(eigvs))
    # evals = deepwalk_filter(evals, window=config.window)
    # evals = multi_deepwalk_filter(evals, min_hop=3, max_hop=config.window)
    X = U @ np.diag(eigvs ** 0.5)
    # X = X[:m-n, :]
    X = normalize(X, norm='l2', axis=1)
    if X.shape[1] < dim:
        padding = np.zeros((X.shape[0], dim-X.shape[1]))
        X = np.hstack((X, padding))
    elif X.shape[1] > dim:
        X = X[:, :dim]
    embed_time = time.time()
    if config.verbose:
        print("Embedding computation time:", embed_time-eigsh_time)
    return X

def hgmf_proximity_base(H, dim=256, hyperedge_weights=None, rank=32, window=10, negative=1.0):
    if config.verbose:
        print(f"parameter test: rank {config.svd_rank} window {config.window} svd_tol {config.svd_tol} negative {config.negative} beta {config.beta}")
    start_time = time.time()
    vol = float(H.sum())
    m = H.shape[0]
    n = H.shape[1]
    if hyperedge_weights is None:
        deg_vec = H.sum(axis=0).A1
    else:
        deg_vec = (H.T@hyperedge_weights[:,np.newaxis]).flatten()
    deg_vec[deg_vec==0] = 1
    deg_alpha = deg_vec**(-1)
    D_v = sparse.diags(deg_alpha, format='csr')
    deg_edge = H.sum(axis=1).A1
    deg_edge[deg_edge==0] = 1
    D_e = sparse.diags(deg_edge**(-1))
    if hyperedge_weights is not None:
        D_e = sparse.diags(hyperedge_weights) @ D_e
    H = H.toarray()
    D_e = D_e.toarray()
    D_v = D_v.toarray()
    P = (H.T @ D_e @ H @ D_v)
    L = P
    # deg_vec1 = (H.T@hyperedge_weights[:,np.newaxis]).flatten()
    # WDerH = np.diag(hyperedge_weights)@np.diag(H.sum(1)**-1)@H
    for i in range(window-1):
        L = L + P @ L
    L = (D_v * (vol/window)) @ L
    lap_time = time.time()
    if config.verbose:
        print("Laplacian computation time:", lap_time-start_time)
    Y = np.log(np.maximum(L, 1.))
    log_time = time.time()
    if config.verbose:
        print("Log computation time:", log_time-lap_time)
    eigvs, U = sparse.linalg.eigsh(Y, min(dim, n//2), which='LA')
    eigsh_time = time.time()
    if config.verbose:
        print("EIGSH computation time:", eigsh_time-log_time)
    logger.info("Maximum eigenvalue %f, minimum eigenvalue %f", np.max(eigvs), np.min(eigvs))
    # evals = deepwalk_filter(evals, window=config.window)
    # evals = multi_deepwalk_filter(evals, min_hop=3, max_hop=config.window)
    X = U @ np.diag(eigvs ** 0.5)
    X = normalize(X, norm='l2', axis=1)
    if X.shape[1] < dim:
        padding = np.zeros((X.shape[0], dim-X.shape[1]))
        X = np.hstack((X, padding))
    elif X.shape[1] > dim:
        X = X[:, :dim]
    embed_time = time.time()
    if config.verbose:
        print("Embedding computation time:", embed_time-eigsh_time)
    return X

def hgmf_proximity_nolog(H, dim=256, hyperedge_weights=None, rank=32, window=10, negative=1.0):
    if config.verbose:
        print(f"parameter test: rank {config.svd_rank} window {config.window} svd_tol {config.svd_tol} negative {config.negative} beta {config.beta}")
    start_time = time.time()
    vol = float(H.sum())
    m = H.shape[0]
    n = H.shape[1]
    if hyperedge_weights is None:
        deg_vec = H.sum(axis=0).A1
    else:
        deg_vec = (H.T@hyperedge_weights[:,np.newaxis]).flatten()
    deg_vec[deg_vec==0] = 1
    deg_alpha = deg_vec**(-1)
    D_v = sparse.diags(deg_alpha, format='csr')
    deg_edge = H.sum(axis=1).A1
    deg_edge[deg_edge==0] = 1
    D_e = sparse.diags(deg_edge**(-1))
    if hyperedge_weights is not None:
        D_e = sparse.diags(hyperedge_weights) @ D_e
    H = H.toarray()
    D_e = D_e.toarray()
    D_v = D_v.toarray()
    P = (H.T @ D_e @ H @ D_v)
    L = P
    # deg_vec1 = (H.T@hyperedge_weights[:,np.newaxis]).flatten()
    # WDerH = np.diag(hyperedge_weights)@np.diag(H.sum(1)**-1)@H
    for i in range(window-1):
        L = L + P @ L
    L = (D_v * (vol/window)) @ L
    lap_time = time.time()
    if config.verbose:
        print("Laplacian computation time:", lap_time-start_time)
    eigvs, U = sparse.linalg.eigsh(L, min(dim, n//2), which='LA')
    eigsh_time = time.time()
    logger.info("Maximum eigenvalue %f, minimum eigenvalue %f", np.max(eigvs), np.min(eigvs))
    # evals = deepwalk_filter(evals, window=config.window)
    # evals = multi_deepwalk_filter(evals, min_hop=3, max_hop=config.window)
    X = U @ np.diag(eigvs ** 0.5)
    X = normalize(X, norm='l2', axis=1)
    if X.shape[1] < dim:
        padding = np.zeros((X.shape[0], dim-X.shape[1]))
        X = np.hstack((X, padding))
    elif X.shape[1] > dim:
        X = X[:, :dim]
    embed_time = time.time()
    if config.verbose:
        print("Embedding computation time:", embed_time-eigsh_time)
    return X

def hgmf_proximity_eiglogeig(H, dim=256, hyperedge_weights=None, rank=32, window=10, negative=1.0):
    if config.verbose:
        print(f"parameter test: rank {config.svd_rank} window {config.window} svd_tol {config.svd_tol} negative {config.negative} beta {config.beta}")
    start_time = time.time()
    vol = float(H.sum())
    m = H.shape[0]
    n = H.shape[1]
    if hyperedge_weights is None:
        deg_vec = H.sum(axis=0).A1
    else:
        deg_vec = (H.T@hyperedge_weights[:,np.newaxis]).flatten()
    deg_vec[deg_vec==0] = 1
    deg_alpha = deg_vec**(-1)
    D_v = sparse.diags(deg_alpha, format='csr')
    deg_edge = H.sum(axis=1).A1
    deg_edge[deg_edge==0] = 1
    D_e = sparse.diags(deg_edge**(-1))
    if hyperedge_weights is not None:
        D_e = sparse.diags(hyperedge_weights) @ D_e
    H = H.toarray()
    D_e = D_e.toarray()
    D_v = D_v.toarray()
    P = (H.T @ D_e @ H @ D_v)
    L = P
    # deg_vec1 = (H.T@hyperedge_weights[:,np.newaxis]).flatten()
    # WDerH = np.diag(hyperedge_weights)@np.diag(H.sum(1)**-1)@H
    for i in range(window-1):
        L = L + P @ L
    L = (D_v * (vol/window)) @ L
    lap_time = time.time()
    if config.verbose:
        print("Laplacian computation time:", lap_time-start_time)
    eigvs, X = sparse.linalg.eigsh(L, min(32, n//2), which='LA')
    Y = np.log(np.maximum(X @ (np.diag(eigvs) @ X.T), 1.))
    log_time = time.time()
    if config.verbose:
        print("Log computation time:", log_time-lap_time)
    eigvs, U = sparse.linalg.eigsh(Y, min(dim, n//2), which='LA')
    eigsh_time = time.time()
    if config.verbose:
        print("EIGSH computation time:", eigsh_time-log_time)
    logger.info("Maximum eigenvalue %f, minimum eigenvalue %f", np.max(eigvs), np.min(eigvs))
    # evals = deepwalk_filter(evals, window=config.window)
    # evals = multi_deepwalk_filter(evals, min_hop=3, max_hop=config.window)
    X = U @ np.diag(eigvs ** 0.5)
    X = normalize(X, norm='l2', axis=1)
    if X.shape[1] < dim:
        padding = np.zeros((X.shape[0], dim-X.shape[1]))
        X = np.hstack((X, padding))
    elif X.shape[1] > dim:
        X = X[:, :dim]
    embed_time = time.time()
    if config.verbose:
        print("Embedding computation time:", embed_time-eigsh_time)
    return X

def hgmf_rwr_1svd(H, dim=256, hyperedge_weights=None, rank=32, window=10, negative=1.0):
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
    D_v = sparse.diags(deg_alpha, format='csr')
    deg_edge = H.sum(axis=1).A1
    deg_edge[deg_edge==0] = 1
    D_e = sparse.diags(deg_edge**(-0.5))
    if hyperedge_weights is not None:
        D_e = sparse.diags(hyperedge_weights**0.5) @ D_e
    L = D_e @ H @ D_v
    lap_time = time.time()
    if config.verbose:
        print("Laplacian computation time:", lap_time-start_time)
    logger.info("SVD decomposition...")
    U, s, VT = sparse.linalg.svds(L, min(config.svd_rank, n//2), tol=config.svd_tol)
    svd_time = time.time()
    if config.verbose:
        print("SVD computation time:", svd_time-lap_time)
    V = VT.T
    evals = s**2
    if config.plot_svd:
        U, s, VT = sparse.linalg.svds(L, 256, tol=config.svd_tol)
        import matplotlib.pyplot as plt
        plt.plot(s**2)
        plt.show()
    logger.info("Maximum eigenvalue %f, minimum eigenvalue %f", np.max(s), np.min(s))
    D_rt_invU = D_v.dot(V)
    evals = rwr_filter(evals)
    X = sparse.diags(np.sqrt(evals)).dot(D_rt_invU.T).T
    scalar = np.sqrt(vol/config.negative)
    # scalar = 1.
    # X = X * scalar
    X = normalize(X, norm='l2', axis=1)
    if X.shape[1] < dim:
        padding = np.zeros((X.shape[0], dim-X.shape[1]))
        X = np.hstack((X, padding))
    elif X.shape[1] > dim:
        X = X[:, :dim]
    embed_time = time.time()
    if config.verbose:
        print("Embedding computation time:", embed_time-svd_time)
    return X

def hgmf_1eigsh(H, dim=256, hyperedge_weights=None, rank=32, window=10, negative=1.0):
    if config.verbose:
        print(f"parameter test: rank {config.svd_rank} window {config.window} svd_tol {config.svd_tol} negative {config.negative} beta {config.beta}")
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
    D_v = sparse.diags(deg_alpha, format='csr')
    deg_edge = H.sum(axis=1).A1
    deg_edge[deg_edge==0] = 1
    D_e = sparse.diags(deg_edge**(-0.5))
    if hyperedge_weights is not None:
        D_e = sparse.diags(hyperedge_weights**0.5) @ D_e
    L = D_e @ H @ D_v
    lap_time = time.time()
    if config.verbose:
        print("Laplacian computation time:", lap_time-start_time)
    logger.info("SVD decomposition...")
    def laplacian_operator(x):
        return L.T @ (L @ x)
    lap_lo = sparse.linalg.LinearOperator((n, n), matvec=laplacian_operator)
    evals, V = sparse.linalg.eigsh(lap_lo, min(config.svd_rank, n//2), tol=config.svd_tol,which='LA')
    s = np.sqrt(evals)
    # U, s, VT = sparse.linalg.svds(L, min(config.svd_rank, n//2), tol=config.svd_tol)
    svd_time = time.time()
    if config.verbose:
        print("Eigsh computation time:", svd_time-lap_time)
    # V = VT.T
    # evals = s**2
    logger.info("Maximum eigenvalue %f, minimum eigenvalue %f", np.max(s), np.min(s))
    D_rt_invU = D_v.dot(V)
    evals = deepwalk_filter(evals, window=config.window)
    X = sparse.diags(np.sqrt(evals)).dot(D_rt_invU.T).T
    scalar = np.sqrt(vol/config.negative)
    # scalar = 1.
    # X = X * scalar
    X = normalize(X, norm='l2', axis=1)
    if X.shape[1] < dim:
        padding = np.zeros((X.shape[0], dim-X.shape[1]))
        X = np.hstack((X, padding))
    elif X.shape[1] > dim:
        X = X[:, :dim]
    embed_time = time.time()
    if config.verbose:
        print("Embedding computation time:", embed_time-svd_time)
    return X

def hgmf(H, dim=256, hyperedge_weights=None, rank=32, window=10, negative=1.0):
    from sklearn.utils.extmath import randomized_svd
    vol = float(H.sum())
    # if hyperedge_weights is None:
    #     vol = float(H.sum())
    # else:
    #     vol = (H.T @ hyperedge_weights[:, np.newaxis]).sum()
    m = H.shape[0]
    n = H.shape[1]
    if hyperedge_weights is None:
        deg_vec = H.sum(axis=0).A1
    else:
        deg_vec = (H.T@hyperedge_weights[:,np.newaxis]).flatten()
    deg_vec[deg_vec==0] = 1
    deg_alpha = deg_vec**(-0.5)
    # L, d_rt = csgraph.laplacian(H, normed=True, return_diag=True)
    D_v = sparse.diags(deg_alpha, format='csr')
    deg_edge = H.sum(axis=1).A1
    deg_edge[deg_edge==0] = 1
    D_e = sparse.diags(deg_edge**(-0.5))
    if hyperedge_weights is not None:
        D_e = sparse.diags(hyperedge_weights**0.5) @ D_e
    L = D_e @ H @ D_v
    logger.info("SVD decomposition...")
    U, s, VT = sparse.linalg.svds(L, min(rank, n//2))
    V = VT.T
    evals = s**2
    logger.info("Maximum eigenvalue %f, minimum eigenvalue %f", np.max(s), np.min(s))
    D_rt_invU = D_v.dot(V)
    # approximate deepwalk matrix
    # deepwalk_matrix = approximate_deepwalk_matrix(evals, D_rt_invU, window=window, vol=vol, b=negative)
    evals = deepwalk_filter(evals, window=window)
    X = sparse.diags(np.sqrt(evals)).dot(D_rt_invU.T).T
    # Y = np.log(np.maximum((X @ X.T) * vol / b, 1.))
    # Y = (X @ X.T) * vol / negative
    # factorize deepwalk matrix with SVD
    # deepwalk_embedding = svd_deepwalk_matrix(deepwalk_matrix, dim=dim)
    scalar = np.sqrt(vol/negative)
    return (X * scalar)
    # scalar = 1
    # u, s, v = randomized_svd(X*scalar,dim,n_oversamples=1,n_iter=1)
    u, s, v = sparse.linalg.svds(X*scalar, min(dim, rank-1))
    # u = D_rt_invU
    # deepwalk_embedding = sparse.diags(np.sqrt(s)).dot(u.T).T
    # deepwalk_embedding = sparse.diags((s)).dot(u.T).T
    # return deepwalk_embedding

def hgmf_pts(H, dim=256, hyperedge_weights=None, rank=32, window=10, negative=1.0):
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
    D_v = sparse.diags(deg_alpha, format='csr')
    deg_edge = H.sum(axis=1).A1
    deg_edge[deg_edge==0] = 1
    D_e = sparse.diags(deg_edge**(-0.5))
    if hyperedge_weights is not None:
        D_e = sparse.diags(hyperedge_weights**0.5) @ D_e
    L = D_e @ H @ D_v
    lap_time = time.time()
    if config.verbose:
        print("Laplacian computation time:", lap_time-start_time)
    logger.info("SVD decomposition...")
    U, s, VT = sparse.linalg.svds(L, min(config.svd_rank, n//2), tol=config.svd_tol)
    svd_time = time.time()
    if config.verbose:
        print("SVD computation time:", svd_time-lap_time)
    V = VT.T
    evals = s**2
    logger.info("Maximum eigenvalue %f, minimum eigenvalue %f", np.max(s), np.min(s))
    D_rt_invU = D_v.dot(V)
    # evals = rwr_filter(evals)
    evals = deepwalk_filter(evals)
    U = sparse.diags(np.sqrt(evals)).dot(D_rt_invU.T).T
    # U = sparse.diags(evals**2).dot(D_rt_invU.T).T
    from pts import poly_tensor_sketch, poly_fit, poly_fit_lsq, poly_fit_minimize
    # coeffs = poly_fit(lambda x: np.log(x) if x > 1 else 0, deg=3, domain=(0, 3))
    # coeffs = poly_fit_minimize(lambda x: np.log(x) if x > 1 else 0, deg=3, domain=(0, 3), visualize=False)
    coeffs = poly_fit_lsq(lambda x: np.log(x) if x > 1 else 0, deg=1, domain=(0, 3), visualize=False)
    # coeffs = poly_fit_lsq(lambda x: np.exp(x/2), deg=3, domain=(0, 3), visualize=False)
    Z, _ = poly_tensor_sketch(U, U, sketch_dim=config.sketch_dim, coeffs=coeffs)
    X, s, v = sparse.linalg.svds(Z, dim)
    # scalar = np.sqrt(vol/config.negative)
    # scalar = 1.
    # X = X * scalar
    X = normalize(X, norm='l2', axis=1)
    if X.shape[1] < dim:
        padding = np.zeros((X.shape[0], dim-X.shape[1]))
        X = np.hstack((X, padding))
    elif X.shape[1] > dim:
        X = X[:, :dim]
    embed_time = time.time()
    if config.verbose:
        print("Embedding computation time:", embed_time-svd_time)
    return X

def hgmf_asym_pts(H, dim=256, hyperedge_weights=None, rank=32, window=10, negative=1.0):
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
    D_v = sparse.diags(deg_alpha, format='csr')
    deg_edge = H.sum(axis=1).A1
    deg_edge[deg_edge==0] = 1
    D_e = sparse.diags(deg_edge**(-0.5))
    if hyperedge_weights is not None:
        D_e = sparse.diags(hyperedge_weights**0.5) @ D_e
    L = D_e @ H @ D_v
    lap_time = time.time()
    if config.verbose:
        print("Laplacian computation time:", lap_time-start_time)
    logger.info("SVD decomposition...")
    U, s, VT = sparse.linalg.svds(L, min(config.svd_rank, n//2), tol=config.svd_tol)
    svd_time = time.time()
    if config.verbose:
        print("SVD computation time:", svd_time-lap_time)
    V = VT.T
    evals = s**2
    logger.info("Maximum eigenvalue %f, minimum eigenvalue %f", np.max(s), np.min(s))
    D_rt_invU = D_v.dot(V)
    # evals = rwr_filter(evals)
    evals = deepwalk_filter(evals)
    U = sparse.diags(np.sqrt(evals)).dot(D_rt_invU.T).T
    # U = sparse.diags(evals**2).dot(D_rt_invU.T).T
    sample_x = None
    U*=np.sqrt(vol/negative)
    if config.fit_sample > 0:
        sample_ids = np.random.choice(n, config.fit_sample, replace=False)
        sample_x = (U[sample_ids] @ U.T).flatten()
    from pts import poly_tensor_sketch, poly_fit, poly_fit_lsq, poly_fit_minimize, asym_poly_tensor_sketch, polynomial_fit, sym_poly_tensor_sketch
    if config.func_name == 'tlog':
        # coeffs = poly_fit_lsq(lambda x: np.log(x) if x > 1 else 0, nonnegative=False, deg=config.poly_deg, domain=(0, 3), sample=sample_x, visualize=False)
        # coeffs = poly_fit_minimize(lambda x: np.log(x) if x > 1 else 0, nonnegative=False, deg=config.poly_deg, domain=(0, 3), sample=sample_x, visualize=True)
        # coeffs = poly_fit(lambda x: np.log(x) if x > 1 else 0, deg=config.poly_deg, domain=(0, 3), sample=sample_x, visualize=False)
        coeffs = polynomial_fit(lambda x: np.log(x) if x > 1 else 0, deg=config.poly_deg, domain=(0, 3), sample=sample_x, visualize=True)
    elif config.func_name == 'relu':
        coeffs = poly_fit_lsq(lambda x: np.maximum(x-1., 0.) if x > 1 else 0, nonnegative=False, deg=config.poly_deg, domain=(0, 3), sample=sample_x, visualize=False)
    elif config.func_name == 'tanh':
        coeffs = poly_fit_lsq(lambda x: np.tanh(x-1) if x > 1 else 0, nonnegative=False, deg=config.poly_deg, domain=(0, 3), sample=sample_x, visualize=False)
    elif config.func_name == 'sigmoid':
        coeffs = poly_fit_lsq(lambda x: 1/(1 + np.exp(2-x)), nonnegative=False, deg=config.poly_deg, domain=(0, 3), sample=sample_x, visualize=False)
    # Y = U @ U.T
    # coeffs = poly_fit_lsq(lambda x: np.log(x) if x > 1 else 0, deg=3, sample=np.sort(Y[:, :].flatten()), nonnegative=False, visualize=False)
    # coeffs = poly_fit_lsq(lambda x: np.exp(x/2), deg=3, domain=(0, 3), visualize=False)
    if config.verbose:
        print(coeffs)
        test_ids = np.random.choice(n,config.fit_sample,replace=False)
        test_x = (U[sample_ids]@U.T).flatten()
        func = np.vectorize(lambda x: np.log(x) if x > 1 else 0, otypes=[np.float64])
        test_y = func(test_x)
        test_lsq = np.polyval(coeffs[::-1],test_x)
        # test_lsq = np.polyval(coeffs,test_x)
        error = np.abs(test_lsq-test_y).sum()/len(test_x)
        mse = np.square(test_lsq-test_y).sum()/len(test_x)
        print(f"Mean absolute error: {error}")
        print(f"Mean squared error: {mse}")
    TU, TV = asym_poly_tensor_sketch(U, U, sketch_dim=config.sketch_dim, coeffs=coeffs)
    # TU, C = sym_poly_tensor_sketch(U, coeffs=coeffs, sketch_dim=config.sketch_dim)
    # TV = TU @ C
    def laplacian_operator(x):
        return TU @ (TV.T @ x)
    sketch_lo = sparse.linalg.LinearOperator((n, n), matvec=laplacian_operator)
    try:
        evals, X = sparse.linalg.eigsh(sketch_lo, min(config.svd_rank, n//2), tol=config.svd_tol,which='LM')
        # evals, X = sparse.linalg.eigsh(sketch_lo, min(config.svd_rank, n//2), tol=config.svd_tol,which='LA', maxiter=1000)
    except sparse.linalg.ArpackError as e:
        print(e)
        evals, X = e.eigenvalues, e.eigenvectors
    # X = X @ np.diag(np.sqrt(evals))
    # X = X @ np.diag(np.sqrt(np.abs(evals)))
    # Xu, su, vu = sparse.linalg.svds(TU, dim//2)
    # Xv, sv, vv = sparse.linalg.svds(TV, dim//2)
    # X = np.hstack((Xu, Xv))
    # X, su, vu = sparse.linalg.svds(TU, dim)
    # X = X @ np.diag(np.sqrt(su))
    # scalar = np.sqrt(vol/config.negative)
    # scalar = 1.
    # X = X * scalar
    X = normalize(X, norm='l2', axis=1)
    if X.shape[1] < dim:
        padding = np.zeros((X.shape[0], dim-X.shape[1]))
        X = np.hstack((X, padding))
    elif X.shape[1] > dim:
        X = X[:, :dim]
    embed_time = time.time()
    if config.verbose:
        print("Embedding computation time:", embed_time-svd_time)
    return X

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
    D_v = sparse.diags(deg_alpha, format='csr')
    deg_edge = H.sum(axis=1).A1
    deg_edge[deg_edge==0] = 1
    D_e = sparse.diags(deg_edge**(-0.5))
    if hyperedge_weights is not None:
        D_e = sparse.diags(hyperedge_weights**0.5) @ D_e
    L = D_e @ H @ D_v
    lap_time = time.time()
    if config.verbose:
        print("Laplacian computation time:", lap_time-start_time)
    logger.info("SVD decomposition...")

    spmv_count, spmv_count1 = 0, 0
    def linear_operator(x):
        nonlocal spmv_count
        spmv_count += 1
        return dot_product_mkl(L, x, cast=True)
    def linear_operator1(x):
        nonlocal spmv_count1
        spmv_count1 += 1
        return dot_product_mkl(L.T, x, cast=True)
    lap_lo = sparse.linalg.LinearOperator((m, n),matvec = linear_operator,rmatvec = linear_operator1)
    U, s, VT = sparse.linalg.svds(lap_lo, min(config.svd_rank, n//2), tol=config.svd_tol, random_state=config.rng)
    # print(hash(U.data.tobytes()))
    # U, s, VT = sparse.linalg.svds(L, min(config.svd_rank, n//2), tol=config.svd_tol)
    ## sklearn truncated svd
    # from sklearn.decomposition import TruncatedSVD
    # svd = TruncatedSVD(n_components=min(config.svd_rank, n//2), n_oversamples=15, n_iter=20)
    # U = svd.fit_transform(L)
    # s = svd.singular_values_
    # VT = svd.components_
    if config.verbose:
        print(f"spmv count: {spmv_count}, {spmv_count1}")
    svd_time = time.time()

    if config.test_time:
        print("SVD computation time:", svd_time-lap_time)
    V = VT.T
    evals = s**2
    logger.info("Maximum eigenvalue %f, minimum eigenvalue %f", np.max(s), np.min(s))
    D_rt_invU = D_v.dot(V)
    evals = rwr_filter(evals)
    # evals = deepwalk_filter(evals)
    # evals = multi_deepwalk_filter(evals, min_hop=5, max_hop=config.window)
    U = sparse.diags(np.sqrt(evals)).dot(D_rt_invU.T).T
    # U = sparse.diags(evals**2).dot(D_rt_invU.T).T
    sample_x = None
    U*=np.sqrt(vol/negative)
    from pts import full_pts, sym_poly_tensor_sketch, sym_poly_tensor_sketch_simple
    coeffs = full_pts(U, visualize=False)
    # TU, C = sym_poly_tensor_sketch(U, coeffs=coeffs, sketch_dim=config.sketch_dim)
    pts_start=time.time()
    TU, C = sym_poly_tensor_sketch_simple(U, coeffs=coeffs, sketch_dim=config.sketch_dim)
    TV = TU @ C
    if config.test_time:
        print("PTS computation time:", time.time()-pts_start)
    if config.test_error and n<10000:
        # print(f"Coefficients: {coeffs}")
        from sklearn.feature_selection import r_regression
        test_x = (U@U.T).flatten()
        func = np.vectorize(lambda x: np.log(x) if x > 1 else 0, otypes=[np.float64])
        test_y = func(test_x)
        poly_y = np.polyval(coeffs[::-1],test_x)
        poly_mae = np.abs(poly_y-test_y).sum()/len(test_x)
        poly_mse = np.square(poly_y-test_y).sum()/len(test_x)
        poly_r = r_regression(poly_y.reshape([-1, 1]), test_y)
        config.poly_mae_error.append(poly_mae)
        config.poly_mse_error.append(poly_mse)
        config.poly_pearson_r.append(poly_r[0])
        # print(f"Polynomial mean absolute error: {poly_mae}")
        # print(f"Polynomial mean squared error: {poly_mse}")
        # print(f"Polynomial pearson's r: {poly_r[0]}")
        sketch_y = (TU @ TV.T).flatten()
        # test_lsq = np.polyval(coeffs,test_x)
        mae = np.abs(sketch_y-test_y).sum()/len(test_x)
        mse = np.square(sketch_y-test_y).sum()/len(test_x)
        sketch_r = r_regression(sketch_y.reshape([-1, 1]), test_y)
        config.mae_error.append(mae)
        config.mse_error.append(mse)
        config.pearson_r.append(sketch_r[0])
        # print(f"Full SKetch mean absolute error: {mae}")
        # print(f"Full SKetch mean squared error: {mse}")
        # print(f"Full SKetch pearson's r: {sketch_r[0]}")
    #to test with dot_product_mkl
    def laplacian_operator(x):
        return TU @ (TV.T @ x)
        # TV_x = TV.T@ x
        # return dot_product_mkl(TU,TV_x,cast=True)
    sketch_lo = sparse.linalg.LinearOperator((n, n), matvec=laplacian_operator)
    try:
        evals, X = sparse.linalg.eigsh(sketch_lo, min(dim, n//2), tol=config.svd_tol,which='LM')
        # evals, X = sparse.linalg.eigsh(sketch_lo, min(config.svd_rank, n//2), tol=config.svd_tol,which='LA', maxiter=1000)
    except sparse.linalg.ArpackError as e:
        print(e)
        evals, X = e.eigenvalues, e.eigenvectors
    X = normalize(X, norm='l2', axis=1)
    embed_time = time.time()
    if config.verbose:
        print("Embedding computation time:", embed_time-svd_time)
    return X

def hgmf_asym_normx_pts(H, dim=256, hyperedge_weights=None, rank=32, window=10, negative=1.0):
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
    D_v = sparse.diags(deg_alpha, format='csr')
    deg_edge = H.sum(axis=1).A1
    deg_edge[deg_edge==0] = 1
    D_e = sparse.diags(deg_edge**(-0.5))
    if hyperedge_weights is not None:
        D_e = sparse.diags(hyperedge_weights**0.5) @ D_e
    L = D_e @ H @ D_v
    lap_time = time.time()
    if config.verbose:
        print("Laplacian computation time:", lap_time-start_time)
    logger.info("SVD decomposition...")
    U, s, VT = sparse.linalg.svds(L, min(config.svd_rank, n//2), tol=config.svd_tol)
    svd_time = time.time()
    if config.verbose:
        print("SVD computation time:", svd_time-lap_time)
    V = VT.T
    evals = s**2
    logger.info("Maximum eigenvalue %f, minimum eigenvalue %f", np.max(s), np.min(s))
    D_rt_invU = D_v.dot(V)
    # evals = rwr_filter(evals)
    evals = deepwalk_filter(evals)
    U = sparse.diags(np.sqrt(evals)).dot(D_rt_invU.T).T
    # U = sparse.diags(evals**2).dot(D_rt_invU.T).T
    from pts import poly_tensor_sketch, poly_fit, poly_fit_lsq, poly_fit_minimize, asym_poly_tensor_sketch
    # coeffs = poly_fit(lambda x: np.log(x) if x > 1 else 0, deg=3, domain=(0, 3))
    # coeffs = poly_fit_minimize(lambda x: np.log(x) if x > 1 else 0, deg=3, domain=(0, 3), visualize=False)
    # coeffs = poly_fit_lsq(lambda x: np.log(x) if x > 1 else 0, deg=10, domain=(-1, 3), nonnegative=False, visualize=False)
    # U*=np.sqrt(vol/negative)
    U = normalize(U, norm='l2', axis=1)
    Y = U @ U.T
    # coeffs = poly_fit_lsq(lambda x: np.log(x) if x > 1 else 0, deg=3, sample=np.sort(Y[:, :].flatten()), nonnegative=False, visualize=False)
    coeffs = poly_fit_lsq(lambda x: np.log(10.0*x) if x > 0.1 else 0, deg=3, sample=np.sort(Y[:, :].flatten()), nonnegative=False, visualize=False)
    # coeffs = poly_fit_lsq(lambda x: np.exp(x/2), deg=3, domain=(0, 3), visualize=False)
    TU, TV = asym_poly_tensor_sketch(U, U, sketch_dim=dim, coeffs=coeffs)
    Xu, su, vu = sparse.linalg.svds(TU, dim//2)
    Xv, sv, vv = sparse.linalg.svds(TV, dim//2)
    X = np.hstack((Xu, Xv))
    # X, su, vu = sparse.linalg.svds(TU, dim)
    # scalar = np.sqrt(vol/config.negative)
    # scalar = 1.
    # X = X * scalar
    X = normalize(X, norm='l2', axis=1)
    if X.shape[1] < dim:
        padding = np.zeros((X.shape[0], dim-X.shape[1]))
        X = np.hstack((X, padding))
    elif X.shape[1] > dim:
        X = X[:, :dim]
    embed_time = time.time()
    if config.verbose:
        print("Embedding computation time:", embed_time-svd_time)
    return X

def hgmf_nystroem(H, dim=256, hyperedge_weights=None, rank=32, window=10, negative=1.0):
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
    D_v = sparse.diags(deg_alpha, format='csr')
    deg_edge = H.sum(axis=1).A1
    deg_edge[deg_edge==0] = 1
    D_e = sparse.diags(deg_edge**(-0.5))
    if hyperedge_weights is not None:
        D_e = sparse.diags(hyperedge_weights**0.5) @ D_e
    L = D_e @ H @ D_v
    lap_time = time.time()
    if config.verbose:
        print("Laplacian computation time:", lap_time-start_time)
    logger.info("SVD decomposition...")
    U, s, VT = sparse.linalg.svds(L, min(config.svd_rank, n//2), tol=config.svd_tol)
    svd_time = time.time()
    if config.verbose:
        print("SVD computation time:", svd_time-lap_time)
    V = VT.T
    evals = s**2
    logger.info("Maximum eigenvalue %f, minimum eigenvalue %f", np.max(s), np.min(s))
    D_rt_invU = D_v.dot(V)
    # evals = rwr_filter(evals)
    evals = deepwalk_filter(evals)
    U = sparse.diags(np.sqrt(evals)).dot(D_rt_invU.T).T
    U*=np.sqrt(vol/negative)
    # U = normalize(U, norm='l2', axis=1)
    from sklearn.kernel_approximation import Nystroem
    def tlog_kernel(x, y):
        return np.log(max(1, x@y))
    # log_nystroem = Nystroem(kernel=tlog_kernel, n_components=2*dim)
    log_nystroem = Nystroem(kernel=config.kernel, n_components=dim)
    UU = log_nystroem.fit_transform(U)
    # UU, _ = np.linalg.qr(UU)
    # UU, su, vu = sparse.linalg.svds(U_n, dim)
    X = normalize(UU, norm='l2', axis=1)
    if X.shape[1] < dim:
        padding = np.zeros((X.shape[0], dim-X.shape[1]))
        X = np.hstack((X, padding))
    elif X.shape[1] > dim:
        X = X[:, :dim]
    embed_time = time.time()
    if config.verbose:
        print("Embedding computation time:", embed_time-svd_time)
    return X

def hgmf_sketchne(H, dim=256, hyperedge_weights=None, rank=32, window=10, negative=1.0):
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
    D_v = sparse.diags(deg_alpha, format='csr')
    deg_edge = H.sum(axis=1).A1
    deg_edge[deg_edge==0] = 1
    D_e = sparse.diags(deg_edge**(-0.5))
    if hyperedge_weights is not None:
        D_e = sparse.diags(hyperedge_weights**0.5) @ D_e
    L = D_e @ H @ D_v
    lap_time = time.time()
    if config.verbose:
        print("Laplacian computation time:", lap_time-start_time)
    logger.info("SVD decomposition...")
    U, s, VT = sparse.linalg.svds(L, min(config.svd_rank, n//2), tol=config.svd_tol)
    svd_time = time.time()
    if config.verbose:
        print("SVD computation time:", svd_time-lap_time)
    V = VT.T
    evals = s**2
    logger.info("Maximum eigenvalue %f, minimum eigenvalue %f", np.max(s), np.min(s))
    D_rt_invU = D_v.dot(V)
    evals = rwr_filter(evals)
    U = sparse.diags(np.sqrt(evals)).dot(D_rt_invU.T).T
    from sketchne import sketch_svds
    X = sketch_svds(U, U.T, dim=dim, s1=100, s2=1000, eta1=8, eta2=8, normalize=True)
    # scalar = np.sqrt(vol/config.negative)
    # scalar = 1.
    # X = X * scalar
    # X = normalize(X, norm='l2', axis=1)
    if X.shape[1] < dim:
        padding = np.zeros((X.shape[0], dim-X.shape[1]))
        X = np.hstack((X, padding))
    elif X.shape[1] > dim:
        X = X[:, :dim]
    embed_time = time.time()
    if config.verbose:
        print("Embedding computation time:", embed_time-svd_time)
    return X

def hgmf_mask_log(H, dim=256, hyperedge_weights=None, rank=32, window=10, negative=1.0, mask=None):
    if hyperedge_weights is None:
        vol = float(H.sum())
    else:
        vol = (H.T @ hyperedge_weights[:, np.newaxis]).sum()
    evals, D_rt_invU = approximate_normalized_graph_laplacian(H, hyperedge_weights=hyperedge_weights, rank=rank, which="LA")
    # approximate deepwalk matrix
    deepwalk_matrix = approximate_deepwalk_matrix(evals, D_rt_invU, window=window, vol=vol, b=negative)
    # factorize deepwalk matrix with SVD
    deepwalk_embedding = svd_deepwalk_matrix(masked_deepwalk_matrix, dim=dim)
    deepwalk_embedding = normalize(deepwalk_embedding, norm='l2', axis=1)
    return deepwalk_embedding

def hgsvdlogsvd(H, dim=256, hyperedge_weights=None, rank=32, window=10, negative=1.0):
    if hyperedge_weights is None:
        vol = float(H.sum())
    else:
        vol = (H.T @ hyperedge_weights[:, np.newaxis]).sum()
    evals, D_rt_invU = approximate_normalized_graph_laplacian(H, hyperedge_weights=hyperedge_weights, rank=config.svd_rank, which="LA")
    # approximate deepwalk matrix
    deepwalk_matrix = approximate_deepwalk_matrix(evals, D_rt_invU, window=window, vol=vol, b=negative)
    # factorize deepwalk matrix with SVD
    deepwalk_embedding = svd_deepwalk_matrix(deepwalk_matrix, dim=dim)
    # deepwalk_embedding = normalize(deepwalk_embedding, norm='l2', axis=1)
    return deepwalk_embedding

def hgsvddw(H, dim=256, hyperedge_weights=None, rank=32, window=10, negative=1.0):
    if hyperedge_weights is None:
        vol = float(H.sum())
    else:
        vol = (H.T @ hyperedge_weights[:, np.newaxis]).sum()
    evals, D_rt_invU = approximate_normalized_graph_laplacian(H, hyperedge_weights=hyperedge_weights, rank=rank, which="LA")
    # approximate deepwalk matrix
    deepwalk_embedding = deepwalk_no_log(evals, D_rt_invU, window=window, vol=vol, b=negative)
    # factorize deepwalk matrix with SVD
    # deepwalk_embedding = svd_deepwalk_matrix(deepwalk_matrix, dim=dim)
    return deepwalk_embedding

def hgsvd(H, dim=256, hyperedge_weights=None, rank=32, window=10, negative=1.0):
    if hyperedge_weights is None:
        vol = float(H.sum())
    else:
        vol = (H.T @ hyperedge_weights[:, np.newaxis]).sum()
    evals, D_rt_invU = approximate_normalized_graph_laplacian(H, hyperedge_weights=hyperedge_weights, rank=dim, which="LA")
    return D_rt_invU

def hgmf_edge_1svd(H, dim=256, hyperedge_weights=None, rank=32, window=10, negative=1.0):
    if config.verbose:
        print(f"parameter test: rank {config.svd_rank} window {config.window} svd_tol {config.svd_tol} negative {config.negative} beta {config.beta}")
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
    D_v = sparse.diags(deg_alpha, format='csr')
    deg_edge = H.sum(axis=1).A1
    deg_edge[deg_edge==0] = 1
    D_e = sparse.diags(deg_edge**(-0.5))
    if hyperedge_weights is not None:
        D_e = sparse.diags(hyperedge_weights**0.5) @ D_e
    L = D_e @ H @ D_v
    lap_time = time.time()
    if config.verbose:
        print("Laplacian computation time:", lap_time-start_time)
    logger.info("SVD decomposition...")
    def linear_operator(x):
        return dot_product_mkl(L, x, cast=True)
    def linear_operator1(x):
        return dot_product_mkl(L.T, x, cast=True)
    lap_lo = sparse.linalg.LinearOperator((m, n),matvec = linear_operator,rmatvec = linear_operator1)
    U, s, VT = sparse.linalg.svds(lap_lo, min(config.svd_rank, n//2), tol=config.svd_tol)
    # U, s, VT = sparse.linalg.svds(L, min(config.svd_rank, n//2), tol=config.svd_tol)
    svd_time = time.time()
    if config.verbose:
        print("SVD computation time:", svd_time-lap_time)
    evals = s**2
    logger.info("Maximum eigenvalue %f, minimum eigenvalue %f", np.max(s), np.min(s))
    D_rt_invV = D_e.dot(U)[:m-n]
    evals = rwr_filter(evals, window=config.window)
    # evals = deepwalk_filter(evals, window=config.window)
    # evals = multi_deepwalk_filter(evals, min_hop=3, max_hop=config.window)
    X = sparse.diags(np.sqrt(evals)).dot(D_rt_invV.T).T
    # X = sparse.diags(evals**2).dot(D_rt_invU.T).T # sqev
    scalar = np.sqrt(vol/config.negative)
    # scalar = 1.
    # X = X * scalar
    X = normalize(X, norm='l2', axis=1)
    # X = X[:m-n]
    if X.shape[1] < dim:
        padding = np.zeros((X.shape[0], dim-X.shape[1]))
        X = np.hstack((X, padding))
    elif X.shape[1] > dim:
        X = X[:, :dim]
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
    D_v = sparse.diags(deg_alpha, format='csr')
    deg_edge = H.sum(axis=1).A1
    deg_edge[deg_edge==0] = 1
    D_e = sparse.diags(deg_edge**(-0.5))
    if hyperedge_weights is not None:
        D_e = sparse.diags(hyperedge_weights**0.5) @ D_e
    L = D_e @ H @ D_v
    lap_time = time.time()
    if config.verbose:
        print("Laplacian computation time:", lap_time-start_time)
    logger.info("SVD decomposition...")

    spmv_count, spmv_count1 = 0, 0
    def linear_operator(x):
        nonlocal spmv_count
        spmv_count += 1
        return dot_product_mkl(L, x, cast=True)
    def linear_operator1(x):
        nonlocal spmv_count1
        spmv_count1 += 1
        return dot_product_mkl(L.T, x, cast=True)
    lap_lo = sparse.linalg.LinearOperator((m, n),matvec = linear_operator,rmatvec = linear_operator1)
    U, s, VT = sparse.linalg.svds(lap_lo, min(config.svd_rank, n//2), tol=config.svd_tol)
    # U, s, VT = sparse.linalg.svds(L, min(config.svd_rank, n//2), tol=config.svd_tol)
    if config.verbose:
        print(f"spmv count: {spmv_count}, {spmv_count1}")
    svd_time = time.time()

    if config.verbose:
        print("SVD computation time:", svd_time-lap_time)
    # V = VT.T
    evals = s**2
    logger.info("Maximum eigenvalue %f, minimum eigenvalue %f", np.max(s), np.min(s))
    # D_rt_invU = D_v.dot(V)
    D_rt_invU = D_e.dot(U)[:m-n]
    evals = rwr_filter(evals)
    # evals = deepwalk_filter(evals)
    # evals = multi_deepwalk_filter(evals, min_hop=5, max_hop=config.window)
    U = sparse.diags(np.sqrt(evals)).dot(D_rt_invU.T).T
    # U = sparse.diags(evals**2).dot(D_rt_invU.T).T
    sample_x = None
    U*=np.sqrt(vol/negative)
    from pts import full_pts, sym_poly_tensor_sketch, sym_poly_tensor_sketch_simple
    coeffs = full_pts(U, visualize=False)
    TU, C = sym_poly_tensor_sketch_simple(U, coeffs=coeffs, sketch_dim=config.sketch_dim)
    # TU, C = sym_poly_tensor_sketch(U, coeffs=coeffs, sketch_dim=config.sketch_dim)
    TV = TU @ C
    if config.test_error and n<10000:
        # print(f"Coefficients: {coeffs}")
        from sklearn.feature_selection import r_regression
        test_x = (U@U.T).flatten()
        func = np.vectorize(lambda x: np.log(x) if x > 1 else 0, otypes=[np.float64])
        test_y = func(test_x)
        poly_y = np.polyval(coeffs[::-1],test_x)
        poly_mae = np.abs(poly_y-test_y).sum()/len(test_x)
        poly_mse = np.square(poly_y-test_y).sum()/len(test_x)
        poly_r = r_regression(poly_y.reshape([-1, 1]), test_y)
        config.poly_mae_error.append(poly_mae)
        config.poly_mse_error.append(poly_mse)
        config.poly_pearson_r.append(poly_r[0])
        # print(f"Polynomial mean absolute error: {poly_mae}")
        # print(f"Polynomial mean squared error: {poly_mse}")
        # print(f"Polynomial pearson's r: {poly_r[0]}")
        sketch_y = (TU @ TV.T).flatten()
        # test_lsq = np.polyval(coeffs,test_x)
        mae = np.abs(sketch_y-test_y).sum()/len(test_x)
        mse = np.square(sketch_y-test_y).sum()/len(test_x)
        sketch_r = r_regression(sketch_y.reshape([-1, 1]), test_y)
        config.mae_error.append(mae)
        config.mse_error.append(mse)
        config.pearson_r.append(sketch_r[0])
        # print(f"Full SKetch mean absolute error: {mae}")
        # print(f"Full SKetch mean squared error: {mse}")
        # print(f"Full SKetch pearson's r: {sketch_r[0]}")
    #to test with dot_product_mkl
    def laplacian_operator(x):
        return TU @ (TV.T @ x)
        # TV_x = TV.T@ x
        # return dot_product_mkl(TU,TV_x,cast=True)
    sketch_lo = sparse.linalg.LinearOperator((m-n, m-n), matvec=laplacian_operator)
    try:
        evals, X = sparse.linalg.eigsh(sketch_lo, min(dim, (m-n)//2), tol=config.svd_tol,which='LM')
        # evals, X = sparse.linalg.eigsh(sketch_lo, min(config.svd_rank, n//2), tol=config.svd_tol,which='LA', maxiter=1000)
    except sparse.linalg.ArpackError as e:
        print(e)
        evals, X = e.eigenvalues, e.eigenvectors
    X = normalize(X, norm='l2', axis=1)
    embed_time = time.time()
    if config.verbose:
        print("Embedding computation time:", embed_time-svd_time)
    return X