import numpy as np
import scipy.sparse as sp
import time
import resource
import config
from sklearn.preprocessing import normalize
from sparse_dot_mkl import dot_product_mkl
from sahe import extend_hypergraph

def base(ahg):
    start_time = time.perf_counter()
    H, hyperedge_weights= extend_hypergraph(ahg, knn=config.knn)
    vol = float(H.sum())
    m, n = H.shape
    deg_vec = (H.T@hyperedge_weights[:,np.newaxis]).flatten()
    deg_vec[deg_vec==0] = 1
    deg_alpha = deg_vec**(-1)
    D_v = sp.diags(deg_alpha, format='csr')
    deg_edge = H.sum(axis=1).A1
    deg_edge[deg_edge==0] = 1
    D_e = sp.diags(deg_edge**(-1))
    if hyperedge_weights is not None:
        D_e = sp.diags(hyperedge_weights) @ D_e
    H_dense = H.toarray()
    D_e = D_e.toarray()
    D_v = D_v.toarray()
    P = (D_v @ H_dense.T @ D_e @ H_dense)
    L = np.eye(n)
    for i in range(config.window):
        L = config.alpha * L + (1-config.alpha) * L @ P
    L = L @ (D_v * vol)
    Y = np.log(np.maximum(L, 1.))
    eigvs, U = sp.linalg.eigsh(Y, min(config.embedding_dim, n//2), which='LA')
    X = U @ np.diag(eigvs ** 0.5)
    Z_v = normalize(X, norm='l2', axis=1)

    Hd = H.multiply(hyperedge_weights[:, np.newaxis])
    vol = float(Hd.sum())
    m = Hd.shape[0]
    n = Hd.shape[1]
    deg_vec = Hd.sum(axis=0).A1
    deg_vec[deg_vec==0] = 1
    deg_alpha = deg_vec**(-1)
    D_v = sp.diags(deg_alpha, format='csr')
    deg_edge = Hd.sum(axis=1).A1
    deg_edge[deg_edge==0] = 1
    D_e = sp.diags(deg_edge**(-1))
    Hd_dense = Hd.toarray()
    D_e = D_e.toarray()
    D_v = D_v.toarray()
    P = (D_e @ Hd_dense @ D_v @ Hd_dense.T)
    L = np.eye(m)
    for i in range(config.window):
        L = config.alpha * L + (1-config.alpha) * P @ L
    L = L @ (D_e * vol)
    L = L[:m-n,:m-n]
    Y = np.log(np.maximum(L, 1.))
    eigvs, U = sp.linalg.eigsh(Y, min(config.embedding_dim, (m-n)//2), which='LA')
    X = U @ np.diag(eigvs ** 0.5)
    Z_e = normalize(X, norm='l2', axis=1)
    return Z_v, Z_e, (time.perf_counter() - start_time, resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 / 1024)