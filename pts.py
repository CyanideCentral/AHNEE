import numpy as np
import scipy.sparse as sp
from scipy.fft import fft, ifft
import config

# Implementation of Polynomial Tensor Sketching (PTS)

def count_sketch_matrix(M, s):
    n, d = M.shape
    hash_indices = config.rng.integers(0, s, d)
    hash_signs = config.rng.choice([-1, 1], d)
    S = sp.csr_matrix((hash_signs, hash_indices, np.arange(d+1)), shape=(d, s))
    return S

def fit_coeffs(U, func):
    sample_ids = config.rng.choice(U.shape[0], config.fit_sample, replace=False)
    ps = np.full(config.fit_sample, U.shape[0]/config.fit_sample)
    U_sample = U[sample_ids]
    X = (U_sample @ U.T).flatten()
    X_vander = np.vander(X, config.poly_deg + 1, increasing=True)
    ps = np.concatenate([np.full(U.shape[0], p) for p in ps])
    Xt_D = X_vander.T * ps
    w = np.zeros(config.poly_deg + 1)
    U_norms = np.square(U).sum(axis=1)
    for i in range(1, config.poly_deg + 1):
        w[i] = np.sqrt(config.poly_deg*(2+3**i)*((U_norms**i).sum()**2)/config.sketch_dim)
    func = np.vectorize(func, otypes=[np.float64])
    Y = func(X)
    coeffs = np.linalg.inv(Xt_D @ X_vander + np.diag(w ** 2)) @ Xt_D @ Y
    return coeffs

def tensor_sketches(U, coeffs=[0.0002, 0.2319, 0.366, 0.001], sketch_dim=10):
    n, d = U.shape
    tu = [np.ones((n, 1), np.float64)]
    S = count_sketch_matrix(U, sketch_dim)
    U_sketch = U @ S
    tu.append(U_sketch)
    fu = fft(tu[-1])
    fu0 = fu
    c_vecs = [[coeffs[0]]]
    c_vecs.append(np.full(sketch_dim, coeffs[1]))
    for i in range(2, len(coeffs)):
        fu = fu0 * fu
        tu.append(ifft(fu).real)
        c_vecs.append(np.full(sketch_dim, coeffs[i]))
    c_diag = sp.diags(np.concatenate(c_vecs))
    tu = np.c_[*tu]
    return tu, c_diag

def pts_tlog(U):
    tlog_func = lambda x: np.log(x) if x > 1 else 0
    coeffs = fit_coeffs(U, func=tlog_func)
    return tensor_sketches(U, coeffs=coeffs, sketch_dim=config.sketch_dim)