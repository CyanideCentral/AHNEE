import numpy as np
import scipy.sparse as sp
from scipy.fft import fft, ifft
from scipy.optimize import minimize, lsq_linear
from numpy.polynomial.polynomial import Polynomial
import config

def count_sketch_matrix(M, s):
    n, d = M.shape
    # hash_indices = np.random.randint(0, s, d)
    hash_indices = config.rng.integers(0, s, d)
    # hash_signs = np.random.choice([-1, 1], d)
    hash_signs = config.rng.choice([-1, 1], d)
    S = sp.csr_matrix((hash_signs, hash_indices, np.arange(d+1)), shape=(d, s))
    return S

def plot_func(func=None):
    import matplotlib.pyplot as plt
    X = np.linspace(-1, 3, 1000)
    if func is None:
        coeffs=[0.0002, 0.2319, 0.366, 0.001]
        func = np.poly1d(coeffs[::-1])
    Y = np.vectorize(func, otypes=[np.float64])(X)
    plt.plot(X, Y)
    plt.show()

def greedy_k_center(U, k):
    n = U.shape[0]
    centers = np.zeros(k, dtype=int)
    dist_list = []
    centers[0] = np.random.randint(0, n)
    dists = np.linalg.norm(U - U[centers[0]], axis=1)
    dist_list.append(dists)
    for i in range(1, k):
        centers[i] = np.argmax(dists)
        new_dist = np.linalg.norm(U - U[centers[i]], axis=1)
        dist_list.append(new_dist)
        dists = np.minimum(dists, new_dist)
    center_dists = np.c_[*dist_list]
    center_assignments = np.argmin(center_dists, axis=1)
    _, counts = np.unique(center_assignments, return_counts=True)
    return centers, counts

def full_pts(U, visualize=False):
    if config.sampling == 'random':
        # sample_ids = np.random.choice(U.shape[0], config.fit_sample, replace=False)
        sample_ids = config.rng.choice(U.shape[0], config.fit_sample, replace=False)
        ps = np.full(config.fit_sample, U.shape[0]/config.fit_sample)
    elif config.sampling == 'kcenter':
        sample_ids, ps = greedy_k_center(U, config.fit_sample)
    else:
        raise ValueError(f"Unknown sampling method: {config.sampling}")
    U_sample = U[sample_ids]
    X = (U_sample @ U.T).flatten()
    X_vander = np.vander(X, config.poly_deg + 1, increasing=True)
    ps = np.concatenate([np.full(U.shape[0], p) for p in ps])
    Xt_D = X_vander.T * ps
    w = np.zeros(config.poly_deg + 1)
    U_norms = np.square(U).sum(axis=1)
    for i in range(1, config.poly_deg + 1):
        w[i] = np.sqrt(config.poly_deg*(2+3**i)*((U_norms**i).sum()**2)/config.sketch_dim)
    func = np.vectorize(config.func, otypes=[np.float64])
    Y = func(X)
    coeffs = np.linalg.inv(Xt_D @ X_vander + np.diag(w ** 2)) @ Xt_D @ Y
    if visualize:
        X.sort()
        Y = func(X)
        polynomial = np.poly1d(coeffs[::-1])
        y_fit = polynomial(X)
        import matplotlib.pyplot as plt
        plt.scatter(X, Y, label="Data", color="blue")
        plt.plot(X, y_fit, label="Fitted Polynomial", color="red")
        plt.legend()
        plt.show()
    return coeffs

def polynomial_fit(func, deg=5, domain=(-1, 1), visualize=False, sample=None):
    if sample is None:
        x = np.linspace(domain[0], domain[1], 1000)
    else:
        x = sample
    func = np.vectorize(func, otypes=[np.float64])
    y = func(x)
    series = Polynomial.fit(x, y, deg, w=np.ones_like(x))
    coeffs = series.convert().coef
    if visualize:
        # x_min, x_max = sample.min(), sample.max()
        # uniform_x = np.linspace(x_min, x_max, 1000)
        # x = np.concatenate([x, uniform_x])
        x.sort()
        y = func(x)
        polynomial = np.poly1d(coeffs[::-1])
        y_fit = polynomial(x)
        import matplotlib.pyplot as plt
        plt.scatter(x, y, label="Data", color="blue")
        plt.plot(x, y_fit, label="Fitted Polynomial", color="red")
        plt.legend()
        plt.show()
    return coeffs

def poly_fit(func, deg=5, domain=(-1, 1), visualize=False, sample=None):
    if sample is None:
        x = np.linspace(domain[0], domain[1], 1000)
    else:
        x = sample
    func = np.vectorize(func, otypes=[np.float64])
    y = func(x)
    coeffs = np.polyfit(x, y, deg)
    if visualize:
        x.sort()
        y = func(x)
        polynomial = np.poly1d(coeffs)
        y_fit = polynomial(x)
        import matplotlib.pyplot as plt
        plt.scatter(x, y, label="Data", color="blue")
        plt.plot(x, y_fit, label="Fitted Polynomial", color="red")
        plt.legend()
        plt.show()
    return coeffs[::-1]

def poly_fit_minimize(func, deg=5, domain=(0, 3), nonnegative=False, visualize=False, sample=None):
    def polynomial(x, coeffs):
        return sum(c * x**i for i, c in enumerate(coeffs))

    def objective(coeffs, x, y):
        return np.sum((polynomial(x, coeffs) - y)**2)

    if sample is None:
        x = np.linspace(domain[0], domain[1], 1000)
    else:
        x = sample
    x = x.astype(np.float128)
    func = np.vectorize(func, otypes=[np.float128])
    y = func(x)
    initial_guess = np.zeros(deg + 1)
    initial_guess[0:2] = 1.0
    constraints = [{'type': 'ineq', 'fun': lambda c: c}]
    result = minimize(objective, initial_guess, args=(x, y), constraints=(constraints if nonnegative else None))
    coefficients = result.x
    coefficients[coefficients < 0] = 0.0
    if visualize:
        x.sort()
        y = func(x)
        y_fit = polynomial(x, result.x)
        import matplotlib.pyplot as plt
        plt.scatter(x, y, label="Data", color="blue")
        plt.plot(x, y_fit, label="Fitted Polynomial", color="red")
        plt.legend()
        plt.show()
    return result.x

def poly_fit_lsq(func, deg=5, domain=(0, 3), nonnegative=False, visualize=False, sample=None):
    def poly_basis(x, deg):
        return np.vander(x, deg + 1, increasing=True)

    func = np.vectorize(func, otypes=[np.float64])
    if sample is None:
        x_data = np.linspace(domain[0], domain[1], 1000)
    else:
        x_data = sample
    y_data = func(x_data)

    A = poly_basis(x_data, deg)
    res = lsq_linear(A, y_data, bounds=((0, np.inf) if nonnegative else (-np.inf, np.inf)))
    coefficients = res.x
    if visualize:
        y_fit = np.dot(A, coefficients)
        x_min, x_max = x_data.min(), x_data.max()
        # x_min, x_max = -30, 600
        uniform_x = np.linspace(x_min, x_max, 1000)
        x_agg = np.concatenate([x_data, uniform_x])
        x_agg.sort()
        y_agg = poly_basis(x_agg, deg) @ coefficients
        import matplotlib.pyplot as plt
        plt.scatter(x_data, y_data, label="Data", color="blue")
        plt.plot(x_agg, y_agg, label="Fitted Polynomial", color="red")
        plt.legend()
        plt.show()
    return coefficients

def poly_tensor_sketch(U, V, coeffs=[0.0002, 0.2319, 0.366, 0.001], sketch_dim=10):
    # follow Alg. 2
    coeffs = np.sqrt(coeffs)
    n, d = U.shape
    tu = [np.ones((n, 1), np.float64)]
    tv = [np.ones((n, 1), np.float64)]
    S = count_sketch_matrix(U, sketch_dim)
    tu.append(U @ S)
    tv.append(V @ S)
    # tu.append(U)
    # tv.append(V)
    fu, fv = fft(tu[-1]), fft(tv[-1])
    for i in range(2, len(coeffs)):
        S = count_sketch_matrix(U, sketch_dim)
        fu, fv = fft(U @ S) * fu, fft(V @ S) * fv
        # fu, fv = fft(U) * fu, fft(V) * fv
        tu.append(ifft(fu).real)
        tv.append(ifft(fv).real)
    for i in range(0, len(coeffs)):
        tu[i] = tu[i] * coeffs[i]
    tu = np.c_[*tu]
    tv = np.c_[*tv]
    return tu, tv

def asym_poly_tensor_sketch(U, V, coeffs=[0.0002, 0.2319, 0.366, 0.001], sketch_dim=10):
    # follow Alg. 2
    # coeffs = np.sqrt(coeffs)
    n, d = U.shape
    tu = [np.ones((n, 1), np.float64)]
    tv = [np.ones((n, 1), np.float64)]
    S = count_sketch_matrix(U, sketch_dim)
    tu.append(U @ S)
    tv.append(V @ S)
    # tu.append(U)
    # tv.append(V)
    fu, fv = fft(tu[-1]), fft(tv[-1])
    # c_vecs = [coeffs[0]]
    for i in range(2, len(coeffs)):
        S = count_sketch_matrix(U, sketch_dim)
        fu, fv = fft(U @ S) * fu, fft(V @ S) * fv
        # fu, fv = fft(U) * fu, fft(V) * fv
        tu.append(ifft(fu).real)
        tv.append(ifft(fv).real)
        # c_vecs.append(np.full(sketch_dim, coeffs[i]))
    for i in range(0, len(coeffs)):
        tu[i] = tu[i] * coeffs[i]
    # c_diag = sp.diags(np.concatenate(c_vecs))
    tu = np.c_[*tu]
    tv = np.c_[*tv]
    return tu, tv

def sym_poly_tensor_sketch(U, coeffs=[0.0002, 0.2319, 0.366, 0.001], sketch_dim=10):
    # follow Alg. 2
    # coeffs = np.sqrt(coeffs)
    n, d = U.shape
    tu = [np.ones((n, 1), np.float64)]
    S = count_sketch_matrix(U, sketch_dim)
    tu.append(U @ S)
    # tu.append(U)
    fu = fft(tu[-1])
    c_vecs = [[coeffs[0]]]
    c_vecs.append(np.full(sketch_dim, coeffs[1]))
    for i in range(2, len(coeffs)):
        S = count_sketch_matrix(U, sketch_dim)
        fu = fft(U @ S) * fu
        tu.append(ifft(fu).real)
        c_vecs.append(np.full(sketch_dim, coeffs[i]))
    # for i in range(0, len(coeffs)):
    #     tu[i] = tu[i] * coeffs[i]
    # c_vecs = [c_vecs[0]] + c_vecs[2:]
    # tu = [tu[0]] + tu[2:]
    c_diag = sp.diags(np.concatenate(c_vecs))
    tu = np.c_[*tu]
    return tu, c_diag

def sym_poly_tensor_sketch_simple(U, coeffs=[0.0002, 0.2319, 0.366, 0.001], sketch_dim=10):
    # follow Alg. 2
    # coeffs = np.sqrt(coeffs)
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
        # S = count_sketch_matrix(U, sketch_dim)
        fu = fu0 * fu
        tu.append(ifft(fu).real)
        c_vecs.append(np.full(sketch_dim, coeffs[i]))
    # for i in range(0, len(coeffs)):
    #     tu[i] = tu[i] * coeffs[i]
    # c_vecs = [c_vecs[0]] + c_vecs[2:]
    # tu = [tu[0]] + tu[2:]
    c_diag = sp.diags(np.concatenate(c_vecs))
    tu = np.c_[*tu]
    return tu, c_diag

def fit_trunc_log(deg=10, matrix=None):
    # fit polynomial to trunc_log
    trunc_log = np.vectorize(lambda x: np.log(x) if x > 1 else 0, otypes=[np.float64])
    if matrix is None:
        matrix = np.load('temp/dw_matrix_sample.npy')
    # x = matrix[matrix.shape[0]//2, :]
    max_norm = (matrix**2).sum(axis=1).max()
    x = np.linspace(-max_norm, max_norm, 1000)
    coeffs = np.polyfit(x, trunc_log(x), deg)
    # coeffs = np.polyfit(x, np.exp(2*x), deg)
    return coeffs[::-1]

# approximate func(U@U.T) with Tu@Tu.T
# def sym_poly_tensor_sketch(U, func, sketch_dim, degree=10):
#     vfunc = np.vectorize(func, otypes=[np.float64])
#     n, d = U.shape
#     w = np.sqrt((U**2).sum(axis=1).max())
#     # row_range = 2
#     U = U / w
#     xs = np.linspace(-1, 1, 1000)
#     # xs = np.linspace(-2, 100, 10000)
#     # xs = np.linspace(-0.5, 2, 10000)
#     ys = vfunc(xs * w ** 2)
#     coeffs = np.polyfit(xs, ys, degree)[::-1]

#     poly = np.poly1d(coeffs[::-1])
#     ys = poly(xs * w ** 2)
#     import matplotlib.pyplot as plt
#     plt.plot(xs * w ** 2, ys)
#     plt.show()

#     # coeffs = np.sqrt(coeffs) # coeffs may be negative
#     tu = [np.ones((n, 1), np.float64)]
#     S = count_sketch_matrix(U, sketch_dim)
#     tu.append(U @ S)
#     fu = fft(tu[-1])
#     for i in range(2, len(coeffs)):
#         S = count_sketch_matrix(U, sketch_dim)
#         fu = fft(U @ S) * fu
#         tu.append(ifft(fu).real)
#     c = np.ones(sketch_dim*degree+1, np.float64)
#     c[0] = coeffs[0]
#     for i in range(1, len(coeffs)):
#         c[(i-1)*sketch_dim+1:i*sketch_dim+1] = coeffs[i]
#     tu = np.c_[*tu] * w
#     return tu, c

if __name__ == '__main__':
    # M = sp.random(100, 1000, density=0.01, format='csr')
    plot_func()
    n, d = 1000, 100
    U = np.random.randn(n, d)
    # coeffs = fit_trunc_log(matrix = U)
    # V = np.random.randn(1000, 100)
    # U, V = U / np.sqrt(1000), V / np.sqrt(1000)
    # coeffs=[1.0002, 0.2319, 0.0366, 0.2271, 0.4328, 1.5164, 4.6589, 14.1915, 40.5976, 110.7718, 287.9291]
    coeffs=[0.0002, 0.2319, 0.366, 0.001]
    # TU, TV = poly_tensor_sketch(U, U, coeffs, 10)
    # TU, c_vec = sym_poly_tensor_sketch(U, lambda x: np.log(x) if x > 1 else 0.0, 10, 5)
    TU, TV = poly_tensor_sketch(U, U, coeffs, 32)
    pass
    # poly_tensor_sketch()