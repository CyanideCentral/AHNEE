import numpy as np
rng = np.random.default_rng(0)

embedding_dim = 32 # k
knn = 10
beta = 1.0
svd_rank = 32 # r
svd_tol = 0
window = 10 # T
alpha = 0.1

sketch_dim = 128 # b
poly_deg = 3 # \tau
fit_sample = 10

verbose = True

# node classification
nc_train_ratio = 0.2
nc_num_splits = 10

# hyperedge link prediction
lp_train_ratio = 0.8
lp_num_splits = 10
set_reduce = 'maxmin' # How to aggregate embeddings: 'sum', 'mean', 'range', 'maxmin'

# hyperedge classification
hec_train_ratio = 0.2
hec_num_splits = 10

