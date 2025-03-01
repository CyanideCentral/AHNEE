import numpy as np

embedding_dim = 32
verbose = True
concise_log = False

knn = 10

# visual
visual = False
plot_svd = False

# aug_hg weight
beta = 1.0

# HGMF
svd_rank = 32
svd_tol = 0
window = 10
negative = 1.0

rwr_alpha = 0.1
rwr_window = 10

# alpha = 0.1
# gamma = 10

# dynamic
qr_iters = 20

# PTS
rng = np.random.default_rng(0)
sketch_dim = 128
poly_deg = 3
func_name = 'tlog' # 'tlog', 'relu', 'tanh', 'sigmoid'
if func_name == 'tlog':
    func = lambda x: np.log(x) if x > 1 else 0
elif func_name == 'relu':
    func = lambda x: np.maximum(x-1., 0.) if x > 1 else 0
elif func_name == 'tanh':
    func = lambda x: np.tanh(x-1) if x > 1 else 0
elif func_name == 'sigmoid':
    func = lambda x: 1/(1 + np.exp(2-x))
else:
    raise ValueError(f"Unknown function: {func_name}")
fit_sample = 10 # -1 or the number of samples
sampling = 'random' # 'random', 'kcenter'

# Nystrom
kernel = ['rbf']

test_time=False
#error test
test_error=False
mae_error = []
mse_error = []
pearson_r=[]
poly_mae_error=[]
poly_mse_error=[]
poly_pearson_r=[]

p_beta = 0.5

# SketchNE
window_size = 10

fix_eval_split = False

# node classification
nc_train_ratio = 0.2
nc_num_splits = 10
# eval with pytorch
nc_epochs = 200
nc_lr = 1e-3
nc_w_decay = 1e-6

# hyperedge prediction
lp_train_ratio = 0.8
lp_num_splits = 10
set_reduce = 'maxmin' # How to aggregate embeddings: 'sum', 'mean', 'range', 'maxmin'

# hyperedge classification
hec_train_ratio = 0.2
hec_num_splits = 10

# #param test
# large_proportion = [0.02,0.05]
# small_proportion = [0.1,0.2]
# num_clusters = [10]
# multi_knn = [5,20]

#dynamic plot result
dy_nc_acc = []
dy_nc_maf1 = []
dy_lp_acc = []
dy_lp_auc = []
dy_visual = True