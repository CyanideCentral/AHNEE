import numpy as np
import scipy.sparse as sp
import time
import resource
import config
from sketchne import sketchne_graph
from hgsketchne import sketchne_hypergraph
from netmf import netmf
from og_netmf import og_netmf
from hgnetmf import netmf_hypergraph
from sklearn.preprocessing import normalize
from augment import *
from hgmf import *
from hgsketchne import chebyshev_expansion
import pickle as pkl

def random_embedding(ahg, dim):
    start_time = time.perf_counter()
    n = ahg.n
    embedding = np.random.rand(n, dim)
    time_cost = time.perf_counter() - start_time
    ram_gbs = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 / 1024
    return embedding, (time_cost, ram_gbs)

def knnshe_embedding(ahg, dim):
    start_time = time.perf_counter()
    import faiss
    # construct exact knn with ahg.features
    features = ahg.features.toarray()
    n, d = features.shape
    faiss.normalize_L2(features)
    index = faiss.IndexFlatIP(d)
    index.add(features)
    D, I = index.search(features, config.knn)

    # print(f"FAISS exact knn search time: {time.perf_counter() - start_time}")
    # knn_start_time = time.perf_counter()
    # from sklearn.neighbors import kneighbors_graph
    # A = kneighbors_graph(features, config.knn, mode='distance', include_self=False, metric='cosine')
    # print(f"sklearn kneighbors_graph time: {time.perf_counter() - knn_start_time}")

    # construct hypergraph
    knn_hg = sp.csr_matrix((np.ones(n * config.knn), I.flatten(), np.arange(0, n*config.knn+1, config.knn)), shape=(n, n))
    H = sp.vstack([ahg.hypergraph, knn_hg]).T
    
    # H = ahg.hypergraph.T
    Dv = sp.diags(H.sum(axis=1).A1)
    De = sp.diags(H.sum(axis=0).A1)
    Z = sp.eye(H.shape[0])
    Sv = lambda H, Se: H * Se * H.transpose()
    W = sp.eye(H.shape[1])
    Se = lambda W: W
    L = Z * (Dv - Sv(H, Se(W))) * Z.transpose()
    k = dim
    eigenval, eigenvec = sp.linalg.eigs(L, k=k, which='SR')
    s_eigenval = np.argsort(eigenval.real)
    indices = range(0, dim)
    vecs = [s_eigenval[index] for index in indices]
    embeddings = eigenvec[:, vecs].real
    time_cost = time.perf_counter() - start_time
    ram_gbs = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 / 1024
    return embeddings, (time_cost, ram_gbs)

def knnrw_sketchne(ahg, dim):
    start_time = time.perf_counter()
    import faiss
    # construct exact knn with ahg.features
    features = ahg.features.toarray()
    n, d = features.shape
    faiss.normalize_L2(features)
    # index = faiss.IndexFlatIP(d)
    quantizer = faiss.IndexFlatL2(d)
    index = faiss.IndexIVFFlat(quantizer, d, 100)
    index.train(features)
    index.add(features)
    D, I = index.search(features, config.knn+1)
    # print(f"FAISS exact knn search time: {time.perf_counter() - start_time}")
    knn_graph = sp.csr_matrix((D[:, 1:].flatten(), I[:, 1:].flatten(), np.arange(0, n*config.knn+1, config.knn)), shape=(n, n))

    H = ahg.hypergraph
    De = sp.diags(1.0/H.sum(axis=1).A1)
    Dv = sp.diags(1.0/H.sum(axis=0).A1)
    hgrw = Dv @ H.T @ De @ H

    Dk = knn_graph.sum(axis=1).A1
    Dk[Dk==0] = 1
    Dk = sp.diags(1.0/Dk)
    # Dk = sp.diags(1.0/knn_graph.sum(axis=1).A1)
    knnrw = Dk @ knn_graph
 
    beta = 0.5
    L = (1-beta) * hgrw + beta * knnrw
    # L = (L + L.T) / 2
    embedding = sketchne_graph(L, dim=dim)
    return embedding, (time.perf_counter() - start_time, resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 / 1024)

def aughg_sketchne(ahg, dim):
    start_time = time.perf_counter()
    import faiss
    # construct exact knn with ahg.features
    features = ahg.features.toarray()
    n, d = features.shape
    faiss.normalize_L2(features)
    index = faiss.IndexFlatIP(d)
    index.add(features)
    D, I = index.search(features, config.knn+1)
    knn_graph = sp.csr_matrix((D[:, 1:].flatten(), I[:, 1:].flatten(), np.arange(0, n*config.knn+1, config.knn)), shape=(n, n))

    H = ahg.hypergraph
    De = sp.diags(1.0/H.sum(axis=1).A1)
    # Dv = sp.diags(1.0/H.sum(axis=0).A1)
    hg_graph = H.T @ De @ H
 
    beta = 0.5
    # Dk = knn_graph.sum(axis=1).A1
    # Dk[Dk==0] = 1
    # knn_graph = sp.diags(Dk) @ knn_graph
    aug_adj = (1-beta) * hg_graph + beta * knn_graph
    embedding = sketchne_graph(aug_adj, dim=dim, window_size=config.window_size)
    return embedding, (time.perf_counter() - start_time, resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 / 1024)

def aughg_netmf(ahg, dim):
    start_time = time.perf_counter()
    import faiss
    # construct exact knn with ahg.features
    features = ahg.features.toarray()
    n, d = features.shape
    faiss.normalize_L2(features)
    index = faiss.IndexFlatIP(d)
    index.add(features)
    D, I = index.search(features, config.knn+1)
    knn_graph = sp.csr_matrix((D[:, 1:].flatten(), I[:, 1:].flatten(), np.arange(0, n*config.knn+1, config.knn)), shape=(n, n))

    H = ahg.hypergraph
    De = sp.diags(1.0/H.sum(axis=1).A1)
    # Dv = sp.diags(1.0/H.sum(axis=0).A1)
    hg_graph = H.T @ De @ H

    beta = 0.5
    # Dk = knn_graph.sum(axis=1).A1
    # Dk[Dk==0] = 1
    # knn_graph = sp.diags(Dk) @ knn_graph
    aug_adj = (1-beta) * hg_graph + beta * knn_graph
    embedding = netmf(aug_adj.toarray(), dim=dim, window=10, negative=1.0)
    return embedding, (time.perf_counter() - start_time, resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 / 1024)

def rgehg_sketchne(ahg, dim):
    start_time = time.perf_counter()
    import faiss
    # construct exact knn with ahg.features
    features = ahg.features.toarray()
    n, d = features.shape
    faiss.normalize_L2(features)
    index = faiss.IndexFlatIP(d)
    index.add(features)
    lims, D, I = index.range_search(features, 0.001)
    rge_graph = sp.csr_matrix((D, I, lims), shape=(n, n))

    H = ahg.hypergraph
    De = sp.diags(1.0/H.sum(axis=1).A1)
    # Dv = sp.diags(1.0/H.sum(axis=0).A1)
    hg_graph = H.T @ De @ H
 
    # beta = 0.5
    beta = hg_graph.sum() / (rge_graph.sum() + hg_graph.sum())
    # Dk = knn_graph.sum(axis=1).A1
    # Dk[Dk==0] = 1
    # knn_graph = sp.diags(Dk) @ knn_graph
    aug_adj = (1-beta) * hg_graph + beta * rge_graph
    embedding = sketchne_graph(aug_adj, dim=dim, window_size=config.window_size)
    return embedding, (time.perf_counter() - start_time, resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 / 1024)

def aughg_planar_pearson_netmf(ahg, dim):
    start_time = time.perf_counter()
    import faiss
    # construct exact knn with ahg.features
    features = ahg.features.toarray()
    planar_graph = sp.load_npz(f"/home/gongyao/par-filtered-graph-clustering/outputs/pearson/{ahg.name}_planar.npz")

    H = ahg.hypergraph
    De = sp.diags(1.0/H.sum(axis=1).A1)
    # Dv = sp.diags(1.0/H.sum(axis=0).A1)
    hg_graph = H.T @ De @ H

    beta = 0.5
    # Dk = planar_graph.sum(axis=1).A1
    # Dk[Dk==0] = 1
    # planar_graph = sp.diags(Dk) @ planar_graph
    aug_adj = (1-beta) * hg_graph + beta * planar_graph
    embedding = netmf(aug_adj.toarray(), dim=dim, window=10, negative=1.0)
    return embedding, (time.perf_counter() - start_time, resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 / 1024)

def aughg_planar_cosine_netmf(ahg, dim):
    start_time = time.perf_counter()
    import faiss
    # construct exact knn with ahg.features
    features = ahg.features.toarray()
    planar_graph = sp.load_npz(f"/home/gongyao/par-filtered-graph-clustering/outputs/cosine/{ahg.name}_planar.npz")

    H = ahg.hypergraph
    De = sp.diags(1.0/H.sum(axis=1).A1)
    # Dv = sp.diags(1.0/H.sum(axis=0).A1)
    hg_graph = H.T @ De @ H

    beta = 0.5
    # Dk = planar_graph.sum(axis=1).A1
    # Dk[Dk==0] = 1
    # planar_graph = sp.diags(Dk) @ planar_graph
    aug_adj = (1-beta) * hg_graph + beta * planar_graph
    embedding = netmf(aug_adj.toarray(), dim=dim, window=10, negative=1.0)
    return embedding, (time.perf_counter() - start_time, resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 / 1024)

def xxt_netmf(ahg, dim):
    start_time = time.perf_counter()
    from sklearn.metrics.pairwise import cosine_similarity
    sim_matrix = cosine_similarity(ahg.features)
    sim_matrix[sim_matrix < 0] = 0

    H = ahg.hypergraph
    De = sp.diags(1.0/H.sum(axis=1).A1)
    # Dv = sp.diags(1.0/H.sum(axis=0).A1)
    hg_graph = H.T @ De @ H

    # beta = 0.03
    # beta = 0.1
    beta = hg_graph.sum() / (sim_matrix.sum() + hg_graph.sum())
    # Dk = knn_graph.sum(axis=1).A1
    # Dk[Dk==0] = 1
    # knn_graph = sp.diags(Dk) @ knn_graph
    aug_adj = (1-beta) * hg_graph + beta * sim_matrix
    embedding = netmf(aug_adj, dim=dim, window=10, negative=1.0)
    return embedding, (time.perf_counter() - start_time, resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 / 1024)

def xxt_sketchne(ahg, dim):
    start_time = time.perf_counter()
    from sklearn.metrics.pairwise import cosine_similarity
    sim_matrix = cosine_similarity(ahg.features)
    sim_matrix[sim_matrix < 0] = 0

    H = ahg.hypergraph
    De = sp.diags(1.0/H.sum(axis=1).A1)
    # Dv = sp.diags(1.0/H.sum(axis=0).A1)
    hg_graph = H.T @ De @ H

    # beta = 0.03
    beta = hg_graph.sum() / (sim_matrix.sum() + hg_graph.sum())
    # Dk = knn_graph.sum(axis=1).A1
    # Dk[Dk==0] = 1
    # knn_graph = sp.diags(Dk) @ knn_graph
    aug_adj = (1-beta) * hg_graph + beta * sim_matrix
    embedding = sketchne_graph(aug_adj, dim=dim, spec_propagation=True)
    return embedding, (time.perf_counter() - start_time, resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 / 1024)

# H -> Hypergraph SketchNE
def hg_sketchne(ahg, dim):
    start_time = time.perf_counter()
    embedding = sketchne_hypergraph(ahg.hypergraph, dim, hyperedge_weights=np.ones(ahg.m))
    return embedding, (time.perf_counter() - start_time, resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 / 1024)

# ATH -> Hypergraph SketchNE
def knnhg_sketchne(ahg, dim):
    start_time = time.perf_counter()
    aug_hg = knn_hypergraph(ahg.features, knn=config.knn, weighted=True, undirected=False)
    aug_hg = config.beta* aug_hg
    if config.verbose:
        print(f"KNN time: {time.perf_counter() - start_time} KNN K: {config.knn}")
    athg = sp.vstack([ahg.hypergraph, aug_hg], format='csr')
    # HH_vol = (ahg.hypergraph.sum(1).A1**2).sum()
    # AA_vol = (aug_hg.sum(1).A1**2).sum()
    # attr_weight = HH_vol / (HH_vol + AA_vol)
    H_vol = (ahg.hypergraph.sum(1).A1).sum()
    A_vol = (aug_hg.sum(1).A1).sum()
    attr_weight = H_vol / (H_vol + A_vol)
    he_weights = np.concatenate([np.ones(ahg.m), np.full(aug_hg.shape[0], attr_weight)])
    # dv_hg = ahg.hypergraph.sum(0).A1
    # dv_knn = aug_hg.sum(0).A1
    # he_weights = np.concatenate([np.ones(ahg.m), dv_hg / dv_knn])
    # embedding = hgsvdlogsvd(athg, dim, hyperedge_weights=he_weights)
    if config.verbose:
        print(f"HG augmentation time: {time.perf_counter() - start_time}")
    embedding = sketchne_hypergraph(athg, dim, hyperedge_weights=he_weights)
    return embedding, (time.perf_counter() - start_time, resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 / 1024)

# H.T@H -> SketchNE
def clex_sketchne(ahg, dim):
    start_time = time.perf_counter()
    clex = ahg.hypergraph.T @ ahg.hypergraph
    embedding = sketchne_graph(clex, dim)
    return embedding, (time.perf_counter() - start_time, resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 / 1024)

# H.T@H -> NetMF
def clex_netmf(ahg, dim):
    from og_netmf import og_netmf, og_netmf_small
    start_time = time.perf_counter()
    clex = ahg.hypergraph.T @ ahg.hypergraph
    # embedding = netmf(clex, dim, rank=256)
    embedding = og_netmf(clex, dim, rank=32)
    # embedding = og_netmf_small(clex, dim, rank=128)
    return embedding, (time.perf_counter() - start_time, resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 / 1024)

# H -> Hypergraph NetMF
def hg_netmf(ahg, dim):
    start_time = time.perf_counter()
    # embedding = netmf_hypergraph(ahg.hypergraph, dim, hyperedge_weights=np.ones(ahg.m), rank=256, window=10)
    # config.svd_rank = 16
    embedding = hgmf_1eigsh(ahg.hypergraph, dim, hyperedge_weights=np.ones(ahg.m))
    return embedding, (time.perf_counter() - start_time, resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 / 1024)

def hg_g_netmf(ahg, dim):
    from og_netmf import netmf_hg
    start_time = time.perf_counter()
    embedding = netmf_hg(ahg.hypergraph, dim, rank=256, window=10)
    return embedding, (time.perf_counter() - start_time, resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 / 1024)

def attr_hg_sketchne(ahg, dim):
    start_time = time.perf_counter()
    athg = sp.vstack([ahg.hypergraph, ahg.features.T], format='csr')
    attr_weight = ahg.hypergraph.sum() / (ahg.hypergraph.sum() + ahg.features.shape[1])
    he_weights = np.concatenate([np.ones(ahg.m), np.full(ahg.features.shape[1], attr_weight)])
    embedding = sketchne_hypergraph(athg, dim, hyperedge_weights=he_weights)
    return embedding, (time.perf_counter() - start_time, resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 / 1024)

def attr_hg_netmf(ahg, dim):
    start_time = time.perf_counter()
    features = normalize(ahg.features, norm='l2', axis=1)
    athg = sp.vstack([ahg.hypergraph, features.T], format='csr')
    # attr_weight = ahg.hypergraph.sum() / (ahg.hypergraph.sum() + features.sum())
    HH_vol = (ahg.hypergraph.sum(1).A1**2).sum()
    features_vol = (features.sum(0).A1**2).sum()
    attr_weight = HH_vol / (HH_vol + features_vol)
    he_weights = np.concatenate([np.ones(ahg.m), np.full(features.shape[1], attr_weight**2)])
    embedding = netmf_hypergraph(athg, dim, hyperedge_weights=he_weights)
    return embedding, (time.perf_counter() - start_time, resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 / 1024)

def knn_hg_clex_netmf(ahg, dim):
    start_time = time.perf_counter()
    aug_hg = knn_hypergraph(ahg.features, weighted=True)
    athg = sp.vstack([ahg.hypergraph, aug_hg], format='csr')
    # HH_vol = (ahg.hypergraph.sum(1).A1**2).sum()
    # AA_vol = (aug_hg.sum(1).A1**2).sum()
    # attr_weight = HH_vol / (HH_vol + AA_vol)
    H_vol = (ahg.hypergraph.sum(1).A1).sum()
    A_vol = (aug_hg.sum(1).A1).sum()
    attr_weight = H_vol / (H_vol + A_vol)
    he_weights = np.concatenate([np.ones(ahg.m), np.full(aug_hg.shape[0], attr_weight)])
    clex = athg.T @ sp.diags(he_weights) @ athg
    embedding = og_netmf(clex, dim, rank=32)
    return embedding, (time.perf_counter() - start_time, resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 / 1024)

def knn_hg_netmf(ahg, dim):
    start_time = time.perf_counter()
    aug_hg = knn_hypergraph(ahg.features, weighted=True)
    athg = sp.vstack([ahg.hypergraph, aug_hg], format='csr')
    # HH_vol = (ahg.hypergraph.sum(1).A1**2).sum()
    # AA_vol = (aug_hg.sum(1).A1**2).sum()
    # attr_weight = HH_vol / (HH_vol + AA_vol)
    H_vol = (ahg.hypergraph.sum(1).A1).sum()
    A_vol = (aug_hg.sum(1).A1).sum()
    attr_weight = H_vol / (H_vol + A_vol)
    he_weights = np.concatenate([np.ones(ahg.m), np.full(aug_hg.shape[0], attr_weight)])
    embedding = netmf_hypergraph(athg, dim, hyperedge_weights=he_weights)
    return embedding, (time.perf_counter() - start_time, resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 / 1024)

def knn_hg_sketchne(ahg, dim):
    start_time = time.perf_counter()
    # aug_hg = knn_hypergraph(ahg.features, weighted=True)
    aug_hg = euc_knn_hypergraph(ahg.features, weighted=True)
    athg = sp.vstack([ahg.hypergraph, aug_hg], format='csr')
    # HH_vol = (ahg.hypergraph.sum(1).A1**2).sum()
    # AA_vol = (aug_hg.sum(1).A1**2).sum()
    # attr_weight = HH_vol / (HH_vol + AA_vol)
    H_vol = (ahg.hypergraph.sum(1).A1).sum()
    A_vol = (aug_hg.sum(1).A1).sum()
    attr_weight = H_vol / (H_vol + A_vol)
    he_weights = np.concatenate([np.ones(ahg.m), np.full(aug_hg.shape[0], attr_weight)])
    embedding = sketchne_hypergraph(athg, dim, hyperedge_weights=he_weights,spec_propagation=True)
    return embedding, (time.perf_counter() - start_time, resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 / 1024)

def hac_hg_hgmf(ahg, dim):
    start_time = time.perf_counter()
    aug_hg = agglomerative_hypergraph(ahg.features)
    athg = sp.vstack([ahg.hypergraph, aug_hg], format='csr')
    # HH_vol = (ahg.hypergraph.sum(1).A1**2).sum()
    # AA_vol = (aug_hg.sum(1).A1**2).sum()
    # attr_weight = HH_vol / (HH_vol + AA_vol)
    H_vol = (ahg.hypergraph.sum(1).A1).sum()
    A_vol = (aug_hg.sum(1).A1).sum()
    attr_weight = H_vol / (H_vol + A_vol)
    he_weights = np.concatenate([np.ones(ahg.m), np.full(aug_hg.shape[0], attr_weight)])
    embedding = hgsvdlogsvd(athg, dim, hyperedge_weights=he_weights)
    # embedding = hgmf_1svd(athg, dim, hyperedge_weights=he_weights)
    # embedding = hgmf(athg, dim, hyperedge_weights=he_weights)
    return embedding, (time.perf_counter() - start_time, resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 / 1024)

def knn_hg_hgmf(ahg, dim):
    start_time = time.perf_counter()
    aug_hg = knn_hypergraph(ahg.features, weighted=True)
    athg = sp.vstack([ahg.hypergraph, aug_hg], format='csr')
    # HH_vol = (ahg.hypergraph.sum(1).A1**2).sum()
    # AA_vol = (aug_hg.sum(1).A1**2).sum()
    # attr_weight = HH_vol / (HH_vol + AA_vol)
    H_vol = (ahg.hypergraph.sum(1).A1).sum()
    A_vol = (aug_hg.sum(1).A1).sum()
    attr_weight = H_vol / (H_vol + A_vol)
    he_weights = np.concatenate([np.ones(ahg.m), np.full(aug_hg.shape[0], attr_weight)])
    embedding = hgsvdlogsvd(athg, dim, hyperedge_weights=he_weights)
    # embedding = hgmf_1svd(athg, dim, hyperedge_weights=he_weights)
    # embedding = hgmf(athg, dim, hyperedge_weights=he_weights)
    return embedding, (time.perf_counter() - start_time, resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 / 1024)

def knn_hg_hgmf_1svd(ahg, dim):
    start_time = time.perf_counter()
    aug_hg = knn_hypergraph(ahg.features, knn=config.knn, weighted=True, undirected=False)
    aug_hg = config.beta* aug_hg
    if config.verbose:
        print(f"KNN time: {time.perf_counter() - start_time} KNN K: {config.knn}")
    athg = sp.vstack([ahg.hypergraph, aug_hg], format='csr')
    # HH_vol = (ahg.hypergraph.sum(1).A1**2).sum()
    # AA_vol = (aug_hg.sum(1).A1**2).sum()
    # attr_weight = HH_vol / (HH_vol + AA_vol)
    H_vol = (ahg.hypergraph.sum(1).A1).sum()
    A_vol = (aug_hg.sum(1).A1).sum()
    attr_weight = H_vol / (H_vol + A_vol)
    # H_e = ahg.hypergraph.shape[0]
    # A_e = aug_hg.shape[0]
    # attr_weight = H_e / (H_e + A_e)
    # dv_hg = ahg.hypergraph.sum(0).A1
    # dv_knn = aug_hg.sum(0).A1
    # Dv_hg = sp.diags(dv_hg)
    # Dv_knn = sp.diags(dv_knn)
    # test = ahg.hypergraph @ Dv_hg
    # test1 = aug_hg @ Dv_knn
    # ev_hg = test.sum(1).A1
    # ev_knn = test1.sum(1).A1
    # he_weights = np.concatenate([ev_hg, ev_knn])
    # ev_hg = ahg.hypergraph.sum(1).A1
    # ev_knn = aug_hg.sum(1).A1 * attr_weight
    # he_weights = np.concatenate([ev_hg, ev_knn])
    he_weights = np.concatenate([np.ones(ahg.m), np.full(aug_hg.shape[0], attr_weight)])
    # embedding = hgsvdlogsvd(athg, dim, hyperedge_weights=he_weights)
    if config.verbose:
        print(f"HG augmentation time: {time.perf_counter() - start_time}")
    embedding = hgmf_1svd(athg, dim, hyperedge_weights=he_weights)
    # embedding = hgmf_rwr_1svd(athg, dim, hyperedge_weights=he_weights)
    # embedding = hgmf(athg, dim, hyperedge_weights=he_weights)
    return embedding, (time.perf_counter() - start_time, resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 / 1024)

def knn_hg_svd(ahg, dim):
    start_time = time.perf_counter()
    aug_hg = knn_hypergraph(ahg.features, knn=config.knn, weighted=True, undirected=False)
    aug_hg = config.beta* aug_hg
    if config.verbose:
        print(f"KNN time: {time.perf_counter() - start_time} KNN K: {config.knn}")
    athg = sp.vstack([ahg.hypergraph, aug_hg], format='csr')
    H_vol = (ahg.hypergraph.sum(1).A1).sum()
    A_vol = (aug_hg.sum(1).A1).sum()
    attr_weight = H_vol / (H_vol + A_vol)
    he_weights = np.concatenate([np.ones(ahg.m), np.full(aug_hg.shape[0], attr_weight)])
    # embedding = hgsvdlogsvd(athg, dim, hyperedge_weights=he_weights)
    if config.verbose:
        print(f"HG augmentation time: {time.perf_counter() - start_time}")
    embedding = hg_svd(athg, dim, hyperedge_weights=he_weights)
    return embedding, (time.perf_counter() - start_time, resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 / 1024)

def knn_hg_hgmf_1eigsh(ahg, dim):
    start_time = time.perf_counter()
    aug_hg = knn_hypergraph(ahg.features, knn=config.knn, weighted=True)
    aug_hg = config.beta* aug_hg
    if config.verbose:
        print(f"KNN time: {time.perf_counter() - start_time} KNN K: {config.knn}")
    athg = sp.vstack([ahg.hypergraph, aug_hg], format='csr')
    # HH_vol = (ahg.hypergraph.sum(1).A1**2).sum()
    # AA_vol = (aug_hg.sum(1).A1**2).sum()
    # attr_weight = HH_vol / (HH_vol + AA_vol)
    H_vol = (ahg.hypergraph.sum(1).A1).sum()
    A_vol = (aug_hg.sum(1).A1).sum()
    attr_weight = H_vol / (H_vol + A_vol)
    he_weights = np.concatenate([np.ones(ahg.m), np.full(aug_hg.shape[0], attr_weight)])
    # embedding = hgsvdlogsvd(athg, dim, hyperedge_weights=he_weights)
    if config.verbose:
        print(f"HG augmentation time: {time.perf_counter() - start_time}")
    embedding = hgmf_1eigsh(athg, dim, hyperedge_weights=he_weights)
    # embedding = hgmf(athg, dim, hyperedge_weights=he_weights)
    return embedding, (time.perf_counter() - start_time, resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 / 1024)

def knn_hg_hgmf_1svd_spec(ahg, dim):
    start_time = time.perf_counter()
    aug_hg = knn_hypergraph(ahg.features, knn=config.knn, weighted=True)
    if config.verbose:
        print(f"KNN time: {time.perf_counter() - start_time} KNN K: {config.knn}")
    athg = sp.vstack([ahg.hypergraph, aug_hg], format='csr')
    # HH_vol = (ahg.hypergraph.sum(1).A1**2).sum()
    # AA_vol = (aug_hg.sum(1).A1**2).sum()
    # attr_weight = HH_vol / (HH_vol + AA_vol)
    H_vol = (ahg.hypergraph.sum(1).A1).sum()
    A_vol = (aug_hg.sum(1).A1).sum()
    attr_weight = H_vol / (H_vol + A_vol)
    he_weights = np.concatenate([np.ones(ahg.m), np.full(aug_hg.shape[0], attr_weight)])
    # embedding = hgsvdlogsvd(athg, dim, hyperedge_weights=he_weights)
    if config.verbose:
        print(f"HG augmentation time: {time.perf_counter() - start_time}")
    embedding = hgmf_1svd(athg, dim, hyperedge_weights=he_weights)
    embedding = chebyshev_expansion(embedding, athg)
    # embedding = hgmf(athg, dim, hyperedge_weights=he_weights)
    return embedding, (time.perf_counter() - start_time, resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 / 1024)

def knn_hg_hgmf_1svd_concatenate(ahg, dim):
    start_time = time.perf_counter()
    aug_hg = knn_hypergraph(ahg.features, weighted=True)
    if config.verbose:
        print(f"KNN time: {time.perf_counter() - start_time} KNN K: {config.knn}")
    athg = sp.vstack([ahg.hypergraph, aug_hg], format='csr')
    # HH_vol = (ahg.hypergraph.sum(1).A1**2).sum()
    # AA_vol = (aug_hg.sum(1).A1**2).sum()
    # attr_weight = HH_vol / (HH_vol + AA_vol)
    H_vol = (ahg.hypergraph.sum(1).A1).sum()
    A_vol = (aug_hg.sum(1).A1).sum()
    attr_weight = H_vol / (H_vol + A_vol)
    he_weights = np.concatenate([np.ones(ahg.m), np.full(aug_hg.shape[0], attr_weight)])
    # embedding = hgsvdlogsvd(athg, dim, hyperedge_weights=he_weights)
    if config.verbose:
        print(f"HG augmentation time: {time.perf_counter() - start_time}")
    embedding_athg = hgmf_1svd(athg, int(dim/2), hyperedge_weights=he_weights)
    embedding_hg = hgmf_1svd(ahg.hypergraph, int(dim/2), hyperedge_weights=np.ones(ahg.m))
    embedding = np.concatenate((embedding_athg, embedding_hg), axis=1)
    return embedding, (time.perf_counter() - start_time, resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 / 1024)

def knn_hg_hgmf_teigs(ahg, dim):
    start_time = time.perf_counter()
    aug_hg = knn_hypergraph(ahg.features, weighted=True)
    athg = sp.vstack([ahg.hypergraph, aug_hg], format='csr')
    # HH_vol = (ahg.hypergraph.sum(1).A1**2).sum()
    # AA_vol = (aug_hg.sum(1).A1**2).sum()
    # attr_weight = HH_vol / (HH_vol + AA_vol)
    H_vol = (ahg.hypergraph.sum(1).A1).sum()
    A_vol = (aug_hg.sum(1).A1).sum()
    attr_weight = H_vol / (H_vol + A_vol)
    he_weights = np.concatenate([np.ones(ahg.m), np.full(aug_hg.shape[0], attr_weight)])
    # embedding = hgsvdlogsvd(athg, dim, hyperedge_weights=he_weights)
    embedding = hgmf_teigs(athg, dim, hyperedge_weights=he_weights)
    # embedding = hgmf(athg, dim, hyperedge_weights=he_weights)
    return embedding, (time.perf_counter() - start_time, resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 / 1024)

def knnrw_proximity_base(ahg, dim, window=5, beta=0.6, scalar = 1e5):
    start_time = time.perf_counter()
    knn = knn_graph(ahg.features, knn=config.knn, weighted=True, undirected=True)
    knn_dv = knn.sum(1).A1
    knn_dv[knn_dv==0] = 1
    p_knn = knn / knn_dv[:, np.newaxis]
    H = ahg.hypergraph
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
    H = H.toarray()
    D_e = D_e.toarray()
    D_v = D_v.toarray()
    p_hg = D_v @ H.T @ D_e @ H
    P = beta * p_knn + (1-beta) * p_hg
    # p = sp.linalg.eigs(P, k=1, which='LR')[1].flatten().real
    # p = np.full(n, 1/n)
    # L = np.diag(p)
    L = np.eye(n)
    # deg_vec1 = (H.T@hyperedge_weights[:,np.newaxis]).flatten()
    # WDerH = np.diag(hyperedge_weights)@np.diag(H.sum(1)**-1)@H
    for i in range(window):
        L = config.rwr_alpha * L + (1-config.rwr_alpha) * P @ L
    L = (L + L.T) * scalar
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
    return X, (time.perf_counter() - start_time, resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 / 1024)

def knn_hg_proximity_base(ahg, dim):
    start_time = time.perf_counter()
    aug_hg = knn_hypergraph(ahg.features, weighted=True)
    athg = sp.vstack([ahg.hypergraph, aug_hg], format='csr')
    # HH_vol = (ahg.hypergraph.sum(1).A1**2).sum()
    # AA_vol = (aug_hg.sum(1).A1**2).sum()
    # attr_weight = HH_vol / (HH_vol + AA_vol)
    H_vol = (ahg.hypergraph.sum(1).A1).sum()
    A_vol = (aug_hg.sum(1).A1).sum()
    attr_weight = H_vol / (H_vol + A_vol)
    he_weights = np.concatenate([np.ones(ahg.m), np.full(aug_hg.shape[0], attr_weight)])
    # embedding = hgsvdlogsvd(athg, dim, hyperedge_weights=he_weights)
    embedding = hgmf_proximity_base(athg, dim, hyperedge_weights=he_weights)
    # embedding = hgmf(athg, dim, hyperedge_weights=he_weights)
    return embedding, (time.perf_counter() - start_time, resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 / 1024)

def knn_hg_with_embedder(hg_embedder):
    def knn_hg_embedder_base(ahg, dim):
        start_time = time.perf_counter()
        aug_hg = knn_hypergraph(ahg.features, weighted=True)
        athg = sp.vstack([ahg.hypergraph, aug_hg], format='csr')
        # HH_vol = (ahg.hypergraph.sum(1).A1**2).sum()
        # AA_vol = (aug_hg.sum(1).A1**2).sum()
        # attr_weight = HH_vol / (HH_vol + AA_vol)
        H_vol = (ahg.hypergraph.sum(1).A1).sum()
        A_vol = (aug_hg.sum(1).A1).sum()
        attr_weight = H_vol / (H_vol + A_vol)
        he_weights = np.concatenate([np.ones(ahg.m), np.full(aug_hg.shape[0], attr_weight)])
        # embedding = hgsvdlogsvd(athg, dim, hyperedge_weights=he_weights)
        embedding = hg_embedder(athg, dim, hyperedge_weights=he_weights)
        # embedding = hgmf(athg, dim, hyperedge_weights=he_weights)
        return embedding, (time.perf_counter() - start_time, resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 / 1024)
    print(f"Base hypergraph embedder: {hg_embedder.__name__}")
    return knn_hg_embedder_base

def knn_hg_hgmf_svdx(ahg, dim):
    start_time = time.perf_counter()
    aug_hg = knn_hypergraph(ahg.features, weighted=True)
    athg = sp.vstack([ahg.hypergraph, aug_hg], format='csr')
    # HH_vol = (ahg.hypergraph.sum(1).A1**2).sum()
    # AA_vol = (aug_hg.sum(1).A1**2).sum()
    # attr_weight = HH_vol / (HH_vol + AA_vol)
    H_vol = (ahg.hypergraph.sum(1).A1).sum()
    A_vol = (aug_hg.sum(1).A1).sum()
    attr_weight = H_vol / (H_vol + A_vol)
    he_weights = np.concatenate([np.ones(ahg.m), np.full(aug_hg.shape[0], attr_weight)])
    # embedding = hgsvdlogsvd(athg, dim, hyperedge_weights=he_weights)
    embedding = hgmf_svdx(athg, dim, hyperedge_weights=he_weights)
    # embedding = hgmf(athg, dim, hyperedge_weights=he_weights)
    return embedding, (time.perf_counter() - start_time, resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 / 1024)

def knn_hg_hgmf_pts(ahg, dim):
    start_time = time.perf_counter()
    aug_hg = knn_hypergraph(ahg.features, knn=config.knn, weighted=True, undirected=False)
    aug_hg = config.beta* aug_hg
    if config.verbose:
        print(f"KNN time: {time.perf_counter() - start_time} KNN K: {config.knn}")
    athg = sp.vstack([ahg.hypergraph, aug_hg], format='csr')
    # HH_vol = (ahg.hypergraph.sum(1).A1**2).sum()
    # AA_vol = (aug_hg.sum(1).A1**2).sum()
    # attr_weight = HH_vol / (HH_vol + AA_vol)
    H_vol = (ahg.hypergraph.sum(1).A1).sum()
    A_vol = (aug_hg.sum(1).A1).sum()
    attr_weight = H_vol / (H_vol + A_vol)
    he_weights = np.concatenate([np.ones(ahg.m), np.full(aug_hg.shape[0], attr_weight)])
    # dv_hg = ahg.hypergraph.sum(0).A1
    # dv_knn = aug_hg.sum(0).A1
    # he_weights = np.concatenate([np.ones(ahg.m), dv_hg / dv_knn])
    # embedding = hgsvdlogsvd(athg, dim, hyperedge_weights=he_weights)
    if config.verbose:
        print(f"HG augmentation time: {time.perf_counter() - start_time}")
    # embedding = hgmf_pts(athg, dim, hyperedge_weights=he_weights)
    embedding = hgmf_asym_pts(athg, dim, hyperedge_weights=he_weights)
    # embedding = hgmf_asym_normx_pts(athg, dim, hyperedge_weights=he_weights)
    # embedding = hgmf_rwr_1svd(athg, dim, hyperedge_weights=he_weights)
    # embedding = hgmf(athg, dim, hyperedge_weights=he_weights)
    return embedding, (time.perf_counter() - start_time, resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 / 1024)

def knn_hg_hgmf_full_pts(ahg, dim):
    start_time = time.perf_counter()
    aug_hg = knn_hypergraph(ahg.features, knn=config.knn, weighted=True, undirected=False)
    aug_hg = config.beta* aug_hg
    if config.verbose:
        print(f"KNN time: {time.perf_counter() - start_time} KNN K: {config.knn}")
    athg = sp.vstack([ahg.hypergraph, aug_hg], format='csr')
    # HH_vol = (ahg.hypergraph.sum(1).A1**2).sum()
    # AA_vol = (aug_hg.sum(1).A1**2).sum()
    # attr_weight = HH_vol / (HH_vol + AA_vol)
    H_vol = (ahg.hypergraph.sum(1).A1).sum()
    A_vol = (aug_hg.sum(1).A1).sum()
    attr_weight = H_vol / (H_vol + A_vol)
    he_weights = np.concatenate([np.ones(ahg.m), np.full(aug_hg.shape[0], attr_weight)])
    # dv_hg = ahg.hypergraph.sum(0).A1
    # dv_knn = aug_hg.sum(0).A1
    # he_weights = np.concatenate([np.ones(ahg.m), dv_hg / dv_knn])
    # embedding = hgsvdlogsvd(athg, dim, hyperedge_weights=he_weights)
    if config.verbose:
        print(f"HG augmentation time: {time.perf_counter() - start_time}")
    embedding = hgmf_full_sym_pts(athg, dim, hyperedge_weights=he_weights)
    return embedding, (time.perf_counter() - start_time, resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 / 1024)

def knn_hg_hgmf_nystroem(ahg, dim):
    start_time = time.perf_counter()
    aug_hg = knn_hypergraph(ahg.features, knn=config.knn, weighted=True, undirected=False)
    aug_hg = config.beta* aug_hg
    if config.verbose:
        print(f"KNN time: {time.perf_counter() - start_time} KNN K: {config.knn}")
    athg = sp.vstack([ahg.hypergraph, aug_hg], format='csr')
    # HH_vol = (ahg.hypergraph.sum(1).A1**2).sum()
    # AA_vol = (aug_hg.sum(1).A1**2).sum()
    # attr_weight = HH_vol / (HH_vol + AA_vol)
    H_vol = (ahg.hypergraph.sum(1).A1).sum()
    A_vol = (aug_hg.sum(1).A1).sum()
    attr_weight = H_vol / (H_vol + A_vol)
    he_weights = np.concatenate([np.ones(ahg.m), np.full(aug_hg.shape[0], attr_weight)])
    # dv_hg = ahg.hypergraph.sum(0).A1
    # dv_knn = aug_hg.sum(0).A1
    # he_weights = np.concatenate([np.ones(ahg.m), dv_hg / dv_knn])
    # embedding = hgsvdlogsvd(athg, dim, hyperedge_weights=he_weights)
    if config.verbose:
        print(f"HG augmentation time: {time.perf_counter() - start_time}")
    embedding = hgmf_nystroem(athg, dim, hyperedge_weights=he_weights)
    return embedding, (time.perf_counter() - start_time, resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 / 1024)

def knn_hg_hgmf_sketchne(ahg, dim):
    start_time = time.perf_counter()
    aug_hg = knn_hypergraph(ahg.features, knn=config.knn, weighted=True, undirected=False)
    aug_hg = config.beta* aug_hg
    if config.verbose:
        print(f"KNN time: {time.perf_counter() - start_time} KNN K: {config.knn}")
    athg = sp.vstack([ahg.hypergraph, aug_hg], format='csr')
    # HH_vol = (ahg.hypergraph.sum(1).A1**2).sum()
    # AA_vol = (aug_hg.sum(1).A1**2).sum()
    # attr_weight = HH_vol / (HH_vol + AA_vol)
    H_vol = (ahg.hypergraph.sum(1).A1).sum()
    A_vol = (aug_hg.sum(1).A1).sum()
    attr_weight = H_vol / (H_vol + A_vol)
    he_weights = np.concatenate([np.ones(ahg.m), np.full(aug_hg.shape[0], attr_weight)])
    # embedding = hgsvdlogsvd(athg, dim, hyperedge_weights=he_weights)
    if config.verbose:
        print(f"HG augmentation time: {time.perf_counter() - start_time}")
    embedding = hgmf_sketchne(athg, dim, hyperedge_weights=he_weights)
    # embedding = hgmf_rwr_1svd(athg, dim, hyperedge_weights=he_weights)
    # embedding = hgmf(athg, dim, hyperedge_weights=he_weights)
    return embedding, (time.perf_counter() - start_time, resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 / 1024)

def knn_hg_hgmf_mask_log(ahg, dim):
    start_time = time.perf_counter()
    aug_hg = knn_hypergraph(ahg.features, weighted=True)
    athg = sp.vstack([ahg.hypergraph, aug_hg], format='csr')
    # HH_vol = (ahg.hypergraph.sum(1).A1**2).sum()
    # AA_vol = (aug_hg.sum(1).A1**2).sum()
    # attr_weight = HH_vol / (HH_vol + AA_vol)
    H_vol = (ahg.hypergraph.sum(1).A1).sum()
    A_vol = (aug_hg.sum(1).A1).sum()
    attr_weight = H_vol / (H_vol + A_vol)
    he_weights = np.concatenate([np.ones(ahg.m), np.full(aug_hg.shape[0], attr_weight)])
    # embedding = hgsvdlogsvd(athg, dim, hyperedge_weights=he_weights)
    embedding = hgmf_mask_log(athg, dim, hyperedge_weights=he_weights, mask = ahg.hypergraph.T@ahg.hypergraph + aug_hg + aug_hg.T)
    # embedding = hgmf_1svd(athg, dim, hyperedge_weights=he_weights)
    # embedding = hgmf(athg, dim, hyperedge_weights=he_weights)
    return embedding, (time.perf_counter() - start_time, resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 / 1024)

def mutual_knn_hg_hgmf_1svd(ahg, dim):
    start_time = time.perf_counter()
    # config.knn = 50
    aug_hg = mutual_knn_hypergraph(ahg.features, knn=50)
    if config.verbose:
        print(f"KNN time: {time.perf_counter() - start_time} KNN K: {config.knn}")
    athg = sp.vstack([ahg.hypergraph, aug_hg], format='csr')
    # HH_vol = (ahg.hypergraph.sum(1).A1**2).sum()
    # AA_vol = (aug_hg.sum(1).A1**2).sum()
    # attr_weight = HH_vol / (HH_vol + AA_vol)
    H_vol = (ahg.hypergraph.sum(1).A1).sum()
    A_vol = (aug_hg.sum(1).A1).sum()
    attr_weight = H_vol / (H_vol + A_vol)
    he_weights = np.concatenate([np.ones(ahg.m), np.full(aug_hg.shape[0], attr_weight)])
    # embedding = hgsvdlogsvd(athg, dim, hyperedge_weights=he_weights)
    if config.verbose:
        print(f"HG augmentation time: {time.perf_counter() - start_time}")
    embedding = hgmf_1svd(athg, dim, hyperedge_weights=he_weights)
    # embedding = hgmf(athg, dim, hyperedge_weights=he_weights)
    return embedding, (time.perf_counter() - start_time, resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 / 1024)

def split_knn_hg_hgmf_1svd(ahg, dim):
    start_time = time.perf_counter()
    # config.knn = 50
    aug_hg = split_knn_hypergraph(ahg.features, knn=50)
    if config.verbose:
        print(f"KNN time: {time.perf_counter() - start_time} KNN K: {config.knn}")
    athg = sp.vstack([ahg.hypergraph, aug_hg], format='csr')
    # HH_vol = (ahg.hypergraph.sum(1).A1**2).sum()
    # AA_vol = (aug_hg.sum(1).A1**2).sum()
    # attr_weight = HH_vol / (HH_vol + AA_vol)
    H_vol = (ahg.hypergraph.sum(1).A1).sum()
    A_vol = (aug_hg.sum(1).A1).sum()
    attr_weight = H_vol / (H_vol + A_vol)
    he_weights = np.concatenate([np.ones(ahg.m), np.full(aug_hg.shape[0], attr_weight)])
    # embedding = hgsvdlogsvd(athg, dim, hyperedge_weights=he_weights)
    if config.verbose:
        print(f"HG augmentation time: {time.perf_counter() - start_time}")
    embedding = hgmf_1svd(athg, dim, hyperedge_weights=he_weights)
    # embedding = hgmf(athg, dim, hyperedge_weights=he_weights)
    return embedding, (time.perf_counter() - start_time, resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 / 1024)

def knn_metis_hg_hgmf_1svd(ahg, dim):
    start_time = time.perf_counter()
    config.knn = 10
    if ahg.n>1000000:
        proportion = config.large_proportion
    else:
        proportion = config.small_proportion

    aug_hg_c, aug_hg_knn = knn_metis_hypergraph(ahg.features,weighted = True, num_clusters = list(map(lambda x: int(x * ahg.n), proportion)),withknn=True)
    aug_hg = sp.vstack([aug_hg_c, aug_hg_knn],format='csr')
    if config.verbose:
        print(f"KNN time: {time.perf_counter() - start_time} KNN K: {config.knn}")
    athg = sp.vstack([ahg.hypergraph, aug_hg], format='csr')

    H_vol = (ahg.hypergraph.sum(1).A1).sum()
    A_vol_knn = (aug_hg_knn.sum(1).A1).sum()
    A_vol_c = (aug_hg_c.sum(1).A1).sum()
    total_vol = H_vol + A_vol_knn + A_vol_c
    attr_weight_c = (H_vol + A_vol_knn) / (2 *total_vol)
    attr_weight_knn = (H_vol + A_vol_c) / (2 *total_vol)
    hg_weight = (A_vol_c + A_vol_knn) / (2 *total_vol)

    # H_e = ahg.m
    # A_e_knn = ahg.n
    # A_e_c = aug_hg_c.shape[0]
    # total_e = H_e + A_e_knn + A_e_c
    # attr_weight_c = (H_e + A_e_knn) / (2 *total_e)
    # attr_weight_knn = (H_e + A_e_c) / (2 *total_e)
    # hg_weight = (A_e_c + A_e_knn) / (2 *total_e)   
    
    # hg_weight = (A_vol_c + A_vol_knn) / (2 *(H_vol + A_vol_c + A_vol_knn)) *(ahg.hypergraph.sum(1)*ahg.hypergraph.shape[0]/H_vol)
    # he_weights = np.concatenate((attr_weight_c,attr_weight_knn,hg_weight)).A1.flatten()
    he_weights = np.concatenate((np.full(aug_hg_c.shape[0], attr_weight_c),np.full(ahg.n, attr_weight_knn),np.full(ahg.m, hg_weight)))
    # he_weights = np.concatenate([np.ones(ahg.m), np.full(aug_hg.shape[0], attr_weight)])
    if config.verbose:
        print(f"HG augmentation time: {time.perf_counter() - start_time}")
    embedding = hgmf_rwr_1svd(athg, dim, hyperedge_weights=he_weights)
    return embedding, (time.perf_counter() - start_time, resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 / 1024)

def knn_kahip_hg_hgmf_1svd(ahg, dim):
    start_time = time.perf_counter()
    config.knn = 10
    if ahg.n>1000000:
        proportion = config.large_proportion
    else:
        proportion = config.small_proportion

    aug_hg_c, aug_hg_knn = knn_metis_hypergraph(ahg.features,weighted = True, num_clusters = list(map(lambda x: int(x * ahg.n), proportion)),withknn=True)
    aug_hg = sp.vstack([aug_hg_c, aug_hg_knn],format='csr')
    if config.verbose:
        print(f"KNN time: {time.perf_counter() - start_time} KNN K: {config.knn}")
    athg = sp.vstack([ahg.hypergraph, aug_hg], format='csr')

    H_vol = (ahg.hypergraph.sum(1).A1).sum()
    A_vol_knn = (aug_hg_knn.sum(1).A1).sum()
    A_vol_c = (aug_hg_c.sum(1).A1).sum()
    total_vol = H_vol + A_vol_knn + A_vol_c
    attr_weight_c = (H_vol + A_vol_knn) / (2 *total_vol)
    attr_weight_knn = (H_vol + A_vol_c) / (2 *total_vol)
    hg_weight = (A_vol_c + A_vol_knn) / (2 *total_vol)
    he_weights = np.concatenate((np.full(aug_hg_c.shape[0], attr_weight_c),np.full(ahg.n, attr_weight_knn),np.full(ahg.m, hg_weight)))
    if config.verbose:
        print(f"HG augmentation time: {time.perf_counter() - start_time}")
    embedding = hgmf_rwr_1svd(athg, dim, hyperedge_weights=he_weights)
    return embedding, (time.perf_counter() - start_time, resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 / 1024)

def knn_kmeans_hg_hgmf_1svd(ahg,dim):
    start_time = time.perf_counter()
    config.knn = 10
    if ahg.n>1000000:
        proportion = config.large_proportion
    else:
        proportion = config.small_proportion

    # aug_hg= kmeans_hypergraph(ahg.features,num_clusters = [10])
    # aug_hg = config.beta* aug_hg
    # if config.verbose:
    #     print(f"KNN time: {time.perf_counter() - start_time} KNN K: {config.knn}")
    # athg = sp.vstack([ahg.hypergraph, aug_hg], format='csr')
    # H_vol = (ahg.hypergraph.sum(1).A1).sum()
    # A_vol = (aug_hg.sum(1).A1).sum()
    # attr_weight = H_vol / (H_vol + A_vol)
    # he_weights = np.concatenate([np.ones(ahg.m), np.full(aug_hg.shape[0], attr_weight)])
    # if config.verbose:
    #     print(f"HG augmentation time: {time.perf_counter() - start_time}")
    # embedding = hgmf_rwr_1svd(athg, dim, hyperedge_weights=he_weights)
    # return embedding, (time.perf_counter() - start_time, resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 / 1024)

    aug_hg_knn = knn_hypergraph(ahg.features, knn=config.knn, weighted=True, undirected=False)
    # aug_hg_c = kmeans_hypergraph(ahg.features,num_clusters = list(map(lambda x: int(x * ahg.n), proportion)))
    aug_hg_c = kmeans_hypergraph(ahg.features,num_clusters = config.num_clusters,weighted=True)
    aug_hg = sp.vstack([aug_hg_c, aug_hg_knn],format='csr')
    if config.verbose:
        print(f"KNN time: {time.perf_counter() - start_time} KNN K: {config.knn}")
    athg = sp.vstack([ahg.hypergraph, aug_hg], format='csr')

    H_vol = (ahg.hypergraph.sum(1).A1).sum()
    A_vol_knn = (aug_hg_knn.sum(1).A1).sum()
    A_vol_c = (aug_hg_c.sum(1).A1).sum()
    total_vol = H_vol + A_vol_knn + A_vol_c
    attr_weight_c = ((H_vol + A_vol_knn) / total_vol)
    attr_weight_knn = ((H_vol + A_vol_c) / total_vol)
    hg_weight = ((A_vol_c + A_vol_knn) / total_vol)
    he_weights = np.concatenate((np.full(aug_hg_c.shape[0], attr_weight_c),np.full(ahg.n, attr_weight_knn),np.full(ahg.m, hg_weight)))
    if config.verbose:
        print(f"HG augmentation time: {time.perf_counter() - start_time}")
    embedding = hgmf_rwr_1svd(athg, dim, hyperedge_weights=he_weights)
    return embedding, (time.perf_counter() - start_time, resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 / 1024)

def allno_hg_hgmf_1svd(ahg,dim):
    start_time = time.perf_counter()
    athg = ahg.hypergraph
    embedding = hgmf_rwr_1svd(athg, dim, hyperedge_weights=np.ones(ahg.m))
    return embedding, (time.perf_counter() - start_time, resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 / 1024)

def select_knn_hg_hgmf_1svd(ahg, dim):
    start_time = time.perf_counter()
    aug_hg = select_knn_hypergraph(ahg.hypergraph,ahg.features, knn=config.knn, weighted=True, undirected=False)
    aug_hg = config.beta* aug_hg
    if config.verbose:
        print(f"KNN time: {time.perf_counter() - start_time} KNN K: {config.knn}")
    athg = sp.vstack([ahg.hypergraph, aug_hg], format='csr')
    # HH_vol = (ahg.hypergraph.sum(1).A1**2).sum()
    # AA_vol = (aug_hg.sum(1).A1**2).sum()
    # attr_weight = HH_vol / (HH_vol + AA_vol)
    H_vol = (ahg.hypergraph.sum(1).A1).sum()
    A_vol = (aug_hg.sum(1).A1).sum()
    attr_weight = H_vol / (H_vol + A_vol)
    # H_e = ahg.hypergraph.shape[0]
    # A_e = aug_hg.shape[0]
    # attr_weight = H_e / (H_e + A_e)
    he_weights = np.concatenate([np.ones(ahg.m), np.full(aug_hg.shape[0], attr_weight)])
    # embedding = hgsvdlogsvd(athg, dim, hyperedge_weights=he_weights)
    if config.verbose:
        print(f"HG augmentation time: {time.perf_counter() - start_time}")
    # embedding = hgmf_1svd(athg, dim, hyperedge_weights=he_weights)
    embedding = hgmf_rwr_1svd(athg, dim, hyperedge_weights=he_weights)
    # embedding = hgmf(athg, dim, hyperedge_weights=he_weights)
    return embedding, (time.perf_counter() - start_time, resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 / 1024)

def multi_knn_hg_hgmf_pts(ahg, dim):
    start_time = time.perf_counter()
    # for i, knn_k in enumerate(config.multi_knn) :
    #     aug_knn = knn_hypergraph(ahg.features, knn=knn_k, weighted=True, undirected=False)
    #     if i !=0:
    #         aug_hg = sp.vstack([aug_hg, aug_knn], format='csr')
    #     else:
    #         aug_hg = aug_knn
    aug_hg = multiknn_hypergraph(ahg.features, k_list=config.multi_knn,weighted=True)
    aug_hg = config.beta* aug_hg
    if config.verbose:
        print(f"KNN time: {time.perf_counter() - start_time} KNN K: {config.knn}")
    athg = sp.vstack([ahg.hypergraph, aug_hg], format='csr')
    H_vol = (ahg.hypergraph.sum(1).A1).sum()
    A_vol = (aug_hg.sum(1).A1).sum()
    attr_weight = H_vol / (H_vol + A_vol)
    he_weights = np.concatenate([np.ones(ahg.m), np.full(aug_hg.shape[0], attr_weight)])
    if config.verbose:
        print(f"HG augmentation time: {time.perf_counter() - start_time}")
    # embedding = hgmf_pts(athg, dim, hyperedge_weights=he_weights)
    # embedding = hgmf_asym_pts(athg, dim, hyperedge_weights=he_weights)
    # embedding = hgmf_asym_normx_pts(athg, dim, hyperedge_weights=he_weights)
    # embedding = hgmf_rwr_1svd(athg, dim, hyperedge_weights=he_weights)
    # embedding = hgmf(athg, dim, hyperedge_weights=he_weights)
    embedding = hgmf_full_sym_pts(athg, dim, hyperedge_weights=he_weights)
    return embedding, (time.perf_counter() - start_time, resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 / 1024)

def knn_kmeans_hg_hgmf_pts(ahg, dim):
    start_time = time.perf_counter()
    config.knn = 10
    if ahg.n>1000000:
        proportion = config.large_proportion
    else:
        proportion = config.small_proportion

    aug_hg_knn = knn_hypergraph(ahg.features, knn=config.knn, weighted=True, undirected=False)
    aug_hg_c = kmeans_hypergraph(ahg.features,num_clusters = config.num_clusters,weighted=True)
    aug_hg = sp.vstack([aug_hg_c, aug_hg_knn],format='csr')
    if config.verbose:
        print(f"KNN time: {time.perf_counter() - start_time} KNN K: {config.knn}")
    athg = sp.vstack([ahg.hypergraph, aug_hg], format='csr')

    H_vol = (ahg.hypergraph.sum(1).A1).sum()
    A_vol_knn = (aug_hg_knn.sum(1).A1).sum()
    A_vol_c = (aug_hg_c.sum(1).A1).sum()
    total_vol = H_vol + A_vol_knn + A_vol_c
    attr_weight_c = ((H_vol + A_vol_knn) / total_vol)
    attr_weight_knn = ((H_vol + A_vol_c) / total_vol)
    hg_weight = ((A_vol_c + A_vol_knn) / total_vol)
    he_weights = np.concatenate((np.full(aug_hg_c.shape[0], attr_weight_c),np.full(ahg.n, attr_weight_knn),np.full(ahg.m, hg_weight)))
    if config.verbose:
        print(f"HG augmentation time: {time.perf_counter() - start_time}")
    # embedding = hgmf_rwr_1svd(athg, dim, hyperedge_weights=he_weights)
    embedding = hgmf_full_sym_pts(athg, dim, hyperedge_weights=he_weights)
    return embedding, (time.perf_counter() - start_time, resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 / 1024)

def knn_hgedge_hgmf_pts(ahg, dim):
    start_time = time.perf_counter()
    # for i, knn_k in enumerate(config.multi_knn) :
    #     aug_knn = knn_hypergraph(ahg.features, knn=knn_k, weighted=True, undirected=False)
    #     if i !=0:
    #         aug_hg = sp.vstack([aug_hg, aug_knn], format='csr')
    #     else:
    #         aug_hg = aug_knn
    aug_hg = knn_hg_hypergraph(ahg.features, knn=config.knn, weighted=False, undirected=False)
    aug_hg = config.beta* aug_hg
    if config.verbose:
        print(f"KNN time: {time.perf_counter() - start_time} KNN K: {config.knn}")
    athg = sp.vstack([ahg.hypergraph, aug_hg], format='csr')
    H_vol = (ahg.hypergraph.sum(1).A1).sum()
    A_vol = (aug_hg.sum(1).A1).sum()
    attr_weight = H_vol / (H_vol + A_vol)
    he_weights = np.concatenate([np.ones(ahg.m), np.full(aug_hg.shape[0], attr_weight)])
    if config.verbose:
        print(f"HG augmentation time: {time.perf_counter() - start_time}")
    # embedding = hgmf_pts(athg, dim, hyperedge_weights=he_weights)
    # embedding = hgmf_asym_pts(athg, dim, hyperedge_weights=he_weights)
    # embedding = hgmf_asym_normx_pts(athg, dim, hyperedge_weights=he_weights)
    # embedding = hgmf_rwr_1svd(athg, dim, hyperedge_weights=he_weights)
    # embedding = hgmf(athg, dim, hyperedge_weights=he_weights)
    embedding = hgmf_full_sym_pts(athg, dim, hyperedge_weights=he_weights)
    return embedding, (time.perf_counter() - start_time, resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 / 1024)

def ghac_hg_hgmf(ahg, dim):
    start_time = time.perf_counter()
    # knn_hg = knn_hypergraph(ahg.features, knn=10)
    # aug_hg = parhac_hypergraph(ahg.features, knn=10, levels=[0.005])
    aug_hg = parhac_hypergraph(ahg.features, knn=50, levels=[0.05, 0.1, 0.2])
    athg = sp.vstack([ahg.hypergraph, aug_hg], format='csr')
    # HH_vol = (ahg.hypergraph.sum(1).A1**2).sum()
    # AA_vol = (aug_hg.sum(1).A1**2).sum()
    # attr_weight = HH_vol / (HH_vol + AA_vol)
    H_vol = (ahg.hypergraph.sum(1).A1).sum()
    A_vol = (aug_hg.sum(1).A1).sum()
    attr_weight = H_vol / (H_vol + A_vol)
    he_weights = np.concatenate([np.ones(ahg.m), np.full(aug_hg.shape[0], attr_weight)])
    # embedding = hgsvdlogsvd(athg, dim, hyperedge_weights=he_weights)
    embedding = hgmf_1svd(athg, dim, hyperedge_weights=he_weights)
    # embedding = hgmf(athg, dim, hyperedge_weights=he_weights)
    return embedding, (time.perf_counter() - start_time, resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 / 1024)

def knn_ghac_hg_hgmf(ahg, dim):
    start_time = time.perf_counter()
    knn_hg = knn_hypergraph(ahg.features, knn=10, weighted=True)
    # aug_hg = parhac_hypergraph(ahg.features, knn=10, levels=[0.005])
    ghac_hg = parhac_hypergraph(ahg.features, knn=50, levels=[0.05])
    athg = sp.vstack([ahg.hypergraph, knn_hg, ghac_hg], format='csr')
    # HH_vol = (ahg.hypergraph.sum(1).A1**2).sum()
    # AA_vol = (aug_hg.sum(1).A1**2).sum()
    # attr_weight = HH_vol / (HH_vol + AA_vol)
    H_vol = (ahg.hypergraph.sum(1).A1).sum()
    knn_vol = (knn_hg.sum(1).A1).sum()
    ghac_vol = (ghac_hg.sum(1).A1).sum()
    knn_weight = H_vol / (H_vol + knn_vol)
    ghac_weight = H_vol / (H_vol + ghac_vol) /2.0
    he_weights = np.concatenate([np.ones(ahg.m), np.full(knn_hg.shape[0], knn_weight), np.full(ghac_hg.shape[0], ghac_weight)])
    # embedding = hgsvdlogsvd(athg, dim, hyperedge_weights=he_weights)
    embedding = hgmf_1svd(athg, dim, hyperedge_weights=he_weights)
    # embedding = hgmf(athg, dim, hyperedge_weights=he_weights)
    return embedding, (time.perf_counter() - start_time, resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 / 1024)

def knn_p_svd(ahg, dim):
    import faiss
    index_type = "IVF1000,PQ40"
    start_time = time.perf_counter()
    # construct exact knn with ahg.features
    features = ahg.features.toarray()
    n, d = features.shape
    faiss.normalize_L2(features)
    if n > 1000000:
        if config.verbose:
            print(f"use specified index type {index_type} for large-scale graphs")
        index = faiss.index_factory(d, index_type, faiss.METRIC_INNER_PRODUCT)
        index.train(features)
    else:
        index = faiss.IndexFlatIP(d)
    index.add(features)
    D, I = index.search(features, config.knn+1)
    knng = sp.csr_matrix((D.flatten(), I.flatten(), np.arange(0, n*(config.knn+1)+1, config.knn+1)), shape=(n, n))
    p_knn = normalize(knng, norm='l1', axis=1)
    p_he = normalize(ahg.hypergraph, norm='l1', axis=0)
    p_hv = normalize(ahg.hypergraph, norm='l1', axis=1)
    def knn_hg_p(mat):
        return (1-config.p_beta)*p_he.T @ (p_hv @ mat) + config.p_beta*p_knn @ mat
    def knn_hg_p_h(mat):
        return (1-config.p_beta)*p_hv.T @ (p_he @ mat) + config.p_beta*p_knn.T @ mat
    p_lo = sp.linalg.LinearOperator(matvec=knn_hg_p, rmatvec=knn_hg_p_h, shape=(n, n))
    u, s, vt = sp.linalg.svds(p_lo, k=dim/2)
    u = u @ np.diag(np.sqrt(s))
    v = vt.T @ np.diag(np.sqrt(s))
    embedding = np.concatenate((u, v), axis=1)
    embedding = normalize(embedding, norm='l2', axis=1)
    return embedding, (time.perf_counter() - start_time, resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 / 1024)

def knn_hg_hgmf_edge_1svd(ahg,dim):

    start_time = time.perf_counter()
    aug_hg = knn_hypergraph(ahg.features, knn=config.knn, weighted=True, undirected=False)
    aug_hg = config.beta* aug_hg
    if config.verbose:
        print(f"KNN time: {time.perf_counter() - start_time} KNN K: {config.knn}")
    athg = sp.vstack([ahg.hypergraph, aug_hg], format='csr')
    H_vol = (ahg.hypergraph.sum(1).A1).sum()
    A_vol = (aug_hg.sum(1).A1).sum()
    attr_weight = H_vol / (H_vol + A_vol)
    he_weights = np.concatenate([np.ones(ahg.m), np.full(aug_hg.shape[0], attr_weight)])
    if config.verbose:
        print(f"HG augmentation time: {time.perf_counter() - start_time}")
    embedding = hgmf_edge_1svd(athg, dim, hyperedge_weights=he_weights)
    return embedding, (time.perf_counter() - start_time, resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 / 1024)

def knn_hg_hgmf_edge_full_pts(ahg,dim):

    start_time = time.perf_counter()
    aug_hg = knn_hypergraph(ahg.features, knn=config.knn, weighted=True, undirected=False)
    aug_hg = config.beta* aug_hg
    if config.verbose:
        print(f"KNN time: {time.perf_counter() - start_time} KNN K: {config.knn}")
    athg = sp.vstack([ahg.hypergraph, aug_hg], format='csr')
    H_vol = (ahg.hypergraph.sum(1).A1).sum()
    A_vol = (aug_hg.sum(1).A1).sum()
    attr_weight = H_vol / (H_vol + A_vol)
    he_weights = np.concatenate([np.ones(ahg.m), np.full(aug_hg.shape[0], attr_weight)])
    if config.verbose:
        print(f"HG augmentation time: {time.perf_counter() - start_time}")
    embedding = hgmf_edge_full_sym_pts(athg, dim, hyperedge_weights=he_weights)
    return embedding, (time.perf_counter() - start_time, resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 / 1024)

def avg_hyperedge_embedder(node_embedder):
    def avg_agg_hyperedge_embedder(ahg, dim):
        start_time = time.perf_counter()
        node_embedding = node_embedder(ahg, dim)[0]
        hyperedge_embedding = ahg.hypergraph @ node_embedding
        return hyperedge_embedding, (time.perf_counter() - start_time, resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 / 1024)
    return avg_agg_hyperedge_embedder

def range_hyperedge_embedder(node_embedder):
    def range_agg_hyperedge_embedder(ahg, dim):
        start_time = time.perf_counter()
        hyperedges = ahg.hypergraph
        node_embedding = node_embedder(ahg, dim)[0]
        embeddings_max = np.maximum.reduceat(node_embedding[hyperedges.indices, :], hyperedges.indptr[:-1], 0)
        embeddings_min = np.minimum.reduceat(node_embedding[hyperedges.indices, :], hyperedges.indptr[:-1], 0)
        hyperedge_embedding = embeddings_max - embeddings_min
        return hyperedge_embedding, (time.perf_counter() - start_time, resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 / 1024)
    return range_agg_hyperedge_embedder

def minmax_hyperedge_embedder(node_embedder):
    def minmax_agg_hyperedge_embedder(ahg, dim):
        start_time = time.perf_counter()
        hyperedges = ahg.hypergraph
        node_embedding = node_embedder(ahg, dim//2)[0]
        embeddings_max = np.maximum.reduceat(node_embedding[hyperedges.indices, :], hyperedges.indptr[:-1], 0)
        embeddings_min = np.minimum.reduceat(node_embedding[hyperedges.indices, :], hyperedges.indptr[:-1], 0)
        hyperedge_embedding = np.hstack([embeddings_max, embeddings_min])
        return hyperedge_embedding, (time.perf_counter() - start_time, resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 / 1024)
    return minmax_agg_hyperedge_embedder

if __name__ == '__main__':

    pkl_file = open(f"temp/ahg.pkl","rb")
    ahg = pkl.load(pkl_file)
    embedding, nc_cost = knn_hg_hgmf_full_pts(ahg, config.embedding_dim)
    print(f"&&&&& {nc_cost[0]} {nc_cost[1]}")
    np.save(f"temp/ours_{ahg.name}_emb.npy",embedding)
    