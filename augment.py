import scipy.sparse as sp
import numpy as np
from scipy.cluster.hierarchy import linkage,fcluster
from scipy.spatial.distance import pdist
import config
import time
from parhac import parhac
from load import Dataset, AttrHypergraph

def knn_hypergraph(X, knn = 10, weighted = True, index_type = "IVF512,PQ10", undirected = False, return_index = False):
    import faiss
    # construct exact knn with ahg.features
    features = X.toarray()
    n, d = features.shape
    faiss.normalize_L2(features)
    if n > 100000:
        if config.verbose:
            print(f"use specified index type {index_type} for large-scale graphs")
        index = faiss.index_factory(d, index_type, faiss.METRIC_INNER_PRODUCT)
        index.train(features)
    else:
        index = faiss.IndexFlatIP(d)
    index.add(features)
    D, I = index.search(features, knn+1)
    # construct hypergraph
    if weighted == True:
        H = sp.csr_matrix((D.flatten(), I.flatten(), np.arange(0, n*(knn+1)+1, knn+1)), shape=(n, n))
    else:
        H = sp.csr_matrix((np.ones(n*(knn+1)), I.flatten(), np.arange(0, n*(knn+1)+1, knn+1)), shape=(n, n))
    self_weight = 1.0
    if undirected == True:
        H = H + H.T
    H.setdiag(self_weight)
    if return_index:
        return H, index
    else:
        return H

def knn_graph(X, knn = 10, weighted = True, index_type = "IVF512,PQ10", undirected = False, return_index = False):
    import faiss
    # construct exact knn with ahg.features
    features = X.toarray()
    n, d = features.shape
    faiss.normalize_L2(features)
    if n > 100000:
        if config.verbose:
            print(f"use specified index type {index_type} for large-scale graphs")
        index = faiss.index_factory(d, index_type, faiss.METRIC_INNER_PRODUCT)
        index.train(features)
    else:
        index = faiss.IndexFlatIP(d)
    index.add(features)
    D, I = index.search(features, knn+1)
    # construct hypergraph
    if weighted == True:
        G = sp.csr_matrix((D.flatten(), I.flatten(), np.arange(0, n*(knn+1)+1, knn+1)), shape=(n, n))
    else:
        G = sp.csr_matrix((np.ones(n*(knn+1)), I.flatten(), np.arange(0, n*(knn+1)+1, knn+1)), shape=(n, n))
    self_weight = 1.0
    if undirected == True:
        G = G + G.T
    G.setdiag(self_weight)
    if return_index:
        return G, index
    else:
        return G

def multiknn_hypergraph(X, k_list, weighted = True, index_type = "IVF1000,PQ40", undirected = False):
    import faiss
    # construct exact knn with ahg.features
    features = X.toarray()
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
    knn = k_list[-1] #search with max k
    D, I = index.search(features, knn+1)
    H_full = None
    for knn in k_list:
        # construct hypergraph
        if weighted == True:
            H = sp.csr_matrix((D[:,0:knn+1].flatten(), I[:,0:knn+1].flatten(), np.arange(0, n*(knn+1)+1, knn+1)), shape=(n, n))
        else:
            H = sp.csr_matrix((np.ones(n*(knn+1)), I[:,0:knn+1].flatten(), np.arange(0, n*(knn+1)+1, knn+1)), shape=(n, n))
        self_weight = 1.0
        if undirected == True:
            H = H + H.T
        H.setdiag(self_weight)
        if H_full == None:
            H_full = H
        else:
            H_full = sp.vstack([H_full,H],format='csr')
    return H_full

def knn_hg_hypergraph(X, knn = 10, weighted = True, index_type = "IVF1000,PQ40", undirected = False):
    import faiss
    # construct exact knn with ahg.features
    features = X.toarray()
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
    D, I = index.search(features, knn+1)
    row = np.concatenate((np.arange(n*knn),np.arange(n*knn)))
    col = np.concatenate((np.repeat(np.arange(n),knn),I[:,1:].flatten()))
    data = np.concatenate((D[:,1:].flatten(),D[:,1:].flatten()))
    # construct hypergraph
    if weighted == True:
        H = sp.csr_matrix((data, (row, col)), shape=(n*knn, n))
    else:
        H = sp.csr_matrix((np.ones(n*knn*2), (row, col)), shape=(n*knn, n))
    return H
# house_hb knn=20, beta=20
def l2_knn_hypergraph(X, knn = 10, weighted = True, index_type = "IVF1000,PQ40"):
    import faiss
    # construct exact knn with ahg.features
    features = X.toarray()
    n, d = features.shape
    faiss.normalize_L2(features)
    if n > 1000000:
        print(f"use specified index type {index_type} for large-scale graphs")
        index = faiss.index_factory(d, index_type, faiss.METRIC_L2)
        index.train(features)
    else:
        index = faiss.IndexFlatL2(d)
    index.add(features)
    D, I = index.search(features, config.knn+1)
    # construct hypergraph
    if weighted == True:
        H = sp.csr_matrix((D.flatten(), I.flatten(), np.arange(0, n*(config.knn+1)+1, config.knn+1)), shape=(n, n))
    else:
        H = sp.csr_matrix((np.ones(n*(config.knn+1)), I.flatten(), np.arange(0, n*(config.knn+1)+1, config.knn+1)), shape=(n, n))
    self_weight = 1.0
    H.setdiag(self_weight)
    return H

def mutual_knn_hypergraph(X, knn = 50, index_type = "IVF1000,PQ40"):
    import faiss
    # construct exact knn with ahg.features
    features = X.toarray()
    n, d = features.shape
    faiss.normalize_L2(features)
    if n > 1000000:
        print(f"use specified index type {index_type} for large-scale graphs")
        index = faiss.index_factory(d, index_type, faiss.METRIC_INNER_PRODUCT)
        index.train(features)
    else:
        index = faiss.IndexFlatIP(d)
    index.add(features)
    D, I = index.search(features, config.knn+1)
    # D[D<0] = 0.0
    # construct hypergraph
    H = sp.csr_matrix((D.flatten(), I.flatten(), np.arange(0, n*(config.knn+1)+1, config.knn+1)), shape=(n, n))
    H = H.multiply(H.T)
    H.data = np.sqrt(H.data)
    return H

def split_knn_hypergraph(X, knn = 10, index_type = "IVF1000,PQ40"):
    import faiss
    # construct exact knn with ahg.features
    features = X.toarray()
    feature_splits = [np.ascontiguousarray(features[:, :features.shape[1]//2]), np.ascontiguousarray(features[:, features.shape[1]//2:])]
    knn_hypergraphs = []
    for features in feature_splits:
        n, d = features.shape
        faiss.normalize_L2(features)
        if n > 1000000:
            print(f"use specified index type {index_type} for large-scale graphs")
            index = faiss.index_factory(d, index_type, faiss.METRIC_INNER_PRODUCT)
            index.train(features)
        else:
            index = faiss.IndexFlatIP(d)
        index.add(features)
        D, I = index.search(features, config.knn+1)
        # construct hypergraph
        H = sp.csr_matrix((D.flatten(), I.flatten(), np.arange(0, n*(config.knn+1)+1, config.knn+1)), shape=(n, n))
        knn_hypergraphs.append(H)
    knn_hg = sp.vstack(knn_hypergraphs)
    return knn_hg

def knn_metis_hypergraph(X, knn = 10, index_type = "IVF1000,PQ40", weighted = False, num_clusters = [10], withknn = False):
    import faiss
    import pymetis
    # construct exact knn with ahg.features
    features = X.toarray()
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
    # construct hypergraph without similarity
    if weighted == True:
        KNN = sp.csr_matrix((D.flatten(), I.flatten(), np.arange(0, n*(config.knn+1)+1, config.knn+1)), shape=(n, n))
        KNN_hg = KNN.copy()
        KNN.setdiag(0.0)
        KNN_hg.setdiag(1.0)
        KNN.data *=100 # 100*similarity weight integer
    else:
        KNN_hg = KNN = sp.csr_matrix((np.ones(n*(config.knn+1)), I.flatten(), np.arange(0, n*(config.knn+1)+1, config.knn+1)), shape=(n, n))
        KNN.setdiag(0.0)
        KNN_hg.setdiag(1.0)
        KNN.eliminate_zeros()
    KNN = KNN+KNN.T
    xadj = KNN.indptr
    adjncy = KNN.indices
    for i , num_cluster in enumerate(num_clusters):
        _, clusters = pymetis.part_graph(num_cluster, xadj=xadj, adjncy=adjncy, eweights=KNN.data.astype(int)) 
        if config.verbose:
            print(f"hyperedge number {num_cluster}" )
        data = np.ones(n)
        row = np.asarray(np.array(clusters))
        col = np.arange(0,n)
        H_level = sp.csr_matrix((data, (row, col)),dtype='float')
        H_level = H_level [H_level.getnnz(axis=1)>1,:] # make sure every hyperedge has at least two nodes
        if i == 0:
            H = H_level
        else:
            H = sp.vstack((H,H_level))
    # if withknn == True:
    #     H = sp.vstack((H,KNN_hg))
    return H, KNN_hg

def knn_kahip_hypergraph(X, knn = 10, index_type = "IVF1000,PQ40", weighted = False, num_clusters = [50], withknn = False):
    import faiss
    import kahip
    # construct exact knn with ahg.features
    features = X.toarray()
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
    # construct hypergraph without similarity
    if weighted == True:
        KNN = sp.csr_matrix((D.flatten(), I.flatten(), np.arange(0, n*(config.knn+1)+1, config.knn+1)), shape=(n, n))
        KNN.setdiag(0.0)
        KNN_hg = KNN.copy()
        KNN_hg.setdiag(1.0)
        KNN.data *=100 # 100*similairty weight integer
    else:
        KNN_hg = KNN = sp.csr_matrix((np.ones(n*(config.knn+1)), I.flatten(), np.arange(0, n*(config.knn+1)+1, config.knn+1)), shape=(n, n))
        KNN.setdiag(0.0)
        KNN_hg.setdiag(1.0)
        KNN.eliminate_zeros()
    KNN = KNN+KNN.T
    xadj = KNN.indptr
    adjncy = KNN.indices
    vwgt = np.ones(n).astype('int32')
    adjcwgt = KNN.data.astype(int)
    supress_output = 0
    imbalance = 0.03
    seed = 0
    mode = 0
    for i , num_cluster in enumerate(num_clusters):
        _, clusters = kahip.kaffpa(vwgt, xadj, adjcwgt, adjncy,  num_cluster, imbalance, supress_output, seed, mode)
        if config.verbose:
            print(f"hyperedge number {num_cluster}" )
        data = np.ones(n)
        row = np.asarray(np.array(clusters))
        col = np.arange(0,n)
        H_level = sp.csr_matrix((data, (row, col)),dtype='float')
        H_level = H_level [H_level.getnnz(axis=1)>1,:] # make sure every hyperedge has at least two nodes
        if i == 0:
            H = H_level
        else:
            H = sp.vstack((H,H_level))
    # if withknn == True:
    #     H = sp.vstack((H,KNN_hg))
    return H,KNN_hg

def kmeans_hypergraph(X,num_clusters = [10],weighted=False):
    from sklearn.cluster import KMeans
    num_node = X.shape[0]
    # return number of clusters = levels[i]*num_node
    for i , num_cluster in enumerate(num_clusters):
        kmeans = KMeans(n_clusters=num_cluster,init="k-means++", n_init="auto").fit(X)
        clusters = kmeans.labels_ 
        if weighted == True:
            centroids = kmeans.cluster_centers_
            from sklearn.metrics.pairwise import cosine_similarity
            from sklearn.metrics import pairwise_distances
            similarity = []
            for j in range(num_node):
                X1 = np.stack((X[j].toarray().squeeze(),centroids[clusters[j]]))
                cos_sim = 1 - pairwise_distances(X1,metric='cosine')[0,1]
                similarity.append(cos_sim)
        num_clusters = np.unique(clusters).shape[0]
        data = np.ones(num_node)
        if weighted == True:
            data = np.asarray(similarity)
        row = np.asarray(clusters)
        col = np.arange(0,num_node)
        H_level = sp.csr_matrix((data, (row, col)),dtype='float')
        H_level = H_level [H_level.getnnz(axis=1)>1,:] # make sure every hyperedge has at least two nodes
        if i == 0:
            H = H_level
        else:
            H = sp.vstack((H,H_level))

    return H

def agglomerative_hypergraph(X, levels = [0.1]):
    
    y = pdist(X.toarray(), 'cosine')
    y [np.isnan(y)] = 1 # fix error nan cos similarity
    Z = linkage(y, method='average', optimal_ordering=False) # par-filtered graph utilize "average" and "complete" for "method"
    num_node = X.shape[0]

    # return number of clusters = levels[i]*num_node
    for i , propotion in enumerate(levels):
        clusters = fcluster(Z, t=int(propotion*num_node), criterion='maxclust')
        num_clusters = np.unique(clusters).shape[0]
        if config.verbose:
            print(f"hyperedge number when propotion = {propotion}: {num_clusters}" )
        data = np.ones(num_node)
        row = np.asarray(clusters-1)
        col = np.arange(0,num_node)
        H_level = sp.csr_matrix((data, (row, col)),dtype='float')
        H_level = H_level [H_level.getnnz(axis=1)>1,:] # make sure every hyperedge has at least two nodes
        if i == 0:
            H = H_level
        else:
            H = sp.vstack((H,H_level))

    return H

def parhac_hypergraph(X, knn = 10, levels = [0.1], unweighted = False):

    # return linkage from C++ implementation
    Z = parhac(X, knn_k = knn, unweighted = unweighted)
    num_node = X.shape[0]
    # return number of clusters = levels[i]*num_node
    for i, proportion in enumerate(levels):
        clusters = fcluster(Z, t=int(proportion*num_node), criterion='maxclust')
        num_clusters = np.unique(clusters).shape[0]
        if config.verbose:
            print(f"hyperedge number when propotion = {proportion}: {num_clusters}" )
        data = np.ones(num_node)
        row = np.asarray(clusters-1)
        col = np.arange(0,num_node)
        H_level = sp.csr_matrix((data, (row, col)),dtype='float')
        H_level = H_level [H_level.getnnz(axis=1)>1,:] # make sure every hyperedge has at least two nodes
        if i == 0:
            H = H_level
        else:
            H = sp.vstack((H,H_level))

    return H

def select_knn_hypergraph(H_o, X, knn = 10, weighted = True, index_type = "IVF1000,PQ40", undirected = False):
    import faiss
    # construct exact knn with ahg.features
    features = X.toarray()
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
    # construct hypergraph
    if weighted == True:
        H = sp.csr_matrix((D.flatten(), I.flatten(), np.arange(0, n*(config.knn+1)+1, config.knn+1)), shape=(n, n))
    else:
        H = sp.csr_matrix((np.ones(n*(config.knn+1)), I.flatten(), np.arange(0, n*(config.knn+1)+1, config.knn+1)), shape=(n, n))
    self_weight = 1.0
    if undirected == True:
        H = H + H.T
    H.setdiag(self_weight)

    col_sum = H_o.sum(axis=0).A1
    top_pro = int(n*0.7)
    sorted_indices = np.argsort(col_sum)[:top_pro]
    H = H[sorted_indices]

    return H

if __name__ == '__main__':
    hb_datasets = ["citeseer_cite_hb", "cora_cite_hb", "pubmed_cite_hb", "cora_coauth_hb", "dblp_copub_hb", "dblp_coauth_hb", "aminer_hb", "imdb_hb", "modelnet_40_hb", "news_hb", "house_hb"]
    ahc_datasets = ['cora-CA', 'dblp-CA', 'cora-CC', 'citeseer-CC', 'pubmed-CC', 'query', '20news']
    large_datasets = ['amazon','magpm']
    all_datasets = ahc_datasets + large_datasets
    for data_name in all_datasets:
        print(f"Evaluating on {data_name}")
        dataset = Dataset(data_name)
        X = dataset.features
        H = dataset.hypergraph
        # start_time = time.time()
        # agglomerative_hypergraph(X,levels = [0.1])
        # time_elapsed = time.time()-start_time
        # print(f'agglomerative construct time: {time_elapsed}')

        # start_time = time.time()
        # knn_hypergraph(X, weighted = True)
        # time_elapsed = time.time()-start_time
        # print(f'knn construct time: {time_elapsed}')

        # start_time = time.time()
        # multiknn_hypergraph(X, k_list= config.multi_knn, weighted = True)
        # time_elapsed = time.time()-start_time
        # print(f'knn construct time: {time_elapsed}')

        start_time = time.time()
        config.knn = 10
        knn_hg_hypergraph(X, knn = config.knn, weighted = True)
        time_elapsed = time.time()-start_time
        print(f'knn construct time: {time_elapsed}')

        # start_time = time.time()
        # config.knn = 10
        # H = knn_metis_hypergraph(X, num_clusters = [int(dataset.features.shape[0]*0.1)],withknn= True, weighted=True)
        # time_elapsed = time.time()-start_time
        # print(f'knn metis construct time: {time_elapsed}')

        # start_time = time.time()
        # config.knn = 10
        # H = knn_kahip_hypergraph(X, weighted = True, num_clusters = [100])
        # time_elapsed = time.time()-start_time
        # print(f'knn kahip construct time: {time_elapsed}')

        # start_time = time.time()
        # H = kmeans_hypergraph(X, num_clusters = [50], weighted = True)
        # time_elapsed = time.time()-start_time
        # print(f'kmeans construct time: {time_elapsed}')

        # start_time = time.time()
        # H = parhac_hypergraph(X, knn = 10, levels = [0.1], unweighted = True)
        # time_elapsed = time.time()-start_time
        # print(f'parhac construct time: {time_elapsed}')

        # start_time = time.time()
        # config.knn = 10
        # H =select_knn_hypergraph(H, X, knn = 10, weighted = True, index_type = "IVF1000,PQ40", undirected = False)
        # time_elapsed = time.time()-start_time
        # print(f'knn metis construct time: {time_elapsed}')


