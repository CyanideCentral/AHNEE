import pickle
import numpy as np
import scipy.sparse as sp
import os
import matplotlib.pyplot as plt
from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import rbf_kernel

def pkl_to_npz():
    dataset_types = {'coauthorship': ['cora', 'dblp'], 'cocitation': ['cora', 'citeseer', 'pubmed']}
    for dataset_type in dataset_types:
        for dataset in dataset_types[dataset_type]:
            hypergraph = pickle.load(open(f'../ahrw/data/{dataset_type}/{dataset}/hypergraph.pickle', 'rb'))
            features = pickle.load(open(f'../ahrw/data/{dataset_type}/{dataset}/features.pickle', 'rb'))
            labels = pickle.load(open(f'../ahrw/data/{dataset_type}/{dataset}/labels.pickle', 'rb'))
            adj = np.zeros((len(hypergraph), features.shape[0]))
            for index, edge in enumerate(hypergraph):
                hypergraph[edge] = list(hypergraph[edge])
                adj[index, hypergraph[edge]] = 1
            adj_sp = sp.csr_matrix(adj)
            folder = f"data/{dataset}-{'CA' if dataset_type == 'coauthorship' else 'CC'}"
            os.makedirs(folder, exist_ok=True)
            sp.save_npz(f'{folder}/hypergraph.npz', adj_sp)
            sp.save_npz(f'{folder}/features.npz', sp.csr_matrix(features))
            np.save(f'{folder}/labels.npy', labels)

def aggregate_by_hyperedges(embeddings, hyperedges, reduce = 'sum'):
    if reduce == 'sum':
        return hyperedges @ embeddings
    elif reduce == 'mean':
        return hyperedges @ embeddings / np.asarray(hyperedges.sum(axis = 1))
    elif reduce == 'range':
        embeddings_max = np.maximum.reduceat(embeddings[hyperedges.indices, :], hyperedges.indptr[:-1], 0)
        embeddings_min = np.minimum.reduceat(embeddings[hyperedges.indices, :], hyperedges.indptr[:-1], 0)
        return embeddings_max - embeddings_min
    elif reduce == 'maxmin':
        embeddings_max = np.maximum.reduceat(embeddings[hyperedges.indices, :], hyperedges.indptr[:-1], 0)
        embeddings_min = np.minimum.reduceat(embeddings[hyperedges.indices, :], hyperedges.indptr[:-1], 0)
        return np.c_[embeddings_max, embeddings_min]

def extract_from_log(log):
    log = log.split('\n')
    time_ram_line = [line for line in log if line.startswith('&&&&&')]
    if len(time_ram_line) == 0:
        return None, None
    time_ram_line = time_ram_line[0]
    time = float(time_ram_line.split(' ')[1])
    ram = float(time_ram_line.split(' ')[2])
    return time, ram

def add_unconnected_singletons(hypergraph):
    n = hypergraph.shape[1]
    singletons = np.setdiff1d(np.arange(n), hypergraph.indices)
    if len(singletons) > 0:
        indices = np.concatenate((hypergraph.indices, singletons))
        indptr = np.r_[hypergraph.indptr, hypergraph.indptr[-1]+1:hypergraph.indptr[-1]+len(singletons)+1]
        data = np.concatenate((hypergraph.data, np.ones(len(singletons))))
        hypergraph = sp.csr_matrix((data, indices, indptr), shape = (hypergraph.shape[0]+len(singletons), n))
    return hypergraph

# Ref HypeBoy Fig. 2
def dimension_collapse_analysis(embeddings):
    U, S, Vh = np.linalg.svd(embeddings, full_matrices=False)
    # plot all singular values
    plt.plot(S)
    plt.savefig('temp/singular_values.png')

def uniformity_metric(embeddings):
    Z = normalize(embeddings, axis = 1, norm = 'l2')
    uniformity = rbf_kernel(Z).mean()
    print(f"Uniformity: {uniformity}")

if __name__ == '__main__':
    pkl_to_npz()