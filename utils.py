import numpy as np
import scipy.sparse as sp

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

def add_unconnected_singletons(hypergraph):
    n = hypergraph.shape[1]
    singletons = np.setdiff1d(np.arange(n), hypergraph.indices)
    if len(singletons) > 0:
        indices = np.concatenate((hypergraph.indices, singletons))
        indptr = np.r_[hypergraph.indptr, hypergraph.indptr[-1]+1:hypergraph.indptr[-1]+len(singletons)+1]
        data = np.concatenate((hypergraph.data, np.ones(len(singletons))))
        hypergraph = sp.csr_matrix((data, indices, indptr), shape = (hypergraph.shape[0]+len(singletons), n))
    return hypergraph
