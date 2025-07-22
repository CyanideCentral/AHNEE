import os
import pickle
import numpy as np
import scipy.sparse as sp

class AttrHypergraph:
    def __init__(self, hypergraph, attributes, name = None, labels = None):
        self.hypergraph = hypergraph
        self.attributes = attributes
        self.labels = labels
        self.n = attributes.shape[0]
        self.m = self.hypergraph.shape[0]
        self.name = name

class Dataset:
    def __init__(self, dataset_name):
        dataset_name = dataset_name.lower()
        self.name = dataset_name
        self.rng = np.random.default_rng()
        data_files = os.listdir(f'data/{dataset_name}')
        self.hypergraph = sp.load_npz(f'data/{dataset_name}/hypergraph.npz')
        self.attributes = sp.load_npz(f'data/{dataset_name}/attributes.npz').astype(np.float32)
        self.n = self.attributes.shape[0]
        self.m = self.hypergraph.shape[0]
        self.labels = np.load(f'data/{dataset_name}/node_labels.npy')
        self.num_classes = len(np.unique(self.labels))
        self.full_ahg = AttrHypergraph(self.hypergraph, self.attributes, self.name, self.labels)
        # match file names with node_split_xxx.pickle
        node_split_files = [f for f in data_files if f.startswith('node_split')]
        self.node_splits = {} if len(node_split_files) > 0 else None
        for file in node_split_files:
            train_ratio = float(file[:-7].split('_')[-1])
            self.node_splits[train_ratio] = (pickle.load(open(f'data/{dataset_name}/{file}', 'rb')))
        if 'neg_hypergraph.npz' in data_files:
            self.edge_splits = {}
            self.neg_hg = sp.load_npz(f'data/{dataset_name}/neg_hypergraph.npz')
            edge_split_files = [f for f in data_files if f.startswith('hyperedge_split')]
            if len(edge_split_files) == 0:
                self.new_hlp_split()
            else:
                for file in edge_split_files:
                    train_ratio = float(file[:-7].split('_')[-1])
                    self.edge_splits[train_ratio] = pickle.load(open(f'data/{dataset_name}/{file}', 'rb'))
        else:
            self.neg_hg = None
            self.edge_splits = None

        self.edge_labels = np.load(f'data/{dataset_name}/hyperedge_labels.npy')

        hyperedge_split_files = [f for f in data_files if f.startswith('hyperedge_split')]
        self.hyperedge_splits = {} if len(hyperedge_split_files) > 0 else None
        for file in hyperedge_split_files:
            train_ratio = float(file[:-7].split('_')[-1])
            self.hyperedge_splits[train_ratio] = (pickle.load(open(f'data/{dataset_name}/{file}', 'rb')))

    def new_node_cls_split(self, train_ratio = 0.6, num_splits = 10):
        node_cls_split = []
        for _ in range(num_splits):
            idx = self.rng.permutation(len(self.labels))
            train_idx = idx[:int(len(self.labels)*train_ratio)]
            test_idx = idx[int(len(self.labels)*train_ratio):]
            node_cls_split.append((train_idx, test_idx))
        self.node_cls_split = node_cls_split
        pickle.dump(node_cls_split, open(f'data/{self.name}/node_split_{train_ratio}.pickle', 'wb'))
        return node_cls_split

    # generate negative hyperedges for hyperedge link prediction
    def generate_negative_hyperedges(self, ratio=1.0):
        # ratio: number of negative hyperedges / number of positive hyperedges
        num_pos = self.hypergraph.shape[0]
        num_neg = int(num_pos * ratio)
        pos_hyperedge_sizes = np.diff(self.hypergraph.indptr)
        # Repeat the positive hyperedge sizes to match the number of negative hyperedges
        neg_hyperedge_sizes = np.tile(pos_hyperedge_sizes, num_neg // num_pos + 1)[:num_neg]
        # Generate the negative hyperedges in batch
        indices = np.concatenate([self.rng.choice(self.n, size, replace=False) for size in neg_hyperedge_sizes])
        # Create the indptr array based on the hyperedge sizes
        indptr = np.cumsum(np.concatenate(([0], neg_hyperedge_sizes)))
        self.neg_hg = sp.csr_matrix((np.ones(len(indices)), indices, indptr), shape=(num_neg, self.n))
        sp.save_npz(f'data/{self.name}/neg_hypergraph.npz', self.neg_hg)

    def new_hlp_split(self, train_ratio = 0.8, num_splits = 10):
        lp_splits = []
        if self.neg_hg is None:
            self.generate_negative_hyperedges()
        num_pos = self.hypergraph.shape[0]
        num_pos_train = int(num_pos * train_ratio)
        num_neg = self.neg_hg.shape[0]
        num_neg_train = int(num_neg * train_ratio)
        for _ in range(num_splits):
            pos_idx = self.rng.permutation(self.hypergraph.shape[0])
            neg_idx = self.rng.permutation(self.neg_hg.shape[0]) + num_pos
            # hyperedges for embedding
            train_pos_idx = pos_idx[:num_pos_train]
            train_idx = np.concatenate((train_pos_idx, neg_idx[:num_neg_train]))
            train_labels = np.concatenate((np.ones(num_pos_train), np.zeros(num_neg_train)))
            # shuffle train samples
            shuffle_idx = self.rng.permutation(len(train_idx))
            train_idx = train_idx[shuffle_idx]
            train_labels = train_labels[shuffle_idx]
            test_idx = np.concatenate((pos_idx[num_pos_train:], neg_idx[num_neg_train:]))
            test_labels = np.concatenate((np.ones(num_pos - num_pos_train), np.zeros(num_neg - num_neg_train)))
            # shuffle test samples
            shuffle_idx = self.rng.permutation(len(test_idx))
            test_idx = test_idx[shuffle_idx]
            test_labels = test_labels[shuffle_idx]
            lp_splits.append((train_pos_idx, (train_idx, train_labels), (test_idx, test_labels)))
        if self.edge_splits is None:
            self.edge_splits = {}
        self.edge_splits[train_ratio] = lp_splits
        pickle.dump(lp_splits, open(f'data/{self.name}/hyperedge_split_{train_ratio}.pickle', 'wb'))

    def new_hyperedge_cls_split(self,train_ratio = 0.8, num_splits = 10):
        labeled_idx = np.arange(self.hypergraph.shape[0])[self.edge_labels >=0]
        hyperedge_cls_split = []
        for _ in range(num_splits):
            idx = labeled_idx[self.rng.permutation(len(labeled_idx))]
            train_idx = idx[:int(len(labeled_idx)*train_ratio)]
            test_idx = idx[int(len(labeled_idx)*train_ratio):]
            hyperedge_cls_split.append((train_idx, test_idx))
        self.hyperedge_cls_split = hyperedge_cls_split
        pickle.dump(hyperedge_cls_split, open(f'data/{self.name}/hyperedge_split_{train_ratio}.pickle', 'wb'))
        return hyperedge_cls_split
