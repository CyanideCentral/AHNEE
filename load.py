import os
import pickle
import numpy as np
import scipy.sparse as sp
from sparse_dot_mkl import dot_product_mkl

class AttrHypergraph:
    def __init__(self, hypergraph, features, name = None, labels = None):
        self.hypergraph = hypergraph
        self.features = features
        self.labels = labels
        self.n = features.shape[0]
        self.m = self.hypergraph.shape[0]
        self.name = name

class Dataset:
    def __init__(self, dataset_name):
        self.name = dataset_name.lower()
        self.rng = np.random.default_rng(0)
        data_files = os.listdir(f'data/{dataset_name}')
        self.hypergraph = sp.load_npz(f'data/{dataset_name}/hypergraph.npz')
        self.features = sp.load_npz(f'data/{dataset_name}/features.npz').astype(np.float32)
        self.n = self.features.shape[0]
        self.m = self.hypergraph.shape[0]
        self.labels = np.load(f'data/{dataset_name}/labels.npy')
        self.num_classes = len(np.unique(self.labels))
        self.full_ahg = AttrHypergraph(self.hypergraph, self.features, self.name, self.labels)
        # match file names with node_split_xxx.pickle
        node_split_files = [f for f in data_files if f.startswith('node_split')]
        self.node_splits = {} if len(node_split_files) > 0 else None
        for file in node_split_files:
            train_ratio = float(file[:-7].split('_')[-1])
            self.node_splits[train_ratio] = (pickle.load(open(f'data/{dataset_name}/{file}', 'rb')))
        # self.node_split = pickle.load(open(f'data/{dataset_name}/node_split_0.01.pickle', 'rb'))
        if 'neg_hypergraph.npz' in data_files:
            self.edge_splits = {}
            self.neg_hg = sp.load_npz(f'data/{dataset_name}/neg_hypergraph.npz')
            edge_split_files = [f for f in data_files if f.startswith('hyperedge_split')]
            if len(edge_split_files) == 0:
                self.new_lp_split()
            else:
                for file in edge_split_files:
                    train_ratio = float(file[:-7].split('_')[-1])
                    self.edge_splits[train_ratio] = pickle.load(open(f'data/{dataset_name}/{file}', 'rb'))
        else:
            self.neg_hg = None
            self.edge_splits = None

        if self.name == "walmart":
            self.edge_labels = np.load(f'data/{dataset_name}/hyperedge_labels.npy')
        elif "edge_single_label.npy" not in data_files:
            self.edge_labels = self.generate_edge_single_label()
        else:
            self.edge_labels = np.load(f'data/{dataset_name}/edge_single_label.npy')

        edge_single_split_files = [f for f in data_files if f.startswith('edge_single_split')]
        self.edge_single_splits = {} if len(edge_single_split_files) > 0 else None
        for file in edge_single_split_files:
            train_ratio = float(file[:-7].split('_')[-1])
            self.edge_single_splits[train_ratio] = (pickle.load(open(f'data/{dataset_name}/{file}', 'rb')))


        if "edge_multi_labels.npy" not in data_files:
            self.edge_multi_labels = self.generate_edge_multi_labels()
        else:
            self.edge_multi_labels = np.load(f'data/{dataset_name}/edge_multi_labels.npy')

        edge_multi_split_files = [f for f in data_files if f.startswith('edge_multi_split')]
        self.edge_multi_splits = {} if len(edge_multi_split_files) > 0 else None
        for file in edge_multi_split_files:
            train_ratio = float(file[:-7].split('_')[-1])
            self.edge_multi_splits[train_ratio] = (pickle.load(open(f'data/{dataset_name}/{file}', 'rb')))


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

    # # generate negative hyperedges for link prediction
    # def generate_negative_hyperedges(self, ratio = 1.0):
    #     # ratio: number of negative hyperedges / number of positive hyperedges
    #     print("start generate negative hyperedges")
    #     num_pos = self.hypergraph.shape[0]
    #     num_neg = int(num_pos * ratio)
    #     indices = []
    #     indptr = [0]
    #     for i in range(num_neg):
    #         hyperedge = self.rng.choice(self.n, self.hypergraph[i%num_pos, :].nnz, replace = False)
    #         indices.extend(hyperedge)
    #         indptr.append(indptr[-1] + len(hyperedge))
    #     self.neg_hg = sp.csr_matrix((np.ones(len(indices)), indices, indptr), shape = (num_neg, self.n))
    #     sp.save_npz(f'data/{self.name}/neg_hypergraph.npz', self.neg_hg)
    #     print("finish generate negative hyperedges")

    # generate negative hyperedges for link prediction rewrite for large graphs
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

    def new_lp_split(self, train_ratio = 0.8, num_splits = 10):
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

    def generate_edge_single_label(self, proportion = 0.0):
        node_label_map = np.eye(self.labels.max()+1)[self.labels,:]
        hyperedge_label_count=dot_product_mkl(self.hypergraph, node_label_map, cast=True)
        if proportion == 0.0:
            edge_label = np.argmax(hyperedge_label_count, axis=1)
        else:
            edge_label = np.where(hyperedge_label_count.max(axis=1) > self.hypergraph.sum(1).A1 * proportion, np.argmax(hyperedge_label_count, axis=1), -1)
        # edge_num = self.hypergraph.shape[0]
        # edge_label = np.repeat(-1,edge_num)
        # for i in np.arange(self.hypergraph.shape[0]):
        #     node_entry = self.hypergraph[i].nonzero()[1]
        #     node_num = len(self.hypergraph[i].nonzero()[1])
        #     node_labels = self.labels[node_entry]
        #     max_frequency = np.max(np.bincount(node_labels))
        #     max_frequency_label = np.argmax(np.bincount(node_labels))
        #     if max_frequency > int(node_num * proportion):
        #         edge_label[i] = max_frequency_label
        np.save(f'data/{self.name}/edge_single_label.npy', edge_label)
        return edge_label
    
    def new_edge_single_cls_split(self,train_ratio = 0.8, num_splits = 10):
        labeled_idx = np.arange(self.hypergraph.shape[0])[self.edge_labels >=0]
        edge_single_cls_split = []
        for _ in range(num_splits):
            idx = labeled_idx[self.rng.permutation(len(labeled_idx))]
            train_idx = idx[:int(len(labeled_idx)*train_ratio)]
            test_idx = idx[int(len(labeled_idx)*train_ratio):]
            edge_single_cls_split.append((train_idx, test_idx))
        self.edge_single_cls_split = edge_single_cls_split
        pickle.dump(edge_single_cls_split, open(f'data/{self.name}/edge_single_split_{train_ratio}.pickle', 'wb'))
        return edge_single_cls_split

    def generate_edge_multi_labels(self):
        node_label_map = np.eye(self.labels.max()+1)[self.labels,:]
        edge_label=dot_product_mkl(self.hypergraph, node_label_map, cast=True)
        edge_label[edge_label>0]=1
        # edge_num = self.hypergraph.shape[0]
        # edge_label = np.zeros(shape=(edge_num,self.num_classes))
        # for i in np.arange(self.hypergraph.shape[0]):
        #     node_entry = self.hypergraph[i].nonzero()[1]
        #     node_labels = self.labels[node_entry]
        #     np.unique(node_labels)
        #     edge_label[i][node_labels] = 1
        np.save(f'data/{self.name}/edge_multi_labels.npy', edge_label)
        return edge_label
    
    def new_edge_multi_cls_split(self, train_ratio = 0.8, num_splits = 10):
        edge_multi_cls_split = []
        for _ in range(num_splits):
            idx = self.rng.permutation(self.hypergraph.shape[0])
            train_idx = idx[:int(self.hypergraph.shape[0]*train_ratio)]
            test_idx = idx[int(self.hypergraph.shape[0]*train_ratio):]
            edge_multi_cls_split.append((train_idx, test_idx))
        self.edge_multi_cls_split = edge_multi_cls_split
        pickle.dump(edge_multi_cls_split, open(f'data/{self.name}/edge_multi_split_{train_ratio}.pickle', 'wb'))
        return edge_multi_cls_split

# def load_data(dataset_name):
#     dataset = {}
#     dataset['hypergraph'] = sp.load_npz(f'data/{dataset_name}/hypergraph.npz')
#     dataset['features'] = sp.load_npz(f'data/{dataset_name}/features.npz')
#     dataset['labels'] = np.load(f'data/{dataset_name}/labels.npy')
#     # dataset['node_split'] = pickle.load(open(f'data/{dataset_name}/node_split_0.01.pickle', 'rb'))
#     # dataset['edge_split'] = pickle.load(open(f'data/{dataset_name}/edge_split.pickle', 'rb'))
#     dataset['name'] = dataset_name
#     return dataset

if __name__== '__main__':

    dataset = Dataset("cora-CA")
    dataset.generate_edge_single_label()
    dataset.new_edge_single_cls_split()
    dataset.generate_edge_multi_labels()
    dataset.new_edge_multi_cls_split()
    # dataset.generate_negative_hyperedges()