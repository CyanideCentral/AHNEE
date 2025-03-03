import numpy as np
import scipy.sparse as sp
import pickle
from tqdm import tqdm
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from load import Dataset, AttrHypergraph
import torch
import torch.nn as nn
import config
import copy
import utils
from munkres import Munkres
from sklearn import metrics
from sahe import sahe

def node_classification_eval(embeddings, dataset):
    if dataset.n > 1000000:
        config.nc_train_ratio = 0.02
    if dataset.node_splits is None or config.nc_train_ratio not in dataset.node_splits:
        data_splits = dataset.new_node_cls_split(config.nc_train_ratio)
    else:
        data_splits = dataset.node_splits[config.nc_train_ratio]
    mif1_results, maf1_results = [], []
    num_splits = min(len(data_splits), config.nc_num_splits)
    for splits in range(num_splits):
        if len(data_splits[splits]) == 3:
            train_idx, valid_idx, test_idx = data_splits[splits]
            train_idx = np.concatenate((train_idx, valid_idx))
        else:
            train_idx, test_idx = data_splits[splits]
        classifier = SGDClassifier(loss = 'log_loss', max_iter = 1000, tol = 1e-5)
        classifier.fit(embeddings[train_idx], dataset.labels[train_idx])
        predict_test_labels = classifier.predict(embeddings[test_idx])
        maf1 = metrics.f1_score(dataset.labels[test_idx], predict_test_labels, average='macro')
        mif1 = metrics.f1_score(dataset.labels[test_idx], predict_test_labels, average='micro')
        maf1_results.append(maf1)
        mif1_results.append(mif1)
    print("Node classification results:")
    print(f"MIF1 avg: {np.mean(mif1_results):.3f} / MIF1 std: {np.std(mif1_results):.3f} / MAF1 avg: {np.mean(maf1_results):.3f} / MAF1 std: {np.std(maf1_results):.3f}")

def hyperedge_link_prediction_eval(embedder, dataset):
    if dataset.n > 1000000:
        config.lp_train_ratio = 0.98
    if dataset.neg_hg is None or (config.lp_train_ratio not in dataset.edge_splits.keys()):
        dataset.new_lp_split(train_ratio = config.lp_train_ratio)
    acc_results = []
    ap_results = []
    roc_auc_macros = []
    num_splits = len(dataset.edge_splits[config.lp_train_ratio])
    num_splits = min(num_splits, config.lp_num_splits)
    all_hyperedges = sp.csr_matrix(sp.vstack([dataset.hypergraph, dataset.neg_hg]))
    time_ram_list = []
    embeddings_list = []
    for spid in range(num_splits):
        train_hg = utils.add_unconnected_singletons(all_hyperedges[dataset.edge_splits[config.lp_train_ratio][spid][0],:])
        input_ahg = AttrHypergraph(train_hg, dataset.features,dataset.name,dataset.labels)
        embeddings, _, _ = embedder(input_ahg)
        embeddings_list.append(embeddings)
        train_idx, train_labels = dataset.edge_splits[config.lp_train_ratio][spid][1]
        train_hyperedges = all_hyperedges[train_idx,:]
        train_embeddings = utils.aggregate_by_hyperedges(embeddings, train_hyperedges, config.set_reduce)
        test_idx, test_labels = dataset.edge_splits[config.lp_train_ratio][spid][2]
        test_hyperedges = all_hyperedges[test_idx,:]
        test_embeddings = utils.aggregate_by_hyperedges(embeddings, test_hyperedges, config.set_reduce)
        classifier = SGDClassifier(loss = 'log_loss', max_iter = 1000, tol = 1e-5)
        classifier.fit(train_embeddings, train_labels)
        predict_test_proba  = classifier.predict_proba(test_embeddings)
        predict_test_proba = predict_test_proba[:,1]
        acc = classifier.score(test_embeddings, test_labels)
        roc_auc_score = metrics.roc_auc_score(test_labels, predict_test_proba, average='macro')
        roc_auc_macros.append(roc_auc_score)
        acc_results.append(acc)
    print("Hyperedge link prediction results:")
    print(f"Acc avg: {np.mean(acc_results):.3f} / Acc std: {np.std(acc_results):.3f} / AUC avg: {np.mean(roc_auc_macros):.3f} / AUC std: {np.std(roc_auc_macros):.3f}")

def hyperedge_classification_eval(embeddings, dataset):
    if dataset.hypergraph.shape[0] > 1000000:
        config.hec_train_ratio = 0.02
    if dataset.edge_single_splits is None or config.hec_train_ratio not in dataset.edge_single_splits:
        data_splits = dataset.new_edge_single_cls_split(config.hec_train_ratio)
    else:
        data_splits = dataset.edge_single_splits[config.hec_train_ratio]
    mif1_results, maf1_results = [], []
    num_splits = min(len(data_splits), config.hec_num_splits)
    for splits in range(num_splits):
        if len(data_splits[splits]) == 3:
            train_idx, valid_idx, test_idx = data_splits[splits]
            train_idx = np.concatenate((train_idx, valid_idx))
        else:
            train_idx, test_idx = data_splits[splits]
        classifier = SGDClassifier(loss = 'log_loss', max_iter = 1000, tol = 1e-5)
        classifier.fit(embeddings[train_idx], dataset.edge_labels[train_idx])
        predict_test_labels = classifier.predict(embeddings[test_idx])
        maf1 = metrics.f1_score(dataset.edge_labels[test_idx], predict_test_labels, average='macro')
        mif1 = metrics.f1_score(dataset.edge_labels[test_idx], predict_test_labels, average='micro') #same with acc
        maf1_results.append(maf1)
        mif1_results.append(mif1)
    print("Hyperedge classification results:")
    print(f"MIF1 avg: {np.mean(mif1_results):.3f} / MIF1 std: {np.std(mif1_results):.3f} / MAF1 avg: {np.mean(maf1_results):.3f} / MAF1 std: {np.std(maf1_results):.3f}")
