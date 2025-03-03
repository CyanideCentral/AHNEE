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

def node_classification(embeddings, dataset):
    if dataset.n > 1000000:
        config.nc_train_ratio = 0.02
    if dataset.node_splits is None or config.nc_train_ratio not in dataset.node_splits:
        data_splits = dataset.new_node_cls_split(config.nc_train_ratio)
    else:
        data_splits = dataset.node_splits[config.nc_train_ratio]
    dim = embeddings.shape[1]
    results = []
    maf1_results = []
    num_splits = min(len(data_splits), config.nc_num_splits)
    for splits in range(num_splits):
        if len(data_splits[splits]) == 3:
            train_idx, valid_idx, test_idx = data_splits[splits]
            train_idx = np.concatenate((train_idx, valid_idx))
            # test_idx = np.concatenate((valid_idx, test_idx))
        else:
            train_idx, test_idx = data_splits[splits]
        classifier = SGDClassifier(loss = 'log_loss', max_iter = 1000, tol = 1e-5)
        # classifier = OneVsRestClassifier(LogisticRegression(solver="liblinear", multi_class="ovr"), n_jobs=-1) # higher Acc
        classifier.fit(embeddings[train_idx], dataset.labels[train_idx])
        acc = classifier.score(embeddings[test_idx], dataset.labels[test_idx])
        results.append(acc)
        predict_test_labels = classifier.predict(embeddings[test_idx])
        maf1 = metrics.f1_score(dataset.labels[test_idx], predict_test_labels, average='macro')
        maf1_results.append(maf1)
    if config.verbose:
        print(f"Data: {dataset.name} / Task: node classification / ACC avg: {np.mean(results)} / ACC std: {np.std(results)}/ MAF1 avg: {np.mean(maf1_results)} / MAF1 std: {np.std(maf1_results)}")
    return np.mean(results), np.std(results),np.mean(maf1_results),np.std(maf1_results)

def node_classification_repeat(embedder, dataset):
    if dataset.n > 1000000:
        config.nc_train_ratio = 0.02
    if dataset.node_splits is None or config.nc_train_ratio not in dataset.node_splits:
        data_splits = dataset.new_node_cls_split(config.nc_train_ratio)
    else:
        data_splits = dataset.node_splits[config.nc_train_ratio]
    results = []
    maf1_results = []
    num_splits = min(len(data_splits), config.nc_num_splits)
    times, rams = [], []
    # embedding_outcomes = [] # no need to return all embeddings
    for splits in range(num_splits):
        if config.fix_eval_split:
            splits = 0
        embeddings, cost = embedder(dataset.full_ahg, config.embedding_dim)
        # embedding_outcomes.append(embeddings)
        times.append(cost[0])
        rams.append(cost[1])
        if len(data_splits[splits]) == 3:
            train_idx, valid_idx, test_idx = data_splits[splits]
            train_idx = np.concatenate((train_idx, valid_idx))
            # test_idx = np.concatenate((valid_idx, test_idx))
        else:
            train_idx, test_idx = data_splits[splits]
        classifier = SGDClassifier(loss = 'log_loss', max_iter = 1000, tol = 1e-5)
        # classifier = OneVsRestClassifier(LogisticRegression(solver="liblinear", multi_class="ovr"), n_jobs=-1) # higher Acc
        classifier.fit(embeddings[train_idx], dataset.labels[train_idx])
        acc = classifier.score(embeddings[test_idx], dataset.labels[test_idx])
        results.append(acc)
        predict_test_labels = classifier.predict(embeddings[test_idx])
        maf1 = metrics.f1_score(dataset.labels[test_idx], predict_test_labels, average='macro')
        mif1 = metrics.f1_score(dataset.labels[test_idx], predict_test_labels, average='micro') #same with acc
        maf1_results.append(maf1)
    if config.verbose:
        print(f"Data: {dataset.name} / Task: node classification / ACC avg: {np.mean(results)} / ACC std: {np.std(results)}/ MAF1 avg: {np.mean(maf1_results)} / MAF1 std: {np.std(maf1_results)}")
    return np.mean(results), np.std(results),np.mean(maf1_results),np.std(maf1_results), np.mean(times), np.mean(rams), embeddings

def eval_hyperedge_link_prediction(embedder, dataset, return_embeddings = False):
    if dataset.n > 1000000:
        config.lp_train_ratio = 0.98
    if dataset.neg_hg is None or (config.lp_train_ratio not in dataset.edge_splits.keys()):
        dataset.new_lp_split(train_ratio = config.lp_train_ratio)
    results = []
    ap_results = []
    roc_auc_macros = [] #same with micros
    num_splits = len(dataset.edge_splits[config.lp_train_ratio])
    num_splits = min(num_splits, config.lp_num_splits)
    all_hyperedges = sp.csr_matrix(sp.vstack([dataset.hypergraph, dataset.neg_hg]))
    time_ram_list = []
    embeddings_list = []
    for spid in range(num_splits):
        if config.fix_eval_split:
            spid = 0
        train_hg = utils.add_unconnected_singletons(all_hyperedges[dataset.edge_splits[config.lp_train_ratio][spid][0],:])
        input_ahg = AttrHypergraph(train_hg, dataset.features,dataset.name,dataset.labels)
        embeddings, cost = embedder(input_ahg, config.embedding_dim)
        embeddings_list.append(embeddings)
        if cost[0] == None or cost[1] == None:
            if config.verbose:
                print(f"Split: {spid} / encounter error in this split")
            continue            
        time_ram_list.append(cost)
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
        predict_test_labels = classifier.predict(test_embeddings)
        ap = metrics.average_precision_score(test_labels, predict_test_labels)        
        roc_auc_score = metrics.roc_auc_score(test_labels, predict_test_proba, average='macro')
        roc_auc_macros.append(roc_auc_score)
        if config.verbose:
            print(f"Split: {spid} / ACC: {acc} AP: {ap} / MA-AUC: {roc_auc_score} / Time: {cost[0]}s / RAM: {cost[1]} GB")
        results.append(acc)
        ap_results.append(ap)
    avg_time, avg_ram = np.c_[time_ram_list].mean(0)
    if config.verbose:
        print(f"Data: {dataset.name} / Task: hyperedge prediction / ACC avg: {np.mean(results)} / ACC std: {np.std(results)} /AUC avg: {np.mean(roc_auc_macros)} / AUC std: {np.std(roc_auc_macros)} / Time: {avg_time}s / RAM: {avg_ram} GB")
        # print(f"{np.mean(results)} {np.std(results)} {np.mean(roc_auc_macros)} {np.std(roc_auc_macros)} {avg_time} {avg_ram}")
    return np.mean(results), np.std(results), np.mean(roc_auc_macros), np.std(roc_auc_macros), np.mean(ap_results), np.std(ap_results), avg_time, avg_ram

class MLP_HENN(nn.Module) :
    
    def __init__(self, in_dim, hidden_dim, p = 0.5) : 
        super(MLP_HENN, self).__init__() 
        
        self.classifier1 = torch.nn.Linear(in_dim, hidden_dim)
        self.classifier2 = torch.nn.Linear(hidden_dim, 1)
        self.dropouts = torch.nn.Dropout(p = p)
        
    def forward(self, X: torch.Tensor) : 
        
        # Z = scatter(src = x[target_nodes, :], index = target_ids, dim = 0, reduce = 'sum')
        Z = X
        Z = (self.classifier1(Z)) # No need of Logits
        Z = torch.relu(Z)
        Z = self.dropouts(Z)
        Z = (self.classifier2(Z)) # No need of Logits
        
        return torch.sigmoid(Z).squeeze(-1) # Edge Prediction Probability

def eval_all(embedder, dataset):

    if config.verbose:
        print(f"Evaluating {embedder.__name__} on {dataset.name}")
    # full_embeddings, nc_cost = embedder(dataset.full_ahg, config.embedding_dim)
    # utils.dimension_collapse_analysis(full_embeddings)
    # if config.verbose:
    #     utils.uniformity_metric(full_embeddings)
    # nc_time, nc_ram = nc_cost
    
    # if nc_time is None and nc_ram is None:
    #     if config.verbose:
    #         print(f"Embedding time and RAM usage not available. Error encountered")
    #     print(f"0 0 0 0 0 0")
    #     return 0, 0, 0, 0, 0, 0, 0, 0
    # nc_acc_mean, nc_acc_std, nc_maf1_mean, nc_maf1_std = node_classification(full_embeddings, dataset)
    nc_acc_mean, nc_acc_std, nc_maf1_mean, nc_maf1_std, nc_time, nc_ram, full_embeddings = node_classification_repeat(embedder, dataset)
    if config.verbose:
        print(f"Average embedding time: {nc_time}s / embedding RAM: {nc_ram} GB")
    hep_acc, hep_std, hep_auc ,hep_auc_std, hep_ap, hep_ap_std, hep_time, hep_ram = eval_hyperedge_link_prediction(embedder, dataset)
    # hep_acc_pytorch, hep_std_pytorch, hep_auc_pytorch ,hep_auc_std_pytorch, hep_time_pytorch, hep_ram_pytorch = eval_hyperedge_prediction_pytorch(embedder, dataset)
    # acc,f1,nmi,ari = node_clustering(embedder, dataset)
    if config.verbose:
        print(f"Evaluating {embedder.__name__} on {dataset.name}")
        print(f"nc train ratio: {config.nc_train_ratio}, lp train ratio: {config.lp_train_ratio}")
        print(f"Original lp result: {hep_acc} {hep_std} {hep_auc} {hep_auc_std} {hep_time} {hep_ram}")
        # print(f"Pytorch lp result: {hep_acc_pytorch} {hep_std_pytorch} {hep_auc_pytorch} {hep_auc_std_pytorch} {hep_time_pytorch} {hep_ram_pytorch}")
    print(f"{nc_acc_mean} {nc_acc_std} {nc_time} {nc_ram} {hep_acc} {hep_std}")
    # with open("temp/acc_std_time_ram_acc_std.txt", "a") as f:
    #     print(f"{nc_acc_mean} {nc_acc_std} {nc_time} {nc_ram} {hep_acc} {hep_std}", file = f)
    # print(f"{nc_acc_mean} {nc_acc_std} {nc_maf1_mean} {nc_maf1_std} {hep_acc} {hep_std} {hep_auc} {hep_auc_std} {nc_time} {nc_ram}")
    if config.verbose:
        print(f"{nc_acc_mean} {nc_maf1_mean} {hep_acc} {hep_ap} {hep_auc} {nc_time} {nc_ram}")
        print(f"{nc_acc_mean} {nc_acc_std} {nc_maf1_mean} {nc_maf1_std} {hep_acc} {hep_std} {hep_ap} {hep_ap_std} {hep_auc} {hep_auc_std} {nc_time} {nc_ram}")
    # print(f"{acc} {f1} {nmi} {ari}")
    if config.test_error:
        print(f"Poly MAE error : {np.asarray(config.poly_mae_error).mean()} {np.asarray(config.poly_mae_error).std()} / MSE error : {np.asarray(config.poly_mse_error).mean()} {np.asarray(config.poly_mse_error).std()} / Pearson r  {np.asarray(config.poly_pearson_r).mean()} {np.asarray(config.poly_pearson_r).std()}")
        print(f"SKetch MAE error : {np.asarray(config.mae_error).mean()} {np.asarray(config.mae_error).std()} / MSE error : {np.asarray(config.mse_error).mean()} {np.asarray(config.mse_error).std()} / Pearson r  {np.asarray(config.pearson_r).mean()} {np.asarray(config.pearson_r).std()}")
        config.mae_error = []
        config.mse_error = []
        config.poly_mae_error = []
        config.poly_mse_error = []
    return nc_acc_mean, nc_acc_std, nc_time, nc_ram, hep_acc, hep_std, hep_time, hep_ram

def edge_single_label_classification_repeat(embedder,dataset):
    if dataset.hypergraph.shape[0] > 1000000:
        config.hec_train_ratio = 0.02
    if dataset.edge_single_splits is None or config.hec_train_ratio not in dataset.edge_single_splits:
        data_splits = dataset.new_edge_single_cls_split(config.hec_train_ratio)
    else:
        data_splits = dataset.edge_single_splits[config.hec_train_ratio]
    results = []
    maf1_results = []
    num_splits = min(len(data_splits), config.hec_num_splits)
    times, rams = [], []

    for splits in range(num_splits):
        if config.fix_eval_split:
            splits = 0
        embeddings, cost = embedder(dataset.full_ahg, config.embedding_dim)
        times.append(cost[0])
        rams.append(cost[1])
        if len(data_splits[splits]) == 3:
            train_idx, valid_idx, test_idx = data_splits[splits]
            train_idx = np.concatenate((train_idx, valid_idx))
            # test_idx = np.concatenate((valid_idx, test_idx))
        else:
            train_idx, test_idx = data_splits[splits]
        classifier = SGDClassifier(loss = 'log_loss', max_iter = 1000, tol = 1e-5)
        # classifier = OneVsRestClassifier(LogisticRegression(solver="liblinear", multi_class="ovr"), n_jobs=-1) # higher Acc
        classifier.fit(embeddings[train_idx], dataset.edge_labels[train_idx])
        acc = classifier.score(embeddings[test_idx], dataset.edge_labels[test_idx])
        results.append(acc)
        predict_test_labels = classifier.predict(embeddings[test_idx])
        maf1 = metrics.f1_score(dataset.edge_labels[test_idx], predict_test_labels, average='macro')
        mif1 = metrics.f1_score(dataset.edge_labels[test_idx], predict_test_labels, average='micro') #same with acc
        maf1_results.append(maf1)
        if config.verbose:
            print(f"Split {splits} / Task: Hyperedge single label classification/ ACC: {mif1} / MAF1: {maf1} ")
    if config.verbose:
        print(f"Data: {dataset.name} / Task: Hyperedge single label classification / ACC avg: {np.mean(results)} / ACC std: {np.std(results)}/ MAF1 avg: {np.mean(maf1_results)} / MAF1 std: {np.std(maf1_results)}")
    # print(f"{np.mean(results)} {np.std(results)} {np.mean(maf1_results)} {np.std(maf1_results)} {np.mean(times)} {np.mean(rams)}")
    return np.mean(results), np.std(results),np.mean(maf1_results),np.std(maf1_results), np.mean(times), np.mean(rams), embeddings

def eval_all_hyperedge(embedder,dataset):
    acc, std, maf1, std_maf1, time, ram, embeddings = edge_single_label_classification_repeat(embedder,dataset)
    print(f"{acc} {maf1} {time} {ram}")
    return embeddings

if __name__ == '__main__':
    config.verbose = True
    # dataset = Dataset('citeseer_cite_hb')
    # dataset = Dataset('cora_cite_hb')
    # dataset = Dataset('cora_coauth_hb')
    # dataset = Dataset('dblp_copub_hb')
    # dataset = Dataset('dblp_coauth_hb')
    # dataset = Dataset('dblp-CA')
    # dataset = Dataset('20news')
    # dataset = Dataset('news_hb')
    dataset = Dataset('citeseer-CC')
    # dataset = Dataset('cora-CC')
    # dataset = Dataset('cora-CA')
    # dataset = Dataset('amazon')
    # dataset = Dataset('magpm')
    eval_all(sahe,dataset)
    # eval_all_hyperedge(sahe,dataset)