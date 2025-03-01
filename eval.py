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

def node_classification_valid(embeddings, dataset):
    if config.nc_train_ratio in dataset.node_splits:
        data_splits = dataset.node_splits[config.nc_train_ratio]
    else:
        data_splits = dataset.new_node_cls_split(config.nc_train_ratio)
    dim = embeddings.shape[1]
    results = []
    device = torch.device('cpu')
    embeddings = torch.tensor(embeddings, dtype = torch.float32, device = device)
    labels = torch.tensor(dataset.labels, dtype = torch.long, device = device)
    num_splits = min(len(data_splits), config.nc_num_splits)
    if len(data_splits[0]) != 3:
        print("Evaluation error: no validation set found")
        return
    for splits in tqdm(range(num_splits)):
        train_idx, valid_idx, test_idx = data_splits[splits]
        classifier = torch.nn.Linear(embeddings.shape[1], dataset.num_classes)
        optimizer = torch.optim.Adam(classifier.parameters(), lr = config.nc_lr, weight_decay = config.nc_w_decay)
        criterion = torch.nn.CrossEntropyLoss()
        valid_acc = 0
        for ep in range(config.nc_epochs):
            classifier.train()
            optimizer.zero_grad()
            pred = classifier(embeddings)[train_idx,:]
            loss = criterion(pred, labels[train_idx])
            loss.backward()
            optimizer.step()
            if ep % 10 == 0:
                with torch.no_grad():
                    classifier.eval()
                    pred = torch.argmax(classifier(embeddings), dim = 1)
                    acc = torch.sum((pred == labels)[valid_idx])/len(valid_idx)
                    cur_valid =  acc.to('cpu').detach().item()
                if cur_valid > valid_acc:
                    valid_acc = cur_valid
                    param = copy.deepcopy(classifier.state_dict())

        classifier.load_state_dict(param)
        with torch.no_grad():
            classifier.eval()
            pred = torch.argmax(classifier(embeddings), dim = 1)
            acc = torch.sum((pred == labels)[test_idx])/len(test_idx)
            test_acc =  acc.to('cpu').detach().item()
        results.append(test_acc)
    print(f"Data: {dataset.name} / Task: node classification / ACC avg: {np.mean(results)} / ACC std: {np.std(results)}")
    return np.mean(results), np.std(results)

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

def eval_hyperedge_prediction(embedder, dataset, return_embeddings = False):
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

def eval_hyperedge_prediction_pytorch(embedder, dataset):
    if dataset.n > 1000000:
        config.lp_train_ratio = 0.98
    if dataset.neg_hg is None or dataset.edge_splits is None:
        dataset.new_lp_split()
    results = []
    roc_auc_macros = [] #same with micros
    num_splits = len(dataset.edge_splits[config.lp_train_ratio])
    num_splits = min(num_splits, config.lp_num_splits)
    all_hyperedges = sp.vstack([dataset.hypergraph, dataset.neg_hg])
    time_ram_list = []
    for spid in range(num_splits):
        train_hg = utils.add_unconnected_singletons(all_hyperedges[dataset.edge_splits[config.lp_train_ratio][spid][0],:])
        input_ahg = AttrHypergraph(train_hg, dataset.features)
        embeddings, cost = embedder(input_ahg, config.embedding_dim)
        time_ram_list.append(cost)
        train_idx, train_labels = dataset.edge_splits[config.lp_train_ratio][spid][1]
        train_hyperedges = all_hyperedges[train_idx,:]
        train_embeddings = utils.aggregate_by_hyperedges(embeddings, train_hyperedges, config.set_reduce)
        test_idx, test_labels = dataset.edge_splits[config.lp_train_ratio][spid][2]
        test_hyperedges = all_hyperedges[test_idx,:]
        test_embeddings = utils.aggregate_by_hyperedges(embeddings, test_hyperedges, config.set_reduce)

        device = torch.device('cpu')
        model = MLP_HENN(in_dim = train_embeddings.shape[1], hidden_dim = 128).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr = config.nc_lr, weight_decay = 1e-6)
        criterion = torch.nn.BCELoss()
        train_embeddings = torch.tensor(train_embeddings, dtype=torch.float32).to(device)
        test_embeddings = torch.tensor(test_embeddings, dtype=torch.float32).to(device)
        train_labels = torch.tensor(train_labels, dtype=torch.float32).to(device)
        test_labels = torch.tensor(test_labels, dtype=torch.float32).to(device)

        for ep in (range(200)): 
            model.train()
            optimizer.zero_grad()
            pred = model(train_embeddings)
            loss = criterion(pred, train_labels)
            loss.backward()
            optimizer.step()
        with torch.no_grad():
            model.eval()
            pred = model(test_embeddings).to('cpu').detach().squeeze(-1).numpy()
        roc_auc_macros.append(metrics.roc_auc_score(test_labels, pred, average='macro'))
        acc = np.sum(((pred > 0.5) == test_labels).numpy()) / len(test_labels)
        if config.verbose:
            print(f"Split: {spid} / ACC: {acc} / MA-AUC: {roc_auc_macros[spid]} / Time: {cost[0]}s / RAM: {cost[1]} GB")
        results.append(acc)
    avg_time, avg_ram = np.c_[time_ram_list].mean(0)
    if config.verbose:
        print(f"Data: {dataset.name} / Task: hyperedge prediction / ACC avg: {np.mean(results)} / ACC std: {np.std(results)} / AUC avg: {np.mean(roc_auc_macros)} / AUC std: {np.std(roc_auc_macros)} / Time: {avg_time}s / RAM: {avg_ram} GB")
        print(f"{np.mean(results)} {np.std(results)} {np.mean(roc_auc_macros)} {np.std(roc_auc_macros)} {avg_time} {avg_ram}")
    return np.mean(results), np.std(results), np.mean(roc_auc_macros), np.std(roc_auc_macros), avg_time, avg_ram

class clustering_metrics():
    def __init__(self, true_label, predict_label):
        self.true_label = true_label
        self.pred_label = predict_label


    def clusteringAcc(self):
        l1 = list(set(self.true_label))
        numclass1 = len(l1)

        l2 = list(set(self.pred_label))
        numclass2 = len(l2)
        
        if numclass1 != numclass2:
            print('Class Not equal!!!!')
            missing_classes = set(l1)-set(l2)
            for i, c in enumerate(missing_classes):
                self.pred_label[i] = c
                l2.append(c)
            numclass2 = len(l2)
        cost = np.zeros((numclass1, numclass2), dtype=np.float64)
        for i, c1 in enumerate(l1):
            mps = [i1 for i1, e1 in enumerate(self.true_label) if e1 == c1]
            for j, c2 in enumerate(l2):
                mps_d = [i1 for i1 in mps if self.pred_label[i1] == c2]

                cost[i][j] = len(mps_d)

        m = Munkres()
        cost = cost.__neg__().tolist()

        indexes = m.compute(cost)

        new_predict = np.zeros(len(self.pred_label))
        for i, c in enumerate(l1):
            c2 = l2[indexes[i][1]]

            ai = [ind for ind, elm in enumerate(self.pred_label) if elm == c2]
            new_predict[ai] = c

        acc = metrics.accuracy_score(self.true_label, new_predict)
        f1_macro = metrics.f1_score(self.true_label, new_predict, average='macro')
        precision_macro = metrics.precision_score(self.true_label, new_predict, average='macro')
        recall_macro = metrics.recall_score(self.true_label, new_predict, average='macro')
        return acc, f1_macro, precision_macro, recall_macro

    def evaluationClusterModelFromLabel(self):
        nmi = metrics.normalized_mutual_info_score(self.true_label, self.pred_label)
        adjscore = metrics.adjusted_rand_score(self.true_label, self.pred_label)
        acc, f1, pre, rc = self.clusteringAcc()

        return acc, nmi, f1, pre, adjscore, rc

def node_clustering(embedder, dataset):
    
    acc_results = []
    f1_results = []
    nmi_results = []
    ari_results = []
    for i in range(config.nc_num_splits): 
        embeddings, cost = embedder(dataset.full_ahg, config.embedding_dim)
        from sklearn.cluster import KMeans
        predict_clusters = KMeans(n_clusters=dataset.num_classes).fit_predict(embeddings)
        cm = clustering_metrics(dataset.labels, predict_clusters)
        acc, nmi, f1, _, ari, _ = cm.evaluationClusterModelFromLabel()
        acc_results.append(acc)
        f1_results.append(f1)
        nmi_results.append(nmi)
        ari_results.append(ari)
        if config.verbose:
            print(f"Acc: {acc} F1: {f1} NMI: {nmi} ARI: {ari}")
    if config.verbose:
        print(f"Data: {dataset.name} / Task: node clustering / ACC avg: {np.mean(acc_results)} / F1 avg: {np.mean(f1_results)}/ NMI avg: {np.mean(nmi_results)} / ARI avg: {np.mean(ari_results)}")
    return np.mean(acc_results), np.mean(f1_results),np.mean(nmi_results),np.mean(ari_results)

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
    hep_acc, hep_std, hep_auc ,hep_auc_std, hep_ap, hep_ap_std, hep_time, hep_ram = eval_hyperedge_prediction(embedder, dataset)
    # hep_acc_pytorch, hep_std_pytorch, hep_auc_pytorch ,hep_auc_std_pytorch, hep_time_pytorch, hep_ram_pytorch = eval_hyperedge_prediction_pytorch(embedder, dataset)
    # acc,f1,nmi,ari = node_clustering(embedder, dataset)
    if config.verbose:
        print(f"Evaluating {embedder.__name__} on {dataset.name}")
        print(f"nc train ratio: {config.nc_train_ratio}, lp train ratio: {config.lp_train_ratio}")
        print(f"Original lp result: {hep_acc} {hep_std} {hep_auc} {hep_auc_std} {hep_time} {hep_ram}")
        # print(f"Pytorch lp result: {hep_acc_pytorch} {hep_std_pytorch} {hep_auc_pytorch} {hep_auc_std_pytorch} {hep_time_pytorch} {hep_ram_pytorch}")
    print(f"{nc_acc_mean} {nc_acc_std} {nc_time} {nc_ram} {hep_acc} {hep_std} ")
    print(f"{nc_acc_mean} {nc_acc_std} {nc_maf1_mean} {nc_maf1_std} {hep_acc} {hep_std} {hep_auc} {hep_auc_std} {nc_time} {nc_ram} ")
    if config.verbose:
        print(f"{nc_acc_mean} {nc_maf1_mean}{hep_acc} {hep_ap} {hep_auc} {nc_time} {nc_ram}")
        print(f"{nc_acc_mean} {nc_acc_std} {nc_maf1_mean} {nc_maf1_std} {hep_acc} {hep_std} {hep_ap} {hep_ap_std} {hep_auc} {hep_auc_std} {nc_time} {nc_ram} ")
    # print(f"{acc} {f1} {nmi} {ari}")
    if config.test_error:
        print(f"Poly MAE error : {np.asarray(config.poly_mae_error).mean()} {np.asarray(config.poly_mae_error).std()} / MSE error : {np.asarray(config.poly_mse_error).mean()} {np.asarray(config.poly_mse_error).std()} / Pearson r  {np.asarray(config.poly_pearson_r).mean()} {np.asarray(config.poly_pearson_r).std()}")
        print(f"SKetch MAE error : {np.asarray(config.mae_error).mean()} {np.asarray(config.mae_error).std()} / MSE error : {np.asarray(config.mse_error).mean()} {np.asarray(config.mse_error).std()} / Pearson r  {np.asarray(config.pearson_r).mean()} {np.asarray(config.pearson_r).std()}")
        config.mae_error = []
        config.mse_error = []
        config.poly_mae_error = []
        config.poly_mse_error = []
    if config.visual:
        from tsne import visual
        visual(full_embeddings,len(np.unique(dataset.labels)),dataset.labels,dataset.name)
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

def edge_multi_label_classification_repeat(embedder,dataset):
    if dataset.hypergraph.shape[0] > 1000000:
        config.hec_train_ratio = 0.02
    if dataset.edge_multi_splits is None or config.hec_train_ratio not in dataset.edge_multi_splits:
        data_splits = dataset.new_edge_multi_cls_split(config.hec_train_ratio)
    else:
        data_splits = dataset.edge_multi_splits[config.hec_train_ratio]
    results = []
    maf1_results = []
    num_splits = min(len(data_splits), config.hec_num_splits)
    times, rams = [], []

    for splits in range(num_splits):
        embeddings, cost = embedder(dataset.full_ahg, config.embedding_dim)
        times.append(cost[0])
        rams.append(cost[1])
        if len(data_splits[splits]) == 3:
            train_idx, valid_idx, test_idx = data_splits[splits]
            train_idx = np.concatenate((train_idx, valid_idx))
            # test_idx = np.concatenate((valid_idx, test_idx))
        else:
            train_idx, test_idx = data_splits[splits]
        classifier = OneVsRestClassifier(LogisticRegression())
        classifier.fit(embeddings[train_idx], dataset.edge_multi_labels[train_idx])

        predict_test_labels = classifier.predict(embeddings[test_idx]) #default threshold
        # predict_test_prob = classifier.predict_proba(embeddings[test_idx])
        # threshold = 1.0/dataset.num_classes
        # predict_test_labels = (predict_test_prob > threshold).astype(int)
        maf1 = metrics.f1_score(dataset.edge_multi_labels[test_idx], predict_test_labels, average='macro')
        mif1 = metrics.f1_score(dataset.edge_multi_labels[test_idx], predict_test_labels, average='micro') #same with acc
        results.append(mif1)
        maf1_results.append(maf1)
        if config.verbose:
            print(f"Split {splits} / Task: Hyperedge multi label classification/ ACC: {mif1} / MAF1: {maf1} ")
    if config.verbose:
        print(f"Data: {dataset.name} / Task: Hyperedge multi label classification / ACC avg: {np.mean(results)} / ACC std: {np.std(results)}/ MAF1 avg: {np.mean(maf1_results)} / MAF1 std: {np.std(maf1_results)}")
    return np.mean(results), np.std(results),np.mean(maf1_results),np.std(maf1_results), np.mean(times), np.mean(rams), embeddings

def eval_all_hyperedge(embedder,dataset):
    acc, std, maf1, std_maf1, time, ram, embeddings = edge_single_label_classification_repeat(embedder,dataset)
    acc_multi, std_multi, maf1_multi, std_maf1_multi, time_multi, ram_multi, embeddings_multi = edge_multi_label_classification_repeat(embedder,dataset)
    print(f"{acc} {maf1} {acc_multi} {maf1_multi} {time} {ram}")
    return embeddings

if __name__ == '__main__':
    config.verbose = True
    # dataset = Dataset('citeseer_cite_hb')
    # dataset = Dataset('cora_cite_hb')
    # dataset = Dataset('pubmed_cite_hb')
    # dataset = Dataset('cora_coauth_hb')
    # dataset = Dataset('dblp_copub_hb')
    # dataset = Dataset('dblp_coauth_hb')
    # dataset = Dataset('aminer_hb')
    # dataset = Dataset('dblp-CA')
    # dataset = Dataset('20news')
    # dataset = Dataset('news_hb')
    # dataset = Dataset('query')
    dataset = Dataset('citeseer-CC')
    # dataset = Dataset('cora-CC')
    # dataset = Dataset('cora-CA')
    # dataset = Dataset('amazon')
    # dataset = Dataset('magpm')
    dataset = Dataset('walmart')
    
    # embeddings = np.load(f'baselines/hypeboy/output/{dataset.name}_embeddings.npy')
    # node_classification_valid(embeddings, dataset)
    # node_classification(embeddings, dataset)
    
    from ahe import *
    from blhg import *
    from baselines.run import *
    # eval_all(random_embedding, dataset)
    # eval_all(knnshe_embedding, dataset)
    # eval_all(knnrw_sketchne, dataset)
    # eval_hyperedge_prediction(random_embedding, dataset)
    # eval_all(hg_sketchne, dataset)
    # eval_all(clex_netmf, dataset)
    # eval_all(xxt_netmf, dataset)
    # eval_all(xxt_sketchne, dataset)
    # eval_all(knn_two_layer_hypergraph_netmf, dataset)
    # eval_all(aughg_planar_pearson_netmf, dataset)
    # eval_all(biased_agg_two_layer_hypergraph, dataset)
    # eval_all(hg_netmf, dataset)
    # eval_all(attr_hg_netmf, dataset)
    # eval_all(knn_hg_netmf, dataset)
    # eval_all(knn_hg_sketchne,dataset)
    # eval_all(ghac_hg_hgmf, dataset)
    # eval_all(hac_hg_hgmf, dataset)
    # eval_all(knn_hg_hgmf_1svd, dataset)
    # eval_all(allno_hg_hgmf_1svd,dataset)
    # eval_all(knn_hg_hgmf_1svd_concatenate, dataset)
    # eval_all(knn_ghac_hg_hgmf, dataset)
    # eval_all(distached_knn_two_layer_hypergraph, dataset)
    # eval_all(distached_agg_two_layer_hypergraph, dataset)
    # eval_all(knn_hg_hgmf_1svd,dataset)
    # eval_all(knn_hg_hgmf_1svd_concatenate,dataset)
    # eval_all(knn_hg_hgmf_1svd_spec,dataset)
    # eval_all(knn_metis_hg_hgmf_1svd,dataset)
    # eval_all(knn_kahip_hg_hgmf_1svd,dataset)
    # eval_all(knn_kmeans_hg_hgmf_1svd,dataset)
    # eval_all(knn_hg_hgmf_pts,dataset)
    # eval_all(select_knn_hg_hgmf_1svd, dataset)
    # eval_all(multi_knn_hg_hgmf_1svd, dataset)
    # eval_all(knn_hg_hgmf_pts,dataset)
    # eval_all(multi_knn_hg_hgmf_pts,dataset)
    # eval_all(knn_hg_hgmf_full_pts,dataset)
    # eval_all(knn_hgedge_hgmf_pts,dataset)

    # eval_all(knn_hg_hgmf, dataset)
    # eval_all(knn_hg_hgmf_full_pts,dataset)

    # eval_all(call_zhou,dataset)
    # eval_all(run_hypeboy, dataset)
    # eval_all(run_pane, dataset)
    # eval_all(run_aneci,dataset)
    # eval_all(run_conn,dataset)
    # eval_all(run_anchorgnn,dataset)
    # eval_all(run_villain,dataset)
    # eval_all(run_tricl,dataset)
    # eval_all(run_sketchne,dataset)
    # eval_all(run_hyper2vec,dataset)
    # eval_all(run_nehe,dataset)
    # eval_all(run_lightne,dataset)
    # eval_all(run_sketchne,dataset)
    # eval_all(run_netmf,dataset)
    # eval_all(run_ours,dataset)
    # eval_all(run_gebe,dataset)
    # eval_all(run_biane, dataset)
    # eval_all(run_hypergrl,dataset)
    # eval_all(run_ours,dataset)
    # hyperedge_prediction(run_hypeboy, dataset)

    # edge_single_label_classification_repeat(knn_hg_hgmf_edge_1svd, dataset)
    edge_single_label_classification_repeat(run_biane, dataset)
    # edge_multi_label_classification_repeat(knn_hg_hgmf_edge_1svd, dataset)
    # edge_multi_label_classification_repeat(run_biane, dataset)