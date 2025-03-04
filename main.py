import argparse
from load import Dataset
from sahe import sahe
from eval import *
import os
import numpy as np
import config

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='DBLP-CA')
    parser.add_argument('--knn', type=int, default='10')
    parser.add_argument('--beta', type=float, default='1.0')
    parser.add_argument('--rank', type=float, default='32')
    args = parser.parse_args()
    config.knn = args.knn
    config.beta = args.beta
    config.svd_rank = args.rank

    print(f"Running SAHE on {args.data} dataset")
    dataset = Dataset(args.data)
    node_emb, hyperedge_emb, cost = sahe(dataset.full_ahg)
    print(f"Embedding time: {cost[0]:.3f}s / RAM: {cost[1]:.3f}GB")
    os.makedirs('output', exist_ok = True)
    np.save(f'output/{args.data}_node_embeddings.npy', node_emb)
    np.save(f'output/{args.data}_hyperedge_embeddings.npy', hyperedge_emb)

    node_classification_eval(node_emb, dataset)
    hyperedge_classification_eval(hyperedge_emb, dataset)
    
    hyperedge_link_prediction_eval(sahe, dataset)
