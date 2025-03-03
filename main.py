import argparse
from load import Dataset
from sahe import sahe
from eval import *

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='cora-ca1')
    args = parser.parse_args()

    dataset = Dataset(args.data)
    node_emb, hyperedge_emb, cost = sahe(dataset.full_ahg)
    print(f"Embedding time: {cost[0]:.3f}s / RAM: {cost[1]:.3f}GB")
    
    node_classification_eval(node_emb, dataset)
    hyperedge_classification_eval(hyperedge_emb, dataset)
    
    hyperedge_link_prediction_eval(sahe, dataset)
