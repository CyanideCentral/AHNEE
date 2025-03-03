import argparse
from load import Dataset
from sahe import sahe
from eval import node_classification_repeat, eval_hyperedge_link_prediction

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='cora-ca')
    parser.add_argument('--task', type=str, default='cls')
    args = parser.parse_args()

    dataset = Dataset(args.data)
    if args.task == 'cls':
        node_classification_repeat(sahe, dataset)
    elif args.task == 'hlp':
        eval_hyperedge_link_prediction(sahe, dataset)
    else:
        print(f'Unknown task: {args.task}')
