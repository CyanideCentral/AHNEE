# SAHE: Scalable Attributed Hypergraph node and hyperedge Embedding

This is the implementation of SAHE algorithm, designed for the attributed hypergraph node and hyperedge embedding (AHNEE) problem.

### Prerequisites

Required packages are installed by running:

    pip install -r requirements.txt

### Usage

The DBLP-CA, Cora-CA, Cora-CC, Citeseer, 20News, DBLP datasets are provided in this repository. Due to the limit of file size on GitHub, Amazon and MAG-PM dataset will be released via external links.

To test SAHE embedding algorithm on a dataset, simply run main.py with the dataset name as argument:

    python main.py --data $DATASET

The program will print evaluation results and save the node and hyperedge embeddings in the "output" folder. Note that the evaluation results may fluctuate across different runs and random dataset splits.

Below is an example outcome from SAHE on the DBLP-CA dataset:

```
python main.py --data DBLP-CA
Embedding time: 1.637s / RAM: 0.222GB
Node classification results:
MIF1 avg: 0.824 / MIF1 std: 0.007 / MAF1 avg: 0.816 / MAF1 std: 0.008
Hyperedge classification results:
MIF1 avg: 0.854 / MIF1 std: 0.006 / MAF1 avg: 0.836 / MAF1 std: 0.005
Hyperedge link prediction results:
Acc avg: 0.776 / Acc std: 0.012 / AUC avg: 0.890 / AUC std: 0.006
```