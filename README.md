# Effective and Efficient Attributed Hypergraph Embedding on Nodes and Hyperedges (VLDB 2025)

This repository contains the implementation of the SAHE algorithm, designed as a scalable solution for the attributed hypergraph node and hyperedge embedding (AHNEE) problem.

### Pre-requisites

Set up a conda environment with required packages by:

    conda create --name AHNEE --file requirements.txt -c pytorch
    conda activate AHNEE

### Usage

To test the SAHE embedding algorithm, please provide the name of the dataset:

    python main.py --data DBLP-CA

The DBLP-CA, Cora-CA, Cora-CC, Citeseer, Mushroom, 20News, DBLP and Recipe datasets are readily provided in this repository. Due to the limit of file size on GitHub, Amazon and MAG-PM datasets are released via [Zenodo](https://zenodo.org/records/16307348).

Supported command line arguments are listed as follows:

| Parameter  | Default | Description                                           |
|------------|---------|-------------------------------------------------------|
| --data    | DBLP-CA      | A dataset from DBLP-CA, Cora-CA, Cora-CC, Citeseer, Mushroom, 20News, DBLP, and Recipe.     |
| --knn    | 10      | $K$ for constructing attribute-based hyperedges.          |
| --beta    | 1.0      | $\beta$ for balancing structural and attribute information.          |
| --rank    | 32      | $r$ for the output rank of truncated SVD.          |
| --base    |       | Run Base embedding algorithm instead of SAHE.          |


The program will print evaluation results and save the node and hyperedge embeddings in the "output" folder. 

Below is an example outcome from SAHE on the DBLP-CA dataset. *Note that the evaluation results may fluctuate across different runs and random dataset splits.*

```
> python main.py --data DBLP-CA
Running SAHE on DBLP-CA dataset
python main.py --data DBLP-CA
Embedding time: 0.341s / RAM: 0.194GB
Node classification results:
MIF1 avg: 0.824 / MIF1 std: 0.007 / MAF1 avg: 0.816 / MAF1 std: 0.008
Hyperedge classification results:
MIF1 avg: 0.854 / MIF1 std: 0.006 / MAF1 avg: 0.836 / MAF1 std: 0.005
Running hyperedge link prediction tests...
Hyperedge link prediction results:
Acc avg: 0.776 / Acc std: 0.012 / AUC avg: 0.890 / AUC std: 0.006
```
