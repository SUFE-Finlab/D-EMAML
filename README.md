# Introduction

This repo is from D-EMAML, that tackles the challenging task of detecting money laundering patterns in financial transaction networks by leveraging dual-edge motifs and graph neural networks for direct edge anomaly detection.

# Data Source

AMLWorld: [AMLWorld]([IBM Transactions for Anti Money Laundering (AML)](https://www.kaggle.com/datasets/ealtman2019/ibm-transactions-for-anti-money-laundering-aml/data))

Dgraph-Fin:[Dgraph-Fin](https://dgraph.xinye.com/dataset#DGraph-Fin)

TransBank is not provided due to compliance.

# Data download

#### AMLWorld:

Before downloading, kaggle key json file should be placed at: **~/.kaggle/kaggle.json**

Then run below command:

```bash
python download.py
```

#### DGraphE:

Download dataset and unzip at:**./data/DGraph-Fin/raw**

The completed file tree should look like this:

```plaintext
.
├── data/
│   ├── AMLWorld/
│   │   ├── processed
│   │   └── raw
│   │	│   ├── HI-Large_accounts.csv
│   │	│   ├── ...
│   ├── DGraph-Fin/
│   │   ├── processed
│   │   └── raw
│   │	│   ├── dgraphfin.npz
...
```

# Data process

Simply run:

```bash
python process_data.py
```

# Environment deploy

Simply run:

```bash
conda env create -f environment.yaml
```

# How to run

For GAT baseline:

```bash
python main.py --ratio 0.01 --model_name GAT --experiment_name GAT --data AMLWorld --trails 5
```

For D-EMAML baseline:

```bash
python main.py --ratio 0.01 --model_name DEMGNN --experiment_name DEMGNN --data AMLWorld --trails 5 --batch_size 16 --add_feature --n_classes 4
```

