# Introduction

This repo is from D-EMAML, that tackles the challenging task of detecting money laundering patterns in financial transaction networks by leveraging dual-edge motifs and graph neural networks for direct edge anomaly detection.

# Data Source

AMLWorld: [AMLWorld]([IBM Transactions for Anti Money Laundering (AML)](https://www.kaggle.com/datasets/ealtman2019/ibm-transactions-for-anti-money-laundering-aml/data))

TransBank is not provided due to supervision.

# Environment deploy

Simply run:

```bash
conda env create -f environment.yaml
```

# Data Processing

For data downloading:

```bash
conda activate torch-env
python download.py
```

It will print out the path to downloading file. Unzip the file to path "/data/raw", then:

```bash
python process_data.py
```
