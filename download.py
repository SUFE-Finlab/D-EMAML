import os
import requests
import zipfile
import kagglehub
import kaggle
import os
import zipfile
from pathlib import Path

def AMLWorld():
    dataset_slug = 'ealtman2019/ibm-transactions-for-anti-money-laundering-aml'
    dest_dir = Path('./data/AMLWorld/raw')
    print(f"Creating_path: {dest_dir}")
    dest_dir.mkdir(parents=True, exist_ok=True)
    print(f"Path '{dest_dir}' ready.")
    try:
        kaggle.api.authenticate()
        print('Downloadin, please wait.')
        kaggle.api.dataset_download_files(dataset_slug, path = dest_dir, unzip=True, quiet=False)
        print(f"Dataset downloaded succuessfuylly: {zip_filepath}")
    except Exception as e:
        return

def main():
    AMLWorld()

main()
