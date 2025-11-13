import kagglehub
import numpy as np
import torch
import torch_geometric
from torch_geometric.data import Data
import pandas as pd
from tqdm import tqdm
from torch_geometric.transforms import RemoveIsolatedNodes
from utils import DataSplit


trans = DataSplit()
def initialize_bank_indices(data, save_path):
    print(f'initializing for file {save_path}')
    lis_dic = {}
    size_dic = {}
    bank_index = data.bank_index.to('cuda:0')
    banks = bank_index.unique()
    for bank in tqdm(banks):
        bank = bank.cpu().item()
        from_is = bank_index[0]==bank
        to_is = bank_index[1]==bank
        temp = torch.where((from_is|to_is))[0]
        lis_dic[bank] = temp.cpu()
        size_dic[bank] = temp.shape[0]
    torch.save((lis_dic, size_dic), save_path)

def processed_data(raw_table, save_path):
    print('Reading Data...')
    try:
        data = pd.read_csv(raw_table)
    except:
        print(f'data:{raw_table} not found')
    print('Constructing Data...')
    Payment_currency = data['Payment Currency'].unique()
    Receiving_currency = data['Receiving Currency'].unique()
    Payment_Format = data['Payment Format'].unique()
    columns = ['Timestamp','Amount Received','Amount Received','Amount Paid','Is Laundering']
    trans_data = pd.DataFrame(index = data.index)
    for column in columns:
        trans_data[column] = data[column]
    trans_data['From'] = data['From Bank'].astype('str')+data['Account']
    trans_data['To'] = data['To Bank'].astype('str')+data['Account.1']
    Payment_currency_columns = [f'Is_Payment_Currency_{i}' for i in Payment_currency]
    Receiving_currency_columns = [f'Is_Receiving_Currency_{i}' for i in Receiving_currency]
    Payment_Format_columns = [f'Is_Payment_Format_{i}' for i in Payment_currency]
    for column,target in tqdm(zip(Payment_currency_columns,Payment_currency)):
        values = np.zeros(len(data),dtype = bool)
        indices = data['Payment Currency']==target
        values[indices] = True
        trans_data[column] = values
    for column,target in tqdm(zip(Receiving_currency_columns,Receiving_currency)):
        values = np.zeros(len(data),dtype = bool)
        indices = data['Receiving Currency']==target
        values[indices] = True
        trans_data[column] = values    
    for column,target in tqdm(zip(Payment_Format_columns,Payment_currency)):
        values = np.zeros(len(data),dtype = bool)
        indices = data['Payment Format']==target
        values[indices] = True
        trans_data[column] = values
    trans_data['From Bank'] = data['From Bank']
    trans_data['To Bank'] = data['To Bank']
    trans_data['Timestamp'] = pd.to_datetime(trans_data['Timestamp'])
    trans_data['Unix_Timestamp'] = trans_data['Timestamp'].astype('int64') // 10**9
    combined_series = pd.concat([trans_data['From'], trans_data['To']], ignore_index=True)
    unique_series = combined_series.drop_duplicates()
    unique_data = pd.DataFrame(unique_series,columns = ['name']).reset_index(drop = True)
    unique_data['id'] = unique_data.index
    from_id = pd.merge(trans_data[['From']],unique_data,how = 'left',left_on = 'From',right_on = 'name')
    to_id = pd.merge(trans_data[['To']],unique_data,how = 'left',left_on = 'To',right_on = 'name')
    trans_data['from_id'] = from_id['id']
    trans_data['to_id'] = to_id['id']
    print('Constructing PT file...')
    data = Data()
    data.bank_index = torch.tensor(trans_data[['From Bank','To Bank']].values,dtype = torch.int64)
    data.edge_index = torch.tensor(trans_data[['from_id','to_id']].values,dtype = torch.int64)
    data.time = torch.tensor(trans_data['Unix_Timestamp'].values,dtype = torch.int64)
    data.y = torch.tensor(trans_data['Is Laundering'],dtype = torch.int64)
    data.edge_attr = torch.tensor(trans_data.drop(columns = ['Timestamp','Is Laundering','From','To','Unix_Timestamp','from_id','to_id','From Bank','To Bank']).values.astype(np.float32),dtype = torch.float32)
    data.x = torch.ones((len(unique_data),1),dtype = torch.float32)
    data.bank_index = data.bank_index.transpose(0,1)
    data.edge_index = data.edge_index.transpose(0,1)
    print(f'Validation for data {save_path}:{data.validate()}')
    train_data, valid_data, test_data = trans(data)
    torch.save(train_data,f'{save_path}/train.pt')
    torch.save(valid_data,f'{save_path}/valid.pt')
    torch.save(test_data,f'{save_path}/test.pt')
    torch.save(data,f'{save_path}/raw_data.pt')
    return
def AML_Proceeding():
    raw_path = './data/AMLWorld/raw'
    processed_path = './data/AMLWorld/processed'
    lis_name = ['HI-Small_Trans']
    for name in lis_name:
        csv_path = f'{raw_path}/{name}.csv'
        pt_path = f'{processed_path}'
        print(f"Processing {name}...")
        processed_data(csv_path, pt_path)
    
    print("Path to dataset files:", processed_path)

def DGraph_Proceeding():
    raw_path = './data/DGraph-Fin/raw'
    processed_path = './data/DGraph-Fin/processed'
    name = 'dgraphfin.npz'
    npz_path = f'{raw_path}/{name}'
    save_path = f"{processed_path}"
    data = np.load(npz_path)
    g = Data()
    g.x = torch.tensor(data['x'], dtype = torch.float32)
    y = torch.tensor(data['y'], dtype = torch.int64)
    g.edge_index = torch.tensor(data['edge_index'], dtype = torch.int64).transpose(0,1)
    g.edge_type = torch.tensor(data['edge_type'], dtype = torch.int64)
    g.edge_attr = torch.zeros((g.edge_index.shape[1],1),dtype = torch.float32)
    g.time = torch.tensor(data['edge_timestamp'], dtype = torch.int64)
    g.y = torch.zeros(g.edge_index.shape[1], dtype = torch.int64)
    node_type = y
    node_type[node_type == 1] = 0
    node_type[node_type == 2] = 1
    node_type[node_type == 3] = 2
    g.node_type = node_type
    g.y[(y[g.edge_index[0]]==1)&(y[g.edge_index[1]]==1)] = 1
    print(f'Validation for data {save_path}:{g.validate()}')
    train_data, valid_data, test_data = trans(g)
    torch.save(train_data,f'{save_path}/train.pt')
    torch.save(valid_data,f'{save_path}/valid.pt')
    torch.save(test_data,f'{save_path}/test.pt')
    torch.save(g,f'{save_path}/raw_data.pt')
if __name__=="__main__":
    AML_Proceeding()
    DGraph_Proceeding()
