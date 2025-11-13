import kagglehub
import numpy as np
import torch
import torch_geometric
from torch_geometric.data import Data
import pandas as pd
from tqdm import tqdm
from torch_geometric.transforms import RemoveIsolatedNodes


def processed_data(raw_table, save_path_train, save_path_test):
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
    # data.bank_index = torch.tensor(trans_data[['From Bank','To Bank']].values,dtype = torch.int64)
    data.edge_index = torch.tensor(trans_data[['from_id','to_id']].values,dtype = torch.int64)
    data.time = torch.tensor(trans_data['Unix_Timestamp'].values,dtype = torch.int64)
    data.y = torch.tensor(trans_data['Is Laundering'],dtype = torch.int64)
    data.edge_attr = torch.tensor(trans_data.drop(columns = ['Timestamp','Is Laundering','From','To','Unix_Timestamp','from_id','to_id','From Bank','To Bank']).values.astype(np.float32),dtype = torch.float32)
    data.x = torch.randn((len(unique_data),8),dtype = torch.float32)
    # data.bank_index = data.bank_index.transpose(0,1)
    data.edge_index = data.edge_index.transpose(0,1)
    print(f'Spliting for data {save_path_train}...')
    data = data.sort_by_time()
    train_size = int(data.num_edges*0.7)
    test_size = int(data.num_edges*0.3)

    transform = RemoveIsolatedNodes()
    train_data = Data()
    train_data.edge_index = data.edge_index[:,:train_size]
    train_data.time = data.time[:train_size]
    train_data.y = data.y[:train_size]
    train_data.edge_attr = data.edge_attr[:train_size]
    train_data.x = data.x
    train_data = transform(train_data)
    # train_data
    test_data = Data()
    test_data.edge_index = data.edge_index[:,-test_size:]
    test_data.time = data.time[-test_size:]
    test_data.y = data.y[-test_size:]
    test_data.edge_attr = data.edge_attr[-test_size:]
    test_data.x = data.x
    test_data = transform(test_data)
    # test_data
    print(f'Saving for data {save_path_train}')
    torch.save(data,save_path_train)
    print(f'Saving for data {save_path_test}')
    torch.save(data,save_path_test)
    return
if __name__=="__main__":
    raw_path = './data/raw'
    processed_path = './data/processed'
    name = 'HI-Small'
    csv_name = 'HI-Small_Trans'
    csv_path = f'{raw_path}/{csv_name}.csv'
    pt_train_path = f'{processed_path}/{name}_train.pt'
    pt_test_path = f'{processed_path}/{name}_test.pt'
    print(f"Processing {csv_name}...")
    processed_data(csv_path, pt_train_path, pt_test_path)
    print("Path to dataset files:", processed_path)