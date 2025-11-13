from torch_geometric.data import Data,Dataset
import torch_geometric.utils as U
from torch import nn
import torch
import os
from torch_geometric.nn import BatchNorm
from torch_geometric.transforms import BaseTransform
from torch_geometric.utils import degree

def get_pos_indices(edge_index, device):
    t1 = edge_index.unsqueeze(2).repeat(1,1,edge_index.shape[1])
    t2 = edge_index.unsqueeze(1).repeat(1,edge_index.shape[1],1)
    indices = torch.empty((2,2,edge_index.shape[1],edge_index.shape[1]),dtype = torch.bool).to(device)
    
    t = ~torch.eye(edge_index.shape[1], dtype = torch.bool).to(device)
    indices[0,0] = (t1[0] == t2[0])&t
    indices[1,0] = (t1[1] == t2[0])&t
    indices[0,1] = (t1[0] == t2[1])&t
    indices[1,1] = (t1[1] == t2[1])&t
    return indices

class AddPorts(BaseTransform): 
    def __init__(self):
        pass
        
    def __call__(self, data):
        src, dst = data.edge_index
        nums_index = src * data.num_nodes + dst
        unique_index = nums_index.unique()
        unique_src = unique_index//data.num_nodes
        unique_dst = unique_index%data.num_nodes
        inports = degree(unique_dst, data.num_nodes).unsqueeze(1)
        outports = degree(unique_src, data.num_nodes).unsqueeze(1)
        data.x = torch.cat([data.x,inports,outports],dim=1)
        return data

class AddEgoIds(BaseTransform):
    def __init__(self):
        pass

    def __call__(self, data):
        x = data.x 
        device = x.device
        ids = torch.zeros((x.shape[0], 1), device=device)
        nodes = torch.unique(data.edge_label_index.view(-1)).to(device)
        ids[nodes] = 1
        data.x = torch.cat([x, ids], dim=1)
        
        return data

class DataSplit(BaseTransform):
    def __call__(self, data):
        data = data.sort_by_time()
        train_size = int(data.num_edges*0.6)
        valid_size = int(data.num_edges*0.1)
        test_size = int(data.num_edges*0.3)
        train_data = data.edge_subgraph(torch.arange(0, train_size,dtype = torch.int64))
        test_data = data.edge_subgraph(torch.arange(train_size+valid_size, data.num_edges,dtype = torch.int64))
        valid_data = data.edge_subgraph(torch.arange(train_size, train_size+valid_size,dtype = torch.int64))
        return train_data, valid_data, test_data

class SelectEdges(BaseTransform):
    def __call__(self, data, ratio, num_select, device):
        if(num_select is not None):
            data.input_indices = construct_selected_nodes(data.y, ratio, num_select)
        else:
            data.input_indices = torch.arange(data.y.shape[0])
        data.edge_weights = data.edge_attr[:,0]
        return data

def construct_selected_nodes(y, ratio, num_select = None):
    t1 = torch.where(y == 1)[0]
    t2 = torch.where(y == 0)[0]
    ra = torch.randperm(t2.shape[0])
    if(ratio is None):
        ratio = t1.shape[0]/y.shape[0]
    num_abnorm = min(t1.shape[0],int(num_select * ratio))
    num_norm = int(num_abnorm*(1-ratio)/ratio)
    t2 = t2[ra][:num_norm]
    ra = torch.randperm(t1.shape[0])
    t1 = t1[ra][:num_abnorm]
    t3 = torch.cat([t1,t2])
    ra = torch.randperm(t3.shape[0])
    t3 = t3[ra]
    return t3

def calculate_metrics(y_pred, y_true):
    tp = torch.sum((y_pred == 1) & (y_true == 1)).item()
    fp = torch.sum((y_pred == 1) & (y_true == 0)).item()
    fn = torch.sum((y_pred == 0) & (y_true == 1)).item()

    if tp + fp == 0:
        precision = 0
    else:
        precision = tp / (tp + fp)

    if tp + fn == 0:
        recall = 0
    else:
        recall = tp / (tp + fn)

    if precision + recall == 0:
        f1 = 0
    else:
        f1 = 2 * (precision * recall) / (precision + recall)

    return precision, recall, f1
