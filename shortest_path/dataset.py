import pickle
import torch
from torch.utils.data import Dataset
import numpy as np
import random
from numpy.random import permutation

def save_pickle(obj, filename):
    with open(filename, "wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
    return None

def load_pickle(filename):
    print(f"Reading {filename}...")
    with open(filename, "rb") as f:
        labels = pickle.load(f)
    return labels

def create_node_edge_info(N):
    N=4
    edges=list()
    
    for row_idx,row in enumerate(range(N)):
        for idx,V in enumerate(range(N-1)):
            V_idx = N*row_idx + idx
            edges.append((V_idx,V_idx+1))
            
    for idx, col in enumerate(range(N)):
        for row_idx,row in enumerate(range(N-1)):
            V_idx = N*row_idx + idx
            edges.append((V_idx,V_idx+N))
    
    node_edges=list()
    for i in range(N**2):
        node_edge=list()
        for edge in edges:
            if i in edge:
                if i==edge[0]:
                    node_edge.append(edge[1])
                else:
                    node_edge.append(edge[0])
        node_edges.append(node_edge)
    return node_edges, edges

def create_edge_base(node_edges):    
    edge_base = np.zeros((16,16))
    for edges,node_inds in zip(edge_base,node_edges):
        edges[node_inds]=1
    return edge_base


class GridData():
    def __init__(self, data_path,num_nodes,num_edges, split, data_limit=None):
        """
        Args:
            split: (total, train, validation, test)
        """
        self.num_nodes=num_nodes
        self.num_edges=num_edges
        np.random.seed(0)
        data = []
        labels = []
        labels_all = []
        labels_ordered = []
        removed_edges = []
        with open(data_path) as file:
            for line_idx,line in enumerate(file):
                print(line_idx)
                if data_limit and data_limit == line_idx:
                    break
                tokens = line.strip().split(',')
                if(tokens[0] != ''):
                    removed = [int(x) for x in tokens[0].split('-')]
                else:
                    removed = []

                inp = [int(x) for x in tokens[1].split('-')]
                paths = tokens[2:]
                s_g_array = to_one_hot(inp, self.num_nodes)
                
                edge_encode = to_one_hot(removed, self.num_edges, True)
                edge_encode[edge_encode==0]=2
                edge_encode[edge_encode==1]=3
                data.append(np.concatenate((s_g_array,edge_encode )))
                pathind = 0
                if len(paths) > 1:
                    pathind = random.randrange(len(paths))
                paths_parsed = [[int(x) for x in path.split('-')] for path in paths]
                labels.append(to_one_hot(paths_parsed[0], self.num_edges))
                labels_all.append([to_one_hot(path, self.num_edges) for path in paths_parsed])
                labels_ordered.append(paths_parsed)
                removed_edges.append(removed)

        # We're going to split 60/20/20 train/test/validation
        perm = permutation(len(data))

        Ntotal, Ntrain, Nvalidation, Ntest = split
        Ntrain = int(len(data) / Ntotal * Ntrain)
        Nvalidation = int(len(data) / Ntotal * Nvalidation)
        Ntest = int(len(data) / Ntotal * Ntest)
        train_inds = perm[:Ntrain]
        valid_inds = perm[Ntrain:Ntrain+Nvalidation]
        test_inds = perm[-Ntest:]

        self.train_inds=train_inds
        self.test_inds=test_inds
        self.test_inds=test_inds
        self.data = np.array(data)
        self.labels = np.array(labels)
        self.labels_all=np.array(labels_all,dtype=object)
        self.labels_ordered=np.array(labels_ordered, dtype=object)
        self.removed_edges=np.array(removed_edges)
        
        self.train_data = self.data[train_inds, :]
        self.valid_data = self.data[valid_inds, :]
        self.test_data = self.data[test_inds, :]
        self.train_labels = self.labels[train_inds, :]
        self.valid_labels = self.labels[valid_inds, :]
        self.test_labels = self.labels[test_inds, :]
        self.train_labels_all = self.labels_all[train_inds]
        self.valid_labels_all = self.labels_all[valid_inds]
        self.test_labels_all = self.labels_all[test_inds]
        
        self.train_labels_ordered = self.labels_ordered[train_inds]
        self.valid_labels_ordered = self.labels_ordered[valid_inds]
        self.test_labels_ordered = self.labels_ordered[test_inds]
        
        self.train_removed_edges = self.removed_edges[train_inds]
        self.valid_removed_edges = self.removed_edges[valid_inds]
        self.test_removed_edges = self.removed_edges[test_inds]

        # Count what part of the batch we're attempt
        self.batch_ind = len(train_inds)
        self.batch_perm = None
        np.random.seed()

    def get_batch(self, size):
        # If we're out:
        if self.batch_ind >= self.train_data.shape[0]:
            # Rerandomize ordering
            self.batch_perm = permutation(self.train_data.shape[0])
            # Reset counter
            self.batch_ind = 0

        # If there's not enough
        if self.train_data.shape[0] - self.batch_ind < size:
            # Get what there is, append whatever else you need
            ret_ind = self.batch_perm[self.batch_ind:]
            d, l = self.train_data[ret_ind, :], self.train_labels[ret_ind, :]
            size -= len(ret_ind)
            self.batch_ind = self.train_data.shape[0]
            nd, nl = self.get_batch(size)
            return np.concatenate(d, nd), np.concatenate(l, nl)

        # Normal case
        ret_ind = self.batch_perm[self.batch_ind: self.batch_ind + size]
        return self.train_data[ret_ind, :], self.train_labels[ret_ind, :]

def to_one_hot(dense, n, inv=False):
    one_hot = np.zeros(n)
    one_hot[dense] = 1
    if inv:
        one_hot = (one_hot + 1) % 2
    return one_hot


class shortest_path_Dataset(Dataset):
    def __init__(self, data,labels,labels_ordered, labels_all, removed_edges, grid_size):
        """
        Args:
            input_dict: a dictionary that maps indices to a Sudoku board
            label_dict: a dictionary that maps indices to a Sudoku solution
            limit: -1; or a positive number denoting the maximum number of data
            seed: an integer denoting random seed
        """
        self.grid_size = grid_size
        self.num_nodes = grid_size**2
        self.num_edges = 2*(self.grid_size-1)*self.grid_size
        self.X=data
        self.labels=labels
        self.labels_ordered=labels_ordered
        self.labels_all=labels_all
        
        self.node_edges,self.edge_list = create_node_edge_info(grid_size)
        
        self.edge_base=create_edge_base(self.node_edges)
        self.removed_edges=removed_edges
        self.process_data()
        
    def process_data(self):
        new_edges_all=list()
        sol_inds_all = list()
        
        Y=list()
        for idx in range(len(self.X)):
            edge_base = np.copy(self.edge_base)
            data = self.X[idx]
            
            node_label = -100*np.ones(self.num_nodes)
            edge_labels=list()
            for sol in self.labels_ordered[idx]:
                edge_label = np.zeros(self.num_edges)
                edge_label[sol]=1
                edge_labels.append(edge_label)
            Y.append(np.concatenate((node_label,edge_labels[0])))
            sol_inds_all.append(edge_labels)
        self.Y=Y
        self.sol_inds_all = np.array(sol_inds_all, dtype=object)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        """
        Each data instance is a pair <board, label> where
            board: a long tensor of shape (81) consisting of {0,...,9}
            label: a long tensor of shape (81) consisting of {0,...,8} and -100 denoting given cells
        """
        return torch.tensor(self.X[idx],dtype=torch.int), torch.tensor(self.Y[idx],dtype=torch.long), self.removed_edges[idx], idx



