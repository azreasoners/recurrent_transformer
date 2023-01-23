import csv
import pickle
import os
import urllib.request
import zipfile
import numpy as np
import torch
from torch.utils.data import Dataset

def save_pickle(obj, filename):
    with open(filename, 'wb') as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
    return None

def load_pickle(filename):
    print(f'Reading {filename}...')
    with open(filename, 'rb') as f:
        labels = pickle.load(f)
    return labels

class Sudoku_Dataset(Dataset):
    def __init__(self, input_path, label_path, limit, seed):
        """
        Args:
            input_path: a dictionary that maps indices to a Sudoku board
            label_path: a dictionary that maps indices to a Sudoku solution
            limit: -1; or a positive number denoting the maximum number of data
            seed: an integer denoting random seed
        """
        # load the data
        input_dict = load_pickle(input_path)
        label_dict = load_pickle(label_path)
        # set the random seed for reproducibility; get the permutated indices of all data
        rng = np.random.RandomState(seed)
        indices = rng.permutation(len(input_dict))
        # limit the number of data
        if 0 < limit < len(indices):
            indices = indices[:limit]
        # construct the dataset consisting of <board, label>
        self.board = torch.zeros((len(indices), 81)).long()
        self.label = torch.zeros((len(indices), 81)).long()
        for i, idx in enumerate(indices):
            self.board[i] = torch.from_numpy(input_dict[idx]).view(-1).long()
            self.label[i] = torch.from_numpy(label_dict[idx]).view(-1).long() - 1
            self.label[i][self.board[i] != 0] = -100
    
    def __len__(self):
        return len(self.board)

    def __getitem__(self, idx):
        """
        Each data instance is a pair <board, label> where
            board: a long tensor of shape (81) consisting of {0,...,9}
            label: a long tensor of shape (81) consisting of {0,...,8} and -100 denoting given cells
        """
        return self.board[idx], self.label[idx]

class Sudoku_Dataset_Palm(Dataset):
    def __init__(self, segment, limit, seed):
        """
        Args:
            segment: a string in {'train', 'test', 'valid'}
            limit: -1; or a positive integer denoting the maximum number of data
            seed: an integer denoting random seed
        """
        assert segment in ['train', 'valid', 'test']
        url = 'https://www.dropbox.com/s/rp3hbjs91xiqdgc/sudoku-hard.zip?dl=1'
        zip_fname = '../data/sudoku-hard.zip'
        dest_dir = '../data/sudoku-hard/'
        # download the data if it doesn't exist
        if not os.path.exists(dest_dir):
            print(f'Downloading Palm dataset into {dest_dir}...')
            urllib.request.urlretrieve(url, zip_fname)
            with zipfile.ZipFile(zip_fname) as f:
                f.extractall('../data/')
        # load the data
        data = self.__read_csv(dest_dir + f'{segment}.csv')
        # set the random seed for reproducibility; get the permutated indices of all data
        rng = np.random.RandomState(seed)
        indices = rng.permutation(len(data))
        # limit the number of data
        if 0 < limit < len(indices):
            indices = indices[:limit]
        # construct the dataset consisting of <board, label>
        self.board = torch.zeros((len(indices), 81)).long()
        self.label = torch.zeros((len(indices), 81)).long()
        for i, idx in enumerate(indices):
            self.board[i] = self.__str_to_tensor(data[idx][0])
            self.label[i] = self.__str_to_tensor(data[idx][1]) - 1
            self.label[i][self.board[i] != 0] = -100
    
    def __len__(self):
        return len(self.board)

    def __getitem__(self, idx):
        """
        Each data instance is a pair <board, label> where
            board: a long tensor of shape (81) consisting of {0,...,9}
            label: a long tensor of shape (81) consisting of {0,...,8} and -100 denoting given cells
        """
        return self.board[idx], self.label[idx]
    
    @staticmethod
    def __read_csv(filename):
        print(f'Reading {filename}...')
        with open(filename) as f:
            reader = csv.reader(f, delimiter=',')
            return [(q, a) for q, a in reader]
    
    @staticmethod
    def __str_to_tensor(x):
        x = [int(e) for e in x]
        return torch.Tensor(x).long()


class Sudoku_Dataset_SATNet(Dataset):
    def __init__(self):
        data = {}
        data_to_path = {
            'board': '../data/satnet/features.pt',
            'board_img': '../data/satnet/features_img.pt',
            'label': '../data/satnet/labels.pt',
            'perm': '../data/satnet/perm.pt',
        }
        for k in data_to_path:
            with open(data_to_path[k], 'rb') as f:
                data[k] = torch.load(f)
        self.board = ((data['board'].sum(-1) != 0) * (data['board'].argmax(-1) + 1)).view(-1, 81).long() # (10000, 81)
        # self.board_img = data['board_img'].view(10000, 81, 28, 28).float() # (10000, 81, 28, 28)
        self.label = data['label'].argmax(-1).view(-1, 81).long() # (10000, 81)
        self.label_ug = self.label.clone() # (10000, 81)
        self.label_ug[self.board != 0] = -100

    def __len__(self):
        return len(self.board)

    def __getitem__(self, idx):
        """
        Each data instance is a tuple <board, board_img, label, label_ug> where
            board: a float tensor of shape (81) consisting of {0,...,9}
            board_img: a float tensor of shape (81, 28, 28) denoting 81 MNIST images
            label: a float tensor of shape (81) consisting of {0,...,8}
            label_ug: a float tensor of shape (81) consisting of {0,...,8} and -100 denoting given cells
        Note:
            We only use the pair <board, label_ug> as a data instance for textual Sudoku
        """
        # return self.board[idx], self.board_img[idx], self.label[idx], self.label_ug[idx]
        return self.board[idx], self.label_ug[idx]
