import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import torchvision
from torchvision.transforms import transforms

import matplotlib.pyplot as plt

class Visual_Sudoku_Dataset_SATNet(Dataset):
    def __init__(self, data_type, seed):
        """
        Args:
            data_type: a string in {'ground', 'unground'} denoting whether the given digits are in the label
            seed: an integer denoting random seed; this is not used
        """
        assert data_type in ('ground', 'unground')
        self.data_type = data_type
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
        self.board_img = data['board_img'].view(10000, 81, 28, 28).float() # (10000, 81, 28, 28)
        self.label = data['label'].argmax(-1).view(-1, 81).long() # (10000, 81)
        self.label_ug = self.label.clone() # (10000, 81)
        self.label_ug[self.board != 0] = -100

    def __len__(self):
        return len(self.board)

    def __getitem__(self, idx):
        """
        Each data instance is a pair <board_img, label> or <board_img, label_ug> where
            board_img: a float tensor of shape (81, 28, 28) denoting 81 MNIST images
            label: a long tensor of shape (81) consisting of {0,...,8}
            label_ug: a float tensor of shape (81) consisting of {0,...,8} and -100 denoting given cells

        Note:
            -100 is the default ignore_index of torch.nn.CrossEntropyLoss, meaning that
            label -100 will not contribute to the loss computation
        """
        if self.data_type == 'ground':
            return self.board_img[idx], self.label[idx]
        return self.board_img[idx], self.label_ug[idx]


class Visual_Sudoku_Dataset_Palm(Dataset):
    def __init__(self, problem_type, segment, data_type, limit, seed):
        """
        Args:
            problem_type: a string in {'i2t', 'i2i'} denoting image->text or image->image dataset
            segment: a string in {'train', 'test', 'valid'}
            data_type: a string in {'ground', 'unground'} denoting whether the given digits are in the label
            limit: -1; or a positive integer denoting the maximum number of data
            seed: an integer denoting random seed
        """
        assert problem_type in ('i2t', 'i2i')
        assert segment in ('train', 'valid', 'test')
        assert data_type in ('ground', 'unground')
        self.problem_type = problem_type
        self.data_type = data_type
        # load the data
        df = pd.read_csv(f'../data/visual_sudoku/palm_{problem_type}_{segment}.csv')
        # transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081, ))])
        transform = transforms.Compose([transforms.ToTensor()])
        mnist_dataset = torchvision.datasets.MNIST(root='../data/', train=(segment!='test'), download=True, transform=transform)
        # set the random seed for reproducibility; get the permutated indices of all data
        rng = np.random.RandomState(seed)
        indices = rng.permutation(len(df))
        # limit the number of data
        if 0 < limit < len(indices):
            indices = indices[:limit]
        # construct the dataset consisting of <board, label, label_ug>
        self.board = torch.zeros((len(indices), 81, 28, 28)).float()
        if problem_type == 'i2t':
            self.label = torch.zeros((len(indices), 81)).long()
            self.label_ug = torch.zeros((len(indices), 81)).long() # label for ungrounded data
            for i, idx in enumerate(indices):
                self.label[i] = self.__str_to_tensor(df.iloc[idx]['label'])
                self.label_ug[i] = self.__str_to_tensor(df.iloc[idx]['label_ug'])
                board_indices = [int(e) for e in df.iloc[idx]['board'].split(',')]
                for cell_index, mnist_index in enumerate(board_indices):
                    self.board[i][cell_index] = mnist_dataset[mnist_index][0].squeeze()
        else: # problem_type == 'i2i'
            self.label = torch.zeros((len(indices), 81, 28, 28)).float()
            self.label_ug = torch.zeros((len(indices), 81, 28, 28)).float() # label for ungrounded data
            for i, idx in enumerate(indices):
                board_indices = [int(e) for e in df.iloc[idx]['board'].split(',')]
                label_indices = [int(e) for e in df.iloc[idx]['label'].split(',')]
                label_ug_indices = [int(e) for e in df.iloc[idx]['label_ug'].split(',')]
                for cell_index in range(81):
                    self.board[i][cell_index] = mnist_dataset[board_indices[cell_index]][0].squeeze()
                    self.label[i][cell_index] = mnist_dataset[label_indices[cell_index]][0].squeeze()
                    if label_ug_indices[cell_index] != -100:
                        self.label_ug[i][cell_index] = mnist_dataset[label_ug_indices[cell_index]][0].squeeze()
        
        # self.visualize_sudoku(idx=3)
        # breakpoint()

    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx):
        """
        Each data instance is a pair <board, label> or <board, label_ug> where
            board: a long tensor of shape (81, 28, 28)
            label: a long tensor of shape (81) consisting of {0,...,8} if self.problem_type is 'i2t'
                   a float tensor of shape (81, 28, 28) if self.problem_type is 'i2i'
            label_ug: a long tensor of shape (81) consisting of {0,...,8, -100} if self.problem_type is 'i2t'
                      where given cells are denoted by -100
                      a float tensor of shape (81, 28, 28) if self.problem_type is 'i2i'
                      where given cells are denoted by (28,28) of all zeros

        Note:
            -100 is the default ignore_index of torch.nn.CrossEntropyLoss, meaning that
            label -100 will not contribute to the loss computation
        """
        if self.data_type == 'ground':
            return self.board[idx], self.label[idx]
        return self.board[idx], self.label_ug[idx]
    
    @staticmethod
    def __str_to_tensor(x):
        x = x.split(',')
        x = [int(e) for e in x]
        return torch.Tensor(x).long()
    
    def visualize_sudoku(self, idx=0, filename=''):
        """
        Visualize a visual-sudoku board and store the figure in filenameXXX.png
        """
        pixels = 1 - self.board[idx]
        pixels = np.array(pixels, dtype='float').reshape(9,9,28,28)
        pixels_all = np.concatenate(np.concatenate(pixels, axis=1), axis=1)
        plt.imshow(pixels_all, cmap='gray')
        plt.axis('off')
        plt.savefig(filename + 'input.png', bbox_inches='tight', pad_inches=0)
        plt.clf()
        # for i in range(9):
        #     for j in range(9):
        #         digit_pixels = pixels[i][j]
        #         plt.imshow(digit_pixels, cmap='gray')
        #         plt.axis('off')
        #         plt.savefig(filename + f'cell{i+1}_{j+1}.png', bbox_inches='tight', pad_inches=0)
        #         plt.clf()

        if self.problem_type == 'i2i':
            pixels = self.label[idx]
            pixels = np.array(pixels, dtype='float').reshape(9,9,28,28)
            pixels = np.concatenate(np.concatenate(pixels, axis=1), axis=1)
            plt.imshow(pixels, cmap='gray')
            plt.savefig(filename + 'label.png')
            plt.clf()

            pixels = self.label_ug[idx]
            pixels = np.array(pixels, dtype='float').reshape(9,9,28,28)
            pixels = np.concatenate(np.concatenate(pixels, axis=1), axis=1)
            plt.imshow(pixels, cmap='gray')
            plt.savefig(filename + 'label_ug.png')
            plt.clf()
