"""
This code creates the MNIST version of the Palm dataset in the form of a csv file
For image->text (i2t) problem_type:
    each row in the csv file is a data instance of the form "board, label, label_ug" where
    board is a string of 81 image indices
    label is a string of 81 digits
    label_ug is a string of 81 digits with -100 denoting given cells
For image->image (i2i) problem_type:
    each row in the csv file is a data instance of the form "board, label, label_ug" where
    board is a string of 81 image indices
    label is a string of 81 image indices
    label_ug is a string of 81 image indices with -100 denoting given cells
"""

import argparse
import csv
import os
import urllib.request
import zipfile
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import torchvision

class Sudoku_Dataset_Palm(Dataset):
    def __init__(self, segment, seed, limit=-1):
        """
        Args:
            segment: a string in {'train', 'test', 'valid'}
            seed: an integer denoting random seed
            limit: -1; or a positive integer denoting the maximum number of data
        """
        assert segment in ['train', 'valid', 'test']
        url = 'https://www.dropbox.com/s/rp3hbjs91xiqdgc/sudoku-hard.zip?dl=1'
        zip_fname = './sudoku-hard.zip'
        dest_dir = './sudoku-hard/'
        # download the data if it doesn't exist
        if not os.path.exists(dest_dir):
            print(f'Downloading Palm dataset into {dest_dir}...')
            urllib.request.urlretrieve(url, zip_fname)
            with zipfile.ZipFile(zip_fname) as f:
                f.extractall('./')
        # load the data
        data = self.__read_csv(dest_dir + f'{segment}.csv')
        # set the random seed for reproducibility; get the permutated indices of all data
        # rng = np.random.RandomState(seed)
        rng = np.random.default_rng(seed)
        indices = rng.permutation(len(data))
        # limit the number of data
        if 0 < limit < len(indices):
            indices = indices[:limit]
        # construct the dataset consisting of <board, label, label_ug>
        self.board = torch.zeros((len(indices), 81)).long()
        self.label = torch.zeros((len(indices), 81)).long()
        self.label_ug = torch.zeros((len(indices), 81)).long() # label for ungrounded data
        for i, idx in enumerate(indices):
            self.board[i] = self.__str_to_tensor(data[idx][0])
            self.label[i] = self.__str_to_tensor(data[idx][1]) - 1
            self.label_ug[i] = self.label[i]
            self.label_ug[i][self.board[i] != 0] = -100
    
    def __len__(self):
        return len(self.board)

    def __getitem__(self, idx):
        """
        Each data instance is a pair <board, label, label_ug> where
            board: a long tensor of shape (81) consisting of {0,...,9}
            label: a long tensor of shape (81) consisting of {0,...,8}
            label_ug: a long tensor of shape (81) consisting of {0,...,8} and -100 denoting given cells
        """
        return self.board[idx], self.label[idx], self.label_ug[idx]
    
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

def matrix_to_text(m):
    text = ''
    for nums in m.view(9,9).tolist():
        nums = [str(num) for num in nums]
        text += ' '.join(nums) + '\n'
    return text

# Create an text->text (t2t) dataset in GPT-3 format
def create_dataset_gpt3(num, segment, seed):
    """
    Args:
        num: an integer denoting the number of examples with every given in 17-34
        segment: a string in {'train', 'test', 'valid'}
        seed: an integer denoting random seed
    """
    palm_dataset = Sudoku_Dataset_Palm(segment, seed)
    total_num = num * 18
    text = ''
    num_to_data = {}
    for i in range(17, 35):
        num_to_data[i] = []
    for (board, label, _) in palm_dataset:
        num_given = (board != 0).sum().item()
        if len(num_to_data[num_given]) < num:
            # create the text version of board and label and append (board, label)
            num_to_data[num_given].append((matrix_to_text(board), matrix_to_text(label+1)))
            total_num -= 1
        if total_num == 0:
            break
    for k in num_to_data:
        for i in range(num):
            board, label = num_to_data[k][i]
            text += f'Human:\n{board}AI:\n{label}'
    with open(f'GPT3_palm_{segment}.txt', 'w') as f:
        f.write(text)

# Create an image->text (i2t) or image->image (i2i) dataset
def create_dataset(problem_type, segment, seed):
    """
    Args:
        problem_type: a string in {'i2t', 'i2i'} denoting image->text or image->image dataset
        segment: a string in {'train', 'test', 'valid'}
        seed: an integer denoting random seed
    """
    # init Palm and MNSIT datasets
    palm_dataset = Sudoku_Dataset_Palm(segment, seed)
    mnist_dataset = torchvision.datasets.MNIST(root='./', train=(segment=='train'), download=True)
    # find the indices of MNIST images for each label
    indices = {}
    for label in range(10):
        indices[label] = (mnist_dataset.targets==label).nonzero().flatten().tolist()
    # init a random number generator and a dataframe
    rng = np.random.default_rng(seed)
    df = pd.DataFrame(columns = ['board', 'label', 'label_ug'], index = range(len(palm_dataset)))
    # for each data in the Palm dataset, turn it into a row in df
    for idx, (board, label, label_ug) in enumerate(palm_dataset):
        board, label, label_ug = board.tolist(), label.tolist(), label_ug.tolist()
        if problem_type == 'i2t':
            board = [rng.choice(indices[digit]) for digit in board]
            board_indices, label_indices, label_ug_indices = board, label, label_ug
        else:
            board_indices = [rng.choice(indices[digit]) for digit in board]
            label_indices = [rng.choice(indices[label[i] + 1]) if board[i] == 0 else board_indices[i] for i in range(81)]
            label_ug_indices = [label_indices[i] if label_ug[i] != -100 else -100 for i in range(81)]
        df.iloc[idx]['board'] = ','.join([str(e) for e in board_indices])
        df.iloc[idx]['label'] = ','.join([str(e) for e in label_indices])
        df.iloc[idx]['label_ug'] = ','.join([str(e) for e in label_ug_indices])
    # save the df
    df.to_csv(f'./visual_sudoku/palm_{problem_type}_{segment}.csv', encoding='utf-8', index=False)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=0, help='Random seed')
    parser.add_argument('--problem_type', type=str, default='i2t', help='Type of problem in \{i2t, i2i\}')
    args = parser.parse_args()

    for segment in ('test', 'train'):
        create_dataset(args.problem_type, segment, args.seed)
        # create_dataset_gpt3(1, segment, args.seed)