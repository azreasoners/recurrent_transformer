import numpy as np
import torch
from torch.utils.data import Dataset

class Sudoku_Dataset(Dataset):
    def __init__(self, input_path, label_path, limit, seed):
        """
        Args:
            input_dict: a dictionary that maps indices to a Sudoku board
            label_dict: a dictionary that maps indices to a Sudoku solution
            limit: -1; or a positive number denoting the maximum number of data
            seed: an integer denoting random seed
        """
        # load the data
        unsolved_arr = np.load(input_path)
        solved_arr = np.load(label_path)
        solved_arr = solved_arr - 1  # 0-index
        # set the random seed for reproducibility; get the permutated indices of all data
        rng = np.random.RandomState(seed)
        indices = rng.permutation(len(unsolved_arr))
        # limit the number of data
        if 0 < limit < len(indices):
            indices = indices[:limit]
        # construct the dataset consisting of <board, label>
        self.board = torch.zeros((len(indices), 256)).long()
        self.label = torch.zeros((len(indices), 256)).long()
        for i, idx in enumerate(indices):
            self.board[i] = torch.from_numpy(unsolved_arr[idx]).view(-1).long()
            self.label[i] = torch.from_numpy(solved_arr[idx]).view(-1).long()
            self.label[i][self.board[i] != 0] = -100

    def __len__(self):
        return len(self.board)

    def __getitem__(self, idx):
        """
        Each data instance is a pair <board, label> where
            board: a long tensor of shape (256) consisting of {0,...,16}
            label: a long tensor of shape (256) consisting of {0,...,15} and -100 denoting given cells
        """
        return self.board[idx], self.label[idx]
