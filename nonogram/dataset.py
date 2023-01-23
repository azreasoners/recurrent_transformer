import csv
import numpy as np
import torch
from torch.utils.data import Dataset

class Nonogram_Dataset(Dataset):
    def __init__(self, data_path, board_dim, max_num_per_hint, limit, seed):
        """
        Args:
            data_path: a string denoting the path to the dataset
            board_dim: an integer denoting the size of the board
            max_num_per_hint: an integer denoting the max number of hint for each row/column
            limit: -1; or a positive integer denoting the maximum number of data
            seed: an integer denoting random seed
        """
        # breakpoint()
        self.board_dim = board_dim
        self.max_num_per_hint = max_num_per_hint
        all_unsolved, all_solved = self.__read_csv(data_path)

        rng = np.random.RandomState(seed)
        indices = rng.permutation(len(all_unsolved))
        # limit the number of data
        if 0 < limit < len(indices):
            indices = indices[:limit]
        self.X, self.Y = self.process_data(all_unsolved, all_solved, indices)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        """
        Each data instance is a pair <board, label> where
            board: a long tensor of shape (81) consisting of {0,...,9}
            label: a long tensor of shape (81) consisting of {0,...,8} and -100 denoting given cells
        """
        return self.X[idx], self.Y[idx]

    @staticmethod
    def __read_csv(filename):
        print(f"Reading {filename}...")
        all_unsolved = list()
        all_solved = list()
        with open(filename) as f:
            reader = csv.reader(f, delimiter=",")
            for row in reader:
                unsolved, solved = row
                all_unsolved.append(unsolved)
                all_solved.append(solved)
        return all_unsolved, all_solved

    def process_data(self, all_unsolved, all_solved, indices):
        X = list()
        Y = list()
        # breakpoint()
        for ind in indices:
            unsolved = all_unsolved[ind]
            numbers = [n.split(".") for n in unsolved.split("/")]

            numbers = [[int(n) for n in number] for number in numbers]
            numbers_arr = np.array(
                [
                    np.pad(number, (0, self.max_num_per_hint - len(number)), "constant")
                    for number in numbers
                ]
            )
            top_arr = numbers_arr[: self.board_dim]
            left_arr = numbers_arr[self.board_dim :]

            cell_numbers = list()
            for top in top_arr:
                for left in left_arr:
                    cell_numbers.append(np.concatenate((top, left)))
            cell_numbers_arr = np.array(cell_numbers)
            X.append(cell_numbers_arr)
        for ind in indices:
            pass
            solved = all_solved[ind]
            solved_arr = np.array([int(s) for s in solved])
            Y.append(solved_arr)
        return torch.tensor(X), torch.tensor(Y, dtype=torch.long)
