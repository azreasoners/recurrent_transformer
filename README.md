# Recurrent Transformer
This repository contains the codes to reproduce the experiments for the submission "Learning to Solve Constraint Satisfaction Problems with Recurrent Transformers".

## Datasets bigger than 100MB or from existing repositories
Please download the following dataset files from the given link and put to the given destination.
| filename | description | from | to |
| --- | --- | --- | --- |
| palm_i2t_train.csv | RRN-V (train) | https://drive.google.com/file/d/1SCBkX_c2Xaxjvkx0P481G3-SnUGMZX_L/view?usp=sharing | data/visual_sudoku/palm_i2t_train.csv |
| features_img.pt | SATNet-V (input) | https://github.com/locuslab/SATNet#getting-the-datasets | data/satnet/features_img.pt |
| features.pt | SATNet (input) | same as above | data/satnet/features.pt |
| labels.pt | SATNet and SATNet-V (label) | same as above | data/satnet/labels.pt |
| perm.pt | cell permutation for SATNet and SATNet-V | same as above | data/satnet/perm.pt |
| train.csv | RRN (train) | https://www.dropbox.com/s/rp3hbjs91xiqdgc/sudoku-hard.zip?dl=1 | data/sudoku-hard/train.csv |
| valid.csv | RRN (valid) | same as above| data/sudoku-hard/valid.csv |
| test.csv | RRN (test) | same as above| data/sudoku-hard/test.csv |

## Installation
1. Install Anaconda according to its [installation page](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html).
2. Create a new environment using the following commands in terminal.
```bash
conda create -n rt python=3.7
conda activate rt
```
3. Install tqdm, Numpy, Pandas, matplotlib, and wandb
```bash
conda install -c anaconda tqdm numpy pandas
conda install -c conda-forge matplotlib
python3 -m pip install wandb
wandb login
```
4. Install Pytorch according to its [Get-Started page](https://pytorch.org/get-started/locally/). Below is an example command we used on Linux with cuda 10.2.
```bash
conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch
```
or for CPU only
```bash
conda install pytorch torchvision torchaudio cpuonly -c pytorch
```

## To generate visual Sudoku dataset
One can download the visual Sudoku training dataset `palm_i2t_train.csv` from the above provided link or generate with the following commands.
```
cd data
python visual_sudoku_data.py --problem_type i2t
```

## To reproduce the experimental results with a single bash file
Note that it will take a long time for training since it assumes a single GPU and runs every experiment for `runs` (specified in line 4 of `all.sh`) times with different random seeds in {1, 2, ..., `runs`}. One can comment out most of the script and run one experiment at a time with a different GPU index.
```
bash all.sh
```

## To run textual Sudoku experiments
- To run the baseline model L1R32H4 on 9k/1k train/test SATNet data on GPU 0.
```
cd sudoku
python main.py --all_layers --n_layer 1 --n_recur 32 --n_head 4 --epochs 200 --eval_interval 1 --lr 0.001 --dataset satnet --gpu 0
```
- To apply the constraint losses L_sudoku `c1` and L_attention `att_c1` with default weights `1` and  `0.1` to the above baseline on GPU 1.
```
cd sudoku
python main.py --all_layers --n_layer 1 --n_recur 32 --n_head 4 --epochs 200 --eval_interval 1 --lr 0.001 --dataset satnet --gpu 1 --loss c1 att_c1 --hyper 1 0.1
```
- One can also test on Palm dataset by specifying `--dataset palm` and/or use `--n_train 180000` to change the number of training data from 9k (default) to 180k.
- One can always specify `--wandb` in the command to visualize the results in [wandb](https://wandb.ai/). This also applies to all experiments below.

## To run visual Sudoku experiments
- To run the baseline model L1R32H4 on 9k/1k train/test SATNet-V data on GPU 2.
```
cd visual_sudoku
python main.py --all_layers --n_layer 1 --n_recur 32 --n_head 4 --epochs 500 --eval_interval 1 --lr 0.001 --dataset satnet --gpu 2
```
- To apply the constraint losses L_sudoku `c1` and L_attention `att_c1` with default weights `1` and  `0.1` to the above baseline on GPU 3.
```
cd visual_sudoku
python main.py --all_layers --n_layer 1 --n_recur 32 --n_head 4 --epochs 500 --eval_interval 1 --lr 0.001 --dataset satnet --gpu 3 --loss c1 att_c1 --hyper 1 0.1
```

## To run 16x16 Sudoku experiments
```
cd sudoku_16
python main.py --dataset easy
python main.py --dataset medium
```

## To run shortest path experiments
```
cd shortest_path
python main.py --gpu 0 --grid_size 4
python main.py --gpu 1 --grid_size 4 --loss path
python main.py --gpu 2 --grid_size 12
python main.py --gpu 3 --grid_size 12 --loss path
```

## To run MNIST mapping experiment

```
cd MNIST_mapping
python main.py
```

## To run nonogram experiments
```
cd nonogram
python main.py --game_size 7 --gpu 0
python main.py --game_size 15 --gpu 1
```



## Acknowledgements
The GPT implementation is from Andrej Karpathy's [minGPT](https://github.com/karpathy/minGPT) repo. Note that we replaced the causal self-attention in GPT model with typical self-attention by setting `causal_mask=False` whenever it is used. In this way, logical variable X_i is able to pay attention to another logical variable X_j when j>i.
