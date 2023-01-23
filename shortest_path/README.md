# Recurrent Transformer for the Shortest Path Problem

## Requirements
PyTorch and tqdm.

## How to run
- To run the baseline model L1R32H4 on 4x4 grid data on GPU 0.
```
python main.py --gpu 0 --grid_size 4
```
- To run the same model L1R32H4 with constriant loss on 4x4 grid data on GPU 1.
```
python main.py --gpu 1 --grid_size 4 --loss path
```
- To run the baseline model L1R32H4 on 12x12 grid data on GPU 2.
```
python main.py --gpu 2 --grid_size 12
```
- To run the same model L1R32H4 with constriant loss on 4x4 grid data on GPU 3.
```
python main.py --gpu 3 --grid_size 12 --loss path
```