import sys 
sys.path.append('..')

import argparse
import torch

from dataset import Nonogram_Dataset
from tok_emb import NonogramEmb
from network import testNN
from helper import print_result, visualize_adjacency
from mingpt.model import GPT, GPTConfig
from mingpt.trainer import Trainer, TrainerConfig
from mingpt.utils import set_seed


def main(args):
    # generate the name of this experiment
    prefix = f'[{args.game_size},{args.n_train//1000}k]'
    if args.label_size < args.batch_size:
        prefix = prefix[:-1] + f',{args.label_size}-{args.batch_size-args.label_size}]'
    if args.loss:
        prefix += '[' + '_'.join(args.loss) + f',{args.alpha}]'
    prefix += f'L{args.n_layer}R{args.n_recur}H{args.n_head}_'

    if args.wandb:
        import wandb
        wandb.init(project=f'nonogram')
        wandb.run.name = prefix[:-1]
    else:
        wandb = None

    #############
    # Seed everything for reproductivity
    #############
    set_seed(args.seed)

    #############
    # Load data
    #############
    if args.game_size == 7:  # for 7x7 board
        max_hint_value = 6
        max_num_per_hint = 4
        args.n_embd = 128  # max_num_per_hint must be a factor of 2*emb_size
    elif args.game_size == 15:  # for 15x15 board
        max_hint_value = 14
        max_num_per_hint = 5
        args.n_embd = 320  # max_num_per_hint must be a factdor of 2*emb_size
    else:
        raise Exception('not a valid game size')

    dataset = Nonogram_Dataset(
        data_path=f'../data/nonogram/nonograms_{args.game_size}.csv',
        board_dim=args.game_size,
        max_num_per_hint=max_num_per_hint,
        limit=(args.n_train + args.n_test),
        seed=args.seed)
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [args.n_train, args.n_test])
    train_dataset_ulb = None
    print(f'[nonogram-{args.game_size}] use {len(train_dataset)} for training and {len(test_dataset)} for testing')

    #############
    # Construct a GPT model and a trainer
    #############
    # vocab_size is the number of different digits in the input, not used if tok_emb is specified
    mconf = GPTConfig(vocab_size=max_hint_value+1, block_size=args.game_size**2, n_layer=args.n_layer, n_head=args.n_head, n_embd=args.n_embd, 
        num_classes=2, causal_mask=False, losses=args.loss, n_recur=args.n_recur, all_layers=args.all_layers,
        tok_emb=NonogramEmb, max_hint_value=max_hint_value, max_num_per_hint=max_num_per_hint, hyper=args.hyper)
    model = GPT(mconf)

    if args.wandb: wandb.watch(model, log_freq=100)
    if args.heatmap: visualize_adjacency()

    tconf = TrainerConfig(
        max_epochs=args.epochs,
        batch_size=args.batch_size,
        label_size=args.batch_size,
        learning_rate=args.lr,
        lr_decay=args.lr_decay,
        warmup_tokens=1024, # until which point we increase lr from 0 to lr; lr decays after this point
        final_tokens=100 * len(train_dataset), # at what point we reach 10% of lr
        eval_funcs=[testNN], # test without inference trick
        eval_interval=args.eval_interval, # test for every eval_interval number of epochs
        gpu=args.gpu,
        heatmap=args.heatmap,
        prefix=prefix,
        wandb=wandb
    )

    trainer = Trainer(model, train_dataset, train_dataset_ulb, test_dataset, tconf)

    #############
    # Start training
    #############
    trainer.train()
    result = trainer.result
    print('Total and single accuracy are the board and cell accuracy respectively.')
    print_result(result)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Training
    parser.add_argument('--epochs', type=int, default=200, help='Number of epochs.')
    parser.add_argument('--eval_interval', type=int, default=1, help='Compute accuracy for how many number of epochs.')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--label_size', type=int, default=16, help='The number of labeled training data in a batch')
    parser.add_argument('--lr', type=float, default=6e-4, help='Learning rate')
    parser.add_argument('--lr_decay', default=False, action='store_true', help='use lr_decay defined in minGPT')
    # Model and loss
    parser.add_argument('--n_layer', type=int, default=1, help='Number of sequential self-attention blocks.')
    parser.add_argument('--n_recur', type=int, default=16, help='Number of recurrency of all self-attention blocks.')
    parser.add_argument('--n_head', type=int, default=4, help='Number of heads in each self-attention block.')
    parser.add_argument('--n_embd', type=int, default=128, help='Vector embedding size.')
    parser.add_argument('--loss', default=[], nargs='+', help='specify constraint losses in \{\}')
    parser.add_argument('--all_layers', default=True, action='store_true', help='apply losses to all self-attention layers')    
    parser.add_argument('--hyper', default=[1, 0.1], nargs='+', type=float, help='Hyper parameters: Weights of [L_sudoku, L_attention]')

    # Data
    parser.add_argument('--game_size', type=int, default=7, help='Size of grid in \{7, 15\}')
    parser.add_argument('--n_train', type=int, default=9000, help='The number of data for train')
    parser.add_argument('--n_test', type=int, default=1000, help='The number of data for test')
    # Other
    parser.add_argument('--seed', type=int, default=0, help='Random seed for reproductivity.')
    parser.add_argument('--gpu', type=int, default=-1, help='gpu index; -1 means using all GPUs or using CPU if no GPU is available')
    parser.add_argument('--debug', default=False, action='store_true', help='debug mode')
    parser.add_argument('--wandb', default=False, action='store_true', help='save all logs on wandb')
    parser.add_argument('--heatmap', default=False, action='store_true', help='save all heatmaps in trainer.result')
    parser.add_argument('--comment', type=str, default='', help='Comment of the experiment')
    args = parser.parse_args()

    # we do not log onto wandb in debug mode
    if args.debug: args.wandb = False
    main(args)