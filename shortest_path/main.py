# set up logging
import logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
import argparse
import wandb
from dataset import GridData,shortest_path_Dataset
from mingpt_sp.model import GPT, GPTConfig
from mingpt_sp.utils import set_seed
from mingpt_sp.trainer import Trainer, TrainerConfig


def main(args):
    print(args)
    # Seed everything for reproductivity
    set_seed(args.seed)

    # generate the name and setting of this experiment
    setting = 'ste' if 'path' in args.loss else 'base'
    if args.grid_size == 4:
        batch_size = 128
        num_data = '1610'
        data_limit = None # one can also limit the dataset size
        split = (10, 6, 2, 2) # (Ntotal, Ntrain, Nvalidation, Ntest)
    elif args.grid_size == 12:
        batch_size = 32
        num_data = '22k'
        data_limit = None # one can also limit the dataset size
        split = (22, 20, 1, 1) # (Ntotal, Ntrain, Nvalidation, Ntest)

    data_limit_str = f'({data_limit})' if data_limit else ''
    name = f'[n={args.grid_size}][{num_data}{data_limit_str}]{setting}'
    ckpt_path = f'shortest_path_{name[1:]}.pt'
    data_path = f'data/{args.grid_size}x{args.grid_size}_{num_data}.data'
    num_nodes = args.grid_size**2
    num_edges = 2 * (args.grid_size - 1) * args.grid_size

    if args.wandb:
        wandb.init(project='rt-sp')
        wandb.run.name = name
    else:
        wandb.init(mode="disabled")  # for testing without wandb

    # initialize dataset
    gd = GridData(data_path,num_nodes, num_edges, split, data_limit)
    train_dataset=shortest_path_Dataset(
        gd.train_data, gd.train_labels, gd.train_labels_ordered,
        gd.train_labels_all, gd.train_removed_edges, args.grid_size)
    test_dataset=shortest_path_Dataset(
        gd.test_data, gd.test_labels, gd.test_labels_ordered,
        gd.test_labels_all, gd.test_removed_edges, args.grid_size)

    # initialize the model
    mconf = GPTConfig(2, num_nodes+num_edges,
        n_layer=args.n_layer, n_head=args.n_head, n_embd=args.n_embd,
        stack_divisor=17, plot_attentions=False, grid_size=args.grid_size, losses=args.loss)
    model = GPT(mconf)

    # initialize a trainer instance and kick off training
    tconf = TrainerConfig(
        max_epochs = args.epochs,
        batch_size = batch_size,
        learning_rate = args.lr,
        lr_decay = args.lr_decay,
        warmup_tokens = 1024,
        final_tokens = 50 * len(train_dataset) * (2 + 1),
        num_workers = 0,
        ckpt_path = ckpt_path,
        num_nodes = num_nodes,
        num_edges = num_edges,
        exam_batch_size = batch_size,
        gpu = args.gpu
    )

    trainer = Trainer(model, train_dataset, test_dataset, tconf)
    if args.resume:
        trainer.load_checkpoint()
    eval_list = trainer.train()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--grid_size', type=int, default=4, help='Size N of the NxN grid.')
    # Training
    parser.add_argument('--epochs', type=int, default=2000, help='Number of epochs.')
    parser.add_argument('--lr', type=float, default=6e-4, help='Learning rate')
    parser.add_argument('--lr_decay', default=False, action='store_true', help='use lr_decay defined in minGPT')
    # Model and loss
    parser.add_argument('--n_layer', type=int, default=1, help='Number of sequential self-attention blocks.')
    # parser.add_argument('--n_recur', type=int, default=32, help='Number of recurrency of all self-attention blocks.')
    parser.add_argument('--n_head', type=int, default=4, help='Number of heads in each self-attention block.')
    parser.add_argument('--n_embd', type=int, default=128, help='Vector embedding size.')
    parser.add_argument('--loss', default=[], nargs='+', help='specify constraint losses in \{path\}')
    parser.add_argument('--resume', default=False, action='store_true', help='load the last trained model of the same setting')
    # Other
    parser.add_argument('--seed', type=int, default=0, help='Random seed for reproductivity.')
    parser.add_argument('--gpu', type=int, default=-1, help='gpu index; -1 means using all GPUs or using CPU if no GPU is available')
    parser.add_argument('--wandb', default=False, action='store_true', help='save all logs on wandb')
    parser.add_argument('--comment', type=str, default='', help='Comment of the experiment')
    args = parser.parse_args()

    main(args)