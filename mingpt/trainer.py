"""
Simple training loop; Boilerplate that could apply to any arbitrary neural network,
so nothing in this file really has anything to do with GPT specifically.
"""

import math
import logging

from tqdm import tqdm
import numpy as np

import torch
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data.dataloader import DataLoader

from helper import visualize_token2token_scores, visualize_cell_attention

logger = logging.getLogger(__name__)

class TrainerConfig:
    # optimization parameters
    max_epochs = 10
    batch_size = 64
    learning_rate = 3e-4
    betas = (0.9, 0.95)
    grad_norm_clip = 1.0
    weight_decay = 0.1 # only applied on matmul weights
    # learning rate decay params: linear warmup followed by cosine decay to 10% of original
    lr_decay = False
    warmup_tokens = 375e6 # these two numbers come from the GPT-3 paper, but may not be good defaults elsewhere
    final_tokens = 260e9 # (at what point we reach 10% of original LR)
    # checkpoint settings
    ckpt_path = None
    num_workers = 0 # for DataLoader

    def __init__(self, **kwargs):
        for k,v in kwargs.items():
            setattr(self, k, v)

class Trainer:

    def __init__(self, model, train_dataset, train_dataset_ulb, test_dataset, config):
        self.model = model
        self.train_dataset = train_dataset
        self.train_dataset_ulb = train_dataset_ulb
        self.test_dataset = test_dataset
        self.config = config
        self.test_dataloader = DataLoader(test_dataset, batch_size=config.batch_size)
        train_dataset_1000 = torch.utils.data.Subset(
            train_dataset,
            list(range(min(1000, len(train_dataset))))
        )
        # dataloaders for evaluation
        self.train_dataloader = DataLoader(train_dataset_1000, batch_size=config.batch_size)
        self.test_dataloader = DataLoader(self.test_dataset, batch_size=config.batch_size)
        # dataloaders for training
        if config.label_size < config.batch_size:
            self.loader_lb = DataLoader(self.train_dataset, shuffle=True, pin_memory=True,
                                batch_size=config.label_size,
                                num_workers=config.num_workers)
            self.loader_ulb = DataLoader(self.train_dataset_ulb, shuffle=True, pin_memory=True,
                                batch_size=config.batch_size - config.label_size,
                                num_workers=config.num_workers)
        else:
            self.loader_lb = DataLoader(self.train_dataset, shuffle=True, pin_memory=True,
                                    batch_size=config.batch_size,
                                    num_workers=config.num_workers)

        self.eval_funcs = config.eval_funcs
        self.eval_interval = config.eval_interval
        self.result = {}
        self.heatmap = config.heatmap
        self.prefix = config.prefix
        self.wandb = config.wandb
        # # we save the attention for the 1st data in the trainloader for every epoch
        # self.atts = [] # a list of atts of shape (num_layers, num_heads, 81, 81)
        for eval_func in self.eval_funcs:
            self.result[eval_func.__name__] = []

        if config.gpu >= 0 and torch.cuda.is_available():
            print(f'Using GPU {config.gpu}')
            self.device = torch.device('cuda', index=config.gpu)
            self.model.to(self.device)
        else:
            # take over whatever gpus are on the system
            self.device = 'cpu'
            if torch.cuda.is_available():
                print('Using all GPUs')
                self.device = torch.cuda.current_device()
                self.model = torch.nn.DataParallel(self.model).to(self.device)
            else:
                print('Using CPU')

    def save_checkpoint(self):
        # DataParallel wrappers keep raw model object in .module attribute
        raw_model = self.model.module if hasattr(self.model, "module") else self.model
        logger.info("saving %s", self.config.ckpt_path)
        torch.save(raw_model.state_dict(), self.config.ckpt_path)

    def train(self):
        model, config = self.model, self.config
        raw_model = model.module if hasattr(self.model, "module") else model
        optimizer = raw_model.configure_optimizers(config)

        def run_epoch(split):
            is_train = split == 'train'
            model.train(is_train)
            losses = []

            # for semi-supervised learning
            if is_train and config.label_size < config.batch_size:
                pbar = tqdm(enumerate(self.loader_lb), total=len(self.loader_lb))
                loader_ulb_iterator = iter(self.loader_ulb)
                for it, data_lb in pbar:
                    data_ulb = next(loader_ulb_iterator)
                    x, y = data_lb # (label_size, 81, 28, 28), (label_size, 81)
                    x_ulb = data_ulb[0] # (batch_size - label_size, 81, 28, 28)
                    # place data on the correct device
                    x = x.to(self.device)
                    y = y.to(self.device)
                    x_ulb = x_ulb.to(self.device)

                    # forward the model
                    with torch.set_grad_enabled(is_train):
                        logits, loss, atts = model(x, y, x_ulb)
                        loss = loss.mean() # collapse all losses if they are scattered on multiple gpus
                        losses.append(loss.item())

                    # backprop and update the parameters
                    model.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_norm_clip)
                    optimizer.step()

                    # decay the learning rate based on our progress
                    if config.lr_decay:
                        self.tokens += (y >= 0).sum() # number of tokens processed this step (i.e. label is not -100)
                        if self.tokens < config.warmup_tokens:
                            # linear warmup
                            lr_mult = float(self.tokens) / float(max(1, config.warmup_tokens))
                        else:
                            # cosine learning rate decay
                            progress = float(self.tokens - config.warmup_tokens) / float(max(1, config.final_tokens - config.warmup_tokens))
                            lr_mult = max(0.1, 0.5 * (1.0 + math.cos(math.pi * progress)))
                        lr = config.learning_rate * lr_mult
                        for param_group in optimizer.param_groups:
                            param_group['lr'] = lr
                    else:
                        lr = config.learning_rate

                    # report progress
                    pbar.set_description(f"epoch {epoch+1} iter {it}: train loss {loss.item():.5f}. lr {lr:e}")
            else:
                pbar = tqdm(enumerate(self.loader_lb), total=len(self.loader_lb)) if is_train else enumerate(self.test_dataloader)
                for it, (x, y) in pbar:

                    # place data on the correct device
                    x = x.to(self.device)
                    y = y.to(self.device)

                    # forward the model
                    with torch.set_grad_enabled(is_train):
                        logits, loss, atts = model(x, y)
                        loss = loss.mean() # collapse all losses if they are scattered on multiple gpus
                        losses.append(loss.item())

                    if is_train:

                        # backprop and update the parameters
                        model.zero_grad()
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_norm_clip)
                        optimizer.step()

                        # decay the learning rate based on our progress
                        if config.lr_decay:
                            self.tokens += (y >= 0).sum() # number of tokens processed this step (i.e. label is not -100)
                            if self.tokens < config.warmup_tokens:
                                # linear warmup
                                lr_mult = float(self.tokens) / float(max(1, config.warmup_tokens))
                            else:
                                # cosine learning rate decay
                                progress = float(self.tokens - config.warmup_tokens) / float(max(1, config.final_tokens - config.warmup_tokens))
                                lr_mult = max(0.1, 0.5 * (1.0 + math.cos(math.pi * progress)))
                            lr = config.learning_rate * lr_mult
                            for param_group in optimizer.param_groups:
                                param_group['lr'] = lr
                        else:
                            lr = config.learning_rate

                        # report progress
                        pbar.set_description(f"epoch {epoch+1} iter {it}: train loss {loss.item():.5f}. lr {lr:e}")

            return float(np.mean(losses))

        best_loss = float('inf')
        self.tokens = 0 # counter used for learning rate decay
        for epoch in range(config.max_epochs):

            train_loss = run_epoch('train')
            if self.wandb:
                self.wandb.log({f'loss/train_loss': train_loss}, step=epoch+1)
            if self.test_dataset is not None:
                test_loss = run_epoch('test')
                if self.wandb:
                    self.wandb.log({f'loss/test_loss': test_loss}, step=epoch+1)

            # supports early stopping based on the test loss, or just save always if no test set is provided
            good_model = self.test_dataset is None or test_loss < best_loss
            if self.config.ckpt_path is not None and good_model:
                best_loss = test_loss
                self.save_checkpoint()
            
            if (epoch + 1) % self.eval_interval == 0:
                for eval_func in self.eval_funcs:
                    # evaluate on testing data
                    result = eval_func(model, self.test_dataloader, self.heatmap)
                    self.result[eval_func.__name__].append(result)
                    if self.wandb:
                        prev_n_recur = model.n_recur
                        correct, total, singleCorrect, singleTotal, _ = result
                        self.wandb.log({f'test_acc/{eval_func.__name__}_board': 100*correct/total}, step=epoch+1)
                        self.wandb.log({f'test_acc/{eval_func.__name__}_cell': 100*singleCorrect/singleTotal}, step=epoch+1)
                        self.wandb.log({f'test_acc/{eval_func.__name__}_board[{model.n_recur}]': 100*correct/total}, step=epoch+1)
                        self.wandb.log({f'test_acc/{eval_func.__name__}_cell[{model.n_recur}]': 100*singleCorrect/singleTotal}, step=epoch+1)
                        # evaluate the accuracy with fixed number of recurrence during inference
                        for n_recur in (16, 32, 64):
                            # if n_recur > prev_n_recur:
                            if n_recur == prev_n_recur * 2:
                                model.n_recur = n_recur
                                correct, total, singleCorrect, singleTotal, _ = eval_func(model, self.test_dataloader, self.heatmap)
                                self.wandb.log({f'test_acc/{eval_func.__name__}_board[{model.n_recur}]': 100*correct/total}, step=epoch+1)
                                self.wandb.log({f'test_acc/{eval_func.__name__}_cell[{model.n_recur}]': 100*singleCorrect/singleTotal}, step=epoch+1)
                        model.n_recur = prev_n_recur
                    
                        # evaluate on training data
                        correct, total, singleCorrect, singleTotal, _ = eval_func(model, self.train_dataloader, self.heatmap)
                        self.wandb.log({f'train_acc/{eval_func.__name__}_board': 100*correct/total}, step=epoch+1)
                        self.wandb.log({f'train_acc/{eval_func.__name__}_cell': 100*singleCorrect/singleTotal}, step=epoch+1)
                        self.wandb.log({f'train_acc/{eval_func.__name__}_board[{model.n_recur}]': 100*correct/total}, step=epoch+1)
                        self.wandb.log({f'train_acc/{eval_func.__name__}_cell[{model.n_recur}]': 100*singleCorrect/singleTotal}, step=epoch+1)
                        # evaluate the accuracy with fixed number of recurrence during inference
                        for n_recur in (16, 32, 64):
                            # if n_recur > prev_n_recur:
                            if n_recur == prev_n_recur * 2:
                                model.n_recur = n_recur
                                correct, total, singleCorrect, singleTotal, _ = eval_func(model, self.train_dataloader, self.heatmap)
                                self.wandb.log({f'train_acc/{eval_func.__name__}_board[{model.n_recur}]': 100*correct/total}, step=epoch+1)
                                self.wandb.log({f'train_acc/{eval_func.__name__}_cell[{model.n_recur}]': 100*singleCorrect/singleTotal}, step=epoch+1)
                        model.n_recur = prev_n_recur

                # store the visualization of attention matrices
                if self.heatmap and (epoch + 1) % 100 == 0 and 'testNN' in self.result:
                    _,_,_,_,att = self.result['testNN'][-1]
                    all_tokens = range(81)
                    cell_tokens = range(9)
                    indices = [[0,0], [4,4]]

                    for l in range(att.shape[0]):
                        visualize_token2token_scores(att[l].numpy(), all_tokens, filename=f'heatmap/{self.prefix}epoch{epoch+1}_layer{l+1}')
                    visualize_cell_attention(att[0].numpy(), indices, cell_tokens, filename=f'heatmap/{self.prefix}epoch{epoch+1}_layer1_cell')
        
                    # visualize positional embedding
                    pos_emb = model.pos_emb.detach().clone().squeeze().cpu() # (81, 128)
                    pos_emb /= pos_emb.norm(dim=1, keepdim=True) # (81, 128)
                    sim = torch.matmul(pos_emb, pos_emb.T).unsqueeze(0).numpy() # (81, 81)
                    all_tokens = range(81)
                    visualize_token2token_scores(sim, range(81), x_label_name='pos emb', filename=f'heatmap/pos_emb_{self.prefix}epoch{epoch+1}')

