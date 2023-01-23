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
import wandb

logger = logging.getLogger(__name__)


class TrainerConfig:
    # optimization parameters
    max_epochs = 10
    batch_size = 64
    learning_rate = 3e-4
    betas = (0.9, 0.95)
    grad_norm_clip = 1.0
    weight_decay = 0.1  # only applied on matmul weights
    # learning rate decay params: linear warmup followed by cosine decay to 10% of original
    lr_decay = False
    warmup_tokens = 375e6  # these two numbers come from the GPT-3 paper, but may not be good defaults elsewhere
    final_tokens = 260e9  # (at what point we reach 10% of original LR)
    # checkpoint settings
    ckpt_path = None
    num_workers = 0  # for DataLoader

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


class Trainer:
    def __init__(self, model, train_dataset, test_dataset, config):
        self.model = model
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.config = config

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

    def load_checkpoint(self):
        if self.config.ckpt_path is not None:
            ckpt_model = (
                self.model.module if hasattr(self.model, "module") else self.model
            )
            logger.info("loading %s", self.config.ckpt_path)
            ck = torch.load(self.config.ckpt_path)
            ckpt_model.load_state_dict(ck)

    def train(self):
        model, config = self.model, self.config
        raw_model = model.module if hasattr(self.model, "module") else model
        optimizer = raw_model.configure_optimizers(config)

        def run_epoch(split):
            is_train = split == "train"
            model.train(is_train)
            data = self.train_dataset if is_train else self.test_dataset
            loader = DataLoader(
                data,
                shuffle=True,
                pin_memory=True,
                batch_size=config.batch_size,
                num_workers=config.num_workers,
            )
            losses = []
            pbar = (
                tqdm(enumerate(loader), total=len(loader))
                if is_train
                else enumerate(loader)
            )
            for it, (x, y, removed, sample_inds) in pbar:
                # place data on the correct device
                x = x.to(self.device)
                y = y.to(self.device); 
                # forward the model
                with torch.set_grad_enabled(is_train):
                    logits, loss = model(x, y)
                    loss = (
                        loss.mean()
                    )  # collapse all losses if they are scattered on multiple gpus
                    losses.append(loss.item())
                if is_train:

                    # backprop and update the parameters
                    model.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), config.grad_norm_clip
                    )
                    optimizer.step()

                    # decay the learning rate based on our progress
                    if config.lr_decay:
                        self.tokens += (
                            y >= 0
                        ).sum()  # number of tokens processed this step (i.e. label is not -100)
                        if self.tokens < config.warmup_tokens:
                            # linear warmup
                            lr_mult = float(self.tokens) / float(
                                max(1, config.warmup_tokens)
                            )
                        else:
                            # cosine learning rate decay
                            progress = float(
                                self.tokens - config.warmup_tokens
                            ) / float(
                                max(1, config.final_tokens - config.warmup_tokens)
                            )
                            lr_mult = max(
                                0.1, 0.5 * (1.0 + math.cos(math.pi * progress))
                            )
                        lr = config.learning_rate * lr_mult
                        for param_group in optimizer.param_groups:
                            param_group["lr"] = lr
                    else:
                        lr = config.learning_rate
                    # report progress
                    pbar.set_description(
                        f"epoch {epoch+1} iter {it}: train loss {loss.item():.5f}. lr {lr:e}"
                    )
            if not is_train:
                test_loss = float(np.mean(losses))
                logger.info("test loss: %f", test_loss)
                return test_loss
            else:
                return float(np.mean(losses))

        best_loss = float("inf")
        self.tokens = 0  # counter used for learning rate decay
        eval_list = []
        best_wb = 0.0

        for epoch in range(1, config.max_epochs + 1):
            train_loss = run_epoch("train")
            if self.test_dataset is not None:
                test_loss = run_epoch("test")

            if epoch % 1 == 0:  # or (epoch < 50 and epoch % 5 == 0):
                (
                    possible_solution_correct_nt,
                    possible_solution_total_nt,
                ) = self.give_exam(
                    self.test_dataset,
                    batch_size=self.config.exam_batch_size,
                    max_batches=999,
                    steps=32
                )

                possible_solution_acc_no_trick = possible_solution_correct_nt/possible_solution_total_nt
                print('TEST ------------')
                print('whole-path_any acc:')
                print(possible_solution_acc_no_trick)

                (
                    possible_solution_correct_nt_32_train,
                    possible_solution_total_nt_32_train,
                ) = self.give_exam(
                    self.train_dataset,
                    batch_size=self.config.exam_batch_size,
                    max_batches=20,
                    steps=32
                )

                possible_solution_acc_no_trick_32_train = possible_solution_correct_nt_32_train/possible_solution_total_nt_32_train
                print('TRAIN ------------')
                print('whole-path_any acc:')
                print(possible_solution_acc_no_trick_32_train)
            wandb.log(
                {
                    "wp_any_acc_32": possible_solution_acc_no_trick,
                    "wp_any_acc_32_train": possible_solution_acc_no_trick_32_train,
                    'train loss': train_loss,
                    'test_loss': test_loss
                }
            )

            # supports early stopping based on the test loss, or just save always if no test set is provided
            if self.config.ckpt_path is not None and possible_solution_acc_no_trick > best_wb:
                best_wb = possible_solution_acc_no_trick
                print(f"saving best model with possible_solution_acc_no_trick {best_wb}")
                best_loss = test_loss
                self.save_checkpoint()
        return eval_list

    def inference_trick(self, x):
        """ takes a single example, cannot do batches since there 
        are a different number of givens per example"""
        assert len(x.shape) == 2

        model = self.model
        B = x.shape[0]
        first_pass = True
        while (x == 0).sum() >= 1:

            current_mask = x.float().masked_fill(x != 0, float("-inf"))
            logits, loss = model(x)
            if first_pass:
                single_pass_pred = logits[0].argmax(-1).unsqueeze(0)
                first_pass = False
            logits = logits + current_mask.unsqueeze(-1)
            logits_flattened = logits.view(B, -1)
            argmaxes = logits_flattened.argmax(-1)
            pos = argmaxes // 9
            rows, cols = pos // 9, pos % 9
            values = argmaxes % 9 + 1  # 1-indexed

            x[torch.arange(B), pos] = values  # update x with most probable predictions
        return x - 1, single_pass_pred  # return to 0-index

    def no_inference_trick(self, x, steps):
        """ takes a single example, cannot do batches since there 
        are a different number of givens per example"""
        model = self.model
        B = x.shape[0]
        logits, loss = model(x, steps=steps)
        single_pass_pred = logits.argmax(-1)
        return x - 1, single_pass_pred  # return to 0-index

    def give_exam(self, dataset, batch_size=32, max_batches=-1, steps=16):
        loader = DataLoader(dataset, batch_size=batch_size)
        self.model.eval()
        possible_solution_correct = 0
        possible_solution_total = 0 
        for b, (x, y, removed, sample_inds) in enumerate(tqdm(loader)):
            x = x.to(self.device)
            y = y.to(self.device)
            _, pred_nt = self.no_inference_trick(x, steps)
            sol_inds = loader.dataset.sol_inds_all[sample_inds]
            (
                possible_solution_count_batch,
                possible_solution_count_total
            ) = self.calc_acc(pred_nt, y, sol_inds)
            possible_solution_correct+= possible_solution_count_batch
            possible_solution_total += possible_solution_count_total
            if max_batches >= 0 and b + 1 >= max_batches:
                break
        return (
            possible_solution_correct,
            possible_solution_total,
        )

    def calc_acc(
        self, outputs, y, sol_inds
    ):
        # overal grid accuracy 
        num_nodes = self.config.num_nodes
        edge_predictions = outputs[:,num_nodes:]
        
        # check for all possible solutions w/ masking bad edges
        possible_solution_count=0 # masked and check any 
        for idx,edge_prediction in enumerate(edge_predictions.cpu()):
            p_solutions = sol_inds[idx]
            for p_solution in p_solutions:
                if all(edge_prediction.numpy()==p_solution):
                    possible_solution_count+=1
                    break
        return possible_solution_count,y.shape[0]
