"""
Please install clingo before running this program
    `conda install -c conda-forge clingo=5.6`
How to run
    `python test.py`
"""

import sys
sys.path.append('neurasp')

import torch

from dataset import GridData, shortest_path_Dataset
from neurasp import NeurASP

from mingpt_sp.model import GPT, GPTConfig
from mingpt_sp.utils import set_seed
# make deterministic
set_seed(0)

##########################################################
# config of the experiment

setting = ['base', 'ste'][0]
gpu = 3 # GPU index
grid_size = 4
batch_size = 128
num_data = '1610'
data_limit = None
split = (10, 6, 2, 2) # (Ntotal, Ntrain, Nvalidation, Ntest)

##########################################################

data_limit_str = f'({data_limit})' if data_limit else ''
name = f'D[n={grid_size}][{num_data}{data_limit_str}]{setting}'
ckpt_path = f'shortest_path_{name[1:]}.pt'
data_path = f'data/{grid_size}x{grid_size}_{num_data}.data'
losses = name.split(']')[-1]
num_nodes = grid_size**2
num_edges = 2*(grid_size-1)*grid_size

gd = GridData(data_path, num_nodes, num_edges, split, data_limit)
test_dataset = shortest_path_Dataset(
    gd.test_data,
    gd.test_labels,
    gd.test_labels_ordered,
    gd.test_labels_all,
    gd.test_removed_edges,
    grid_size)

dataListTest, obsListTest = [], []
for inp, label, removed, idx in test_dataset:
    obs = ':- mistake.\n'
    startEnd = [i for i, x in enumerate(inp[:24]) if x == 1]
    for edge in removed:
        obs += 'removed({}).\n'.format(edge)
    for node in startEnd:
        obs += 'sp(external, {}).\n'.format(node)
    obsListTest.append(obs)
    dataListTest.append({'g': inp.unsqueeze(0)})

# initialize a baby GPT model
mconf = GPTConfig(2, num_nodes+num_edges, n_layer=1, n_head=4, n_embd=128, stack_divisor=17, plot_attentions=False, grid_size=grid_size, losses=losses)
m = GPT(mconf)


######################################
# The NeurASP program can be written in the scope of ''' Rules '''
# It can also be written in a file
######################################

nnRule = '''
nn(sp(24, g), [true, false]).
'''

aspRule = '''
sp(X) :- sp(X,g,true).

sp(0,1) :- sp(0).
sp(1,2) :- sp(1).
sp(2,3) :- sp(2).
sp(4,5) :- sp(3).
sp(5,6) :- sp(4).
sp(6,7) :- sp(5).
sp(8,9) :- sp(6).
sp(9,10) :- sp(7).
sp(10,11) :- sp(8).
sp(12,13) :- sp(9).
sp(13,14) :- sp(10).
sp(14,15) :- sp(11).
sp(0,4) :- sp(12).
sp(4,8) :- sp(13).
sp(8,12) :- sp(14).
sp(1,5) :- sp(15).
sp(5,9) :- sp(16).
sp(9,13) :- sp(17).
sp(2,6) :- sp(18).
sp(6,10) :- sp(19).
sp(10,14) :- sp(20).
sp(3,7) :- sp(21).
sp(7,11) :- sp(22).
sp(11,15) :- sp(23).

sp(X,Y) :- sp(Y,X).
'''

constraints = {}

constraints['nr'] = '''
% [nr] 1. No removed edges should be predicted
mistake :- sp(X), removed(X).
'''

constraints['p'] = '''
% [p] 2. Prediction must form simple path(s)
% that is: the degree of nodes should be either 0 or 2
mistake :- X=0..15, #count{Y: sp(X,Y)} = 1.
mistake :- X=0..15, #count{Y: sp(X,Y)} >= 3.
'''

constraints['r'] = '''
% [r] 3. Every 2 nodes in the prediction must be reachable
reachable(X, Y) :- sp(X, Y).
reachable(X, Y) :- reachable(X, Z), sp(Z, Y).
mistake :- sp(X, _), sp(Y, _), not reachable(X, Y).
'''

constraints['o'] = '''
% [o] 4. Predicted path should contain least edges
:~ sp(X). [1, X]
'''

########
# Set up the list of constraint combinations for testing accuracy
########

combinations = [['nr'], ['p'], ['r'], ['nr', 'p'], ['nr', 'r'], ['p', 'r'], ['nr', 'p', 'r'], ['nr', 'p', 'r', 'o']]
combinations = [aspRule + ''.join([constraints[c] for c in combination]) for combination in combinations]

########
# Define nnMapping and initialze NeurASP object
########

nnMapping = {'sp': m}
NeurASPobj = NeurASP(nnRule, nnMapping, optimizers=None)

########
# Load pretrained model
########

m.load_state_dict(torch.load(ckpt_path, map_location='cpu'))

########
# Start testing
########
NeurASPobj.testConstraint(dataList=dataListTest, obsList=obsListTest, mvppList=combinations)
