import torch

####################################################################################
# Definitions of General Functions (under binary logic)
####################################################################################

def bp(x):
    """ binarization function bp(x) = 1 if x >= 0.5; bp(x) = 0 if x < 0.5

    @param x: a real number in [0,1] denoting a probability
    """
    return torch.clamp(torch.sign(x-0.5) + 1, max=1)

def binarize(x):
    """ binarization function binarize(x) = 1 if x >= 0; binarize(x) = -1 if x < 0

    Remark:
        This function is indeed the b(x) function in the paper.
        We use binarize(x) instead of b(x) here to differentiate function B(x) later.

    @param x: a real number of any value
    """
    return torch.clamp(torch.sign(x) + 1, max=1)

def sSTE(grad_output, x=None):
    """
    @param grad_output: a tensor denoting the gradient of loss w.r.t. Bs(x)
    @param x: the value of input x
    """
    return grad_output * (torch.le(x, 1) * torch.ge(x, -1)).float() # clipped Relu with range [-1,1]

# B(x) denotes bp(x) with iSTE
class Disc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return bp(x)
    @staticmethod
    def backward(ctx, grad_output):
        return grad_output
B = Disc.apply

# Bi(x) denotes binarize(x) with iSTE
class DiscBi(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return binarize(x)
    @staticmethod
    def backward(ctx, grad_output):
        return grad_output
Bi = DiscBi.apply

# Bs(x) denotes binarize(x) with sSTE
class DiscBs(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return binarize(x)
    @staticmethod
    def backward(ctx, grad_output):
        x, = ctx.saved_tensors
        grad_input = sSTE(grad_output, x)
        return grad_input
Bs = DiscBs.apply

def one(x):
    return (x == 1).float()

def minusOne(x):
    return (x == -1).float()

def zero(x):
    return (x == 0).float()

def noneZero(x):
    return (x != 0).float()

####################################################################################
# Definitions of regularizers
####################################################################################


##########
# Bound
# we limit the size of NN output values
##########

def reg_bound(output):
    return output.pow(2).mean()

##########
# Matrix Form: Condition * Literal
##########

# the uniqueness constraint on each row of x
def reg_uc(x):
    """
    @param x: a tensor of shape (m, n), denoting the probabilities
              of n possible values of m random variables
    """
    A_range = 1 - torch.eye(x.shape[-1], device=x.device)
    # condition, literal = noneZero(torch.mm(bp(x), A_range)), B(x)
    condition, literal = torch.mm(bp(x), A_range), B(x)
    return (condition * literal).mean()

# the existence constraint on each row of x
def reg_ec(x):
    """
    @param x: a tensor of shape (m, n), denoting the probabilities
              of n possible values of m random variables
    """
    A_range = 1 - torch.eye(x.shape[-1], device=x.device)
    condition, literal = zero(torch.mm(bp(x), A_range)), 1 - B(x)
    return (condition * literal).mean()

# the uniqueness and existence constraint on some values in y
def reg_uec(x):
    """
    @param x: a tensor of shape (m, n), denoting the probabilities
              of n possible values of m random variables
    """
    return reg_uc(x) + reg_ec(x)

##########
# Cardinality Form
##########

def reg_cardinality(x, num):
    """
    @param x: a tensor of shape (m, n), denoting the probabilities
              of n possible values of m random variables
    @param num: an integer denoting the expected number 1s in each row of x
    """
    return (B(x).sum(dim=-1) - num).pow(2).mean()

# define A_adj in {0,1}^{81 * 81} as the adjacency matrix for all cells
A_adj = torch.zeros([81, 81], dtype=torch.int32)
for i in range(81):
    for j in range(81):
        ix, iy = i // 9, i % 9
        jx, jy = j // 9, j % 9
        ic = 3 * (ix // 3) + iy // 3
        jc = 3 * (jx // 3) + jy // 3
        if i == j or ix == jx or iy == jy or ic == jc:
            A_adj[i,j] = 1

def reg_att_sudoku_c1(x):
    """
    @param x: a tensor of shape (n_layer * n_recur * batch_size * num_heads, 81, 81)
              denoting the probabilities in the attention matrices
    """
    test = (x * A_adj.unsqueeze(0).to(x.device)).sum(dim=-1,keepdim=True)
    return reg_cardinality(test, 1)
