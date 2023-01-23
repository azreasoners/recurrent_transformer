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
# Cardinality Form
##########

def reg_cardinality(x, num):
    """
    @param x: a tensor of shape ([batch], m, n), denoting the probabilities
              of n possible values of m random variables
    @param num: an integer denoting the expected number 1s in each row of x
    """
    return (B(x).sum(dim=-1) - num).pow(2).mean()

def gen_node2edges(n):
    numNodes = n * n
    numEdges = 2 * n * (n-1)
    node2edges = torch.zeros(numNodes, numEdges)
    # define the mapping from node <x,y> coordinates to node index
    idx = 0
    nodeIdx = {}
    for i in range(n):
        for j in range(n):
            nodeIdx[(i,j)] = idx
            idx += 1
    # define the mapping from edge index to node indices
    idx = 0
    for i1 in range(n):
        for j1 in range(n - 1):
            i2, j2 = i1, j1 + 1
            node1, node2 = nodeIdx[(i1,j1)], nodeIdx[(i2,j2)]
            node2edges[[node1,node2], idx] = 1
            idx += 1
    for j1 in range(n):
        for i1 in range(n - 1):
            i2, j2 = i1 + 1, j1
            node1, node2 = nodeIdx[(i1,j1)], nodeIdx[(i2,j2)]
            node2edges[[node1,node2], idx] = 1
            idx += 1
    return node2edges

def reg_sp(output, inp, target, node2edges):
    """
    @param output: output tensor (of probs) of shape (batchSize, numNodes+numEdges, 2)
        where the last numEdges vectors are for edge predictions
    @param inp: input tensor of shape (batchSize, numNodes+numEdges)
    @param target: label tensor of shape (batchSize, numNodes+numEdges)
    """
    numNodes, numEdges = node2edges.shape
    pred = B(output[:,-numEdges:,1].view(-1, numEdges)) # (batchSize, numEdges)
    countEdges = torch.mm(pred, node2edges.t()) # (batchSize, numNodes)
    endNodes = noneZero(inp[:, :numNodes]) # (batchSize, numNodes)
    countEdges = countEdges + endNodes # (batchSize, numNodes)

    target_edges = target[:,-numEdges:].unsqueeze(1) # (batchSize, 1, numEdges)
    target_nodes = target_edges * node2edges.unsqueeze(0) # (batchSize, numNodes, numEdges)
    target_nodes = noneZero(target_nodes.sum(-1))
    return (target_nodes * (countEdges - 2).pow(2)).mean()
