import sys
import torch
from tqdm import tqdm

# the function to test a neural network model using a test data loader
def testNN(model, testLoader, check_att=False):
    """
    Args:
        model: a Pytorch model
        testLoader: a PyTorch dataLoader object, including (input, output) pairs for model
    """
    device = next(model.parameters()).device
    # set up testing mode
    model.eval()
    # check if total prediction is correct
    correct = total = 0
    # check if each single prediction is correct
    singleCorrect = singleTotal = 0
    # save the attenetion for the 1st data instance
    atts = None
    with torch.no_grad():
        for data, target in testLoader:
            output = model(data.to(device))
            if isinstance(output, tuple):
                if check_att and atts is None:
                    atts = output[-1]
                output = output[0]
            if target.shape == output.shape[:-1]:
                pred = output.argmax(dim=-1) # get the index of the max value
            elif target.shape == output.shape:
                pred = (output >= 0).int()
            else:
                print(f'Error: none considered case for output with shape {output.shape} v.s. label with shape {target.shape}')
                sys.exit()
            target = target.to(device).view_as(pred)
            correctionMatrix = torch.logical_or(target.int() == pred.int(), target < 0).view(target.shape[0], -1)
            correct += correctionMatrix.all(1).sum().item()
            total += target.shape[0]
            singleCorrect += correctionMatrix[target >= 0].sum().item()
            singleTotal += (target >= 0).sum().item()
    return correct, total, singleCorrect, singleTotal, atts

def inference_trick(model, X):
    """
    Args:
        model: a Pytorch model
        X: a tensor of shape (batchSize, 81), denoting the input to the NN
    """
    model.eval()
    batch_size = X.shape[0]
    pred = X.clone().view(batch_size, 81) # values {0,1,...,9} of shape (batch_size, 81)
    while 0 in pred:
        output = model(pred) # (batch_size, 81, 9)
        if isinstance(output, tuple):
            logits = output[0]
        probs = torch.nn.functional.softmax(logits, dim=-1) # (batch_size, 81, 9)
        values, indices = probs.max(dim=-1) # (batch_size, 81), (batch_size, 81)
        values[pred != 0] = 0
        cell_indices = values.argmax(dim=-1) # (batch_size)
        for batch_idx, cell_idx in enumerate(cell_indices.tolist()):
            if pred[batch_idx,cell_idx] == 0:
                # pred contains number 0-9, where 1-9 are labels
                pred[batch_idx,cell_idx] = indices[batch_idx,cell_idx] + 1
    return pred - 1

def testNN_trick(model, test_dataloader, check_att=False):
    device = next(model.parameters()).device
    # check if total prediction is correct
    correct = total = 0
    # check if each single prediction is correct
    singleCorrect = singleTotal = 0
    # start evaluation
    pbar = tqdm(test_dataloader)
    for (data, target) in pbar:
        pred = inference_trick(model, data.to(device))
        target = target.to(device).view_as(pred)
        correctionMatrix = torch.logical_or(target.int() == pred.int(), target < 0).view(target.shape[0], -1)
        correct += correctionMatrix.all(1).sum().item()
        total += target.shape[0]
        singleCorrect += correctionMatrix[target >= 0].sum().item()
        singleTotal += (target >= 0).sum().item()
        # report progress
        pbar.set_description(f'inference trick: board acc {correct/total:0.4f}')
    return correct, total, singleCorrect, singleTotal, None