# The helper functions

import matplotlib.pyplot as plt
import numpy as np
import torch

def print_result(result):
    for eval_func in result:
        print(f'Printing the accuracy (%) using {eval_func} as the evaluation function:')
        print(('{:<6}'+'{:<15}' * 4).format('idx', 'total_acc', 'single_acc', 'total_count', 'single_count'))
        row_format = '{:<6}' + '{:<15.2f}' * 2 + '{:<15}' * 2
        for idx, (correct, total, singleCorrect, singleTotal, _) in enumerate(result[eval_func]):
            print(row_format.format(idx+1, 100*correct/total, 100*singleCorrect/singleTotal,\
                  f'{correct}/{total}', f'{singleCorrect}/{singleTotal}'))

# this function is from https://captum.ai/tutorials/Bert_SQUAD_Interpret2 with some adjustment
def visualize_token2token_scores(scores_mat, all_tokens, x_label_name='Head', filename='heatmap'):
    """
    Args:
        scores_mat: of shape (num_heads, 81, 81)
    """
    num_heads = scores_mat.shape[0]
    fig = plt.figure(figsize=(50, 50))

    for idx, scores in enumerate(scores_mat):
        scores_np = np.array(scores)
        ax = fig.add_subplot((num_heads+1) // 2, 2, idx+1)
        # append the attention weights
        im = ax.imshow(scores, cmap='viridis')

        fontdict = {'fontsize': 10}

        ax.set_xticks(range(len(all_tokens)))
        ax.set_yticks(range(len(all_tokens)))

        ax.set_xticklabels(all_tokens, fontdict=fontdict, rotation=90)
        ax.set_yticklabels(all_tokens, fontdict=fontdict)
        ax.set_xlabel('{} {}'.format(x_label_name, idx+1))

        fig.colorbar(im, fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig(filename + '.png')
    plt.clf()
    plt.close(fig)

def visualize_cell_attention(scores_mat, indices, cell_tokens, filename='heatmap'):
    """
    Args:
        scores_mat: of shape (num_heads, 81, 81)
    """
    num_heads = scores_mat.shape[0]
    num_cells = len(indices)
    fig = plt.figure(figsize=(num_heads*10, num_cells*10))

    subplot_idx = 1
    for idx, scores in enumerate(scores_mat):
        for i,j in indices:
            scores_np = scores[i*9+j].reshape(9,9)
            ax = fig.add_subplot(num_heads, num_cells, subplot_idx)
            # append the attention weights
            im = ax.imshow(scores_np, cmap='viridis')
            fontdict = {'fontsize': 15}
            ax.set_xticks(range(len(cell_tokens)))
            ax.set_yticks(range(len(cell_tokens)))
            ax.set_xticklabels(cell_tokens, fontdict=fontdict, rotation=90)
            ax.set_yticklabels(cell_tokens, fontdict=fontdict)
            ax.set_xlabel('({},{}); Head {}'.format(i+1, j+1, subplot_idx))
            fig.colorbar(im, fraction=0.046, pad=0.04)
            subplot_idx += 1
    plt.tight_layout()
    plt.savefig(filename + '.png', bbox_inches='tight', pad_inches = 0)
    plt.clf()
    plt.close(fig)

def visualize_adjacency(filename='heatmap/adjacency'):
    A = {}
    A['row'] = torch.zeros([81, 81], dtype=torch.float32)
    A['col'] = torch.zeros([81, 81], dtype=torch.float32)
    A['box'] = torch.zeros([81, 81], dtype=torch.float32)
    for i in range(81):
        for j in range(81):
            ix, iy = i // 9, i % 9
            jx, jy = j // 9, j % 9
            ic = 3 * (ix // 3) + iy // 3
            jc = 3 * (jx // 3) + jy // 3
            if ix == jx:
                A['row'][i, j] = 1
            if iy == jy:
                A['col'][i, j] = 1
            if ic == jc:
                A['box'][i, j] = 1
    all_tokens = range(81)
    fig = plt.figure(figsize=(50, 50))
    for idx, k in enumerate(A.keys()):
        ax = fig.add_subplot(2, 2, idx+1)
        im = ax.imshow(A[k], cmap='viridis')
        fontdict = {'fontsize': 10}
        ax.set_xticks(range(len(all_tokens)))
        ax.set_yticks(range(len(all_tokens)))
        ax.set_xticklabels(all_tokens, fontdict=fontdict, rotation=90)
        ax.set_yticklabels(all_tokens, fontdict=fontdict)
        ax.set_xlabel('{}'.format(k), fontsize=40)
        fig.colorbar(im, fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig(filename + '.png')
    plt.clf()
    plt.close(fig)

def create_heatmap(result, prefix=''):
    if 'testNN' in result:
        all_tokens = range(81)
        cell_tokens = range(9)
        indices = [[0,0], [4,4]]
        for i, (_,_,_,_,att) in enumerate(result['testNN']):
            # we save attention matrices for every 10 results
            if i % 10 == 0:
                # att is of shape (n_layer * n_recur, num_heads, 81, 81)
                for l in range(att.shape[0]):
                    visualize_token2token_scores(att[l].detach().cpu().numpy(), all_tokens, filename=f'heatmap/{prefix}idx{i+1}_layer{l+1}')
                visualize_cell_attention(att[0].detach().cpu().numpy(), indices, cell_tokens, filename=f'heatmap/{prefix}idx{i+1}_layer1_cell')
    else:
        print('Cannot find the result under evaluation method testNN')