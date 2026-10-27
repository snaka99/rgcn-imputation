# The composite loss of the paper:
#   L = l1 * Huber + l2 * Pinball(median) + l3 * MAE in original units + l4 * graph smoothness

import torch
import torch.nn.functional as F


def huber(pred, target, delta=1.0):
    return F.huber_loss(pred, target, delta=delta)


def pinball(pred, target, q=0.5):
    # quantile loss; q = 0.5 is the median
    diff = target - pred
    return torch.mean(torch.maximum(q * diff, (q - 1) * diff))


def mae_original(pred_z, target_z, mean, std):
    # back to the original scale: z -> log1p value -> value
    pred = torch.expm1(pred_z * std + mean)
    true = torch.expm1(target_z * std + mean)
    return torch.mean(torch.abs(pred - true)) / (torch.mean(torch.abs(true)) + 1e-8)   # relative, so it has the same size as the others


def smoothness(pred, src, dst):
    # neighbours in the graph should not get very different values
    if len(src) == 0:
        return pred.sum() * 0
    return torch.mean((pred[src] - pred[dst]) ** 2)


def composite_loss(pred, target, mask, mean, std, src, dst, weights=(1.0, 1.0, 0.5, 0.1)):
    l1, l2, l3, l4 = weights
    p, t = pred[mask], target[mask]
    loss = l1 * huber(p, t) + l2 * pinball(p, t) + l3 * mae_original(p, t, mean[mask], std[mask])
    return loss + l4 * smoothness(pred, src, dst)
