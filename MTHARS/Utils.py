import math
import torch
import torch.nn.functional as F
from typing import List
import matplotlib.pyplot as plt
import numpy as np

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
REDUCTION = 3


def compute_iou_1d(w_center, w_length, t_center, t_length):
    """
    Compute IoU between one window and one truth segment in 1D.
    All values are in the same coordinate space (feature indices).
    """
    # Convert center length to [start, end]
    w_start = w_center - 0.5 * w_length
    w_end   = w_center + 0.5 * w_length
    t_start = t_center - 0.5 * t_length
    t_end   = t_center + 0.5 * t_length

    # intersection
    inter_start = max(w_start, t_start)
    inter_end   = min(w_end, t_end)
    inter_len   = max(0.0, inter_end - inter_start)

    # union
    union_len = (w_end - w_start) + (t_end - t_start) - inter_len
    if union_len <= 0:
        return 0.0
    return inter_len / union_len

def assign_window_labels(
    window_centers: torch.Tensor,
    window_lengths: torch.Tensor,
    gt_centers: torch.Tensor,
    gt_lengths: torch.Tensor,
    gt_classes: torch.Tensor,
    iou_threshold: float = 0.5
):
    """
    Assign labels and offsets to windows based on ground truth segments.

    Args:
        window_centers: Tensor (na,) of float centers for each anchor
        window_lengths: Tensor (na,) of float lengths for each anchor
        gt_centers: Tensor (nb,) of true segment centers
        gt_lengths: Tensor (nb,) of true segment lengths
        gt_classes: Tensor (nb,) of true segment class indices
        iou_threshold: IoU threshold for matching remaining windows

    Returns:
        labels:  Tensor (na,)
        offsets: Tensor (na,2) with (f_x, f_l) targets for each window
    """
    na = window_centers.size(0)
    nb = gt_centers.size(0)
    device = window_centers.device

    # Compute IoU matrix M[na, nb]
    M = torch.zeros(na, nb, device=device)
    for i in range(na):
        for j in range(nb):
            M[i,j] = compute_iou_1d(
                window_centers[i].item(),
                window_lengths[i].item(),
                gt_centers[j].item(),
                gt_lengths[j].item()
            )

    labels  = torch.zeros(na, dtype=torch.long, device=device)
    offsets = torch.zeros(na, 2, device=device)
    if nb == 0:
        print("ALERT, no gound truth segments!")
        return labels, offsets

    available_rows = set(range(na))
    available_cols = set(range(nb))
    # print("colds: ",len(available_cols))
    # print("rows: ",len(available_rows))

    # phase 1: force one match per ground truth
    for j in list(available_cols):
        # pick the anchor i (from available rows) with highest IoU[i,j]
        best_i = max(available_rows, key=lambda i: M[i,j].item())
        labels[best_i] = gt_classes[j]
        w_x, w_l = window_centers[best_i], window_lengths[best_i]
        t_x, t_l = gt_centers[j],         gt_lengths[j]
        offsets[best_i,0] = (t_x - w_x) / w_l
        offsets[best_i,1] = torch.log(t_l / w_l)
        available_rows.remove(best_i)
        # available_cols.remove(j)

    # phase 2: thresholded matching on the rest
    all_gt_cols = set(range(nb)) # iterate all GTs again
    candidates = [(i,j,M[i,j].item())
                for i in available_rows
                for j in all_gt_cols
                if M[i,j] >= iou_threshold]
    candidates.sort(key=lambda x: x[2], reverse=True)
    # print(candidates)
    extra_matches = 0
    for i,j,score in candidates:
        if i in available_rows and j in available_cols:
            labels[i] = gt_classes[j]
            w_x, w_l = window_centers[i], window_lengths[i]
            t_x, t_l = gt_centers[j],         gt_lengths[j]
            offsets[i,0] = (t_x - w_x) / w_l
            offsets[i,1] = torch.log(t_l / w_l)
            available_rows.remove(i)
            available_cols.remove(j)
            extra_matches += 1
    # print(f"added {extra_matches} extra matches who pass the IoU threshold of {iou_threshold}")

    return labels, offsets


def hard_negative_mining(
    logits: torch.Tensor,       # (n_w, k+1) raw scores
    labels: torch.Tensor,       # (n_w,) integer labels
    neg_pos_ratio: float = 1.0  # ratio of negatives to positives
) -> List[int]:
    """
    Returns a list of negative indices selected by hard negative mining.
    """
    # get all negatives
    neg_mask = (labels == 0)
    neg_indices = torch.nonzero(neg_mask, as_tuple=False).view(-1)

    # if no positives, return empty:
    num_pos = int((labels > 0).sum().item())
    if num_pos == 0 or neg_indices.numel() == 0:
        return []

    # compute hardness of each negative = highest non background score
    neg_logits = logits[neg_indices] # (n_neg, k+1)
    hardness  = neg_logits[:, 1:].max(dim=1)[0]  # (n_neg,)

    # sort negatives by descending hardness
    _, order = hardness.sort(descending=True)

    # keep up to neg_pos_ratio * num_pos negatives
    max_neg = int(neg_pos_ratio * num_pos)
    keep    = order[:max_neg]
    # print("negative amount: ", len(neg_indices[keep]))
    return neg_indices[keep].tolist()



def localization_loss(pred_offsets, target_offsets):
    """
    Smooth L1 loss for offset regression.

    Args:
        pred_offsets: Tensor of shape (N,2)
        target_offsets: Tensor of shape (N,2)
    Returns:
        Scalar loss (sum over all elements)
    """
    return F.smooth_l1_loss(pred_offsets, target_offsets, reduction='sum')


def classification_loss(logits, labels, pos_indices, neg_indices):
    """
    Cross-entropy loss over selected positives and negatives.

    Args:
        logits: Tensor of shape (n_w, k+1)
        labels: Tensor of shape (n_w,) with class indices (0..k)
        pos_indices: list of positive anchor indices
        neg_indices: list of negative anchor indices
    Returns:
        Scalar loss (sum over selected windows)
    """
    indices = pos_indices + neg_indices
    selected_logits = logits[indices]
    selected_labels = labels[indices]
    return F.cross_entropy(selected_logits, selected_labels, reduction='mean')


def combined_loss(logits, labels, pos_indices, neg_indices,
                  pred_offsets, target_offsets,
                  alpha=1.0, beta=1.0):
    """
    Combined classification and localization loss.

    Args:
        logits: Tensor of shape (n_w, k+1)
        labels: Tensor of shape (n_w,)
        pos_indices: list of positive indices
        neg_indices: list of negative indices
        pred_offsets: Tensor of shape (N_pos,2)
        target_offsets: Tensor of shape (N_pos,2)
        alpha: weight for classification loss
        beta: weight for localization loss
    Returns:
        Scalar combined loss
    """
    # classification
    L_conf = classification_loss(logits, labels, pos_indices, neg_indices)
    # print(f" L_conf (classification) = {L_conf.item():.4f}  |  #pos={len(pos_indices)}  #neg={len(neg_indices)}")
    # localization
    L_loc = localization_loss(pred_offsets, target_offsets)
    # print(f" L_loc (regression) = {L_loc.item():.4f}")
    N = max(len(pos_indices), 1)
    return (alpha * L_conf + beta * L_loc) / N


def nms_1d(
    centers: torch.Tensor,
    lengths: torch.Tensor,
    scores: torch.Tensor,
    classes: torch.Tensor,
    iou_threshold: float = 0.5
) -> torch.Tensor:
    """
    Apply per class 1D non maximum suppression.

    Args:
        centers: (N,) tensor of predicted window centers
        lengths: (N,) tensor of predicted window lengths
        scores: (N,) tensor of max class probability per window
        classes: (N,) tensor of predicted class indices
        iou_threshold: IoU threshold to suppress

    Returns:
        keep: 1D tensor of indices into the original N windows to keep
    """
    # compute start/end for each window
    starts = centers - lengths / 2
    ends   = centers + lengths / 2

    prelim_keep = []
    # loop over each class (skip background=0)
    for c in classes.unique():
        if c.item() == 0:
            continue
        # select only this class
        cls_mask = (classes == c)
        idxs     = torch.nonzero(cls_mask, as_tuple=False).view(-1)
        if idxs.numel() == 0:
            continue

        # sort by descending score
        cls_scores = scores[idxs]
        _, order = torch.sort(cls_scores, descending=True)
        cls_idxs = idxs[order]

        # perform NMS
        while cls_idxs.numel() > 0:
            i = cls_idxs[0].item()
            prelim_keep.append(i)

            if cls_idxs.numel() == 1:
                break

            # compute iou of this window vs the rest
            rest = cls_idxs[1:]
            st_i, en_i = starts[i], ends[i]
            st_rest = starts[rest]
            en_rest = ends[rest]

            inter = torch.min(en_i, en_rest) - torch.max(st_i, st_rest)
            inter = torch.clamp(inter, min=0.0)
            union = lengths[i] + lengths[rest] - inter
            iou   = inter / union

            # keep only those below threshold
            keep_mask = iou <= iou_threshold
            cls_idxs = rest[keep_mask]

   # global NMS over prelim_keep:
    starts = centers[prelim_keep] - lengths[prelim_keep]/2
    ends   = centers[prelim_keep] + lengths[prelim_keep]/2
    scores2 = scores[prelim_keep]

    # sort prelim_keep by score descending
    order = torch.argsort(scores2, descending=True)
    global_keep = []
    for idx in order.tolist():
        i = prelim_keep[idx]
        # check overlap with anything already in global_keep
        st_i, en_i = centers[i] - lengths[i]/2, centers[i] + lengths[i]/2
        conflict = False
        for j in global_keep:
            st_j = centers[j] - lengths[j]/2
            en_j = centers[j] + lengths[j]/2
            inter = max(0, min(en_i,en_j) - max(st_i,st_j))
            union = lengths[i] + lengths[j] - inter
            if inter/union > iou_threshold:
                conflict = True
                break
        if not conflict:
            global_keep.append(i)

    return torch.tensor(global_keep, dtype=torch.long)

def decode_windows(model, accel_w, window_start):
    """
    Run model on one accel window and return raw proposals
    as (class_id, score, global_center, global_length).
    """
    x = torch.from_numpy(np.stack(accel_w, axis=1)) \
             .float().unsqueeze(0).to(DEVICE) # (1,3,450)
    with torch.no_grad():
        cls_logits, offs = model(x)
    probs = torch.softmax(cls_logits[0], dim=0) # (K, T')
    offs  = offs[0] # (2, T')
    K, Tprime = probs.shape

    proposals = []
    for t in range(Tprime):
        score, cls = probs[:,t].max(0)
        fx, fl = offs[:,t]
        # local -> window frames
        local_center = t
        local_length = REDUCTION
        p_center = local_center + fx.item()*local_length
        p_length = local_length * math.exp(fl.item())
        # map into original signal frames
        global_center = window_start + p_center*REDUCTION
        global_length = p_length*REDUCTION
        proposals.append((int(cls.item()), score.item(),
                           global_center, global_length))
    return proposals

def decode_windows_accAndGyro(model, accel_w, gyro_w, window_start):
    """
    Run model on one accel+gyro window and return raw proposals
    as (class_id, score, global_center, global_length).
    """
    a = torch.from_numpy(np.stack(accel_w, axis=1))
    g = torch.from_numpy(np.stack(gyro_w,  axis=1))
    x = torch.cat([a, g], dim=0) # (6, T)
    x = x.unsqueeze(0).float().to(DEVICE) # (1, 6, T)

    with torch.no_grad():
        cls_logits, offs = model(x)

    probs = torch.softmax(cls_logits[0], dim=0) # (K, T')
    offs  = offs[0] # (2, T')
    K, Tprime = probs.shape

    proposals = []
    for t in range(Tprime):
        score, cls = probs[:, t].max(0)
        fx, fl = offs[:, t]

        # local -> window frames
        local_center = t
        local_length = REDUCTION
        p_center = local_center + fx.item() * local_length
        p_length = local_length * math.exp(fl.item())

        # map into original signal frames
        global_center = window_start + p_center * REDUCTION
        global_length = p_length * REDUCTION

        proposals.append((
            int(cls.item()),
            score.item(),
            global_center,
            global_length
        ))

    return proposals


if __name__ == "__main__":
    B, C, N = 4, 256, 150
    feature_seq = torch.arange(B*C*N, dtype=torch.float32).view(B, C, N)

    window_centers = torch.tensor([10, 20, 30, 40, 50, 60], dtype=torch.float)
    window_lengths = torch.tensor([10, 10, 20, 20, 30, 30], dtype=torch.float)
    gt_centers     = torch.tensor([25, 55], dtype=torch.float)
    gt_lengths     = torch.tensor([20, 30], dtype=torch.float)
    gt_classes     = torch.tensor([1, 2], dtype=torch.long)

    labels, offsets = assign_window_labels(
        window_centers, window_lengths,
        gt_centers, gt_lengths, gt_classes,
        iou_threshold=0.5
    )
    print("Labels: ", labels)
    print("Offsets:\n", offsets)


    N = 50
    window_centers = torch.linspace(0, 100, N)
    window_lengths = torch.full((N,), 10.0)
    gt_centers = torch.tensor([30.0, 70.0])
    gt_lengths = torch.tensor([20.0, 15.0])
    gt_classes = torch.tensor([1, 2])

    labels, offsets = assign_window_labels(
        window_centers, window_lengths,
        gt_centers, gt_lengths, gt_classes,
        iou_threshold=0.4
    )

    fig, ax = plt.subplots(figsize=(10,3))
    for c, lbl in zip(window_centers, labels):
        color = 'gray' if lbl==0 else f'C{lbl}'
        ax.vlines(c, 0, 1, color=color, linewidth=2)

    for cx, L, cls in zip(gt_centers, gt_lengths, gt_classes):
        start, end = cx-L/2, cx+L/2
        ax.hlines(1.2, start, end, color=f'C{cls}', linewidth=6)

    ax.set_ylim(0,1.5)
    ax.set_xlabel('time')
    ax.set_yticks([])
    ax.set_title('anchor to-GT Matching (gray=background)')
    plt.show()
