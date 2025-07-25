import math
import numpy as np
import torch
import time
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
import pandas as pd

from dataloader import SensorDataset, LABEL2ID, LABEL2ID_woRest

from Utils import (
    assign_window_labels,
    hard_negative_mining,
    combined_loss,
    decode_windows,
    decode_windows_accAndGyro,
    nms_1d
)
from MTHARS import MTHARS

ROOT_DIR = '../cleanedTrainDataWORest'
VAL_DIR = '../cleanedValDataWORest'
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
WINDOW_SIZE = 450
OVERLAP = 0.5
REDUCTION = 3
scales=[2,3,4]
NUM_CLASSES = len(LABEL2ID_woRest) + 1  # + 1 for background (0)
IN_CHANNELS = 6 # 3 for only accel, 6 for accel+gyro
LR = 1e-3
NUM_EPOCHS = 50
ALPHA = 1.0
BETA = 2.0

def train():
    ds = SensorDataset(ROOT_DIR)
    vds = SensorDataset(VAL_DIR)
    model = MTHARS(in_channels=IN_CHANNELS, num_classes=NUM_CLASSES).to(DEVICE)
    opt = Adam(model.parameters(), lr=LR)
    scheduler = StepLR(opt, step_size=30, gamma=0.1)

    rows = []

    for epoch in range(1, NUM_EPOCHS+1):
        t0 = time.time()

        batches = ds.getBatches(
            window_size=WINDOW_SIZE,
            overlap=OVERLAP,
            reduction=REDUCTION,
            batch_size=8,
            shuffle=True
        )
        print("Starting epoch num ", epoch,"with training batches: ", len(batches))

        total_loss = 0.0
        total_iou = 0.0
        total_pos = 0
        total_pos_correct = 0
        total_pos_anchors = 0

        model.train()

        conf_mat = torch.zeros(NUM_CLASSES, NUM_CLASSES, dtype=torch.long)

        for batch in batches:
            accel_windows = [w[0] for w in batch]
            gyro_windows  = [w[1] for w in batch]
            label_lists   = [w[2] for w in batch]
            # prepare signal tensor [1, C, T]
            B = len(accel_windows)
            # sig = torch.stack([
            #     torch.from_numpy(np.stack(a, axis=1)).float()   # for only accel
            #     for a in accel_windows
            # ], dim=0).to(DEVICE) # -> (B,3,WINDOW_SIZE)
            sig = torch.stack([
                torch.cat([
                torch.from_numpy(np.stack(a,axis=1)),   # for accel+gyro
                torch.from_numpy(np.stack(g,axis=1))
                ], dim=0).float()
                for a, g in zip(accel_windows, gyro_windows)
            ], dim=0).to(DEVICE)  # (B, 6, T)

            # forward pass
            cls_logits, offset_preds = model(sig)


            B, K, N_feat = cls_logits.shape

            num_anchors =  N_feat

            flat_logits = cls_logits.permute(0,2,1).reshape(-1, K)
            flat_offsets = offset_preds.permute(0,2,1).reshape(-1,2)

            gt_lists = [w[2] for w in batch]


            window_centers = torch.arange(N_feat, device=DEVICE).float()
            window_lengths = torch.full((N_feat,), REDUCTION, device=DEVICE).float()               

            batch_loss = 0.0
            for b in range(B):
                gt = gt_lists[b]
                if len(gt)==0:
                    gt_centers = torch.empty(0, device=DEVICE)
                    gt_lengths = torch.empty(0, device=DEVICE)
                    gt_classes = torch.empty(0, device=DEVICE, dtype=torch.long)
                else:
                    gt_centers = torch.tensor([t_x for (_,t_l,t_x) in gt], device=DEVICE, dtype=torch.float)
                    gt_lengths = torch.tensor([t_l for (_,t_l,t_x) in gt], device=DEVICE, dtype=torch.float)
                    gt_classes = torch.tensor([c   for (c,_,_) in gt], device=DEVICE, dtype=torch.long)

                # assign labels & offsets
                labels, offsets = assign_window_labels(
                    window_centers, window_lengths,
                    gt_centers, gt_lengths, gt_classes,
                    iou_threshold=0.5                                                       # iou threshold
                )

                # slice this sample predictionss
                start = b * num_anchors
                end = start + num_anchors
                logits_b = flat_logits[start:end]
                offs_b = flat_offsets[start:end]

                # pick positives & negatives
                pos_inds = (labels>0).nonzero(as_tuple=False).view(-1).tolist()
                neg_inds = hard_negative_mining(logits_b, labels, neg_pos_ratio=0.4)        # negative ratio

                # compute combined loss
                if len(pos_inds) > 0:
                    loss = combined_loss(
                        logits=logits_b, labels=labels,
                        pos_indices=pos_inds, neg_indices=neg_inds,
                        pred_offsets=offs_b[pos_inds], target_offsets=offsets[pos_inds],
                        alpha=ALPHA, beta=BETA
                    )
                else:
                    print("positive inds is 0")
                batch_loss += loss

                # classification accuracy over ALL anchors
                preds = logits_b.argmax(dim=1)
                pos_mask = labels > 0   
                total_pos_correct += (preds[pos_mask] == labels[pos_mask]).sum().item()
                total_pos_anchors += pos_mask.sum().item()

                # update confusion
                for t,p in zip(labels.tolist(), preds.tolist()):
                    conf_mat[t, p] += 1

                # mean iou over the positives
                # for each positive anchor, decode pred & target segments and compute iou
                for i in pos_inds:
                    w_x = window_centers[i].item()
                    w_l = window_lengths[i].item()
                    # predicted offsets
                    f_px, f_pl = offs_b[i].cpu().tolist()
                    p_center = w_x + f_px * w_l
                    p_length = w_l * math.exp(f_pl)
                    # target offsets
                    f_tx, f_tl = offsets[i].cpu().tolist()
                    t_center = w_x + f_tx * w_l
                    t_length = w_l * math.exp(f_tl)
                    # iou
                    start_i = max(p_center - p_length/2, t_center - t_length/2)
                    end_i = min(p_center + p_length/2, t_center + t_length/2)
                    inter = max(0.0, end_i - start_i)
                    union = p_length + t_length - inter
                    total_iou += (inter/union if union>0 else 0.0)
                total_pos += len(pos_inds)

            pos_acc = total_pos_correct / total_pos_anchors if total_pos_anchors else 0.0

            opt.zero_grad()
            batch_loss.backward()
            opt.step()
            total_loss += batch_loss.item()

        scheduler.step()
        if epoch == 1 or epoch == 50 or epoch % 10 == 0:
            print(f" |-_-| Epoch {epoch}, lr = {scheduler.get_last_lr()[0]:.2e}")

        avg_loss = total_loss / len(batches)
        epoch_iou  = total_iou / total_pos     if total_pos>0     else 0.0

        epoch_time = time.time() - t0
        print(f"|-_-| Epoch took {epoch_time} |-_-|")


        print(f" |-_-| Epoch {epoch:03d} |-_-| loss {avg_loss:.4f} |-_-| pos‐acc {pos_acc:.3f} |-_-| mIoU {epoch_iou:.3f} |-_-| #pos {total_pos_anchors} |-_-|")
        t1 = time.time()
        # nms part
        model.eval()
        with torch.no_grad():
            seg_correct = 0
            seg_iou_sum = 0.0
            seg_count   = 0

            # iterate over *samples* (not mini‐batches)
            for sample_idx in range(len(vds)):
                accel_all, gyro_all, gt_all = vds[sample_idx]
                # convert GT to (class, length, center)
                gt = [(LABEL2ID[lbl], t_l, t_x) for (lbl, t_l, t_x) in gt_all]

                # break into windows exactly as in evaluator
                step = int(WINDOW_SIZE*(1-OVERLAP))
                windows = []
                end = 0
                total = len(accel_all)
                while end < total:
                    if end == 0 and end + WINDOW_SIZE <= total:
                        start, stop = 0, WINDOW_SIZE
                    elif end + (WINDOW_SIZE - step) <= total:
                        start = end - step
                        stop = end + (WINDOW_SIZE - step)
                    else:
                        start = max(0, total - WINDOW_SIZE)
                        stop = total
                    # windows.append((accel_all[start:stop], start))  # for accel only
                    windows.append((
                        accel_all[start:stop],    # for accel+gyro
                        gyro_all[start:stop],
                        start
                    ))
                    end = stop

                # collect all proposals
                all_props = []
                # for accel_w, ws in windows:
                #     props = decode_windows(model, accel_w, ws)    # for accel only
                #     all_props.extend(props)

                for accel_w, gyro_w, ws in windows:
                    props = decode_windows_accAndGyro(model, accel_w, gyro_w, ws)  # for accel+gyro
                    all_props.extend(props)

                # NMS
                classes = torch.tensor([c for (c,_,_,_) in all_props])
                scores  = torch.tensor([s for (_,s,_,_) in all_props])
                centers = torch.tensor([cx for (_,_,cx,_) in all_props])
                lengths = torch.tensor([L for (_,_,_,L) in all_props])
                keep = nms_1d(centers, lengths, scores, classes, iou_threshold=0.0)
                preds = [all_props[i] for i in keep.tolist()]  # (class,score,center,length)

                # now match predictions to GT one‐to‐one by best iou
                for (gt_c, gt_L, gt_cx) in gt:
                    # compute ious against *all* predictions, remembering their classes
                    best_iou = 0.0
                    best_cls = None
                    for (p_c, _, p_cx, p_L) in preds:
                        # iou between GT and this pred
                        st = max(gt_cx - gt_L/2, p_cx - p_L/2)
                        en = min(gt_cx + gt_L/2, p_cx + p_L/2)
                        inter = max(0.0, en - st)
                        union = gt_L + p_L - inter
                        iou = inter/union if union>0 else 0.0

                        if iou > best_iou:
                            best_iou = iou
                            best_cls = p_c

                    seg_iou_sum += best_iou
                    seg_count += 1

                    if best_cls == gt_c:
                        seg_correct += 1

                avg_iou = seg_iou_sum / seg_count
                class_acc = seg_correct / seg_count
            
            rows.append({
                "epoch": epoch,
                "loss": avg_loss,
                "train acc": pos_acc,
                "train mean IoU": epoch_iou,
                "total pos anchors": total_pos_anchors,
                "val acc": class_acc,
                "val IoU": avg_iou,
                "val # segments": seg_count
            })

            print(f" ── NMS‐Eval @epoch {epoch:03d} ── class_acc {class_acc:.3f} | avg‐IoU {avg_iou:.3f} over {seg_count} segments")
            val_time = time.time() - t1
            print(f"|-_-| evaluation nms took {val_time} |-_-|")
            total_time = time.time() - t0
            print(f"Total time for this epoch + val: {total_time}")

    df = pd.DataFrame(rows)
    df.to_csv("accelAndGyro_data.csv", index=False)
    torch.save(model.state_dict(), "accelAndGyro.pth")
    print("Training complete. PIZDETS")

if __name__ == "__main__":
    train()
