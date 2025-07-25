import math
import torch
import numpy as np
from collections import Counter
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt


from dataloader import SensorDataset, LABEL2ID_woRest
from MTHARS import MTHARS
from newUtils import nms_1d, decode_windows, decode_windows_accAndGyro

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
ROOT_DIR = '../cleanedValDataCut'
MODEL_PATH = 'accel.pth'
WINDOW_SIZE = 450
OVERLAP = 0.5
REDUCTION = 3
NUM_CLASSES = len(LABEL2ID_woRest) + 1


def compute_iou(seg1, seg2):
    """
    Compute IoU between two 1D segments.
    seg = (center, length)
    """
    c1, l1 = seg1
    c2, l2 = seg2
    s1, e1 = c1 - l1/2, c1 + l1/2
    s2, e2 = c2 - l2/2, c2 + l2/2

    inter = max(0.0, min(e1, e2) - max(s1, s2))
    union = l1 + l2 - inter
    return inter / union if union > 0 else 0.0


def evaluate_sample(accel_all, gyro_all, gt, model):
    """
    Run inference on one sample, count TP/FP/FN per class, and
    collect each GT's best-IoU for mean-IoU.
    Definitions:
      TP: pred overlaps a GT (IoU>0) & correct class
      FP: pred overlaps GT & wrong class, OR no overlap at all
      FN: GT that never overlaps any pred (IoU=0)
    Returns:
      TP, FP, FN arrays, plus list of best-IoUs (one per GT).
    """
    # slice into overlapping windows
    step = int(WINDOW_SIZE * (1 - OVERLAP))
    total = len(accel_all)
    end = 0
    windows = []
    while end < total:
        if end == 0 and end + WINDOW_SIZE <= total:
            start, stop = 0, WINDOW_SIZE
        elif end + (WINDOW_SIZE - step) <= total:
            start = end - step
            stop = end + (WINDOW_SIZE - step)
        else:
            start = max(0, total - WINDOW_SIZE)
            stop = total

        windows.append((accel_all[start:stop], start))     # for accel only
        # windows.append((
        #     accel_all[start:stop],    # for accel+gyro
        #     gyro_all[start:stop],
        #     start
        # ))
        end = stop
        if end == total:
            break

    # decode & collect raw proposals
    all_props = []
    for accel_w, w_start in windows:
        props = decode_windows(model, accel_w, w_start) # for accel only
        all_props.extend(props)

    # for accel_w, gyro_w, ws in windows:
    #     props = decode_windows_accAndGyro(model, accel_w, gyro_w, ws)  # for accel+gyro
    #     all_props.extend(props)

    # NMS and final predictions
    classes = torch.tensor([c for (c,_,_,_) in all_props])
    scores = torch.tensor([s for (_,s,_,_) in all_props])
    centers = torch.tensor([cx for (_,_,cx,_) in all_props])
    lengths = torch.tensor([L for (_,_,_,L) in all_props])

    keep = nms_1d(
        centers=centers,
        lengths=lengths,
        scores=scores,
        classes=classes,
        iou_threshold=0.0
    )
    final_preds = [all_props[i] for i in keep.tolist()]

    # count error for strokes and kicks
    stroke_ids = [1,2,3,4]
    kick_id = 6

    # ground truth counts
    gt_stroke_count = sum(1 for (c,_,_) in gt if c in stroke_ids)
    gt_kick_count = sum(1 for (c,_,_) in gt if c == kick_id)

    # predicted counts
    pred_stroke_count = sum(1 for (c,_,_,_) in final_preds if c in stroke_ids)
    pred_kick_count = sum(1 for (c,_,_,_) in final_preds if c == kick_id)

    # absolute errors
    stroke_error = abs(pred_stroke_count - gt_stroke_count)
    kick_error = abs(pred_kick_count - gt_kick_count)

    TP = np.zeros(NUM_CLASSES, dtype=int)
    FP = np.zeros(NUM_CLASSES, dtype=int)
    FN = np.zeros(NUM_CLASSES, dtype=int)

    # for each prediction, find best-overlap GT (if any) and count TP/FP
    for pred_cls, _, pred_c, pred_l in final_preds:
        best_iou = 0.0
        best_gt_cls = None
        for gt_cls, gt_len, gt_c in gt:
            iou = compute_iou((pred_c, pred_l), (gt_c, gt_len))
            if iou > best_iou:
                best_iou = iou
                best_gt_cls = gt_cls

        if best_iou > 0.0:
            if pred_cls == best_gt_cls:
                TP[pred_cls] += 1
            else:
                FP[pred_cls] += 1
                FN[best_gt_cls] += 1
        else:
            FP[pred_cls] += 1
            FN[gt_cls] += 1

    # for each GT, record best-IoU and count FN if never overlapped
    best_ious = []
    best_pred_cls = []
    for gt_cls, gt_len, gt_c in gt:
        best_iou = 0.0
        pred_for_this_gt = 0
        for pred_cls, _, pred_c, pred_l in final_preds:
            iou = compute_iou((pred_c, pred_l), (gt_c, gt_len))
            if iou > best_iou:
                best_iou = iou
                pred_for_this_gt = pred_cls
        best_ious.append(best_iou)
        best_pred_cls.append(pred_for_this_gt)

        # if best_iou == 0.0:
        #     FN[gt_cls] += 1

    return TP, FP, FN, best_ious, stroke_error, kick_error, best_pred_cls

def plot_confusion(y_true, y_pred, inv_label_map):

    print("lengths of y true and y pred:: ")
    print(len(y_true))
    print(len(y_pred))
    labels = list(range(1, NUM_CLASSES))
    class_names = [inv_label_map[c] for c in labels]
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    disp = ConfusionMatrixDisplay(cm, display_labels=class_names)
    fig, ax = plt.subplots(figsize=(5,5))
    disp.plot(ax=ax, cmap='Blues', colorbar=False)
    plt.xticks(rotation=45, ha='right')
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.show()

def main():
    model = MTHARS(in_channels=3, num_classes=NUM_CLASSES).to(DEVICE)   # accel only
    # model = MTHARS(in_channels=6, num_classes=NUM_CLASSES).to(DEVICE)   # accel and gyro
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()

    ds = SensorDataset(ROOT_DIR)
    print(f"Dataset size: {len(ds)} samples\n")

    total_TP = np.zeros(NUM_CLASSES, dtype=int)
    total_FP = np.zeros(NUM_CLASSES, dtype=int)
    total_FN = np.zeros(NUM_CLASSES, dtype=int)
    all_ious = []
    stroke_errors = []
    kick_errors = []
    y_true = []
    y_pred = []

    for idx in range(len(ds)):
        accel_all, gyro_all, gt_raw = ds[idx]
        gt = [(LABEL2ID_woRest[lbl], length, center) for (lbl, length, center) in gt_raw]

        TP, FP, FN, best_ious, stroke_err, kick_err, best_pred_cls = evaluate_sample(accel_all, gyro_all, gt, model)
        total_TP += TP
        total_FP += FP
        total_FN += FN
        all_ious.extend(best_ious)

        stroke_errors.append(stroke_err)
        kick_errors.append(kick_err)

        print(f"Sample {idx:03d} -> TP: {TP.sum():3d}, FP: {FP.sum():3d}, FN: {FN.sum():3d}, Stroke error {stroke_err}, Kick error {kick_err}")
        for (gt_cls, _l, _c), pred_cls in zip(gt, best_pred_cls):
            y_true.append(gt_cls)
            y_pred.append(pred_cls)


    mean_iou = float(np.mean(all_ious)) if all_ious else 0.0
    print(f"\nMean IoU (over all GT segments): {mean_iou:.4f}\n")

    precisions, recalls, f1s = [], [], []
    for c in range(1, NUM_CLASSES):  # skip background class = 0
        tp, fp, fn = total_TP[c], total_FP[c], total_FN[c]
        prec = tp / (tp + fp) if tp + fp > 0 else 0.0
        rec  = tp / (tp + fn) if tp + fn > 0 else 0.0
        f1   = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0

        precisions.append(prec)
        recalls.append(rec)
        f1s.append(f1)
        print(f"Class {c:2d} -> P: {prec:.3f}, R: {rec:.3f}, F1: {f1:.3f}")

    # per class F1 scores
    inv_label_map = {v: k for k, v in LABEL2ID_woRest.items()}
    print("\nPer class F1:")
    for c, f1 in zip(range(1, NUM_CLASSES), f1s):
        name = inv_label_map.get(c, f"Class{c}")
        print(f" {name:15s}: {f1:.3f}")

    # Macro-F1
    macro_f1 = np.mean(f1s)
    # Micro-metrics
    TP_sum, FP_sum, FN_sum = total_TP.sum(), total_FP.sum(), total_FN.sum()
    micro_prec = TP_sum / (TP_sum + FP_sum) if TP_sum + FP_sum > 0 else 0.0
    micro_rec  = TP_sum / (TP_sum + FN_sum) if TP_sum + FN_sum > 0 else 0.0
    micro_f1   = (2 * micro_prec * micro_rec / (micro_prec + micro_rec)) if (micro_prec + micro_rec) > 0 else 0.0
    stroke_mae = float(np.mean(stroke_errors)) if stroke_errors else 0.0
    kick_mae = float(np.mean(kick_errors)) if kick_errors else 0.0

    print("\n=== Summary ===")
    print(f"Mean IoU : {mean_iou:.4f}")
    print(f"Macro-F1 : {macro_f1:.4f}")
    print(f"Micro-F1 : {micro_f1:.4f}")
    print(f"Stroke Count MAE: {stroke_mae}")
    print(f"Kick MAE: {kick_mae}")
    print("\n") 

    plot_confusion(y_true, y_pred, inv_label_map)
    print(f"Micro-Precision: {micro_prec:.4f}")
    print(f"Micro-Recall : {micro_rec:.4f}")


if __name__ == "__main__":
    main()
