import torch
import numpy as np
import matplotlib.pyplot as plt

from dataloader import SensorDataset, LABEL2ID_woRest
from MTHARS import MTHARS
from newUtils import nms_1d, decode_windows

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
ROOT_DIR = '../cleanedValDataWORest'
MODEL_PATH = 'accel.pth'
WINDOW_SIZE = 450
OVERLAP = 0.5
REDUCTION = 3
NUM_CLASSES = len(LABEL2ID_woRest) + 1

def plot_segments(gt, preds, accel, title):

    fig, ax = plt.subplots(figsize=(12,3))

    # plot each accel axis in the background
    accel = np.asarray(accel)  # shape (T,3)
    t = np.arange(accel.shape[0])
    for i, col in enumerate(['x','y','z']):
        trace = accel[:, i]
        # normalize each channel to [–0.5..0]
        lo, hi = trace.min(), trace.max()
        tn = (trace - lo) / (hi - lo + 1e-6) * 0.5 - 0.5
        ax.plot(t, tn, alpha=0.6, label=f"accel {col}" if i==0 else None)
    ax.legend(loc='upper right', fontsize='small')

    # ground truth bars
    for c, L, cx in gt:
        start, end = cx - L/2, cx + L/2
        ax.fill_between([start,end],[0,0],[1,1],
                        color='lightgray', alpha=0.7)
        ax.text((start+end)/2, 0.5, str(c),
                ha='center', va='center', color='black')

    # predicted bars
    for c, score, cx, L in preds:
        start, end = cx - L/2, cx + L/2
        ax.hlines(1.2, start, end, lw=4,
                  color=f'C{c%10}', alpha=(0.3+0.7*score))
        ax.text((start+end)/2, 1.25, str(c),
                ha='center', va='bottom',
                color=f'C{c%10}', fontsize=8)

    ax.set_ylim(-0.6, 1.4)
    ax.set_xlim(0, len(accel))
    ax.set_xlabel("Frame index")
    ax.set_yticks([])
    ax.set_title(title)
    ax.legend(loc="upper right")

    plt.tight_layout()
    plt.show()



def main():
    model = MTHARS(in_channels=3, num_classes=NUM_CLASSES).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()

    ds = SensorDataset(ROOT_DIR)
    print(len(ds))

    step = int(WINDOW_SIZE*(1-OVERLAP))
    

    for sample_idx in range(0,len(ds)):
        accel_all, _, gt_all = ds[sample_idx]

        print(sample_idx)
        accel_all, gyro_all, gt_all = ds[sample_idx]
        # convert GT to (id, length, center)
        gt = [(LABEL2ID_woRest[lbl], t_l, t_x) for (lbl, t_l, t_x) in gt_all]

        # slice sample into overlapping windows
        windows = []
        total = len(accel_all)
        end = 0
        while end < total:
            if end == 0 and end + WINDOW_SIZE <= total:
                start, stop = 0, WINDOW_SIZE
            elif end + (WINDOW_SIZE - step) <= total:
                start = end - step
                stop = end + (WINDOW_SIZE - step)
            else:
                start = max(0, total - WINDOW_SIZE)
                stop = total

            accel_w = accel_all[start:stop]
            windows.append((accel_w, start))
            end = stop
            if end == total:
                break

        # decode each window, collect proposals
        all_props = []
        for accel_w, window_start in windows:
            props = decode_windows(model, accel_w, window_start)
            all_props.extend(props)

        # NMS
        classes = torch.tensor([c for (c,_,_,_) in all_props])
        scores = torch.tensor([s for (_,s,_,_) in all_props])
        centers = torch.tensor([cx for (_,_,cx,_) in all_props])
        lengths = torch.tensor([L for (_,_,_,L) in all_props])

        keep = nms_1d(
            centers=centers,
            lengths=lengths,
            scores=scores,
            classes=classes,
            iou_threshold=0.00
        )
        final_preds = [ all_props[i] for i in keep.tolist() ]

        # plot 
        title = f"Sample {sample_idx}: GT vs. Pred"

        accel_arr = np.array(accel_all)  # shape (T,3)
        plot_segments(gt, final_preds, accel_arr, title)

        # if sample_idx >= 2:
        #     break

if __name__ == "__main__":
    main()
