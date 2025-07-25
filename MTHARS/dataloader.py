import os
import glob
import pandas as pd
import torch
from torch.utils.data import Dataset
import random

LABEL2ID = {
    'Freestyle':        1,
    'Backstroke':       2,
    'Butterfly':        3,
    'Breaststroke':     4,
    'Underwater glide': 5,
    'Underwater kick':  6,
    'Push-off':         7,
    'Turn':             8,
    'Wall touch':       9,
    'Rest':            10
}

LABEL2ID_woRest = {
    'Freestyle':        1,
    'Backstroke':       2,
    'Butterfly':        3,
    'Breaststroke':     4,
    'Underwater glide': 5,
    'Underwater kick':  6,
    'Push-off':         7,
    'Turn':             8,
    'Wall touch':       9
}

class SensorDataset(Dataset):

    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.samples = [
            d for d in os.listdir(root_dir)
            if os.path.isdir(os.path.join(root_dir, d))
        ]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            return [ self[i] for i in range(*idx.indices(len(self))) ]

        sample_dir = os.path.join(self.root_dir, self.samples[idx])
        accel_df = pd.read_csv(glob.glob(f"{sample_dir}/accel_*.csv")[0])
        gyro_df = pd.read_csv(glob.glob(f"{sample_dir}/gyro_*.csv")[0])

        T = min(len(accel_df), len(gyro_df))
        accel_df, gyro_df = accel_df.iloc[:T], gyro_df.iloc[:T]

        accel_list = accel_df[['x','y','z']].values.tolist()
        gyro_list = gyro_df [['x','y','z']].values.tolist()

        labels_df = pd.read_csv(glob.glob(f"{sample_dir}/labels_*.csv")[0])
        labels_list = sorted([
            (row['label'], int(row['t_l']), int(row['t_x']))
            for _, row in labels_df.iterrows()
        ], key=lambda tup: tup[2])

        return [accel_list, gyro_list, labels_list]

    def getWindows(self, window_size=450, overlap=0.5):
        """
        Slide fixed length windows (with 50% overlap)
        over every sample in the dataset and collect those windows
        plus any labels fully contained in each window.
        """
        step = int(window_size*(1-overlap))
        all_samples = self[:]
        windows = []
        for accel_list, gyro_list, labels_list in all_samples:
            
            total, end = len(accel_list), 0
            while end < total:
                if end == 0 and end + window_size <= total:
                    start, stop = 0, window_size
                elif end + (window_size - step) <= total:
                    start = end - step
                    stop = end + (window_size - step)
                else:
                    start = max(0, total - window_size)
                    stop = total

                end = stop

                accel_w = accel_list[start:stop]
                gyro_w = gyro_list[start:stop]

                window_labels = []
                for label, t_l, t_x in labels_list:
                    half = t_l // 2
                    if (t_x - half) >= start and (t_x + half) <= stop:
                        adj_center = t_x - start
                        window_labels.append((label, t_l, adj_center))

                if not window_labels:
                    # no ground-truth in this window -> skip it
                    end = stop
                    if end == total:
                        break
                    else:
                        continue

                windows.append([accel_w, gyro_w, window_labels])

                if end == total:
                    break

        return windows

    def getBatches(self,
                   window_size=450,
                   overlap=0.5,
                   reduction=3,
                   batch_size=8,
                   shuffle=True):

        # window creation
        raw = self.getWindows(window_size, overlap)

        # match to feature frame and numeric IDs
        matched = []
        for accel_w, gyro_w, labels in raw:
            new_labels = []
            for lbl, t_l, t_x in labels:
                feat_l = t_l // reduction
                feat_x = t_x // reduction
                class_id = LABEL2ID.get(lbl, 0)
                new_labels.append((class_id, feat_l, feat_x))
            matched.append([accel_w, gyro_w, new_labels])

        # shuffle
        if shuffle:
            random.shuffle(matched)

        # chunk into smaller batches
        batches = [
            matched[i:i+batch_size]
            for i in range(0, len(matched), batch_size)
        ]

        if batches and len(batches[-1]) < batch_size:
            batches = batches[:-1]
        return batches



if __name__ == "__main__":
    ds = SensorDataset('../cleanedTrainData')
    batches = ds.getBatches(
        window_size=450,
        overlap=0.5,
        reduction=3,
        batch_size=8,
        shuffle=True
    )
    print(f"Total windows: {sum(len(b) for b in batches)}")
    print(f"Number of mini batches of size 8: {len(batches)}")
    # inspect first mini-batch
    first = batches[0]
    print("First mini-batch contains", len(first), "windows")

    first = batches[0]
    for widx, (accel_w, gyro_w, labels) in enumerate(first):
        L = len(accel_w)
        print(f"\n Window {widx} has length {L} frames and labels:")
        for lbl, t_l, t_x in labels:
            half  = t_l // 2
            start = t_x - half
            end   = t_x + half
            valid = (start >= 0) and (end <= L)
            print(f"class={lbl:2d}  t_l={t_l:3d}  t_x={t_x:3d}  -> range [{start:3d},{end:3d}]  valid={valid}")

    ds = SensorDataset('../cleanedTrainData')
    b1 = ds.getBatches(batch_size=4, shuffle=True)
    b2 = ds.getBatches(batch_size=4, shuffle=True)

    print([win[0][0][0] for win in b1[0]]) 
    print([win[0][0][0] for win in b2[0]])

