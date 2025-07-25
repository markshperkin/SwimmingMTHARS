import os
import glob
import numpy as np
import pandas as pd

data_dir = './train_data'
clean_dir = './cleanedTrainData'

os.makedirs(clean_dir, exist_ok=True)

# iterate over each sample directory
for sample in os.listdir(data_dir):
    sample_dir = os.path.join(data_dir, sample)
    if not os.path.isdir(sample_dir):
        continue
    # create parallel clean subdir
    target_subdir = os.path.join(clean_dir, sample)
    os.makedirs(target_subdir, exist_ok=True)

    accel_file = glob.glob(os.path.join(sample_dir, 'accel_*.csv'))[0]
    gyro_file  = glob.glob(os.path.join(sample_dir, 'gyro_*.csv'))[0]
    labels_file= glob.glob(os.path.join(sample_dir, 'labels_*.csv'))[0]

    accel_df = pd.read_csv(accel_file)
    gyro_df  = pd.read_csv(gyro_file)
    labels_df= pd.read_csv(labels_file)

    # build relative-time array for matching
    t0    = accel_df['timestamp'].iloc[0]
    t_rel = accel_df['timestamp'] - t0

    # convert labels to sample indices
    centers = []
    lengths = []
    ends = []
    for _, row in labels_df.iterrows():
        # label times are in same units as accel timestamps
        center_time = row['t_x']
        half_len    = row['t_l'] / 2.0
        start_time  = center_time - half_len
        end_time    = center_time + half_len
        # nearest-neighbor to find sample indices
        start_idx = (t_rel - start_time).abs().idxmin()
        end_idx   = (t_rel - end_time).abs().idxmin()

        ends.append(end_idx)
        # recompute center and length in samples
        center_idx = int((start_idx + end_idx) / 2)
        length_idx = int(end_idx - start_idx)
        centers.append(center_idx)
        lengths.append(length_idx)
    
    last_end = max(ends) + 1
    accel_df = accel_df.iloc[:last_end].reset_index(drop=True)
    gyro_df  = gyro_df.iloc[:last_end].reset_index(drop=True)

    # overwrite timestamps with sample indices
    accel_df['timestamp'] = np.arange(len(accel_df))
    gyro_df['timestamp']  = np.arange(len(gyro_df))

    # update labels t_x, t_l
    labels_df['t_x'] = centers
    labels_df['t_l'] = lengths

    accel_name = os.path.basename(accel_file)
    gyro_name  = os.path.basename(gyro_file)
    labels_name= os.path.basename(labels_file)

    accel_df.to_csv(os.path.join(target_subdir, accel_name), index=False)
    gyro_df.to_csv(os.path.join(target_subdir, gyro_name),   index=False)
    labels_df.to_csv(os.path.join(target_subdir, labels_name),index=False)

print(f"Cleaned data written to {clean_dir}")
