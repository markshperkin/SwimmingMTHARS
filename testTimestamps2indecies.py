import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# color map for each label
colors = {
    'Freestyle':        '#e74c3c',
    'Backstroke':       '#3498db',
    'Butterfly':        '#f39c12',
    'Breaststroke':     '#2ecc71',
    'Underwater glide': '#1abc9c',
    'Underwater kick':  '#9b59b6',
    'Push-off':         '#e67e22',
    'Turn':             '#d35400',
    'Wall touch':       '#c0392b',
    'Rest':             '#7f8c8d'
}

root = './data'
sample = next(d for d in os.listdir(root) 
              if os.path.isdir(os.path.join(root, d)))
sample_dir = os.path.join(root, sample)

accel_df = pd.read_csv(glob.glob(os.path.join(sample_dir, 'accel_*.csv'))[0])
gyro_df = pd.read_csv(glob.glob(os.path.join(sample_dir, 'gyro_*.csv'))[0])
labels_df = pd.read_csv(glob.glob(os.path.join(sample_dir, 'labels_*.csv'))[0])

t0 = accel_df['timestamp'].iloc[0]
t_rel = accel_df['timestamp'] - t0

segments = []
for _, row in labels_df.iterrows():
    lbl         = row['label']
    center_time = row['t_x']
    half_time   = row['t_l'] / 2.0

    start_time = center_time - half_time
    end_time   = center_time + half_time

    start_idx = (t_rel - start_time).abs().idxmin()
    end_idx   = (t_rel - end_time).abs().idxmin()

    segments.append((lbl, start_idx, end_idx))

accel_df['timestamp'] = np.arange(len(accel_df))
gyro_df['timestamp']  = np.arange(len(gyro_df))

plt.figure(figsize=(12, 5))

plt.plot(accel_df['timestamp'], accel_df['x'], color='red',   label='Accel X')
plt.plot(accel_df['timestamp'], accel_df['y'], color='green', label='Accel Y')
plt.plot(accel_df['timestamp'], accel_df['z'], color='blue',  label='Accel Z')

seen = set()
for lbl, start, end in segments:
    c = colors.get(lbl, '#000000')
    if lbl not in seen:
        plt.axvspan(start, end, color=c, alpha=0.3, label=lbl)
        seen.add(lbl)
    else:
        plt.axvspan(start, end, color=c, alpha=0.3)

plt.xlabel('Sample Index')
plt.ylabel('Sensor Reading')
plt.title(f'Sample {sample}: Accel & Label Segments')
plt.legend(ncol=2, bbox_to_anchor=(1.05,1), loc='upper left')
plt.tight_layout()
plt.show()
