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

root = './train_data'
sample = '49'
sample_dir = os.path.join(root, sample)

accel_df  = pd.read_csv(glob.glob(os.path.join(sample_dir, 'accel_*.csv'))[0])
labels_df = pd.read_csv(glob.glob(os.path.join(sample_dir, 'labels_*.csv'))[0])

accel_df['timestamp'] = accel_df['timestamp'] - accel_df['timestamp'].iloc[0]

plt.figure(figsize=(12,5))
plt.plot(accel_df['timestamp'], accel_df['x'], color='red',   label='Accel X')
plt.plot(accel_df['timestamp'], accel_df['y'], color='green', label='Accel Y')
plt.plot(accel_df['timestamp'], accel_df['z'], color='blue',  label='Accel Z')

seen = set()
for _, row in labels_df.iterrows():
    lbl    = row['label']
    center = row['t_x']
    length = row['t_l']
    half   = length // 2
    start  = center - half
    end    = center + half
    color  = colors.get(lbl)
    if lbl not in seen:
        plt.axvspan(start, end, color=color, alpha=0.3, label=lbl)
        seen.add(lbl)
    else:
        plt.axvspan(start, end, color=color, alpha=0.3)

plt.xlabel('Timestamp (sample index)')
plt.ylabel('Accelerometer Reading')
plt.title(f'Sample {sample}: Accel with Label Segments')
plt.legend(ncol=2, bbox_to_anchor=(1.05,1), loc='upper left')
plt.tight_layout()
plt.show()


