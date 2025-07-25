import os
import pandas as pd
import numpy as np
from glob import glob

DATA_DIR = './cleanedValDataCut'


def process_sample(sample_dir):
    """
    Load accel and label files from one sample directory,
    compute counts, per-label durations (frames), and push-off distances (frames).
    """
    labels_file = glob(os.path.join(sample_dir, 'labels_*.csv'))[0]

    labels = pd.read_csv(labels_file)

    labels['duration_s'] = labels['t_l'] 

    # total count per label in this sample
    counts = labels['label'].value_counts().to_dict()

    # list of durations per label
    durations_by_label = labels.groupby('label')['duration_s'].apply(list).to_dict()

    # compute push off distances: sorted t_x differences in seconds
    centers = labels.loc[labels['label'] == 'Push-off', 't_x'].sort_values().values
    if len(centers) >= 2:
        diffs_s = np.diff(centers)
        push_distances_s = diffs_s.tolist()
    else:
        push_distances_s = []

    return counts, durations_by_label, push_distances_s


def analyze_all():
    total_counts = {}
    durations_all_s = []
    durations_per_label_s = {}
    push_distances_all_s = []

    # iterate over each sample
    for d in os.listdir(DATA_DIR):
        sample_dir = os.path.join(DATA_DIR, d)
        if not os.path.isdir(sample_dir):
            continue

        counts, dur_by_lbl, push_dists_s = process_sample(sample_dir)

        # counts
        for lbl, cnt in counts.items():
            total_counts[lbl] = total_counts.get(lbl, 0) + cnt

        # durations
        for lbl, durs in dur_by_lbl.items():
            durations_per_label_s.setdefault(lbl, []).extend(durs)
            durations_all_s.extend(durs)

        # push off distances
        push_distances_all_s.extend(push_dists_s)

    # average duration per label
    avg_duration_per_label_s = {lbl: np.mean(durs) for lbl, durs in durations_per_label_s.items()}
    # total duration per label
    total_duration_per_label_s = {lbl: np.sum(durs) for lbl, durs in durations_per_label_s.items()}
    # Compute overall average duration
    avg_duration_all_s = np.mean(durations_all_s) if durations_all_s else 0
    # average push-off distance
    avg_push_distance_s = np.mean(push_distances_all_s) if push_distances_all_s else 0

    print("Total count of each label:")
    for lbl, cnt in total_counts.items():
        print(f"  {lbl}: {cnt}")

    print("\nTotal duration per label (frames):")
    for lbl, total_dur in total_duration_per_label_s.items():
        print(f"  {lbl}: {total_dur:.2f}")

    print("\nAverage duration per label (frames):")
    for lbl, dur in avg_duration_per_label_s.items():
        print(f"  {lbl}: {dur:.2f}")

    print(f"\nAverage duration across all labels (frames): {avg_duration_all_s:.2f}")
    print(f"Average distance between consecutive Push-off events (frames): {avg_push_distance_s:.2f}")


if __name__ == '__main__':
    analyze_all()
