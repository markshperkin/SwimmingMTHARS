# plot_results.py

import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt

def config_label_from_filename(fname: str) -> str:
    """Heuristic: if 'gyro' in name → 'Accel+Gyro', else 'Accel'."""
    lower = fname.lower()
    if "gyro" in lower:
        return "Accel + Gyro"
    return "Accel Only"

def main():
    parser = argparse.ArgumentParser(
        description="Plot training & validation acc/IoU from two CSVs"
    )
    parser.add_argument(
        "csv_files",
        nargs=2,
        help="Two CSV files (e.g. accel.csv accelGyro.csv) containing 'epoch','train acc','train mean IoU','val acc','val IoU'"
    )
    args = parser.parse_args()
    csv_files = args.csv_files

    # pick two distinct colors
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    cfg_colors = [colors[0], colors[1]]

    plt.figure(figsize=(10,6))

    for idx, csv_path in enumerate(csv_files):
        if not os.path.isfile(csv_path):
            print(f"⚠️  File not found: {csv_path}")
            continue

        df = pd.read_csv(csv_path)
        required = {"epoch","train acc","train mean IoU","val acc","val IoU"}
        if not required.issubset(df.columns):
            print(f"⚠️  Skipping {csv_path}, missing: {required - set(df.columns)}")
            continue

        epoch     = df["epoch"]
        train_acc = df["train acc"]
        train_iou = df["train mean IoU"]
        val_acc   = df["val acc"]
        val_iou   = df["val IoU"]
        loss     = df["loss"]  # uncomment when desired

        base     = os.path.splitext(os.path.basename(csv_path))[0]
        cfg_name = config_label_from_filename(base)
        color    = cfg_colors[idx]

        # plot with one color per config
        # plt.plot(epoch, val_acc,   label=f"{cfg_name} val acc",  color=color, linestyle="-")
        # plt.plot(epoch, val_iou,   label=f"{cfg_name} val IoU",  color=color, linestyle="--")
        # plt.plot(epoch, train_acc, label=f"{cfg_name} train acc",color=color, linestyle="-.")
        # plt.plot(epoch, train_iou, label=f"{cfg_name} train IoU",color=color, linestyle=":")
        plt.plot(epoch, loss,     label=f"{cfg_name} loss",     color=color, linestyle=":")

    plt.xlabel("Epoch")
    plt.ylabel("Value")
    plt.title("Train Loss")
    plt.legend(fontsize="small", loc="best", ncol=2)
    plt.grid(True)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
