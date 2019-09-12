import argparse
import os

head = ["#!/bin/bash", "\n"]


def submit_sh(scripts_path="/home/maoss2/srkpa/side_effects/scripts", output="/home/maoss2/srkpa/side_effects/submit", step=100):

    files = os.listdir(scripts_path)
    n_step = len(files) / step
    print(n_step)
    for i in range(0, len(files), step):
        f = files[i: i + step]
        with open(f"{output}-{int(i / step)}.sh", "w") as OUT:
            OUT.writelines("\n".join(head))
            for file in f:
                if file.endswith(".sh"):
                    OUT.write(f"sbatch {os.path.join(scripts_path, file)} \n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dir', type=str, default="", help=""" The config dir.""")
    parser.add_argument('-o', '--output', type=str, help=""" The Output file.""")
    args = parser.parse_args()

    submit_sh(scripts_path=args.dir, output=args.output)
