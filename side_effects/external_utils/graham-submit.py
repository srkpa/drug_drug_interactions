import os
import argparse

head = [
    "#!/bin/bash",
    "#SBATCH --nodes=1",
    "#SBATCH --ntasks-per-node=16",
    "#SBATCH --gres=gpu:p100:1",
    "#SBATCH --mem=15G",
    "#SBATCH --time=24:00:00",
    "#SBATCH --account=rrg-corbeilj-ac",
    "#SBATCH --mail-user=sewagnouin-rogia.kpanou.1@ulaval.ca",
    "#SBATCH --mail-type=BEGIN",
    "#SBATCH --mail-type=END",
    "#SBATCH --mail-type=FAIL",
    "#SBATCH --mail-type=REQUEUE",
    "#SBATCH --mail-type=ALL",
    "\n"
]


def submit_sh(config_path, output, expt_output, step=5, project="/home/maoss2/rog/side_effects", input="data/violette/", ):
    files = os.listdir(config_path)
    n_step = len(files) /step
    print(n_step)
    for i in range(0, len(files), step):
        f = files[i: i + step]
        with open(f"{output}-{int(i / step)}.sh", "w") as OUT:
            OUT.writelines("\n".join(head))
            for file in f[:-1]:
                if file.endswith("_config.json"):
                    OUT.write(f"python {project}/utils.py -p {os.path.join(config_path, file)} -o {expt_output} -i {project}/{input} & \n")

            OUT.write(f"python {project}/utils.py -p {os.path.join(config_path, f[-1])} -o {expt_output} -i {project}/{input}\n")
            OUT.write("wait")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dir', type=str, default="", help=""" The config dir.""")
    parser.add_argument('-o', '--output', type=str, help=""" The Output file.""")
    parser.add_argument('-s', '--step', type=int, help=""" The number of files per task.""")
    parser.add_argument('-i', '--input', type=str, help=""" The number of files per task.""")
    parser.add_argument('-e', '--expt', type=str, help=""" The expt Output folder.""")
    args = parser.parse_args()

    submit_sh(config_path=args.dir, output=args.output, step=args.step, input=args.input, expt_output=args.expt)
