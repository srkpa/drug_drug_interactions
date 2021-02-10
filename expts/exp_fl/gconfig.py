import hashlib
import json
import os
import shutil
import subprocess
import tempfile
from os.path import expanduser

import click
from sklearn.model_selection import ParameterGrid

EXP_FOLDER_RESUTS = f"{expanduser('~')}/expts"


@click.command()
def generate_config():
    subprocess.run(["python", f"{os.getcwd()}/config.py"])
    out_path = os.path.join(EXP_FOLDER_RESUTS, os.path.basename(os.getcwd()), 'configs')
    os.makedirs(out_path, exist_ok=True)
    with open('configs.json', 'r') as fd:
        task_config = json.load(fd)
    task_grid = list(ParameterGrid(task_config))
    task_grid = {
        hashlib.sha1(json.dumps(task, sort_keys=True).encode()).hexdigest(): task
        for task in task_grid
    }
    print(f"- Experiment has {len(task_grid)} different tasks:")
    with open(f"{out_path}/configs.txt", "w") as cf:
        for task_id in task_grid:
            temp_file = tempfile.NamedTemporaryFile(mode='w')
            with open(temp_file.name, 'w') as f:
                json.dump(task_grid[task_id], f)
            shutil.copy(temp_file.name, f"{out_path}/{task_id}.json")
            assert os.path.exists(f"{out_path}/{task_id}.json")
            temp_file.close()
            cf.write(f"{out_path}/{task_id}.json\n")


if __name__ == '__main__':
    generate_config()
