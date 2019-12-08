#!/bin/bash
#SBATCH --cpus-per-task=16
#SBATCH --mem=16000M
#SBATCH --time=6:00:00
#SBATCH --account=rrg-corbeilj-ac
#SBATCH --array=3-86
#SBATCH --mail-user=prudencio@invivoai.com

date
SECONDS=0
which python
# source $HOME/venv3/bin/activate
# project_dir=
graham_dispatcher run --exp_name multitask --hpid $SLURM_ARRAY_TASK_ID -e /home/ptossou/anaconda3/bin/train
diff=$SECONDS
echo "$(($diff / 60)) minutes and $(($diff % 60)) seconds elapsed."
date
