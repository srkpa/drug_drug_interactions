#!/bin/bash
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
##SBATCH --gres=gpu:p100:1
#SBATCH --mem=32G
#SBATCH --time=72:00:00
#SBATCH --account=rrg-corbeilj-ac
#SBATCH --array=0-6
#SBATCH --mail-user=sewagnouin-rogia.kpanou.1@ulaval.ca
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --mail-type=REQUEUE
#SBATCH --mail-type=ALL

date
SECONDS=0
which python
# source $HOME/venv3/bin/activate
# project_dir=
graham_dispatcher run --exp_name multitask --hpid $SLURM_ARRAY_TASK_ID
diff=$SECONDS
echo "$(($diff / 60)) minutes and $(($diff % 60)) seconds elapsed."
date
