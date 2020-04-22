#!/bin/bash
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
##SBATCH --gres=gpu:v100:1
#SBATCH --mem=32G
#SBATCH --time=42:00:00
#SBATCH --account=rrg-corbeilj-ac
#SBATCH --array=0-2
##SBATCH --mail-user=sewagnouin-rogia.kpanou.1@ulaval.ca
##SBATCH --mail-type=ALL

date
SECONDS=0

graham_dispatcher run --exp_name $1 --hpid $SLURM_ARRAY_TASK_ID
diff=$SECONDS
echo "$(($diff / 60)) minutes and $(($diff % 60)) seconds elapsed."
date
