#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=16
##SBATCH --gres=gpu:p100:1
#SBATCH --mem=32G
#SBATCH --time=72:00:00
#SBATCH --account=rrg-corbeilj-ac
#SBATCH --mail-user=sewagnouin-rogia.kpanou.1@ulaval.ca
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --mail-type=REQUEUE
#SBATCH --mail-type=ALL

conda activate base
#bash launcher.sh -e $1 -s gra
python $HOME/drug_drug_interactions/drug_drug_interactions/side_effects/utility/exp_utils
#deactivate
