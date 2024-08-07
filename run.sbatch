#!/bin/bash
#SBATCH --time=72:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=5
#SBATCH --gres=gpu:1
#SBATCH --job-name=python-ex
#SBATCH --mail-user=s319439@studenti.polito.it
#SBATCH --mail-type=ALL
#SBATCH --partition=cuda
#SBATCH --output=/home/nberardo/moco/train.log
#SBATCH --error=/home/nberardo/moco/train.log
###### 1 Load the module
module load nvidia/cudasdk
module load intel/python/3
# 2 Activate the virtual environment
source /home/nberardo/myenv/bin/activate
# 3 Run the python script
python /home/nberardo/moco/tools/train.py -c configs/moco/moco_v1_r50.yaml

