#!/bin/bash
#SBATCH --time=72:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:4
#SBATCH --job-name=moco
#SBATCH --mail-user=s319439@studenti.polito.it
#SBATCH --mail-type=END
#SBATCH --partition=cuda
#SBATCH --output=/home/nberardo/moco/train.log
#SBATCH --error=/home/nberardo/moco/train.err
###### 1 Load the module
module load nvidia/cudasdk
module load intel/python/3
# 2 Activate the virtual environment
source /home/nberardo/myenv/bin/activate
# 3 Run the python script
#Linear classifier
#python tools_v110/train.py -c configs/moco/moco_clas_r50.yaml --pretrained /home/nberardo/moco/output_dir/moco_v1_r50/weights.pd
#Extract weights
#python tools_v110/passl2ppclas/convert.py --type res50 --checkpoint /home/nberardo/moco/output_dir/moco_v1_r50/latest.pd --output /home/nberardo/moco/output_dir/moco_v1_r50/weights.pdparams
#python tools_v110/extract_weight.py /home/nberardo/moco/output_dir/moco_v1_r50/latest.pd --output /home/nberardo/moco/output_dir/moco_v1_r50/weights.pd
#Train
#python /home/nberardo/moco/tools_v110/train.py -c configs/moco/moco_v1_r50.yaml --resume /home/nberardo/moco/output_dir/moco_v1_r50/epoch_195.pd

python extract_weights_to_pth.py
