#!/bin/bash
#SBATCH --time=72:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --job-name=mocov2
#SBATCH --mail-user=s319439@studenti.polito.it
#SBATCH --mail-type=END
#SBATCH --partition=cuda
#SBATCH --output=/home/nberardo/moco/trainv2.log
#SBATCH --error=/home/nberardo/moco/trainv2.err
###### 1 Load the module
module load nvidia/cudasdk
module load intel/python/3
# 2 Activate the virtual environment
source /home/nberardo/myenv/bin/activate
# 3 Run the python script
#Linear classifier
  #Train
  #python tools_v110/train.py -c configs/moco/moco_clas_r50.yaml --pretrained /home/nberardo/moco/output_dir/moco_v2_r50/resnet50_extracted.pd --resume /home/nberardo/moco/output_dir/moco_v2_r50/moco_clas_r50/epoch_58.pd
  #Eval
  #python tools_v110/train.py -c configs/moco/moco_clas_r50.yaml --load /home/nberardo/moco/output_dir/moco_v2_r50/moco_clas_r50/epoch_100.pd --evaluate-only

#Extract weights
  #First extract the weights from the checkpoint. This will extract the weights related to the "backbone"
  #python tools_v110/extract_weight.py /home/nberardo/moco/output_dir/moco_v2_r50/epoch_200.pd --output /home/nberardo/moco/output_dir/moco_v2_r50/weights.pd

  #This one converts the extracted "backbone" extracted previously into a specific format usable for classification
  #python extract_weights_for_class.py

  #This one converts the .pd into .pth
  python extract_weights_to_pth.py


#Train
  #python /home/nberardo/moco/tools_v110/train.py -c configs/moco/moco_v1_r50.yaml --resume /home/nberardo/moco/output_dir/moco_v1_r50/epoch_195.pd


