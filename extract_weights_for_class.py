import torch
import paddle

model = paddle.load('/home/nberardo/moco/output_dir/moco_v1_r50/weights.pd')
new_state_dict = {}

for x, item in model.items():
    if x.startswith("backbone."):
        x=x.replace("backbone.","")
        new_state_dict[x]=item
paddle.save(new_state_dict, '/home/nberardo/moco/output_dir/moco_v1_r50/resnet50_extracted.pd')