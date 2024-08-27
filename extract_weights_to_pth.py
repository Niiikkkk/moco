import torch
import paddle

model = paddle.load('/home/nberardo/moco/output_dir/moco_v1_r50/resnet50_extracted.pd')
new_state_dict = {}
for name,weights in model.items():
    new_state_dict[name] = weights
torch.save(new_state_dict, 'resnet50_extracted.pth')