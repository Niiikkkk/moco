import torch
import paddle

model = paddle.load('/home/nberardo/moco/output_dir/moco_v1_r50/epoch_200.pd')
new_state_dict = {}
"""for name,weights in model["state_dict"].items():
    if name.startswith("module.encoder."):
        name = name.replace("module.encoder.", "")
        new_state_dict[name] = weights
torch.save(new_state_dict, 'resnet50.pth')"""
for x,item in model["state_dict"].items():
    new_state_dict[x] = item.numpy()
torch.save(new_state_dict, '/home/nberardo/moco/output_dir/moco_v1_r50/resnet50.pth')