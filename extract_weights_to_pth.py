import torch
import paddle


"""
model = paddle.load('/home/nberardo/moco/output_dir/moco_v1_r50/resnet50_extracted.pd')
new_state_dict = {}
for name,weights in model.items():
    new_state_dict[name] = torch.from_numpy(weights.numpy())
torch.save(new_state_dict, '/home/nberardo/moco/output_dir/moco_v1_r50/resnet50_extracted.pth')"""


model = torch.load('/home/nberardo/moco/output_dir/moco_v1_r50/resnet50_extracted.pth')

for name,weights in model.items():
    print(name)