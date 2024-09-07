import torch
import paddle

model = paddle.load('/home/nberardo/moco/output_dir/moco_v2_r50/resnet50_extracted.pd')
new_state_dict = {}
for x in model:
    print(x)
for name,weights in model.items():
    new_state_dict[name] = weights
print("")
for x in new_state_dict:
    print(x)
#torch.save(new_state_dict, '/home/nberardo/moco/output_dir/moco_v2_r50/resnet50_extracted.pth')