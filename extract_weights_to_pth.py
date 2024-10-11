import torch
import paddle

model = paddle.load('/home/nberardo/moco/output_dir/moco_v1_r50/epoch_200.pd')
new_state_dict = {}
for name,weights in model["state_dict"].items():
    if name.startswith('backbone.'):
        name = name.replace('backbone.','')
        name = name.replace('_mean', 'running_mean')
        name = name.replace('_variance', 'running_var')
        new_state_dict[name] = torch.from_numpy(weights.numpy())
        print(name)
torch.save(new_state_dict, '/home/nberardo/moco/output_dir/moco_v1_r50/epoch_pth.pd')