import torch
import paddle

model = paddle.load('/home/nberardo/moco/moco_v1_r50_clas.pdparams')
new_state_dict = {}
for name,weights in model["state_dict"].items():
    if name.startswith('backbone.'):
        name = name.replace('backbone.','')
        name = name.replace('_mean', 'running_mean')
        name = name.replace('_variance', 'running_var')
        new_state_dict[name] = torch.from_numpy(weights.numpy())
torch.save(new_state_dict, '/home/nberardo/moco/moco_v1_r50_downloaded.pth')