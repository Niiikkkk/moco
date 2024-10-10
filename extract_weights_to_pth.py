import torch
import paddle

model = paddle.load('/home/nberardo/moco/moco_v2_r50_clas.pdparams')
new_state_dict = {}
print("moco")
for name,weights in model["state_dict"].items():
    if name.startswith('backbone.'):
        name = name.replace('backbone.','')
        name = name.replace('_mean', 'running_mean')
        name = name.replace('_variance', 'running_var')
        new_state_dict[name] = torch.from_numpy(weights.numpy())
    print(name)
print("")
model = torch.load('/home/nberardo/vicreg/experiment/resnet50.pth')
for name,weights in model.items():
    print(name)
# torch.save(new_state_dict, '/home/nberardo/moco/moco_v2_r50_clas.pth')