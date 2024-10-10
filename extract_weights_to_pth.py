import torch
import paddle

model = paddle.load('/home/nberardo/moco/moco_v2_r50_clas.pdparams')
new_state_dict = {}
print("moco")
for name,weights in model["state_dict"].items():
    print(name)
    # if name.startswith('backbone.'):
    #     name = name.replace('backbone.','')
    #     new_state_dict[name] = torch.from_numpy(weights.numpy())
print("")
model = torch.load('/home/nberardo/vicreg/experiment/resnet50.pth')
for name,weights in model["state_dict"].items():
    print(name)
# torch.save(new_state_dict, '/home/nberardo/moco/moco_v2_r50_clas.pth')