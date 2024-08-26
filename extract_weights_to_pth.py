import torch
import paddle

#CONVERTIRE il weights.pd senza backbone.


model = paddle.load('/home/nberardo/moco/output_dir/moco_v1_r50/weights.pd')
new_state_dict = {}
"""for name,weights in model["state_dict"].items():
    if name.startswith("module.encoder."):
        name = name.replace("module.encoder.", "")
        new_state_dict[name] = weights
torch.save(new_state_dict, 'resnet50.pth')"""
for x in model:
    print(x)

"""for x, item in model.items():
    if x.startswith("backbone."):
        x=x.replace("backbone.","")
        new_state_dict[x]=item
paddle.save(new_state_dict, '/home/nberardo/moco/output_dir/moco_v1_r50/resnet50_extracted.pd')"""