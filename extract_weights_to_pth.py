import torch
import paddle



model = paddle.load('/home/nberardo/moco/output_dir/moco_v1_r50/epoch_200.pd')
new_state_dict = {}
for name,weights in model.items():
    print(name)
    #name = "backbone." + name
    #new_state_dict[name] = torch.from_numpy(weights.numpy())
#torch.save(new_state_dict, '/home/nberardo/moco/output_dir/moco_v2_r50/resnet50_extracted.pth')
