import torch
import paddle



model = paddle.load('/home/nberardo/moco/output_dir/moco_v2_r50/moco_v2_r50_e200_ckpt.pdparams.1')
new_state_dict = {}
for name,weights in model["state_dict"].items():
    print(name)
    new_state_dict[name] = torch.from_numpy(weights.numpy())
#torch.save(new_state_dict, '/home/nberardo/moco/output_dir/moco_v2_r50/moco_v2_r50_e200_ckpt.pth')