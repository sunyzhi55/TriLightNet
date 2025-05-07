from torch.nn import CrossEntropyLoss
from torch.optim import *
from Net import *
# from MDL_Net.MDL_Net import generate_model
# from RLAD_Net.taad import get_model
from loss_function import joint_loss, loss_in_IMF
from utils.basic import get_scheduler
from thop import profile, clever_format
# from torchsummary import summary
# compute the flops and params

# IMF
# mri_demo = torch.ones(1, 1, 96, 128, 96)
# pet_demo = torch.ones(1, 1, 96, 128, 96)
# cli_demo = torch.ones(1, 9)
# model_demo = Interactive_Multimodal_Fusion_Model()
# model_demo.eval()
# flops, params = profile(model_demo, inputs=(mri_demo, pet_demo, cli_demo,), verbose=False)
# flops, params = clever_format([flops, params], "%.3f")
# params_result = f'flops: {flops}, params: {params}'
# # summary_result = summary(model_demo, (3, 224, 224))
# print(f"IMF:{params_result}")


# HFBSurv
# mri_demo = torch.ones(2, 1, 96, 128, 96)
# pet_demo = torch.ones(2, 1, 96, 128, 96)
# cli_demo = torch.ones(2, 9)
# model_demo = HFBSurv()
# model_demo.eval()
# result = model_demo(mri_demo, pet_demo, cli_demo)
# print("result", result.shape)
# flops, params = profile(model_demo, inputs=(mri_demo, pet_demo, cli_demo,), verbose=False)
# flops, params = clever_format([flops, params], "%.3f")
# params_result = f'flops: {flops}, params: {params}'
# print(f"HFBSurv:{params_result}")
#
# # Resnet
# mri_demo = torch.ones(1, 1, 96, 128, 96)
# pet_demo = torch.ones(1, 1, 96, 128, 96)
# model_demo = ResnetMriPet()
# model_demo.eval()
# inputs = torch.cat([mri_demo, pet_demo], dim=1)
# flops, params = profile(model_demo, inputs=(inputs,), verbose=False)
# flops, params = clever_format([flops, params], "%.3f")
# params_result = f'flops: {flops}, params: {params}'
# print(f"Resnet:{params_result}")

# AweSomeNet
mri_demo = torch.ones(1, 1, 96, 128, 96)
pet_demo = torch.ones(1, 1, 96, 128, 96)
cli_demo = torch.ones(1, 9)
model_demo = AweSomeNet()
model_demo.eval()
flops, params = profile(model_demo, inputs=(mri_demo, pet_demo, cli_demo,), verbose=False)
flops, params = clever_format([flops, params], "%.3f")
params_result = f'flops: {flops}, params: {params}'
print(f"AweSomeNet:{params_result}")
"""
IMF:flops: 70.925G, params: 67.843M
MDL:flops: 9.353G, params: 2.827M
RLAD:flops: 260.882G, params: 30.624M
result torch.Size([2, 2])
HFBSurv:flops: 141.849G, params: 34.123M
mri shape: torch.Size([1, 1, 96, 128, 96])
pet shape: torch.Size([1, 1, 96, 128, 96])
Resnet:flops: 70.924G, params: 66.952M


AweSomeNet:flops: 10.517G, params: 17.405M
"""