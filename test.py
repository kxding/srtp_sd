import torch 
import json
from diffusers import UNet2DConditionModel

if __name__=='__main__':
    #concept_cfg = 'vis.json'
    unet_model = UNet2DConditionModel.from_pretrained("../s", subfolder="unet")
    # unet_model.CrossAttention = torch.nn.Sequential(
    #     ('to_qv', torch.nn.Linear())
    # )
    print(unet_model)
    #model_path = '/home/dingkaixin/visual-concept-translator/output/concept_image_09_10_2023_1027/checkpoint-500/pytorch_model.bin'
    #model_path = '/home/dingkaixin/Mix-of-Show/experiments/EDLoRA_barbaracle_Anyv4_B4_Iter1K/models/net_g_1000.pth'
    #model_path = '../s/unet/diffusion_pytorch_model.bin'
    #model = torch.load(model_path)
    print(model)
 