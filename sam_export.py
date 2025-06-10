import torch
from segment_anything import sam_model_registry

# 无法正常导出，很多坑
sam = sam_model_registry["vit_b"](checkpoint="../model/sam_vit_b_01ec64.pth")
sam.eval()
scripted = torch.jit.script(sam)
scripted.save("../model/sam_vit_b_01ec64.pt")