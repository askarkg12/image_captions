import torch
import torch.nn as nn
import torchvision.models
import PIL
import matplotlib.pyplot as plt


class Looker(nn.Module):
    def __init__(self):
        super().__init__()
        self.weights = torchvision.models.ViT_B_16_Weights.IMAGENET1K_V1
        self.model = torchvision.models.vit_b_16(weights=self.weights)

        # self.e = self.model.encoder
        self.p = self.weights.transforms()

        self.img_embeds = None

        self._register_hooks()

    def _register_hooks(self):
        self.model.encoder.layers[-1].register_forward_hook(self._hook)

        # self.model.encoder.layers[0].reg

    def _hook(self, module, input, output):
        self.img_embeds = output[:, 1:, :]

    def forward(self, x):
        prd = self.model(x)
        return self.img_embeds, prd


l = Looker()

img_og = PIL.Image.open("dog.png").convert("RGB")
img_2 = PIL.Image.open("image.png").convert("RGB")
img = l.p(img_og).unsqueeze(0)

to_pil = torchvision.transforms.ToPILImage()

x = to_pil(img.squeeze(0))


h, prd = l(img)

print("Shapes:", h.shape, prd.shape)


_, idx = prd.max(dim=1)

name = l.weights.meta["categories"][idx.item()]

print("Prediction:", name)

pass
