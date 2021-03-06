# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.11.1
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Changelog
#
# ## Season 1, Episode 01
# ...
#
# ## Season 1, Episode 02
# ...
#
# ## Season 1, Episode 03
# ...
#
# ## Season 1, Episode 04
#
# Tue May  4 21:09:45 CEST 2021
#
# Finished adding ContentLoss and StyleLoss layers in the model.
#
# Thus, we've completed the model \o/
#
#
# ## Season 1, Episode 05
#
# Thu May 20 19:53:52 CEST 2021
#
#

# %%
import numpy as np

# %%
import torch

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
device = torch.device(DEVICE)

# %%
PIC_SIZE = 512 if torch.cuda.is_available() else 128

# %%
from torchvision import transforms

loader = transforms.Compose([
    transforms.Resize(PIC_SIZE),
    transforms.ToTensor(),
])


# %%
from PIL import Image

krakow = Image.open("./pics/krakow.jpg")
picasso = Image.open('./pics/picasso.jpg')

# %%
npkrakow = np.array(krakow)
krakowtensor = torch.Tensor(npkrakow)

krakowloaded = loader(krakow)
krakowtensor.shape

# %%
cztery = krakowloaded.unsqueeze(0)
cztery = cztery.to(device, dtype=torch.float)


# %%
def imgloader(fp) -> torch.Tensor:
    img = Image.open(fp)
    loaded = loader(img) 
    loaded = loaded.unsqueeze(0)  # [..., np.newaxis]
    loaded_with_device = loaded.to(device, dtype=torch.float)
    return loaded_with_device


# %%
picasso = imgloader('./pics/picasso.jpg')
krakow = imgloader('./pics/krakow.jpg')
krakow.shape

# %%
unloader = transforms.ToPILImage()

# %%
import matplotlib.pyplot as plt
# %matplotlib inline

plt.ion()
picasso.shape


# %%
def showtensor(tensor: torch.Tensor) -> None:
    tensor = tensor.cpu().clone()
    unloaded = unloader(tensor.squeeze(0))
    plt.figure()
    plt.imshow(unloaded)
    
showtensor(picasso)
showtensor(krakow)

# %%
## pytorch module
import torch.nn as nn
import torch.nn.functional as F

# Ltotal(~p,~a,~x) =??Lcontent(~p,~x) +??Lstyle(~a,~x)

class ContentLoss(nn.Module):
    
    def __init__(self, p: torch.Tensor):
        nn.Module.__init__(self)
        self.p = p
        self._loss = None
        
    def forward(self, x) -> torch.Tensor:
        # Lcontent(???p,???x, l) = 1/2 ??? i,j (Flij ??? Plij)??.
        # return .5 * sum((x - p)**2)
        self._loss = F.mse_loss(x, self.p)
        return x

# t1 = torch.Tensor([2, 2])
# t2 = torch.Tensor([4, 4])
# ContentLoss(t1).forward(t2)


# %%
def gram(t: torch.Tensor) -> torch.Tensor:
    a, b, c, d = t.size()
    g = t.view(a*c, b*d)
    return g @ g.T / (a*b*c*d)
    
gram(picasso)

class StyleLoss(nn.Module):
    
    def __init__(self, p: torch.Tensor):
        nn.Module.__init__(self)
        self.p = gram(p).detach()
        self._loss = None
        
    def forward(self, x) -> torch.Tensor:
        # Lstyle(~a,~x) = ??? wl E
        self._loss = F.mse_loss(gram(x), self.p)
        return x
    
StyleLoss(picasso).forward(krakow)

# %%
from torchvision.models import vgg19

foo = vgg19(pretrained=True, progress=True)
network = foo.features.to(device).eval()


# %%
class Normalisation(nn.Module):
    def __init__(self):
        self.cnn_normalization_mean = torch.tensor([0.485, 0.456, 0.406]).to(device)
        self.cnn_normalization_std = torch.tensor([0.229, 0.224, 0.225]).to(device)
        
    def forward(self, x):
        return (x - self.cnn_normlization_mean) / self.cnn_normlization_std


# %%
model = nn.Sequential()
conv_counter = 0
style_losses, content_losses = [], []
for n, x in network.named_children():        

    
    if isinstance(x, nn.ReLU):
        model.add_module(name=n, module=nn.ReLU(inplace=False))
        
    elif isinstance(x, nn.Conv2d):
        # 4 -> Content
        # 1,2,3,4,5 -> Style
        
        conv_counter += 1
        print(n, conv_counter)

        model.add_module(name=n, module=x)
        
        if conv_counter == 4:
            # print("CONTENT")
            p_l  = model(krakow).detach()
            content_loss = ContentLoss(p_l)
            model.add_module(name="ContentLoss", module=content_loss)
            content_losses.append(content_loss)
            max_content = n
            
        if conv_counter in [1,2,3,4,5]:
            # print("STYLE")
            a_l  = model(picasso).detach()
            style_loss = StyleLoss(a_l)
            model.add_module(name=f"StyleLoss{n}_{conv_counter}", module=style_loss)
            style_losses.append(style_loss)
            max_style = n
            
            # NOTE: this works because it's bigger than '4' in Content xD
            if conv_counter == 5:
                break

    else:
        model.add_module(name=n, module=x)

model

# %%
krakows = []


# %%
def weighted_sum(styles, contents):
    style = sum(s._loss for s in styles)
    content = sum(c._loss for c in contents)
    
    return style * 10e6 + content * 1


# w = weighted_sum(style_losses, content_losses)

whatever = 10

optimizer = torch.optim.LBFGS([krakow.requires_grad_()])
optimizer

for _ in range(whatever):
    
    def closure():
        krakow.data.clamp_(0, 1)
        optimizer.zero_grad()
        output = model(krakow)
        loss = weighted_sum(style_losses, content_losses)
        print(loss)
        loss.backward()
        return loss
    
    optimizer.step(closure)
    krakow.data.clamp_(0, 1)
    
krakows.append(krakow.clone().detach())

# %%
for krk in krakows:
    showtensor(krk)
