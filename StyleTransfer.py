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
