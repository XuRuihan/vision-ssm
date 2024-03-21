import os
import sys

sys.path.append(os.getcwd())

import torch

from models.vision_ssm import Hydra

batch, dim, height, width = 2, 320, 5, 7
x = torch.randn(batch, dim, height, width).to("cuda")
model = Hydra(dim).to("cuda")
print(model)
y = model(x)
assert y.shape == x.shape
