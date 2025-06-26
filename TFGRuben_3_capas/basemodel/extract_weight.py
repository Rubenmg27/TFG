import os
import sys
sys.path = [
    '../input/efficientnet-pytorch/EfficientNet-PyTorch/EfficientNet-PyTorch-master',
] + sys.path
sys.path = [
    '../input/ttach-kaggle/ttach/',
] + sys.path

import argparse

# Manejo de argumentos de línea de comandos
parser = argparse.ArgumentParser(description="Selecciona el modelo a usar")
parser.add_argument('--model_idx', type=int, choices=[0, 1, 2, 3, 4], default=0,
                    help="Índice del modelo a utilizar (0, 1, 2, 3 o 4)")

args = parser.parse_args()
import ttach as tta
import skimage.io
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import torchvision
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from efficientnet_pytorch import model as enet

import matplotlib.pyplot as plt
from tqdm import tqdm

# Argumento de línea de comando para seleccionar el modelo
parser = argparse.ArgumentParser(description="Selecciona el modelo a usar")
parser.add_argument('--model_idx', type=int, choices=[0, 1, 2, 3, 4], default=0,
                    help="Índice del modelo a utilizar (0, 1, 2, 3 o 4)")
args = parser.parse_args()

# Configuración del dispositivo
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Definición de la clase de modelo simplificada solo para cargar y extraer pesos
class enetv3(nn.Module):
    def __init__(self, modelname, out_dim=5):
        super(enetv3, self).__init__()

        from efficientnet_pytorch import EfficientNet
        self.basemodel = EfficientNet.from_name(modelname)
        self.myfc = nn.Linear(self.basemodel._fc.in_features, out_dim)
        self.basemodel._fc = nn.Identity()
        self.basemodel._avg_pooling = nn.AdaptiveAvgPool2d(1)  # GeM()
    def extract(self, x):
        return self.basemodel(x)
    def forward(self, x):
        x = self.basemodel(x)
        x = self.myfc(x)
        return x

def load_models(model_files):
    models = []

    for model_f in model_files:
        print(model_f)
        model_f = os.path.join("../../kaggle/input/latesubspanda", model_f)
        if "eff" in model_f:
            model = enetv3(modelname, out_dim=5)
        model.load_state_dict(torch.load(model_f))
        model.eval()
        model.to(device)
        models.append(model)
        print(f'{model_f} loaded!')
    return models


# Configuración del modelo y archivo
modelname = "efficientnet-b0"
model_files = [
    "efficientnet-b0famlabelsmodelsub_avgpool_tile36_imsize256_mixup_final_epoch20_fold0.pth",
    "efficientnet-b0famlabelsmodelsub_avgpool_tile36_imsize256_mixup_final_epoch20_fold1.pth",
    "efficientnet-b0famlabelsmodelsub_avgpool_tile36_imsize256_mixup_final_epoch20_fold2.pth",
    "efficientnet-b0famlabelsmodelsub_avgpool_tile36_imsize256_mixup_final_epoch20_fold3.pth",
    "efficientnet-b0famlabelsmodelsub_avgpool_tile36_imsize256_mixup_final_epoch20_fold4.pth"
]
selected_model_file = model_files[args.model_idx]
print(f"Usando el modelo {selected_model_file}")

# Cargar el modelo y los pesos
models = load_models([selected_model_file])
model = models[0]


print(model)

# Alternativamente, imprimir cada capa y sus dimensiones de peso
for name, param in model.named_parameters():
    if param.requires_grad:
        print(f'Layer: {name} | Size: {param.size()} ')  # Muestra los primeros dos valores

#exit()
# Extraer los pesos de la capa `_conv_head`
project_conv_weights = model.basemodel._blocks[15]._project_conv.weight.detach().cpu().numpy()

# Guardar los pesos y sesgos en archivos separados
np.save("/home/ruben/TFGRuben_3_capas/Data/hook/_project_conv_head_weights.npy", project_conv_weights)
#np.save("/home/ruben/TFGRuben/Data/hook/conv_head_bias.npy", conv_head_bias)
print("Pesos de la capa _conv_head guardados en /Data/hook/")
