from torchvision import models, transforms
import torch
from vitcnn import SmartBinSegregationModel
'''
Post processing, we can reduce bit-quantisation
''' 

#just paste the model script here if you want a deployable stage.

model = SmartBinSegregationModel()

model.eval()
model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
model = torch.quantization.prepare(model)
model = torch.quantization.convert(model)