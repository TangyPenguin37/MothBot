import torch

model = torch.hub.load('pytorch/vision:v0.10.0', 'alexnet', pretrained=True)

torch.save(model, "alexnet.pt")