import torch
import torch.nn as nn
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from deepxplore import deepXplore
from utils import *

# Load pretrained ResNet models for CIFAR-10
model1 = torch.hub.load("chenyaofo/pytorch-cifar-models", "cifar10_resnet56", pretrained=True).eval() 
model2 = torch.hub.load("chenyaofo/pytorch-cifar-models", "cifar10_resnet44", pretrained=True).eval()

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])
testset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=True)

# Initialize DeepXplore with hyperparameters
# lambda_1: weight for maximizing neuron coverage, lambda_2: weight for maximizing model disagreement
dxp = deepXplore([model1, model2], lambda_1=2, lambda_2=1, threshold=0.6, s=0.1)
def constraint(x): return x.sign() 

disagreement_inputs = [] 

print("DeepXplore in progress...")
for i, (images, labels) in enumerate(testloader):
    gen_x = dxp.generate(images, constraint)
    
    with torch.no_grad():
        out1 = model1(gen_x).argmax(dim=1).item()
        out2 = model2(gen_x).argmax(dim=1).item()
    
    # Check if models disagree on the generated input
    if out1 != out2:
        print(f"[Found] Input {i}: Model 1 Prediction({out1}) != Model 2 Prediction({out2})")
        disagreement_inputs.append((gen_x.cpu(), out1, out2))
    
    # Stop once we find 5 disagreement cases
    if len(disagreement_inputs) >= 5:
        break

print(f"Final Neuron Coverage: {dxp.get_coverage()}")

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
plt.figure(figsize=(15, 10))
for idx, (img, p1, p2) in enumerate(disagreement_inputs):
    img_disp = to_image(img) 
    plt.subplot(1, 5, idx+1)
    plt.imshow(img_disp)
    plt.title(f"M1: {classes[p1]}\nM2: {classes[p2]}")
    plt.axis('off')

plt.tight_layout()
plt.savefig('disagreement_results.png')
plt.show()