import torch
import torch.nn as nn
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from deepxplore import deepXplore
from utils import *
import os

model1 = torch.hub.load("chenyaofo/pytorch-cifar-models", "cifar10_resnet56", pretrained=True).eval() 
model2 = torch.hub.load("chenyaofo/pytorch-cifar-models", "cifar10_resnet44", pretrained=True).eval()

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])
testset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=True)

dxp = deepXplore([model1, model2], lambda_1=2, lambda_2=1, threshold=0.6, s=0.1)
def constraint(x): return x.sign() 

disagreement_inputs = [] 
total_disagreement_count = 0 

max_samples = 1000 

print(f"DeepXplore in progress (Sampling first {max_samples} images)...")
for i, (images, labels) in enumerate(testloader):
    if i >= max_samples:
        break
        
    gen_x = dxp.generate(images, constraint)
    
    with torch.no_grad():
        out1 = model1(gen_x).argmax(dim=1).item()
        out2 = model2(gen_x).argmax(dim=1).item()
    
    if out1 != out2:
        total_disagreement_count += 1
        if len(disagreement_inputs) < 5:
            disagreement_inputs.append((gen_x.cpu(), out1, out2))
        
        if total_disagreement_count % 5 == 0:
            print(f"Found {total_disagreement_count} disagreements...")

print("-" * 40)
print(f"Total Images Tested: {i}")
print(f"Total Disagreement-inducing Inputs Found: {total_disagreement_count}")
print(f"Final Neuron Coverage: {dxp.get_coverage()}")
print("-" * 40)

if not os.path.exists('results'):
    os.makedirs('results')

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
plt.figure(figsize=(15, 5)) 
for idx, (img, p1, p2) in enumerate(disagreement_inputs):
    img_disp = to_image(img) 
    plt.subplot(1, 5, idx+1)
    plt.imshow(img_disp)
    plt.title(f"M1: {classes[p1]}\nM2: {classes[p2]}")
    plt.axis('off')

plt.tight_layout()
plt.savefig('results/disagreement_results.png') 
print("Visualization saved to results/disagreement_results.png")
plt.show()