import torch
import torchvision.transforms
from PIL import Image
from model_test import *


device = torch.device("cuda:0")
image_path = "dog.png"
image = Image.open(image_path)
print(image)
image = image.convert('RGB')

transform = torchvision.transforms.Compose([torchvision.transforms.Resize((32, 32)),
                                            torchvision.transforms.ToTensor()])

image = transform(image)
image = image.to(device)
print(image.shape)


# 采用第一种方法保存的模型 使用第一种方法加载
model = torch.load("model_9.pth")

print(model)
image = torch.reshape(image, (1, 3, 32, 32))
model.eval()
with torch.no_grad():
    output = model(image)
print(output)


print(output.argmax(1))
