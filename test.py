# -*- coding = utf-8 -*-
# @File : test.py
# @Software : PyCharm
import torch.utils.data.distributed
import torchvision.transforms as transforms
from PIL import Image
from torch.autograd import Variable
import os

classes = ('anger', 'contempt', 'disgust', 'fear', 'happiness', 'neutrality', 'sadness', 'surprise')
transform_test = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5242406, 0.5242406, 0.5242406], [0.19945478, 0.19945478, 0.19945478])
])

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = torch.load("model.pth")
model.eval()
model.to(DEVICE)

path = 'test/'
testList = os.listdir(path)
for file in testList:
    img = Image.open(path + file)
    img = transform_test(img)
    img.unsqueeze_(0)
    img = Variable(img).to(DEVICE)
    out = model(img)
    # Predict
    _, pred = torch.max(out.data, 1)
    print('Image Name:{},predict:{}'.format(file, classes[pred.data.item()]))
