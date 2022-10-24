import os
import matplotlib.pyplot as plt
import torch.utils.data.distributed
import torchvision.transforms as transforms

from PIL import Image
from torch.autograd import Variable
from tqdm import tqdm

classes = ('negative', 'neutral', 'positive')
transform_test = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.5143256, 0.40016153, 0.347045], [0.2527371, 0.21547443, 0.2000117])
])

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = torch.load("./savedModels/model_23_0.868.pth")
model.eval()
model.to(DEVICE)

path = 'test/2022_07_23_20_00_hanyanfei_M_36_VS/'
testList = os.listdir(path)
testList.sort()

negativeNum = 0
negativeList = []
neutralNum = 0
neutralList = []
positiveNum = 0
positiveList = []
emotionList = []

for file in tqdm(testList):
    img = Image.open(path + file)
    img = transform_test(img)
    img.unsqueeze_(0)
    img = Variable(img).to(DEVICE)
    out = model(img)
    # Predict
    _, pred = torch.max(out.data, 1)
    # print(out.data[0])
    # print('Image Name:{},predict:{}'.format(file, classes[pred.data.item()]))
    negativeList.append(out.data[0][0].tolist())
    neutralList.append(out.data[0][1].tolist())
    positiveList.append(out.data[0][2].tolist())
    if classes[pred.data.item()] == 'negative':
        negativeNum += 1
        emotionList.append(0)
    elif classes[pred.data.item()] == 'neutral':
        neutralNum += 1
        emotionList.append(1)
    elif classes[pred.data.item()] == 'positive':
        positiveNum += 1
        emotionList.append(2)

print('predict: negative:{} neutral:{} positive:{}'.format(negativeNum, neutralNum, positiveNum))

emotionChangeList = [emotionList[0]]
lastEmotion = emotionList[0]
tempCount = 0

for i in emotionList:
    if lastEmotion == i:
        tempCount += 1  # 相同情绪计数
    else:
        tempCount = 0  # 重新计数
    lastEmotion = i
    if tempCount > 2 and i != emotionChangeList[-1]:  # 达到微表情阈值 且 跟之前的情绪不同
        emotionChangeList.append(i)

for i in emotionChangeList:
    print(i, end=" ")

# (1, 2, 1) 其中1, 2表示图的位置分布为一行两列，1表示序号为1的图
plt.figure(figsize=(15, 7))  # 1500 * 700 px

plt.subplot(1, 2, 1)
plt.plot(range(len(negativeList)), negativeList, lw=2, c='b', alpha=0.6)
plt.plot(range(len(neutralList)), neutralList, lw=2, c='g', alpha=0.6)
plt.plot(range(len(positiveList)), positiveList, lw=2, c='r', alpha=0.6)
plt.legend(['negative', 'neutral', 'positive'])

plt.subplot(1, 2, 2)
plt.xlabel('frame')
plt.ylim(-1, 3)
plt.yticks([0, 1, 2], ['negative', 'neutral', 'positive'])
plt.plot(range(len(emotionList)), emotionList, lw=2, alpha=0.6)

plt.show()
