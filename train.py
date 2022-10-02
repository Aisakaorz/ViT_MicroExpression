import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.datasets as datasets
import torchvision.transforms as transforms

from torchtoolbox.tools import mixup_data, mixup_criterion
from torchtoolbox.transform import Cutout
from tqdm import tqdm

import timm.models as models

# 设置全局参数
LR = 1e-4
BATCH_SIZE = 32
EPOCHS = 100
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
ACC = 0

if __name__ == '__main__':
    # 数据预处理
    transform_train = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.RandomHorizontalFlip(),
        # transforms.RandomVerticalFlip(),
        Cutout(),
        transforms.ToTensor(),
        transforms.Normalize([0.5325965, 0.43569186, 0.39240554], [0.23322931, 0.21355365, 0.20632775])
    ])
    transform_test = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.5325965, 0.43569186, 0.39240554], [0.23322931, 0.21355365, 0.20632775])
    ])

    # 读取数据
    dataset_train = datasets.ImageFolder('dataset/train', transform=transform_train)
    dataset_test = datasets.ImageFolder("dataset/val", transform=transform_test)

    # 导入数据
    train_loader = torch.utils.data.DataLoader(dataset_train, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset_test, batch_size=BATCH_SIZE, shuffle=False)

    # 实例化模型并且移动到GPU
    criterion = nn.CrossEntropyLoss()
    model = models.vit_base_patch16_224(pretrained=True)
    num_ftrs = model.head.in_features
    model.head = nn.Linear(num_ftrs, 3, bias=True)
    nn.init.xavier_uniform_(model.head.weight)
    model.to(DEVICE)

    # 选择简单暴力的Adam优化器，学习率调低
    adam = optim.Adam(model.parameters(), lr=LR)
    optimizer = optim.lr_scheduler.CosineAnnealingLR(optimizer=adam, T_max=20, eta_min=1e-9)

    for epoch in range(1, EPOCHS + 1):
        # 训练
        model.train()
        train_sum_loss = 0
        loop = tqdm(enumerate(train_loader), total=len(train_loader), leave=True)
        for batch_idx, (data, target) in loop:
            data, target = data.to(DEVICE, non_blocking=True), target.to(DEVICE, non_blocking=True)
            data, labels_a, labels_b, lam = mixup_data(data, target)
            adam.zero_grad()
            output = model(data)
            loss = mixup_criterion(criterion, output, labels_a, labels_b, lam)
            loss.backward()
            adam.step()
            lr = adam.state_dict()['param_groups'][0]['lr']
            train_sum_loss += loss.data.item()
            # 更新信息
            loop.set_description(f'Epoch [{epoch}/{EPOCHS}] Batch')
            loop.set_postfix(train_loss=loss.item(),
                             train_avg_loss=train_sum_loss / len(train_loader),
                             lr=lr)
        optimizer.step()

        # 验证
        model.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in tqdm(test_loader):
                data, target = data.to(DEVICE), target.to(DEVICE)
                output = model(data)
                loss = criterion(output, target)
                _, pred = torch.max(output.data, 1)
                correct += torch.sum(pred == target)
                print_loss = loss.data.item()
                test_loss += print_loss
            correct = correct.data.item()
            acc = correct / len(test_loader.dataset)
            avgloss = test_loss / len(test_loader)
            print('Val set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
                avgloss, correct, len(test_loader.dataset), 100 * acc))
            if acc > ACC:
                torch.save(model, 'model_' + str(epoch) + '_' + str(round(acc, 3)) + '.pth')
                ACC = acc
