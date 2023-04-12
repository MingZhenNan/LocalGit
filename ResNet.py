import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms

# 定义ResNet块
from torchvision.transforms import Compose


class ResNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResNetBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels))
        else:
            self.shortcut = nn.Sequential()

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        identity = self.shortcut(identity)

        out += identity
        out = self.relu(out)

        return out


# 定义ResNet网络
class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=10):
        super(ResNet, self).__init__()

        self.in_channels = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)

        self.layer1 = self._make_layer(block, 64, layers[0], stride=1)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)

    def _make_layer(self, block, out_channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []

        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels

        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)

        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        out = self.fc(out)

        return out


# 定义ResNet18
def resnet18(num_classes):
    return ResNet(ResNetBlock, [2, 2, 2, 2], num_classes)


# 定义ResNet50
def resnet50(num_classes):
    return ResNet(ResNetBlock, [3, 4, 6, 3], num_classes)


# 定义ResNet34
def resnet34(num_classes):
    return ResNet(ResNetBlock, [3, 4, 6, 3], num_classes)


# 定义ResNet101
def resnet101(num_classes):
    return ResNet(ResNetBlock, [3, 4, 23, 3], num_classes)


# 定义ResNet152
def resnet152(num_classes):
    return ResNet(ResNetBlock, [3, 8, 36, 3], num_classes)


# 训练函数
def train(model, data_loader, criterion, optimizer):
    model.train()
    train_loss = 0.0
    train_acc = 0.0

    for inputs, targets in data_loader:
        inputs, targets = inputs.cuda(), targets.cuda()

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item() * inputs.size(0)
        train_acc += torch.sum(torch.argmax(outputs, dim=1) == targets).item()

    train_loss /= len(data_loader.dataset)
    train_acc /= len(data_loader.dataset)

    return train_loss, train_acc


# 测试函数
def test(model, data_loader, criterion):
    model.eval()
    test_loss = 0.0
    test_acc = 0.0

    with torch.no_grad():
        for inputs, targets in data_loader:
            inputs, targets = inputs.cuda(), targets.cuda()

            outputs = model(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item() * inputs.size(0)
            test_acc += torch.sum(torch.argmax(outputs, dim=1) == targets).item()

        test_loss /= len(data_loader.dataset)
        test_acc /= len(data_loader.dataset)

    return test_loss, test_acc


if __name__ == '__main__':
    transform_train = Compose([transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    transform_test = Compose([transforms.ToTensor(),
                              transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    train_set = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    test_set = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)

    # 定义数据加载器
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=128, shuffle=True, num_workers=2)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=128, shuffle=False, num_workers=2)

    # 定义模型、损失函数和优化器
    model = resnet50(num_classes=10).cuda()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)

    # 开始训练和测试
    num_epochs = 1
    pth = './moder/resnet50_cifar10.pth'
    if not os.path.exists(pth):
        for epoch in range(num_epochs):
            train_loss, train_acc = train(model, train_loader, criterion, optimizer)
            print(f"Epoch [{epoch + 1}/{num_epochs}] \n Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")

            test_loss, test_acc = test(model, test_loader, criterion)
            print(f" Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}")
        # 保存模型参数
        torch.save(model.state_dict(), pth)
    else:
        model.load_state_dict(torch.load(pth))
        test_loss, test_acc = test(model, test_loader, criterion)
        print(f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}")
