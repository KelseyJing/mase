# # Original code: https://raw.githubusercontent.com/pytorch/vision/main/torchvision/models/vgg.py
# # Paper code https://github.com/Thinklab-SJTU/twns/blob/2c192c38559ffe168139c9e519a053c295ca1313/cls/litenet.py#L86

# import torch
# import torch.nn as nn
# import argparse
# import torch.optim as optim
# import torchvision
# import torchvision.transforms as transforms
# from torch.utils.data import DataLoader

# class VGG7(nn.Module):
#     def __init__(self, num_classes: int) -> None:
#         super().__init__()
#         self.feature_layers = nn.Sequential(
#             nn.Conv2d(3, 128, kernel_size=3, padding=1),
#             nn.BatchNorm2d(128, momentum=0.9),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(128, 128, kernel_size=3, padding=1),
#             nn.BatchNorm2d(128, momentum=0.9),
#             nn.ReLU(inplace=True),
#             nn.MaxPool2d(kernel_size=2),
#             nn.Conv2d(128, 256, kernel_size=3, padding=1),
#             nn.BatchNorm2d(256, momentum=0.9),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(256, 256, kernel_size=3, padding=1),
#             nn.BatchNorm2d(256, momentum=0.9),
#             nn.ReLU(inplace=True),
#             nn.MaxPool2d(kernel_size=2),
#             nn.Conv2d(256, 512, kernel_size=3, padding=1),
#             nn.BatchNorm2d(512, momentum=0.9),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(512, 512, kernel_size=3, padding=1),
#             nn.BatchNorm2d(512, momentum=0.9),
#             nn.ReLU(inplace=True),
#             nn.MaxPool2d(kernel_size=2),
#         )
#         self.classifier = nn.Sequential(
#             nn.Linear(8192, 1024),
#             nn.ReLU(inplace=True),
#             nn.Linear(1024, 1024),
#             nn.ReLU(inplace=True),
#         )

#         self.last_layer = nn.Linear(1024, num_classes)

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         x = self.feature_layers(x)
#         x = x.view(-1, 512 * 4 * 4)
#         x = self.classifier(x)
#         x = self.last_layer(x)
#         return x
    

# # 初始化模型、损失函数和优化器
# model = VGG7(num_classes=10)
# criterion = nn.CrossEntropyLoss()
# optimizer = optim.Adam(model.parameters(), lr=0.001)

# # 训练循环
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# model.to(device)


# # def get_vgg7(info, pretrained=False) -> VGG7:
# #     image_size = info.image_size
# #     num_classes = info.num_classes
# #     return VGG7(image_size, num_classes)


# def train_one_epoch(model, trainloader, device, optimizer, criterion):
#     model.train()
#     running_loss = 0.0
#     correct = 0
#     total = 0
#     for i, (inputs, labels) in enumerate(trainloader):
#         inputs, labels = inputs.to(device), labels.to(device)

#         optimizer.zero_grad()

#         outputs = model(inputs)
#         loss = criterion(outputs, labels)
#         loss.backward()
#         optimizer.step()

#         running_loss += loss.item()

#         _, predicted = torch.max(outputs.data, 1)
#         total += labels.size(0)
#         correct += (predicted == labels).sum().item()

#     epoch_loss = running_loss / len(trainloader)
#     epoch_acc = 100 * correct / total
#     return epoch_loss, epoch_acc

# def evaluate(model, testloader, device, criterion):
#     model.eval()
#     running_loss = 0.0
#     correct = 0
#     total = 0
#     with torch.no_grad():
#         for data in testloader:
#             images, labels = data[0].to(device), data[1].to(device)
#             outputs = model(images)
#             loss = criterion(outputs, labels)

#             running_loss += loss.item()

#             _, predicted = torch.max(outputs.data, 1)
#             total += labels.size(0)
#             correct += (predicted == labels).sum().item()

#     test_loss = running_loss / len(testloader)
#     test_acc = 100 * correct / total
#     return test_loss, test_acc

# def main(args):
#     transform = transforms.Compose([
#         transforms.ToTensor(),
#         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
#     ])

#     trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
#     trainloader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True)

#     testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
#     testloader = DataLoader(testset, batch_size=args.batch_size, shuffle=False)

#     device = torch.device(args.device if torch.cuda.is_available() else "cpu")

#     model = VGG7(num_classes=10).to(device)
#     criterion = nn.CrossEntropyLoss()
#     optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    

#     for epoch in range(args.epochs):
#         _, train_acc = train_one_epoch(model, trainloader, device, optimizer, criterion)
#         _, test_acc = evaluate(model, testloader, device, criterion)
        
#         print(f'Epoch {epoch+1}, Train Acc: {train_acc:.2f}%, Test Acc: {test_acc:.2f}%')
#         # train_loss, train_acc = train_one_epoch(model, trainloader, device, optimizer, criterion)
#         # test_loss, test_acc = evaluate(model, testloader, device, criterion)
        
#         # print(f'Epoch {epoch+1}, Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%')

# # 其余代码保持不变

#     # 训练循环
#     # for epoch in range(args.epochs):
#     #     model.train()
#     #     running_loss = 0.0
#     #     for i, (inputs, labels) in enumerate(trainloader):
#     #         inputs, labels = inputs.to(args.device), labels.to(args.device)

#     #         optimizer.zero_grad()

#     #         outputs = model(inputs)
#     #         loss = criterion(outputs, labels)
#     #         loss.backward()
#     #         optimizer.step()

#     #         running_loss += loss.item()
        
#     #     print(f'Epoch {epoch+1}, Loss: {running_loss/len(trainloader)}')

#     # 简单测试循环，计算测试集上的准确率
#     model.eval()
#     correct = 0
#     total = 0
#     with torch.no_grad():
#         for data in testloader:
#             images, labels = data
#             images, labels = images.to(args.device), labels.to(args.device)
#             outputs = model(images)
#             _, predicted = torch.max(outputs.data, 1)
#             total += labels.size(0)
#             correct += (predicted == labels).sum().item()
    
#     print(f'Accuracy of the network on the 10000 test images: {100 * correct / total}%')

# if __name__ == '__main__':
#     parser = argparse.ArgumentParser(description='Train VGG model on CIFAR-10')
#     parser.add_argument('--epochs', type=int, default=10, help='Number of epochs to train')
#     parser.add_argument('--batch_size', type=int, default=64, help='Batch size for training')
#     parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate for optimizer')
#     parser.add_argument('--save_path', type=str, default='vgg_cifar10.pth', help='Path to save the trained model')
#     parser.add_argument('--device', type=str, choices=['cpu', 'cuda'], default='cuda' if torch.cuda.is_available() else 'cpu', help='Device to train on')
#     args = parser.parse_args()
#     main(args)

import torch
import torch.nn as nn


class VGG7(nn.Module):
    def __init__(self, image_size: list[int], num_classes: int) -> None:
        super().__init__()
        self.feature_layers = nn.Sequential(
            nn.Conv2d(image_size[0], 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128, momentum=0.9),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128, momentum=0.9),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256, momentum=0.9),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256, momentum=0.9),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512, momentum=0.9),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512, momentum=0.9),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
        )
        self.classifier = nn.Sequential(
            nn.Linear(8192, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 1024),
            nn.ReLU(inplace=True),
        )

        self.last_layer = nn.Linear(1024, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.feature_layers(x)
        x = x.view(-1, 512 * 4 * 4)
        x = self.classifier(x)
        x = self.last_layer(x)
        return x


def get_vgg7(info, pretrained=False) -> VGG7:
    image_size = info.image_size
    num_classes = info.num_classes
    return VGG7(image_size, num_classes)