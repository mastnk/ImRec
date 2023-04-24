
# https://colab.research.google.com/github/YutaroOgawa/pytorch_tutorials_jp/blob/main/notebook/1_Learning%20PyTorch/1_4_cifar10_tutorial_jp.ipynb
# https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import torch.optim.lr_scheduler as sch

import torch.backends.cudnn as cudnn

import time

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print( device )

batch_size = 32
epochs = 10

# training data preparation
transform = transforms.Compose(
    [transforms.RandomHorizontalFlip(),
     transforms.RandomCrop(32, padding=4),
     transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)
testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# network definition
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.conv1 = nn.Conv2d(3, 16, 3, padding=1, padding_mode='replicate')
        self.conv1_bn = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1, padding_mode='zeros')
        self.conv2_bn = nn.BatchNorm2d(32)

        self.pool = nn.MaxPool2d(2, 2)
        self.flatten = nn.Flatten()

        self.fc1 = nn.Linear(32 * 8 * 8, 128)
        self.fc1_bn = nn.BatchNorm1d(128)
        self.fc2 = nn.Linear(128, 64)
        self.fc2_bn = nn.BatchNorm1d(64)
        self.fc3 = nn.Linear(64, 10)

        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.conv1(x) # (B,3,32,32) -> (B,16,32,32)
        x = self.conv1_bn(x)
        x = F.relu(x)
        x = self.pool(x) # (B,16,32,32) -> (B,16,16,16)

        x = self.conv2(x) # (B,16,16,16) -> (B,32,16,16)
        x = self.conv2_bn(x)
        x = F.relu(x)
        x = self.pool(x) # (B,32,16,16) -> (B,32,8,8)

        x = self.flatten(x) # (B,32,8,8) -> (B,32*8*8)

        x = self.fc1(x) # (B,32*8*8) -> (B,128)
        x = self.fc1_bn(x)
        x = F.relu(x)

        x = self.fc2(x) # (B,128) -> (B,64)
        x = self.fc2_bn(x)
        x = F.relu(x)

        x = self.dropout(x)
        x = self.fc3(x) # (B,64) -> (B,32)

        return x

net = Net()
net = net.to(device)
if device == 'cuda':
    cudnn.benchmark = True
    print( 'Run with GPU' )
else:
    print( 'Run with CPU' )

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
scheduler = sch.StepLR(optimizer, step_size=1, gamma=0.9)

net.train()
t0 = time.perf_counter()
for epoch in range(epochs):

    running_loss = 0.0
    for i, data in enumerate(trainloader):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        if i % 100 == 99:
            t1 = time.perf_counter()
            print('[%d, %5d, %.2e] loss: %.3f, time: %.3f' %
                  (epoch + 1, i + 1, optimizer.param_groups[0]['lr'], running_loss / 2000, t1-t0))
            running_loss = 0.0
            t0 = t1
    scheduler.step()

print('Finished Training')

net.eval()
correct = 0
total = 0
with torch.no_grad():
    for data in testloader:
        images, labels = data
        images, labels = images.to(device), labels.to(device)
        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print('Accuracy of the network on the 10000 test images: %.3f %%' % (100 * correct / total))
