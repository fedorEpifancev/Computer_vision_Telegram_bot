import torch 
import torchvision 
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import bot_conv_img
train = True

transform = transforms.Compose(
    [transforms.ToTensor(), 
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
)
batch_size_train = 4
batch_size_test = 4


trainset = torchvision.datasets.CIFAR10(root='./data', train = True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size = batch_size_train, shuffle = True, num_workers = 0)

testset = torchvision.datasets.CIFAR10(root='./data', train = False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size = batch_size_test, shuffle = True, num_workers = 0)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
classes_translate = ('самолёт', 'машина', 'птица', 'кошка', 'олень', 'собака', 'лягушка', 'лошадь', 'корабль', 'машина')




class Net(nn.Module):
    def __init__(self):
        super().__init__()
    
        self.batch_norm0 = torch.nn.BatchNorm2d(3)

        self.conv1 = torch.nn.Conv2d(3, 16, 3, padding=1)
        self.act1  = torch.nn.ReLU()
        self.pool1 = torch.nn.MaxPool2d(2, 2)
        
        self.conv2 = torch.nn.Conv2d(16, 32, 3, padding=1)
        self.act2  = torch.nn.ReLU()
        self.pool2 = torch.nn.MaxPool2d(2, 2)
        
        self.conv3 = torch.nn.Conv2d(32, 64, 3, padding=1)
        self.act3  = torch.nn.ReLU()

        self.fc1   = torch.nn.Linear(8 * 8 * 64, 256)
        self.act4  = torch.nn.Tanh()
        
        self.fc2   = torch.nn.Linear(256, 64)
        self.act5  = torch.nn.Tanh()
        
        self.fc3   = torch.nn.Linear(64, 10)

    def forward(self, x):
    
        x = self.batch_norm0(x)
        x = self.conv1(x)
        x = self.act1(x)
        x = self.pool1(x)
        
        x = self.conv2(x)
        x = self.act2(x)
        x = self.pool2(x)
        
        x = self.conv3(x)
        x = self.act3(x)
        
        x = x.view(x.size(0), x.size(1) * x.size(2) * x.size(3))
        x = self.fc1(x)
        x = self.act4(x)
        x = self.fc2(x)
        x = self.act5(x)
        x = self.fc3(x)
        return x
    
net = Net()

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

def train():
    for epoch in range(15):
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            
            inputs, labels = data
            optimizer.zero_grad()

        
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        
            running_loss += loss.item()
            if i % 1000 == 999:    
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 1000:.3f}')
                running_loss = 0.0

    print('Finished Training')
def test(img_float_tensor):
    train = False
    output_tensor = net(img_float_tensor)
    output_index = torch.argmax(output_tensor)
    print(output_index)
    output_pred = classes_translate[output_index]
    output_phrase = f"Я думаю на этой картинке есть {output_pred}"
    return output_phrase
