import torch
import matplotlib.pyplot as plt
import numpy as np
import time

#Download CIFAR-10
from torchvision import datasets
from torchvision import transforms
data_path = '../data-unversioned/p1ch7/'
cifar10 = datasets.CIFAR10(data_path, train=True, download=True, transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.4915, 0.4823, 0.4468),(0.2470, 0.2435, 0.2616))]))
cifar10_val = datasets.CIFAR10(data_path, train=False, download=True, transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.4915, 0.4823, 0.4468),(0.2470, 0.2435, 0.2616))]))

#cifar10_tensor = datasets.CIFAR10(data_path, train=True, download=False, transform=transforms.ToTensor())

# Limit number of classes (Build Dataset)
label_map = {6: 0, 7:1, 8:2, 9: 3}
class_names = ['frog', 'horse', 'ship', 'truck']
cifar4 = [(img, label_map[label])
        for img, label in cifar10
        if label in [6,7,8,9]]
cifar4_val = [(img, label_map[label])
        for img, label in cifar10_val
        if label in [6,7,8,9]]

#Normalize

import torch.nn as nn
import torch.nn.functional as F
n_out = 4;

#Net
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 8, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(8 * 8 * 8, 32)
        self.fc2 = nn.Linear(32, n_out)
        
    def forward(self, x):
        out = F.max_pool2d(torch.tanh(self.conv1(x)), 2)
        out = F.max_pool2d(torch.tanh(self.conv2(out)), 2)
        out = out.view(-1, 8 * 8 * 8)
        out = torch.tanh(self.fc1(out))
        out = self.fc2(out)
        return out

# Model
model = Net()

#Calculate loss
loss=nn.NLLLoss()

#Test output
#img, label = cifar2[0]
#out = model(img.view(-1).unsqueeze(0))
#print(loss(out, torch.tensor([label])))

#Train
import torch.optim as optim
import datetime
device = (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
print(f"Training on device {device}")

train_loader = torch.utils.data.DataLoader(cifar4, batch_size=64, shuffle=True)
learning_rate = 1e-2
optimizer = optim.SGD(model.parameters(), lr=learning_rate)
loss_fn = nn.CrossEntropyLoss()
total_loss=np.array([])
n_epochs = 200

def training_loop(n_epochs, optimizer, model, loss_fn, train_loader, total_loss):
    for epoch in range(1, n_epochs+1):
        loss_train=0.0
        for imgs, labels in train_loader:
            imgs=imgs.to(device=device)
            labels=labels.to(device=device)
            outputs = model(imgs)
            loss = loss_fn(outputs, labels)

            l2_lambda = 0.001
            l2_norm = sum(p.pow(2.0).sum()
                    for p in model.parameters())
            loss = loss+l2_lambda*l2_norm

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_train += loss.item()
        total_loss = np.append(total_loss, float(loss_train/len(train_loader)))
        if epoch == 1 or epoch %10 == 0:
            #total_loss = np.append(total_loss, float(loss_train/len(train_loader)))
            print('{} Epoch: {}, Loss {}'.format(datetime.datetime.now(), epoch, float(loss_train/len(train_loader))))
    return total_loss

model = Net().to(device=device)
start = time.time()
total_loss = training_loop(n_epochs, optimizer, model, loss_fn, train_loader,total_loss)
finish = time.time()
print('Training time: %f' % (finish - start))

#Validate
import collections
val_loader = torch.utils.data.DataLoader(cifar4_val, batch_size=64, shuffle=False)

all_acc_dict = collections.OrderedDict()

def validate(model, train_loader, val_loader):
    accdict={}
    for name, loader in [("train", train_loader), ("val", val_loader)]:
        correct = 0
        total = 0
        
        with torch.no_grad():
            for imgs, labels in loader:
                imgs = imgs.to(device=device)
                labels = labels.to(device=device)
                outputs = model(imgs)
                _, predicted = torch.max(outputs, dim=1)
                total += labels.shape[0]
                correct += int((predicted == labels).sum())

        print("Accuracy {}: {:.2f}".format(name, correct/total))
        accdict[name] = correct/total
    return accdict

numel_list = [p.numel()
        for p in model.parameters()
        if p.requires_grad == True]
print(sum(numel_list), numel_list)

all_acc_dict["baseline"] = validate(model, train_loader, val_loader)
#plot loss
plt.plot(range(len(total_loss)),total_loss,color="blue")
plt.title("Model Loss")
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.show()
plt.savefig('problem2a_loss.png')
