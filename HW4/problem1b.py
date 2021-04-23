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
#imgs = torch.stack([img_t for img_t, _ in cifar10_tensor], dim=3)
#print(imgs.shape)

# Model
import torch.nn as nn
n_out = 4

model = nn.Sequential(
        nn.Linear(
            3072,#Input features
            1024,#Hidden layer size
            ),
        nn.Tanh(),
        nn.Linear(
            1024,#Input features
            512,#Hidden layer size
            ),
        nn.Tanh(),
        nn.Linear(
            512,#Hidden layer size
            n_out,#output classes
            ),
        nn.LogSoftmax(dim=1))
#Calculate loss
loss=nn.NLLLoss()

#Test output
#img, label = cifar2[0]
#out = model(img.view(-1).unsqueeze(0))
#print(loss(out, torch.tensor([label])))

#Train
import torch.optim as optim
train_loader = torch.utils.data.DataLoader(cifar4, batch_size=64, shuffle=True)
learning_rate = 1e-2
optimizer = optim.SGD(model.parameters(), lr=learning_rate)
loss_fn = nn.NLLLoss()
total_loss=np.array([])
n_epochs = 200
start = time.time()
for epoch in range(n_epochs):
    for imgs, labels in train_loader:
        batch_size = imgs.shape[0]
        outputs = model(imgs.view(batch_size,-1))
        loss = loss_fn(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    total_loss = np.append(total_loss, float(loss))
    print("Epoch: %d, Loss %f" % (epoch, float(loss)))

finish = time.time()
print('Training time: %f' % (finish - start))
#Validate
val_loader = torch.utils.data.DataLoader(cifar4_val, batch_size=64, shuffle=False)

correct = 0
total = 0

with torch.no_grad():
    for imgs, labels in val_loader:
        batch_size = imgs.shape[0]
        outputs = model(imgs.view(batch_size, -1))
        _, predicted = torch.max(outputs, dim=1)
        total += labels.shape[0]
        correct += int((predicted == labels).sum())

print("Accuracy: %f", correct/total)

numel_list = [p.numel()
        for p in model.parameters()
        if p.requires_grad == True]
print(sum(numel_list), numel_list)

#plot loss
plt.plot(range(n_epochs),total_loss,color="blue")
plt.title("Model Loss")
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.show()
plt.savefig('problem1b_loss.png')
