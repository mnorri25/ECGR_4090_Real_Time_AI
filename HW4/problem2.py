import torch

#Download CIFAR-10
from torchvision import datasets
from torchvision import transforms
data_path = '../data-unversioned/p1ch7/'
cifar10 = datasets.CIFAR10(data_path, train=True, download=True, transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.4915, 0.4823, 0.4468),(0.2470, 0.2435, 0.2616))]))
cifar10_val = datasets.CIFAR10(data_path, train=False, download=True, transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.4915, 0.4823, 0.4468),(0.2470, 0.2435, 0.2616))]))

#cifar10_tensor = datasets.CIFAR10(data_path, train=True, download=False, transform=transforms.ToTensor())

# Limit number of classes (Build Dataset)
label_map = {0: 0,2 :1}
class_names = ['airplane', 'bird']
cifar2 = [(img, label_map[label])
        for img, label in cifar10
        if label in [0,2]]
cifar2_val = [(img, label_map[label])
        for img, label in cifar10_val
        if label in [0, 2]]

#Normalize
#imgs = torch.stack([img_t for img_t, _ in cifar10_tensor], dim=3)
#print(imgs.shape)

# Model
import torch.nn as nn
n_out = 2

with torch.no_grad():
    conv.bias.zero()
    conv.weight.fill_(1.0/9.0)

model = nn.Sequential(
        #nn.Linear(
        #    3072,#Input features
        #    512,#Hidden layer size
        #    ),
        nn.Conv2d(3, 16, kernel_size=3, padding=1)
        nn.Tanh(),
        nn.MaxPool2d(2)
        nn.Conv2d(16, 8, kernel_size=3, padding1)
        nn.Tanh()
        #nn.Linear(
        #    512,#Hidden layer size
        #    n_out,#output classes
        #    ),
        nn.Linear(8*8*8, 32),
        nn.Tanh(),
        nn.Linear(32, 2),
        nn.LogSoftmax(dim=1))

#Calculate loss
loss=nn.NLLLoss()

#Test output
#img, label = cifar2[0]
#out = model(img.view(-1).unsqueeze(0))
#print(loss(out, torch.tensor([label])))

#Train
import torch.optim as optim
train_loader = torch.utils.data.DataLoader(cifar2, batch_size=64, shuffle=True)
learning_rate = 1e-2
optimizer = optim.SGD(model.parameters(), lr=learning_rate)
loss_fn = nn.NLLLoss()
n_epochs = 100
for epoch in range(n_epochs):
    for imgs, labels in train_loader:
        batch_size = imgs.shape[0]
        outputs = model(imgs.view(batch_size,-1))
        loss = loss_fn(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print("Epoch: %d, Loss %f" % (epoch, float(loss)))

#Validate
val_loader = torch.utils.data.DataLoader(cifar2_val, batch_size=64, shuffle=False)

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
