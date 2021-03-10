import torch
from PIL import Image
from torchvision import transforms

# Import images
dir = './pictures/'
img_list = ['bird.jpg', 'car.jpg', 'cherry.jpg', 'forest.jpg', 'frog.jpg', 'grouch.jpg', 'jellyfish.jpg', 'kermit.jpg', 'phone.jpg', 'shoes.jpg', 'watermelon.jpg', 'avacado.jpg']
imgs = {}
tensor_list = {}

for f in range(len(img_list)):
    imgs[f] = Image.open(dir+img_list[f])
    tensor_list[f] = transforms.ToTensor()(imgs[f]).unsqueeze(0)
    red = tensor_list[f][:,:,:,0].mean()
    blue = tensor_list[f][:,:,:,1].mean()
    green = tensor_list[f][:,:,:,2].mean()
    print(img_list[f] + "\t\t" + str(tensor_list[f].mean()) + "\t" +str(red) + "\t" + str(blue) + "\t" + str(green))
