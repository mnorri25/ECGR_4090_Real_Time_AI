import torch
import time

# Models
models = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
modelm = torch.hub.load('ultralytics/yolov5', 'yolov5m', pretrained=True)
modell = torch.hub.load('ultralytics/yolov5', 'yolov5l', pretrained=True)
modelx = torch.hub.load('ultralytics/yolov5', 'yolov5x', pretrained=True)

# Images
#dir = 'https://github.com/ultralytics/yolov5/raw/master/data/images/'
dir = './pictures/'
imgs = [dir + f for f in ('cat.jpg', 'city.jpg', 'coin.jpg', 'dice.jpg','dinner.jpg', 'zidane.jpg', 'bus.jpg', 'groundhog.jpg', 'owl.jpg', 'panda.jpg', 'park.jpg', 'people.jpg', 'person.jpg', 'professional.jpg', 'rodent.jpg', 'sloth.jpg', 'street.jpg')]  # batched list of images
#img = dir + 'bus.jpg'
#img = 'D:\intersection(nyc_street).jpg'
print("YOLO5_S")
# Inference
start1 = time.time()
result1 = models(imgs[0])
start2 = time.time()
result2 = models(imgs[:7])
start3 = time.time()
result3 = models(imgs[:15])
start4 = time.time()
#result4 = modelx(img)
#stop = time.time()

dur1 = start2 - start1
dur2 = start3 - start2
dur3 = start4 - start3
#dur4 = stop - start3

result1.print()
result2.print()
result3.print()
#result4.print()

print(dur1)
print(dur2)
print(dur3)
#print(dur4)

print("YOLO5_M")
# Inference
start1 = time.time()
result1 = modelm(imgs[0])
start2 = time.time()
result2 = modelm(imgs[:7])
start3 = time.time()
result3 = modelm(imgs[:15])
start4 = time.time()
#result4 = modelx(img)
#stop = time.time()

dur1 = start2 - start1
dur2 = start3 - start2
dur3 = start4 - start3
#dur4 = stop - start3

result1.print()
result2.print()
result3.print()
#result4.print()

print(dur1)
print(dur2)
print(dur3)
#print(dur4)
print("YOLO5_L")
# Inference
start1 = time.time()
result1 = modell(imgs[0])
start2 = time.time()
result2 = modell(imgs[:7])
start3 = time.time()
result3 = modell(imgs[:15])
start4 = time.time()
#result4 = modelx(img)
#stop = time.time()

dur1 = start2 - start1
dur2 = start3 - start2
dur3 = start4 - start3
#dur4 = stop - start3

result1.print()
result2.print()
result3.print()
#result4.print()

print(dur1)
print(dur2)
print(dur3)
#print(dur4)
print("YOLO5_X")
# Inference
start1 = time.time()
result1 = modelx(imgs[0])
start2 = time.time()
result2 = modelx(imgs[:7])
start3 = time.time()
result3 = modelx(imgs[:15])
start4 = time.time()
#result4 = modelx(img)
#stop = time.time()

dur1 = start2 - start1
dur2 = start3 - start2
dur3 = start4 - start3
#dur4 = stop - start3

result1.print()
result2.print()
result3.print()
#result4.print()

print(dur1)
print(dur2)
print(dur3)
#print(dur4)

