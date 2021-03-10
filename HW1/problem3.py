from jetcam.usb_camera import USBCamera
import torch
from PIL import Image
from matplotlib import cm

model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

camera = USBCamera(width=224, height=224, capture_width=640, capture_height=480, capture_device=0)

image = camera.read()

print(image.shape)


print(camera.value.shape)
#import ipywidgets      #Widget not working
from IPython.display import display
from jetcam.utils import bgr8_to_jpeg

# Since Widget does not work, these are commented out 
#image_widget = ipywidgets.Image(format='jpeg')
#image_widget.value = bgr8_to_jpeg(image)
#display(image_widget)

# Set camera to running
camera.running = True

# Callback fiunction to update imatge and process using YOLO5
def update_image(change):
    image = change['new']
    #image_widget.value = bgr8_to_jpeg(image)   # Widget not working
    im = Image.fromarray(image)                 # Since widget not working, display PIL Image in new window
    im.show()
    result = model(image)                   # Determine output based on model and print result
    result.print()
    im.close()                  # Close PIL Image 

# Tell camera to start calling callback
camera.observe(update_image, names='value')
