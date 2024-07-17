"""
Quantization: convert floating point weights and activation values to integer values. A weight with 
range [-1, 1] may be converted to an integer [-127, 127]. 
reference https://pytorch.org/blog/quantization-in-practice/#post-training-static-quantization-ptq  
This is static quantization: converting both weights and activations to lowered to int8 during quantization
"""
from tqdm import tqdm 
import torch
from torch import nn
import os
from PIL import Image
from fuse_modules import Fusion
from facenet_pytorch import InceptionResnetV1
from torchvision import transforms


class CustomDataset:
    def __init__ (self):
        self.folder_path = 'C:\\Daniel\\Python\\OnClick Face Recognition\\Optimize_FaceNet\\Humans'
        self.transform = transforms.Compose([
            transforms.Resize((500, 500)),
            transforms.ToTensor(),
            transforms.Lambda(self.grayscale_to_rgb)
    ])
    def load_images(self):
        images = []
        for filename in os.listdir(self.folder_path):
            if filename.endswith('jpg') or filename.endswith('png'):
                img_path = os.path.join(self.folder_path, filename)
                image = Image.open(img_path)
                image = self.transform(image)
                images.append(image)
        return images 

    def grayscale_to_rgb(self, image_tensor):
        #assuming image is a grayscale PIL Image
        if image_tensor.size(0) == 1: #if grayscale 
            image_tensor = torch.cat([image_tensor, image_tensor, image_tensor], dim=0)
        elif image_tensor.size(0) == 4: #if RGBA, convert RGB
            image_tensor = image_tensor[:3, :, :]
        image_tensor = image_tensor.unsqueeze(0)
        return image_tensor
        
        
model = InceptionResnetV1(pretrained='vggface2').eval()
torch.save(model.state_dict(), 'unoptimized_torch_model')
fuse = Fusion()
model = fuse.fuse() #fuse layers within the model

#Insert quant and dequant stubs (input/output gates that convert input tensor from floating point to int8)
model = nn.Sequential(torch.quantization.QuantStub(), 
                      model, 
                      torch.quantization.DeQuantStub()) 

#prepare settings, such as type of observer used, (default params) for quantization
model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
torch.backends.quantized.engine = 'fbgemm'
model_static_quantized = torch.quantization.prepare(model, inplace=False)

#calibration with validation dataset of 1000 images
dataset = CustomDataset()
images = dataset.load_images()
with torch.inference_mode():
    for img in tqdm(images, desc="Calibrating model"):
        model_static_quantized(img)

#convert to quantized model
model_static_quantized = torch.quantization.convert(model_static_quantized, inplace=False) #inplace indicates it replaces previous model with the quantized one

print(model_static_quantized)
torch.save(model_static_quantized.state_dict(), 'quant_torch_model.pth')
            

