"""
Ref article: https://docs.openvino.ai/2022.3/notebooks/112-pytorch-post-training-quantization-nncf-with-output.html 
"""
import os
import time
from pathlib import Path 

import nncf #Neural Network Compression Framework
from openvino.runtime import Core, serialize 
from openvino.tools import mo

import torch
from torchvision.datasets import ImageFolder
from torchvision.transforms import transforms
from torch.utils.data import DataLoader 
from facenet_pytorch import InceptionResnetV1

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DATASET_DIR = Path('C:\\Daniel\\Python\\OnClick Face Recognition\\Optimize_FaceNet\\Humans')
OUTPUT_DIR = Path('C:\\Daniel\\Python\\OnClick Face Recognition\\Optimize_FaceNet\\Quant_OV_InceptionResNetV1')
BASE_MODEL = 'InceptionResNetV1'
IMAGE_SIZE = [299, 299]
#define paths where pytorch and openvino IR (intermediate representation) models will be stored
fp32_ir_path = OUTPUT_DIR / Path(BASE_MODEL + "_fp32").with_suffix(".xml")
int8_ir_path = OUTPUT_DIR / Path(BASE_MODEL + "_int8").with_suffix(".xml")

def grayscale_to_rgb(image_tensor):
        #assuming image is a grayscale PIL Image
        if image_tensor.size(0) == 1: #if grayscale 
            image_tensor = torch.cat([image_tensor, image_tensor, image_tensor], dim=0)
        elif image_tensor.size(0) == 4: #if RGBA, convert RGB
            image_tensor = image_tensor[:3, :, :]
        return image_tensor
    
def create_dataloader(batch_size: int = 8):
    """Creates calibration dataloader that is used for quantization initialization"""
    calibrate_dir = DATASET_DIR
    normalize = transforms.Normalize(
        mean=[0.364, 0.407, 0.497], std=[0.268, 0.243, 0.234] #mean and std values sourced from vggface2 github and study by University of Sao Paulo
    )
    calibrate_dataset = ImageFolder(
        calibrate_dir,
        transforms.Compose(
            [transforms.Resize(IMAGE_SIZE), transforms.ToTensor(), transforms.Lambda(grayscale_to_rgb), normalize]
        ),
    )
    calibration_loader = DataLoader(
        calibrate_dataset,
        batch_size=batch_size,
        num_workers=0, 
    )
    return calibration_loader

def transform_fn(data_item):
    images, _ = data_item
    return images

#initialize quantization
model = InceptionResnetV1(pretrained='vggface2')
calibration_loader = create_dataloader()
calibration_dataset = nncf.Dataset(calibration_loader, transform_fn)
quantized_model = nncf.quantize(model, calibration_dataset)

quantized_model_ir = mo.convert_model(
    quantized_model,
    example_input=torch.randn([3, 299, 299]).unsqueeze(0)
)

serialize(quantized_model_ir, str(int8_ir_path))