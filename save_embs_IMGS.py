import cv2
import time
import sys
import pickle
import os
import sqlite3 #database
import numpy as np
from PIL import Image
import torch
from torchvision import transforms
from Optimize_FaceNet.quantize_torch_model.fuse_modules import Fusion
from facenet_pytorch import InceptionResnetV1
import torch.nn as nn

"""
    Run this file, insert the name of a folder containing 3 or more pictures of the person
    Folder's name must be the person's name, ie folder named "Daniel"
    This file will save these faces to the database "StoredFaces.db" USES SAVED IMAGES
    
"""
con = sqlite3.connect('StoredFaces.db') #create a database, establish connection to it
cur = con.cursor() #cursor allws us to access and edit the database

'''
    creates a table in the database named "embeddings", with columns "person", "emb1", "emb2", "emb3", 
'''
cur.execute('''CREATE TABLE IF NOT EXISTS embeddings 
                    (person TEXT PRIMARY KEY, emb1 BLOB, emb2 BLOB, emb3 BLOB)''') #execute sql queries onto database
#setting person as a PRIMARY KEY means there can only be one instance of each person in the database

def record_embeddings(folder_name):
    global cur
    global con
    #load the InceptionResnetV1 quantized model
    model = InceptionResnetV1(pretrained='vggface2').eval()
    fusion = Fusion()
    model = fusion.fuse()
    model = nn.Sequential(torch.quantization.QuantStub(), 
                        model, 
                        torch.quantization.DeQuantStub()) 
    model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
    torch.quantization.prepare(model, inplace=True)
    torch.quantization.convert(model, inplace=True)
    model.load_state_dict(torch.load('Optimize_FaceNet\\model_versions\\quant_torch_model.pth'), strict=False)
    
    transform = transforms.Compose([
        transforms.Resize((400, 400)),
        transforms.ToTensor(),
    ])
    
    #get list of images from the folder
    cropped_imgs = []
    folder_path = f'Faces_Database\\{folder_name}'
    for filename in os.listdir(folder_path):
        if filename.endswith(('.jpg', '.png', '.jpeg')):
            img_path = os.path.join(folder_path, filename)
            img = cv2.imread(img_path)
            if img is not None:
                cropped_imgs.append(cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    embs = [folder_name]
    
    #generate embeddings for each cropped face
    for img in cropped_imgs:
        img = Image.fromarray(img)
        img_tensor = transform(img).unsqueeze(0)
        with torch.no_grad():
            embedding = model(img_tensor)
            embedding_binary = pickle.dumps(embedding)
            embs.append(embedding_binary)
    
    #insert embeddings into database
    cur.execute("INSERT OR IGNORE INTO embeddings (person, emb1, emb2, emb3) VALUES (?, ?, ?, ?)", embs)
    con.commit()
    
def access_ref_imgs():
    haar_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    folder_name = (str)(input())
    folder_path = f'Faces_Database\\{folder_name}'
    images = []
    for filename in os.listdir(folder_path):
        if filename.endswith(('.jpg', '.png', '.jpeg')):
            img_path = os.path.join(folder_path, filename)
            img = cv2.imread(img_path)
            os.remove(img_path)
            if img is not None:
                images.append(cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    
    if len(images) < 3:
        print('Must give 3 or more photos of the person')
        sys.exit()
    
    cropped_imgs = []

    for i, img in enumerate(images):                    
        faces = haar_cascade.detectMultiScale(
            img, scaleFactor=1.05, minNeighbors=2, minSize=(100,100)
        ) #should only yield one result, as ref images must only contain one face
        
        """
            scaleFactor:    used to find different sized faces in the img. Defines how much to scale 
                            the img down by each time it searches for a smaller sized face. Smaller scaleFactor for
                            increased accuracy, decreased speed
            minNeighbors:   How many overlapping detections of the same face needed for it qualify to be
                            considered a face. Higher minNeighbors prevents false positives, but causes more false negatives. 
        """
        
        for j, (x, y, w, h) in enumerate(faces):
            cropped_img = img[y : y+h, x : x+w]
            cropped_imgs.append(cropped_img)
            cv2.imwrite(
                f'Faces_Database\\{folder_name}\\' + folder_name + str(i) + '.jpg',
                cropped_img
            )
        
    record_embeddings(folder_name)

access_ref_imgs()