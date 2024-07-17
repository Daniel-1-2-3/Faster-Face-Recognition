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

class FacesDatabase:
    def __init__(self):
        """
            Run method take_photo_crop_ref_imgs(), then insert the name of a folder containing 3 or more pictures of the person
            when prompted. Folder's name must be the person's name, ie folder named "Daniel"
            This file will save these faces to the database "StoredFaces.db" TAKES PICTURES THROUGH WEBCAM.
            
        """
        self.con = sqlite3.connect('StoredFaces.db') #create a database, establish connection to it
        self.cur = self.con.cursor() #cursor allws us to access and edit the database

        '''
            creates a table in the database named "embeddings", with columns "person", "emb1", "emb2", "emb3", "emb4", "emb5", "emb6", "emb7", "emb8", "emb9", "emb10", 
        '''
        self.cur.execute('''CREATE TABLE IF NOT EXISTS embeddings 
                            (person TEXT PRIMARY KEY, emb1 BLOB, emb2 BLOB, emb3 BLOB, emb4 BLOB, emb5 BLOB, emb6 BLOB, emb7 BLOB, emb8 BLOB, emb9 BLOB, emb10 BLOB)''') #execute sql queries onto database
        #setting person as a PRIMARY KEY means there can only be one instance of each person in the database

        #load the InceptionResnetV1 quantized model
        self.model = InceptionResnetV1(pretrained='vggface2').eval()
        fusion = Fusion()
        self.model = fusion.fuse()
        self.model = nn.Sequential(torch.quantization.QuantStub(), 
                            self.model, 
                            torch.quantization.DeQuantStub()) 
        self.model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
        torch.quantization.prepare(self.model, inplace=True)
        torch.quantization.convert(self.model, inplace=True)
        self.model.load_state_dict(torch.load('Optimize_FaceNet\\model_versions\\quant_torch_model.pth'), strict=False)
        
    def record_embeddings(self, folder_name):
        transform = transforms.Compose([
            transforms.Resize((500, 500)),
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
                embedding = self.model(img_tensor)
                embedding_binary = pickle.dumps(embedding)
                embs.append(embedding_binary)
        
        #insert embeddings into database
        self.cur.execute("INSERT OR REPLACE INTO embeddings (person, emb1, emb2, emb3, emb4, emb5, emb6, emb7, emb8, emb9, emb10) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", embs)
        self.con.commit()
        
    def take_photo_crop_ref_imgs(self):
        haar_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
        folder_name = (str)(input('Enter folder name: '))
        folder_path = f'Faces_Database\\{folder_name}'
        images = []
        
        cap = cv2.VideoCapture(0)
        img_count = 0
        
        while True:
            ret, frame = cap.read()
            frame = cv2.flip(frame, 1)
            
            if cv2.waitKey(1) & 0xFF == 32:
                frame_copy = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = haar_cascade.detectMultiScale(
                    frame_copy, scaleFactor=1.05, minNeighbors=2, minSize=(100,100)
                ) #should only yield one result, as ref images must only contain one face
                if len(faces)>0:
                    images.append(frame_copy)
                    img_count += 1
                    frame = np.ones_like(frame) * 255 #make screen white, flash effect to show photo taken
                else:
                    frame = np.ones_like(frame) * 0
            
            _, width, _ = frame.shape
            cv2.rectangle(frame, (0, 0), (width, 90), (250, 250, 250), thickness=cv2.FILLED)
            cv2.putText(frame, f'Press space to take photo, q to exit camera', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 2)
            cv2.putText(frame, f'Number of photos taken: {img_count}', (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 2)
            cv2.rectangle(frame, ((width-300)//2, 130), ((width-300)//2+300, 410), (100, 100, 200), 1)
            cv2.imshow('Webcam', frame)
            
            if cv2.waitKey(1) == ord('q'):
                if img_count >= 10:
                    break
            
        cap.release()
        cv2.destroyAllWindows()
        
        cropped_imgs = []

        for i, img in enumerate(images[:10]):                    
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
            
        self.record_embeddings(folder_name)

if __name__ == '__main__':
    databaseFaceAdder = FacesDatabase()
    databaseFaceAdder.take_photo_crop_ref_imgs()