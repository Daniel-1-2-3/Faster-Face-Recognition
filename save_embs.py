import cv2, time, math, sys, os, json, pickle, copy
import sqlite3 #database
import numpy as np
from PIL import Image
import torch
from torchvision import transforms
from Optimize_FaceNet.quantize_torch_model.fuse_modules import Fusion
from facenet_pytorch import InceptionResnetV1
import torch.nn as nn
from sympy import symbols, solve, Eq
from sympy.geometry import Circle, Line, Point

class FacesDatabase:
    def __init__(self):
        """
            Run method take_photo_crop_ref_imgs(), then insert the name of a folder to store the pictures taken into
            when prompted. Folder's name must be the person's name, ie folder named "Daniel". Program will save about 70
            pictures of the person to this folder, extract faces from the pictures, then save these faces to the database "StoredFaces.db". 
            
        """
        self.con = sqlite3.connect('StoredFaces.db') #create a database if not exsit, establish connection to it
        self.cur = self.con.cursor() #cursor allws us to access and edit the database

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
        print('Processing faces, this may take a while...')
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
        
        #creates a table in the database named "embeddings" is not exist, with columns "person", "emb0", "emb1", "emb2", "emb3", "emb4" ...)
        columns = 'emb0 BLOB'
        for i in range (1, 80):
            columns = columns + f', emb{i} BLOB'
        self.cur.execute(f'''CREATE TABLE IF NOT EXISTS embeddings 
                            (person TEXT PRIMARY KEY, {columns})''') #execute sql queries onto database
        #setting person as a PRIMARY KEY means there can only be one instance of each person in the database
        
        #generate embeddings for each cropped face
        for img in cropped_imgs:
            img = Image.fromarray(img)
            img_tensor = transform(img).unsqueeze(0)
            with torch.no_grad():
                embedding = self.model(img_tensor)
                embedding_binary = pickle.dumps(embedding)
                embs.append(embedding_binary)
        
        #insert embeddings into database
        columns = 'emb0'
        placeholders = '?, ?'
        for i in range (1, len(cropped_imgs)):
            columns += f', emb{i}'
            placeholders += f', ?'

        self.cur.execute(f"INSERT OR REPLACE INTO embeddings (person, {columns}) VALUES ({placeholders})", embs)
        self.con.commit()
        
    def take_photo_crop_ref_imgs(self):
        haar_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
        folder_name = (str)(input('Enter folder name: '))
        folder_path = f'Faces_Database\\{folder_name}'
        images = []
        
        cap = cv2.VideoCapture(0)
        ret, frame = cap.read()
        height, width, _ = frame.shape
        
        images = []
        while True:
            ret, frame = cap.read()
            frame = cv2.flip(frame, 1)
            cv2.circle(frame, (int(width//2), int(height//2)), 200, (200, 200, 200), 1)
            cv2.putText(frame, 'Press space to start', (10, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (225, 225, 225), 1)
            cv2.imshow('Webcam', frame)
            if cv2.waitKey(1) & 0xFF == ord(' '):
                with open('animation_points.json', 'r') as f:
                    dots = json.load(f)
                #display the points
                for i in range(0, len(dots)):
                    ret, frame = cap.read()
                    frame = cv2.flip(frame, 1)
                    if i%5==0: #save every 5th frame
                        images.append(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
                    frame_copy = copy.deepcopy(frame)
                    cv2.circle(frame_copy, (int(width//2), int(height//2)), 200, (200, 200, 200), 1)
                    for j in range (0, i):
                        cv2.circle(frame_copy, (dots[j][0], dots[j][1]), 4, (0, 0, 255), -1)
                    cv2.putText(frame_copy, 'Collecting pictures...', (10, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (225, 225, 225), 1)
                    cv2.imshow('Webcam', frame_copy)
                    cv2.waitKey(7)
                break
        cap.release()
        cv2.destroyAllWindows()
        
        print('Pictures collected, extracting faces...')
        
        cropped_imgs = []
        for i, img in enumerate(images):                    
            faces = haar_cascade.detectMultiScale(
                img, scaleFactor=1.05, minNeighbors=3, minSize=(100,100)
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
        
        blank_frame = np.ones((height, width, 3), dtype=np.uint8) * 255
        if len(cropped_imgs)==0:
            print('No faces found')
            time.sleep(1)
            sys.exit()
        else:
            print(f'Extracted {len(cropped_imgs)} faces')
            print(f'Check folder "{folder_name}" for extracted faces, remove any faulty extractions')
            proceed = input(f'Ready to proceed? (Y/N): ')
            
            if proceed.lower() == 'n':
                print('Aborting...')
                time.sleep(1)
                sys.exit()
            
            self.record_embeddings(folder_name)

if __name__ == '__main__':
    databaseFaceAdder = FacesDatabase()
    databaseFaceAdder.take_photo_crop_ref_imgs()
    print('Saved')