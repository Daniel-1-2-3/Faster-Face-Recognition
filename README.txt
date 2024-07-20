Instructions for running the software:
1.  Run save_embs.py to add a person into the database
    a.  Create a folder inside the Faces_Database folder, give the folder the 
        name of the person you are adding into the database
    b.  Input the name of the folder when prompted
    c.  Take 5 or more photos of the person in different expressions and angles. Should be facing forwards.
2.  Run recognize_faces.py to recognize faces
3.  Some errors would likely arise due to the fact that we are using the quantized version of the
    InceptionResnetV1 model. Check Important Notes document in the quantize_torch_model folder
    to address those errors.