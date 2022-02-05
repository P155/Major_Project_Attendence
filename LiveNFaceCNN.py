#from imutils.video import VideoStream
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import numpy as np
import argparse
import imutils
import pickle
import time
import cv2
import os

ap =argparse.ArgumentParser()
ap.add_argument("-m1","--model_liveness",type=str, default="liveness1.model",help="path of Liveness trained model")
ap.add_argument("-m2","--model_faceRec",type=str,default="faceRec.model",help="path to train model of Face Recognization using CNN")
ap.add_argument("-d","--face_detector",type= str, default="face_detector",help="path toi the DNN opencv face detector")
ap.add_argument("-le1","--le_liveness",type=str,default="le1.pickle",help="path to the label encoder of liveness network")
ap.add_argument("-le2","--le_faceRec",type=str,default="le_face.pickle",help="path to the label encoder of face Recognization network")
ap.add_argument("-c","--confidence",type=float,default=0.5,help="threshold of confidence level")
arg=vars(ap.parse_args())

protopath = os.path.sep.join([arg["face_detector"],"deploy.prototxt"])
modelpath = os.path.sep.join([arg["face_detector"],"res10_300x300_ssd_iter_140000.caffemodel"])
net = cv2.dnn.readNetFromCaffe(protopath,modelpath)
model_live = load_model(arg["model_liveness"])
model_faceCNN = load_model(arg["model_faceRec"])
le_live = pickle.loads(open(arg["le_liveness"],"rb").read())
le_face = pickle.loads(open(arg["le_faceRec"],"rb").read())

def Start_camera():
    print("starting camera...")
    print("press q to exit")
    cap=cv2.VideoCapture(0)
    time.sleep(2.0)
    while True:
        my_list={}
        ret, frame=cap.read()
        frame = imutils.resize(frame, width=600)#it resize to pixel width 600
        (h, w) = frame.shape[:2]
        blob=cv2.dnn.blobFromImage(cv2.resize(frame,(300,300)),1,(300, 300), (104.0, 177.0, 123.0))#preprocessing
        net.setInput(blob)
        detection=net.forward()
        for i in range(0, detection.shape[2]):
            confidence=detection[0, 0, i, 2]
            if confidence >arg["confidence"]:
                box = detection[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")
                startX = max(0, startX)
                startY = max(0, startY)
                endX = min(w, endX)
                endY = min(h, endY)
                face = frame[startY:endY, startX:endX]
                face = cv2.resize(face, (32, 32))
                face = face.astype("float") / 255.0
                face = img_to_array(face)
                face = np.expand_dims(face, axis=0)
                preds_face=model_faceCNN.predict(face)[0]
                k=np.argmax(preds_face)
                label_face=le_face.classes_[k]
                preds = model_live.predict(face)[0]
                j = np.argmax(preds)
                label_live = le_live.classes_[j]
                #label=[label_face,label_live]
                #for i in label:
                my_list[label_face]=label_live
                print(label_face+" : "+ label_live)
                if label_live=="real":
                    cv2.putText(frame, label_face+" is "+label_live, (startX, startY - 10),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 2)
                    cv2.rectangle(frame, (startX, startY), (endX, endY),(0, 255,0), 2)
                elif label_live=="fake":
                    cv2.putText(frame, label_face+" is "+label_live, (startX, startY - 10),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0, 255), 2)
                    cv2.rectangle(frame, (startX, startY), (endX, endY),(0, 255, 0), 2)
        cv2.imshow("Testing both Liveness and Face Recognization Model", frame)
        if cv2.waitKey(1) & 0XFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
    return (my_list)
