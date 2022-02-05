from imutils import paths
import argparse
import imutils
import numpy as np
import cv2
import os

ap=argparse.ArgumentParser()
ap.add_argument("-i","--input",type=str,default="Dipak",help="path of image input")
ap.add_argument("-o","--output",type=str,default="D",help="path to save image")
ap.add_argument("-d","--detector",type=str,default="face_detector",help="path to face detector")
args=vars(ap.parse_args())

saved=0

#imagepath = os.path.sep.join([args["input"], "Dipak (272).jpg"])
#print(imagepath)
imagepath = list(paths.list_images(args["input"]))
#print(imagepath)
protoPath = os.path.sep.join([args["detector"], "deploy.prototxt"])
modelPath = os.path.sep.join([args["detector"],"res10_300x300_ssd_iter_140000.caffemodel"])
detec=cv2.dnn.readNetFromCaffe(protoPath,modelPath)

print("start savinfg face to disk")
for images in imagepath:
    print(["from{}", format(images)])
    img=cv2.imread(images)
    img = imutils.resize(img, width=600)
    (h, w) = img.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(img, (300, 300)), 1.0,
                             (300, 300), (104.0, 177.0, 123.0))
    detec.setInput(blob)
    detections = detec.forward()
    if (len(detections)>0):
        i = np.argmax(detections[0, 0, :, 2])
        confidence = detections[0, 0, i, 2]
        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
        (startX, startY, endX, endY) = box.astype("int")
        face = img[startY:endY, startX:endX]
        p = os.path.sep.join([args["output"],"{}.png".format(saved)])
        cv2.imwrite(p,face)
        """text = "{:.2f}%".format(confidence * 100)
        cv2.putText(img, text, (startX, startY), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
        cv2.rectangle(img, (startX, startY), (endX, endY), (0, 255, 0), 2)"""
        saved+=1
        print(p)
        print("saved images no", saved)

print("done")
