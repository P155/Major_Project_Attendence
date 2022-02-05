# import the necessary packages
import numpy as np
import argparse
import cv2
import os
"""# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", type=str, required=True, help="path to input video")
ap.add_argument("-o", "--output", type=str, required=True, help="path to output directory of cropped faces")
ap.add_argument("-d", "--detector", type=str, required=True, help="path to OpenCV's deep learning face detector")
ap.add_argument("-c", "--confidence", type=float, default=0.5,help="minimum probability to filter weak detections")
ap.add_argument("-s", "--skip", type=int, default=16, help="# of frames to skip before applying face detection")
args = vars(ap.parse_args())"""
ap = argparse.ArgumentParser()
ap.add_argument("-o", "--output", type=str, required=True, help="path to output directory of cropped faces")
args = vars(ap.parse_args())
# load our serialized face detector from disk
print("[INFO] loading face detector...")
#protoPath = os.path.sep.join([args["detector"], "deploy.prototxt"])
#modelPath = os.path.sep.join([args["detector"], "res10_300x300_ssd_iter_140000.caffemodel"])
net = cv2.dnn.readNetFromCaffe('face_detector\deploy.prototxt', r'face_detector\res10_300x300_ssd_iter_140000.caffemodel')
# open a pointer to the video file stream and initialize the total
# number of frames read and saved thus far
#vs = cv2.VideoCapture(args["input"])
# note for camera write 0
#vs = cv2.VideoCapture('dipak.mp4')# name of vedio
vs = cv2.VideoCapture(0)
read = 0
saved = 1354
# make it zero

# loop over frames from the video file stream
while True:
    # grab the frame from the file
    (grabbed, frame) = vs.read()
    if not grabbed:
        break
    read += 1
    if read % 3 != 0:#skip argument
        continue
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
    net.setInput(blob)
    detections = net.forward()
    if len(detections) > 0:
        # we're making the assumption that each image has only ONE
        # face, so find the bounding box with the largest probability
        i = np.argmax(detections[0, 0, :, 2])
        confidence = detections[0, 0, i, 2]
        if confidence > 0.5:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            face = frame[startY:endY, startX:endX]
            # write the frame to disk

            p = os.path.sep.join([args["output"],"{}.png".format(saved)])
            cv2.imwrite(p,face)
            text="{:.2f}%".format(confidence * 100)
            cv2.putText(frame, text, (startX, startY), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
            cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)

            saved += 1
            print("[INFO] saved {} to disk".format(p))
    cv2.imshow('fake', frame)
    k = cv2.waitKey(10) & 0xff  # Press 'ESC' for exiting video
    if k == 27:
        break
vs.release()
cv2.destroyAllWindows()

