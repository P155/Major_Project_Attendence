import cv2
import argparse


ap = argparse.ArgumentParser()
ap.add_argument("-m", "--output", type=str, default=r"video\face_detc_chk.mp4",
	help="path to store video")
args = vars(ap.parse_args())

cap = cv2.VideoCapture(0)
vid_cod = cv2.VideoWriter_fourcc(*'XVID')
output = cv2.VideoWriter(args["output"], vid_cod, 20.0, (640,480))

while(True):
     ret,frame = cap.read()
     cv2.imshow("My cam video", frame)
     output.write(frame)
     if cv2.waitKey(1) &0XFF == ord('x'):
         break

cap.release()
output.release()
cv2.destroyAllWindows()