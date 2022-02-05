
import matplotlib
matplotlib.use("Agg")# this is used to store matplot plot in backend
from tensorflow.keras.preprocessing.image import img_to_array
from LivenessCNN import LivenessNet
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import argparse
import pickle
import datetime
import cv2
import os


ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", type=str, default="faceRec.model",
	help="path to trained model")
ap.add_argument("-l", "--le", type=str, default="le_face.pickle",
	help="path to label encoder")
ap.add_argument("-d", "--detector", type=str, default="face_detector",
	help="path to OpenCV's deep learning face detector")
ap.add_argument("-p", "--plot", type=str, default=r"plot\faceRec.png",
	help="path to output loss/accuracy plot")
args = vars(ap.parse_args())


INIT_LR = 1e-4
BS = 8
EPOCHS = 25

print("[INFO] loading images...")
#FaceDS isthe path to the dataset
imagePaths = list(paths.list_images('FaceDS'))
data = []#this holds all the dataset for face from FaceDS
labels = []

"""print("loading open cv Dnn module face detector...")
protoPath = os.path.sep.join([args["detector"], "deploy.prototxt"])
modelPath = os.path.sep.join([args["detector"],
	"res10_300x300_ssd_iter_140000.caffemodel"])
net = cv2.dnn.readNetFromCaffe(protoPath, modelPath)"""


for imagePath in imagePaths:
    label = imagePath.split(os.path.sep)[-2]
    image = cv2.imread(imagePath)
    image = cv2.resize(image, (32, 32))
    data.append(image)
    labels.append(label)

le = LabelEncoder()
labels = le.fit_transform(labels)
labels = to_categorical(labels, 4)
data = np.array(data, dtype="float") / 255.0
print(labels)

#split FaceDS into training and Testing Dataset trainx represent data and trainy represent labels
(trainX, testX, trainY, testY) = train_test_split(data, labels,
	test_size=0.25,random_state=42)
print(trainX[1].shape)
print(trainY[1].shape)



#Data augumentation to reduce over fitting
aug = ImageDataGenerator(rotation_range=20, zoom_range=0.15,
	width_shift_range=0.2, height_shift_range=0.2, shear_range=0.15,
	horizontal_flip=True, fill_mode="nearest")

print("[INFO] compiling model...")

opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)
model = LivenessNet.build(width=32, height=32, depth=3,
	classes=len(le.classes_))
model.compile(loss="binary_crossentropy", optimizer=opt,
	metrics=["accuracy"])

print(" training network for {} epochs...".format(EPOCHS))
train_start =datetime.datetime.now()
H = model.fit(x=aug.flow(trainX, trainY, batch_size=BS),
	validation_data=(testX, testY), steps_per_epoch=len(trainX) // BS,
	epochs=EPOCHS)
train_end =datetime.datetime.now()
tot_time = train_end - train_start
#true positive wala report
print("[INFO] evaluating network...")
predictions = model.predict(x=testX, batch_size=BS)
print(classification_report(testY.argmax(axis=1),
	predictions.argmax(axis=1), target_names=le.classes_))
print("Time taken for training Face_rec model ", tot_time.total_seconds(), "s")

print(" serializing network to '{}'...".format(args["model"]))
model.save(args["model"], save_format="h5")

# save the label encoder to disk
f = open(args["le"], "wb")
f.write(pickle.dumps(le))
f.close()

model.summary()
# plot the training loss and accuracy
#plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, EPOCHS), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, EPOCHS), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, EPOCHS), H.history["accuracy"], label="train_acc")
plt.plot(np.arange(0, EPOCHS), H.history["val_accuracy"], label="val_acc")
plt.title("Training Loss / Accuracy of Face Rercognization Model")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="center")
plt.savefig(args["plot"])
#plt.subplot(239),plt.plot(np.arange(0, EPOCHS), H.history["val_accuracy"], label="val_acc"),plt.xlabel("Epoch #"),plt.ylabel("Loss/Accuracy")

#plt.subplot(233),plt.imshow(reflect,'gray'),plt.title('REFLECT')





