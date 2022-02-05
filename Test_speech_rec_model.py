import sounddevice as sd
from scipy.io.wavfile import write
import wavio as wv
import pandas as pd
from tensorflow.keras.models import load_model
from sklearn.preprocessing import LabelEncoder, StandardScaler
import numpy as np
import argparse
import imutils
import pickle
import librosa
import time
import csv
import os

ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", type=str, default="Speaker_rec.model",
	help="path to trained model")
ap.add_argument("-l", "--le", type=str, default="le_speech.pickle",
	help="path to label encoder")
ap.add_argument("-p","--path",type=str, default="Testing_speech.wav",help="path where we are going to store test voice ")
ap.add_argument("-c", "--confidence", type=float, default=0.5,
	help="minimum probability to filter weak detections")
args = vars(ap.parse_args())


# Sampling freq
fs=16000
# Recording duration
duration = 5
def start_rec():
	myrecording = sd.rec(duration * fs, samplerate=fs, channels=2, dtype='float64')
	print("Recording Audio")
	print("start...")
	sd.wait()
	#p = os.path.sep.join([args["path"], ".wav"])
	wv.write("Testing_speech.wav", myrecording, fs, sampwidth=2)
	time.sleep(2)
	sd.play(myrecording, fs)
	sd.wait()
	print ("Play Audio Complete")


def Feature_extract():
	header = 'filename chroma_stft rmse spectral_centroid spectral_bandwidth rolloff zero_crossing_rate'
	for i in range(1, 21):
		header += f' mfcc{i}'
	header += ' label'
	header = header.split()#convert header or string into list
	file = open('Feature_Eng_of_speech.csv', 'w', newline='')
	with file:
		writer = csv.writer(file)
		writer.writerow(header)

	filename="Testing_speech.wav"
	y, sr = librosa.load(filename, mono=True, duration=30)
	chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr)
	rmse = librosa.feature.rms(y=y)
	spec_cent = librosa.feature.spectral_centroid(y=y, sr=sr)
	spec_bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)
	rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
	zcr = librosa.feature.zero_crossing_rate(y)
	mfcc = librosa.feature.mfcc(y=y, sr=sr)
	to_append = f'{filename} {np.mean(chroma_stft)} {np.mean(rmse)} {np.mean(spec_cent)} {np.mean(spec_bw)} {np.mean(rolloff)} {np.mean(zcr)}'
	for e in mfcc:
		to_append += f' {np.mean(e)}'

	file = open('Feature_Eng_of_speech.csv', 'a', newline='')
	with file:
		writer = csv.writer(file)
		writer.writerow(to_append.split())

def read_csv_load_model():
	data = pd.read_csv('Feature_Eng_of_speech.csv')
	data.head()

	# Dropping unneccesary columns
	data = data.drop(['filename'], axis=1)

	print(" loading Speech rec model...")
	model = load_model(args["model"])
	le = pickle.loads(open(args["le"], "rb").read())
	# normalizing
	scaler = StandardScaler()
	X = scaler.fit_transform(np.array(data.iloc[:, :-1], dtype=float))

	preds = model.predict(X)[0]
	j = np.argmax(preds)
	label_voice = le.classes_[j]
	print ("spoken by",label_voice)
	return (label_voice)

