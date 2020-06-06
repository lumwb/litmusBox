import numpy as np
import argparse
import cv2
import json
import time
import pyaudio
import wave
import csv
from imutils.video.pivideostream import PiVideoStream
from imutils.video import FPS
from picamera.array import PiRGBArray
from picamera import PiCamera
from datetime import datetime
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tkinter import *
import tkinter.font
import os


class emoElmo:
    def __init__(self):
        print("constructing emo elmo")
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
        
        self.os = os

        # Create the model
        model = Sequential()

        model.add(Conv2D(32, kernel_size=(3, 3),
                         activation='relu', input_shape=(48, 48, 1)))
        model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))

        model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))

        model.add(Flatten())
        model.add(Dense(1024, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(7, activation='softmax'))

        model.load_weights('model.h5')

        self.model = model

        self.cv2 = cv2

        # prevents openCL usage and unnecessary logging messages
        self.cv2.ocl.setUseOpenCL(False)

        # dictionary which assigns each label an emotion (alphabetical order)
        self.emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful",
                             3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}

        # dictionary to calculate total emotion score at the end of the run
        self.emotion_score_dict = {"Angry": 0, "Disgusted": 0, "Fearful": 0,
                                   "Happy": 0, "Neutral": 0, "Sad": 0, "Surprised": 0}

        # get start datetime
        self.startDateTime = datetime.now()
        self.dateTimeString = self.startDateTime.strftime("%Y-%m-%d_%H:%M")
        self.videoFileName = self.dateTimeString + '.avi'
        self.audioFileName = self.dateTimeString + '.wav'

        # initialize audio recording
        self.form_1 = pyaudio.paInt16  # 16-bit resolution
        self.chans = 1  # 1 channel
        self.samp_rate = 44100  # 44.1kHz sampling rate
        self.chunk = 4096  # 2^12 samples for buffer
        # device index found by p.get_device_info_by_index(ii)
        self.dev_index = 2
        self.wav_output_filename = self.audioFileName  # name of .wav file

        self.audio = pyaudio.PyAudio()  # create pyaudio instantiation

        # create pyaudio stream
        self.stream = self.audio.open(format=self.form_1, rate=self.samp_rate, channels=self.chans,
                                 input_device_index=self.dev_index, input=True,
                                 frames_per_buffer=self.chunk)

        self.audioFrames = []

        # start the webcam feed
        self.vs = PiVideoStream().start()
        time.sleep(2.0)
        self.fps = FPS().start()

        # define codec and create VideoWriter Object
        self.fourcc = cv2.VideoWriter_fourcc(*'XVID')
        self.out = cv2.VideoWriter(self.videoFileName, self.fourcc, 30.0, (320, 240))

        self.facecasc = cv2.CascadeClassifier(
            '/home/pi/opencv/data/haarcascades/haarcascade_frontalface_default.xml')

        self.wave = wave

        # array to store all frames capture for post-processing
        self.videoFrameStore = []

        # boolean flag for recording
        self.recording = 1

        # tkinter settings
        win = Tk()

        win.title("Litmus Box")
        win.geometry('800x480')

        self.win = win

        myFont = tkinter.font.Font(family='Helvetica', size=36, weight='bold')
        
        self.stopRecordButton = Button(self.win, text = "STOP RECORD", font = myFont, command = lambda: self.stop_recording(), height = 2, width = 15)
        self.stopRecordButton.pack(side = BOTTOM)

        self.recordButton = Button(self.win, text = "RECORD", font = myFont, command = lambda: self.start_recording(), height = 2, width = 15)
        self.recordButton.pack()

        mainloop()

    def start_recording(self):
        print("call start recording")
        self.recording = 1
        while self.recording == 1:
            # Find haar cascade to draw bounding box around face
            print("inside whle loop")
            frame = self.vs.read()

            self.videoFrameStore.append(frame)

            # fps stuff
            self.out.write(frame)
            self.fps.update()

            # audio stuff
            data = self.stream.read(self.chunk, exception_on_overflow=False)
            self.audioFrames.append(data)
            print("end of one iteration")
            self.win.update()

        print("exit while loop")

        for currentVideoFrame in self.videoFrameStore:
            # convert to gray and get face position
            gray = self.cv2.cvtColor(currentVideoFrame, cv2.COLOR_BGR2GRAY)
            faces = self.facecasc.detectMultiScale(
                gray, scaleFactor=1.3, minNeighbors=5)

            for (x, y, w, h) in faces:
                roi_gray = gray[y:y + h, x:x + w]
                cropped_img = np.expand_dims(np.expand_dims(
                    cv2.resize(roi_gray, (48, 48)), -1), 0)

                prediction = self.model.predict(cropped_img)
                maxindex = int(np.argmax(prediction))
                # increment score
                emotion = self.emotion_dict[maxindex]
                self.emotion_score_dict[emotion] += 1

        # write to audio file
        self.stream.stop_stream()
        self.stream.close()
        self.audio.terminate()
        self.fps.stop()

        # save the audio frames as .wav file
        wavefile = self.wave.open(self.wav_output_filename, 'wb')
        wavefile.setnchannels(self.chans)
        wavefile.setsampwidth(self.audio.get_sample_size(self.form_1))
        wavefile.setframerate(self.samp_rate)
        wavefile.writeframes(b''.join(self.audioFrames))
        wavefile.close()

        # write final emotion score to txt file
        print(self.emotion_score_dict)
        with open(self.dateTimeString + '.txt', 'w') as file:
            file.write(json.dumps(self.emotion_score_dict))

        print("[INFO] elasped time: {:.2f}".format(self.fps.elapsed()))
        print("[INFO] approx. FPS: {:.2f}".format(self.fps.fps()))

        # cap.release()
        self.out.release()
        self.cv2.destroyAllWindows()
        self.vs.stop()            

    def stop_recording(self):
        print("call stop record button")
        self.recording = 0
        
if __name__ == "__main__":
    emoElmo()