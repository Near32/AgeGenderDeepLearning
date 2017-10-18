import cv2
import sys
import logging as log
import datetime as dt
from time import sleep

import os
import numpy as np
import time
import sys
import caffe

GPU_ID = 0 # Switch between 0 and 1 depending on the GPU you want to use.
caffe.set_mode_gpu()
caffe.set_device(GPU_ID)

# Loading the mean image
path = './models/'
mean_filename=path+'./mean.binaryproto'
proto_data = open(mean_filename, "rb").read()
a = caffe.io.caffe_pb2.BlobProto.FromString(proto_data)
mean  = caffe.io.blobproto_to_array(a)[0]


# Loading the age network
pathage = './age_net_definitions/'
age_net_pretrained=path+'./age_net.caffemodel'
age_net_model_file=pathage+'./deploy.prototxt'
age_net = caffe.Classifier(age_net_model_file, age_net_pretrained,
                       mean=mean,
                       channel_swap=(2,1,0),
                       raw_scale=255,
                       image_dims=(256, 256))


# Loading the gender network
pathgender = './gender_net_definitions/'
gender_net_pretrained=path+'./gender_net.caffemodel'
gender_net_model_file=pathgender+'./deploy.prototxt'
gender_net = caffe.Classifier(gender_net_model_file, gender_net_pretrained,
                       mean=mean,
                       channel_swap=(2,1,0),
                       raw_scale=255,
                       image_dims=(256, 256))


# Labels
age_list=['(0, 2)','(4, 6)','(8, 12)','(15, 20)','(25, 32)','(38, 43)','(48, 53)','(60, 100)']
gender_list=['Male','Female']

def predict(img) :# ## Reading and plotting the input image
    # Age prediction
    prediction = age_net.predict([img])
    age = age_list[prediction[0].argmax()] 
    #print 'predicted age:', age

    # Gender prediction
    prediction = gender_net.predict([img])
    gender = gender_list[prediction[0].argmax() ] 
    #print 'predicted gender:', gender

    return age,gender


cascPath = "haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascPath)
log.basicConfig(filename='webcam.log',level=log.INFO)

video_inp = './testvideo.mp4'
#video_inp = './testvideo2.mp4'
#video_capture = cv2.VideoCapture(0)
video_capture = cv2.VideoCapture(video_inp)
anterior = 0

freq = 10.0
latency = 0.1
skipframe = 0

while True:
    since = time.time()
    if not video_capture.isOpened():
        print('Unable to load camera.')
        sleep(5)
        pass

    # Capture frame-by-frame
    ret, frame = video_capture.read()
    for i in range(skipframe) :
        ret, frame = video_capture.read()
        
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30)
    )

    # Extract and Draw a rectangle around the faces
    facelist = []
    for (x, y, w, h) in faces:
        facelist.append( frame[x-int(w*0.75):x+int(w*1.75),y-int(h*0.75):y+int(h*1.75)] )
        cv2.rectangle(frame, (x-int(w*0.75), y-int(h*0.75)), (x+int(w*1.75),y+int(h*1.75)), (0, 255, 0), 2)

    if anterior != len(faces):
        anterior = len(faces)
        log.info("faces: "+str(len(faces))+" at "+str(dt.datetime.now()))

    for face, coord in zip(facelist,faces) :
        if face.size > 0 :
            age, gender = predict(face)
            cv2.putText(frame,text='{}:{}'.format(gender,age), org=(coord[0],coord[1]), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(255,255,255), thickness=2)

    # Display the resulting frame
    cv2.putText(frame,text='{} Hz : {}'.format(int(freq),latency), org=(10,50), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(255,255,255), thickness=2)
    cv2.imshow('Video', frame)
    
    key = cv2.waitKey(1) & 0xFF
    if  key == ord('q'):
        break
    elif key == ord("s") :
        skipframe +=1
    elif key == ord("d") :
        skipframe -=1
        
    #print("Latency = {} seconds.".format( time.time() - since) )
    latency = time.time() - since
    freq = 1.0/latency

    
# When everything is done, release the capture
video_capture.release()
cv2.destroyAllWindows()


