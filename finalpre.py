import cv2 
import numpy as np
import face_recognition
from operator import itemgetter 
import PIL
import imageio
import scipy.misc
import random
import math
import os

dirpath = os.listdir(os.path.dirname(os.path.abspath(__file__)) + '\\totest\\')
for direc in dirpath:
    path = os.path.dirname(os.path.abspath(__file__)) + '\\totest\\'+direc + '\\'+direc+'.mp4'
    print(path)
    vidcap = cv2.VideoCapture(path)
    success,image = vidcap.read()
    count = 0
    totalframes =  vidcap.get(cv2.CAP_PROP_FRAME_COUNT)
    print(direc + " Total frames : " + str(totalframes))
    while success:
        cv2.imwrite(os.path.dirname(os.path.abspath(__file__)) + '\\totest\\'+ direc +'\\frames\\'
                    + "frame%d.jpg" % count, image)# save frame as JPEG file
        success,image = vidcap.read()
        print (str(int(count/totalframes * 100)) + "%" , success)
        count += 1
    print(direc + " Finished")

for direc in dirpath:
    frames = os.listdir(os.path.dirname(os.path.abspath(__file__)) + '\\totest\\'+ direc +'\\frames\\')
    print(direc + " Total frames : " + str(len(frames)))
    indexoff = 0
    for f in frames:
        if os.path.isfile(os.path.dirname(os.path.abspath(__file__)) + '\\totest\\'+ direc +'\\frames\\' + f):
            image = face_recognition.load_image_file(os.path.dirname(os.path.abspath(__file__)) + '\\totest\\'+ direc +'\\frames\\' + f)
            face_landmarks = face_recognition.face_landmarks(image)
            if len(face_landmarks) > 0:
                postop = face_landmarks[0]["top_lip"]
                posbot = face_landmarks[0]["bottom_lip"]
                pos = postop+posbot
                padding = 10
                vertical_offset = 10
                maxh = 200
                maxw = 200
                top = (min(pos, key = itemgetter(1))[1] - padding + vertical_offset)
                bot = (max(pos, key = itemgetter(1))[1] + padding + vertical_offset)
                left = (min(pos, key = itemgetter(0))[0] - padding)
                right = (max(pos, key = itemgetter(0))[0] + padding)
                h = bot-top
                w = right - left
                marginhight = (maxh - h)/2
                marginwidth = (maxw - w)/2
                im = PIL.Image.open(os.path.dirname(os.path.abspath(__file__)) + '\\totest\\'+ direc +'\\frames\\' + f)
                region = im.crop((int(left-marginwidth), int(top-marginhight), int(right+marginwidth), int(bot+marginhight)))
                region = region.resize((256,256), PIL.Image.ANTIALIAS)
                region.save(os.path.dirname(os.path.abspath(__file__)) + '\\totest\\'+ direc +'\\lips\\' + f)
                print(str(int(indexoff/len(frames)*100))+"%")
                indexoff+=1
    print(direc + " Finished")



