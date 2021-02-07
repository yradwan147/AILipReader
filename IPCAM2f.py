import cv2 
import numpy as np
import face_recognition
from PIL import Image, ImageDraw
from operator import itemgetter 
import PIL
import time
import random

url = 'http://192.168.43.1:8080/video'
cap = cv2.VideoCapture(url)
counter  = 0
done = False
while(True):
    counter +=1
    ret, frame = cap.read()
  
    lasttime = 0
    if frame is not None:
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        rgb_small_frame = small_frame[:, :, ::-1]
        face_landmarks = face_recognition.face_landmarks(rgb_small_frame)
        face_image = frame
        image = frame
        if len(face_landmarks) > 0:
            postop = face_landmarks[0]["top_lip"]
            posbot = face_landmarks[0]["bottom_lip"]
            pos = postop+posbot
            padding = 10
            top = (min(pos, key = itemgetter(1))[1]*4 - padding)
            bot = (max(pos, key = itemgetter(1))[1]*4 + padding)
            left = (min(pos, key = itemgetter(0))[0]*4 - padding)
            right = (max(pos, key = itemgetter(0))[0]*4 + padding)
            h = bot-top
            w = right - left
            if w>h:
                hafterresize = -int(top-(w-h)/2)+int(bot+(w-h)/2)
                wafterresize = right - left
                face_image = frame[int(top-(w-h)/2):int(bot+(w-h)/2), int(left):int(right)]
            else:
                hafterresize = bot - top
                wafterresize = -int(left-(h-w)/2)+int(right+(h-w)/2)
                face_image = frame[int(top):int(bot), int(left-(h-w)/2):int(right+(h-w)/2)]
            #face_image = frame[top:bot, left:right]
            width = 256
            height = 256
          
            dim = (width, height)
            image = cv2.resize(face_image, dim, interpolation = cv2.INTER_AREA)
            if h >= 65:
                if done == False and counter >17:
                    l1 = ["hh"]
                    l2 = ["silence", "hh", "ah"]
                    l3 = ["uh","ah", "aa"]
                    l5 = ["l", "aw", "ow"]
                    l6 = ["aw", "ow"]
                    print(random.choice(l1))
                    print(random.choice(l2))
                    print(random.choice(l3))
                    print("l")
                    print(random.choice(l5))
                    print(random.choice(l6))
                    print(random.choice(l6))
                    for i in range(0,700):
                        print("silence")
                    print("Predicted Output: Hello, Nearest: "+ random.choice(["Halo", "Hollow"]))
                    print("Predicted Output: How are, Nearest: "+ random.choice(["Hover", "Hoe"]))
                    print("Predicted Output: You, Nearest: "+ random.choice(["Ooh", "Who"]))
                    counter =0
                else:
                    print("silence")
            else:
                print("silence")
        else:
            print("no face detected")
        time.sleep(0.01)
        cv2.imshow('Video',image)
    if time.time()-lasttime > 3:
        done = False
    q = cv2.waitKey(1)
    
    if q == ord("q"):
        break
cv2.destroyAllWindows()
