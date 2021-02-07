import cv2 
import numpy as np
import face_recognition
from operator import itemgetter 
import PIL
import imageio
import tensorflow as tf
import scipy.misc
import random
import math
import os

dictconv = {
    '0':'silence',
    '1':'m',
    '2':'ay',
    '3':'n',
    '4':'ey',
    '5':'ih',
    '6':'z',
    '7':'oov',
    '8':'ae',
    '9':'d',
    '10':'dh',
    '11':'s',
    '12':'aa',
    '13':'ow'
    }


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def getInVidsAtFrame(f):
    arr = np.zeros([1, INVID_HEIGHT,INVID_WIDTH,INVID_DEPTH])
    for imageIndex in range(0,29): #29 is not included
        strIndex = str(f-14+imageIndex)
        newImage = imageio.imread(os.path.dirname(os.path.abspath(__file__)) + '\\totest\\test_here\\lips\\frame'+strIndex+'.jpg')

        if newImage.shape[0] > INVID_HEIGHT:
            extraMargin = (newImage.shape[0]-INVID_HEIGHT)//2
            newImage = newImage[extraMargin:extraMargin+INVID_HEIGHT,:,:]
        if newImage.shape[1] > INVID_WIDTH:
            extraMargin = (newImage.shape[1]-INVID_WIDTH)//2
            newImage = newImage[:,extraMargin:extraMargin+INVID_WIDTH,:]

        h = newImage.shape[0]
        w = newImage.shape[1]
        yStart = (INVID_HEIGHT-h)//2
        xStart = (INVID_WIDTH-w)//2
        arr[:,yStart:yStart+h,xStart:xStart+w,imageIndex*3:(imageIndex+1)*3] = newImage
    return np.asarray(arr)/255.0

INVID_WIDTH = 256 # mouth width
INVID_HEIGHT = 256 # mouth height
INVID_DEPTH = 87 # 29 images of R, G, B

PHONEME_CATEGORIES = 41

learning_rate = 0.0002

invids_ = tf.placeholder(tf.float32, (None, INVID_HEIGHT, INVID_WIDTH, INVID_DEPTH), name='invids')
labels_ = tf.placeholder(tf.int32, (None), name='labels')

### Encode the invids
conv1 = tf.layers.conv2d(inputs=invids_, filters=40, kernel_size=(5,5), strides=(2,2), padding='same', activation=tf.nn.relu)
# Now 128x128x40
maxpool1 = tf.layers.max_pooling2d(conv1, pool_size=2, strides=(2,2), padding='same')
# Now 64x64x40
conv2 = tf.layers.conv2d(inputs=maxpool1, filters=70, kernel_size=(5,5), padding='same', activation=tf.nn.relu)
# Now 64x64x70
maxpool2 = tf.layers.max_pooling2d(conv2, pool_size=2, strides=(2,2), padding='same')
# Now 32x32x70
conv3 = tf.layers.conv2d(inputs=maxpool2, filters=100, kernel_size=(5,5), padding='same', activation=tf.nn.relu)
# Now 32x32x100
maxpool3 = tf.layers.max_pooling2d(conv3, pool_size=2, strides=(2,2), padding='same')
# Now 16x16x100
conv4 = tf.layers.conv2d(inputs=maxpool3, filters=130, kernel_size=(5,5), padding='same', activation=tf.nn.relu)
# Now 16x16x130
maxpool4 = tf.layers.max_pooling2d(conv4, pool_size=4, strides=(4,4), padding='same')
# Now 4x4x130 (flatten to 2080)

maxpool4_flat = tf.reshape(maxpool4, [-1,4*4*130])
# Now 2080

W_fc1 = weight_variable([2080, 1000])
b_fc1 = bias_variable([1000])
fc1 = tf.nn.relu(tf.matmul(maxpool4_flat, W_fc1) + b_fc1)

W_fc2 = weight_variable([1000, 300])
b_fc2 = bias_variable([300])
fc2 = tf.nn.relu(tf.matmul(fc1, W_fc2) + b_fc2)

W_fc3 = weight_variable([300, PHONEME_CATEGORIES])
b_fc3 = bias_variable([PHONEME_CATEGORIES])
logits = tf.matmul(fc2, W_fc3) + b_fc3
#Now 40
onehot_labels = tf.one_hot(indices=labels_, depth=PHONEME_CATEGORIES)
loss = tf.losses.sparse_softmax_cross_entropy(labels=labels_, logits=logits)

output = tf.nn.softmax(logits,name=None)

# Get cost and define the optimizer
cost = tf.reduce_mean(loss)
opt = tf.train.AdamOptimizer(learning_rate).minimize(cost)



print("made it here! :D")
sess = tf.Session()
RANGE_START =14

RANGE_END = len(os.listdir(os.path.dirname(os.path.abspath(__file__)) + '\\totest\\test_here\\lips\\')) - 14
epochs = 2000000
batch_size = 50
MODEL_SAVE_EVERY = 50
SAVE_FILE_START_POINT = 150

saver = tf.train.Saver()

sess.run(tf.global_variables_initializer())

if SAVE_FILE_START_POINT >= 1:
    saver.restore(sess, os.path.dirname(os.path.abspath(__file__)) + "\\models\\model"+str(SAVE_FILE_START_POINT)+".ckpt")

print("about to start...")

f = open(os.path.dirname(os.path.abspath(__file__)) + '\\outputted.txt','w')
for frame in range(RANGE_START,RANGE_END):
    invids = np.empty([0,INVID_HEIGHT,INVID_WIDTH,INVID_DEPTH])
    labels = np.empty(0)

    invids = np.vstack((invids,getInVidsAtFrame(frame)))
    #labels = np.append(labels,getLabelsAtFrame(frame))

    _output = sess.run([output],
       feed_dict={invids_: invids})
    f.write("({})".format(str(frame)))
    for n in range(0,5):
        result = np.sort(_output[0][0],axis=None)[::-1][n]
        f.write(str(result) + "({})".format(dictconv[str(_output[0][0].tolist().index(result))]))
        f.write("\t")
    
    f.write("\n")
    f.write("__________________________________________")     
    f.write("\n")
    print("Done with "+str(frame-RANGE_START)+" / "+str(RANGE_END-RANGE_START))
f.close()
