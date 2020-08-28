import torchvision as tv
#import phototour
import torch
from tqdm import tqdm 
import numpy as np
import torch.nn as nn
import math 
import HardNet
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import os

import cv2
import tfeat_utils
import numpy as np
from matplotlib import pyplot as plt


ratio = 0.8


def mp4_to_avi(src_dir, dst_dir):
    #src_dir = "1.mp4"
    #dst_dir = "2.avi"

    video_cap = cv2.VideoCapture(src_dir)
    fps = video_cap.get(cv2.CAP_PROP_FPS)
    size = (int(video_cap.get(cv2.CAP_PROP_FRAME_WIDTH)),   
            int(video_cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))  
    video_writer = cv2.VideoWriter(dst_dir, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), fps, size) 

    success, frame = video_cap.read()
    while success:
        video_writer.write(frame)
        success, frame = video_cap.read()

    return dst_dir

#init tfeat and load the trained weights
tfeat = HardNet.SOSNet32x32()
models_path = 'pretrained-models'
net_name = 'sos_reg.pth'
if torch.cuda.is_available():
    location=lambda storage, loc: storage.cuda()
    tfeat.cuda()
else:
    location='cpu'

print(location)
load_path = os.path.join(models_path,net_name)
print(load_path)
checkpoint = torch.load(load_path, map_location=location)


tfeat.load_state_dict(checkpoint['state_dict'])


tfeat.eval()

#Runing tfeat with openCV for image matching
#Below we show how to use the openCV pipeline to match two images using TFeat.


#jiangshi = cv2.imread('images/Jiangshi_lg.png', 0)
#vetalas = cv2.imread('images/Vetalas_lg.png', 0)
draugr = cv2.imread('images/Draugr_lg.png', 0)
img1  = draugr

scale_percent = 20

width = int(img1.shape[1] * scale_percent / 100)
height = int(img1.shape[0] * scale_percent /100)
dim = (width,height)

#jiangshi = cv2.resize(jiangshi, dim, interpolation = cv2.INTER_AREA)
#vetalas = cv2.resize(vetalas, dim, interpolation = cv2.INTER_AREA)
img1 = cv2.resize(img1, dim, interpolation = cv2.INTER_AREA)

sift = cv2.xfeatures2d.SIFT_create()

kp1,des1 = sift.detectAndCompute(img1,None)
#kp2,des2 = sift.detectAndCompute(vetalas,None)

FLANN_INDEX_KDTREE = 0
index_params = dict(algorithm=FLANN_INDEX_KDTREE,trees=5)
search_params = dict(checks=50)

flann = cv2.FlannBasedMatcher(index_params,search_params)

#cap =cv2.VideoCapture(0)
src_dir='2019-08-02/00091.MTS'
dst_dir='2019-08-02/00091.avi'
#cap = cv2.VideoCapture(mp4_to_avi(src_dir, dst_dir))
cap = cv2.VideoCapture(src_dir)


width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

print "entering while loop"
while cap.isOpened():
    print "while cap.isOpened():"
    ret, frame = cap.read()
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    print "gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)"
    #cv2.imshow('frame', gray)
    #print "cv2.imshow('frame', gray)"
    #cv2.waitKey(0) 
    #print "cv2.waitKey(0) "

    
    # Operations
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    img2 = gray_frame
    kp2,des2 = sift.detectAndCompute(img2,None)

    #matches = flann.knnMatch(des1,des2,k=2)
    print "des2"
    print des2
    print "des2.shape"
    print np.array(des2).shape
    print "des1.shape"
    print np.array(des1).shape
    if des2 is None:
        continue


    # mag_factor is how many times the original keypoint scale
    # is enlarged to generate a patch from a keypoint
    mag_factor = 3
    desc_tfeat1 = tfeat_utils.describe_opencv(tfeat, img1, kp1, 32,mag_factor)
    desc_tfeat2 = tfeat_utils.describe_opencv(tfeat, img2, kp2, 32,mag_factor)

    bf = cv2.BFMatcher(cv2.NORM_L2)
    matches = bf.knnMatch(desc_tfeat1,desc_tfeat2, k=2)
    # Apply ratio test
    good = []
    for m,n in matches:
        if m.distance < 0.8*n.distance:
            good.append([m])

    matchesMask = [[0,0] for i in range(len(matches))]

    for i, (match1,match2) in enumerate(matches):
        if match1.distance < 0.7*match2.distance:
            matchesMask[i] = [1,0]
    
    draw_params = dict(matchColor=(0,255,0),
                singlePointColor=(255,0,0),
                matchesMask=matchesMask,
                flags=0)
            
    flann_matches = cv2.drawMatchesKnn(img1,kp1,img2,kp2,good,0, flags=2)

    # Show image
    cv2.namedWindow('detection', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('detection', 600,600)
    cv2.imshow('frame', flann_matches)
    ratio = 0.7
    good = []
    for p, q in matches:
        if p.distance > q.distance*ratio:
            good.append(p)
    print "len(good)"
    print len(good)
    if cv2.waitKey(1) & 0xFF == ord('q'):
    	break


cap.release()
cv2.destroyAllWindows()