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

#exit(0)
tfeat.load_state_dict(checkpoint['state_dict'])


tfeat.eval()

#Runing tfeat with openCV for image matching
#Below we show how to use the openCV pipeline to match two images using TFeat.




img1 = cv2.imread('imgs/v_churchill/1.ppm',0) 
img2 = cv2.imread('imgs/v_churchill/6.ppm',0)

brisk = cv2.BRISK_create()
kp1, des1 = brisk.detectAndCompute(img1,None)
kp2, des2 = brisk.detectAndCompute(img2,None)

bf = cv2.BFMatcher(cv2.NORM_HAMMING)
matches = bf.knnMatch(des1,des2, k=2)
# Apply ratio test
good = []
for m,n in matches:
    if m.distance < ratio*n.distance:
        good.append([m])
        
img3 = cv2.drawMatchesKnn(img1,kp1,img2,kp2,good,0, flags=2)
plt.imshow(img3),plt.show()


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
        
img3 = cv2.drawMatchesKnn(img1,kp1,img2,kp2,good,0, flags=2)

plt.imshow(img3),plt.show()