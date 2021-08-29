import cv2
import glob
import PIL
import os,sys
from PIL import Image
import re
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.utils import shuffle
import json

from skimage.io import imread, imshow
from skimage.transform import resize
from skimage.feature import hog
from skimage import exposure
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm, datasets


train_hog_images = []
train_hog_features = []


#path=glob.glob("F:\\lectcures and labs fourth year\\1st term\\Machine Learning(ML)\\labs\\Lab7\\assignment\\train\\*.jpg")
train_path=glob.glob("train\\*.jpg")

for img in train_path:
    img_read=cv2.imread(img)
    resized_img=resize(img_read,(128,64))
    fd, hog_image = hog(resized_img, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), visualize=True,multichannel=True)
    train_hog_images.append(hog_image)
    train_hog_features.append(fd)

train_hog_features=np.array(train_hog_features)
X_train= train_hog_features[:, :2]
Y_train=[]
for i in train_path:
    if 'dog' in i:

       Y_train.append(1)
    else:
        Y_train.append(-1)

X_train = np.array(X_train)
Y_train = np.array(Y_train)

C = 0.1
svc = svm.SVC(kernel='linear', C=C).fit(X_train, Y_train)
lin_svc = svm.LinearSVC(C=C).fit(X_train, Y_train)
rbf_svc = svm.SVC(kernel='rbf', gamma=0.8, C=C).fit(X_train, Y_train)
poly_svc = svm.SVC(kernel='poly', degree=2, C=C).fit(X_train, Y_train)


x_min, x_max = X_train[:, 0].min() - 1, X_train[:, 0].max() + 1
y_min, y_max = X_train[:, 1].min() - 1, X_train[:, 1].max() + 1
h = .02
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))

svc_type=''
test_path = glob.glob("test1\\*.jpg")
test_hog_images = []
test_hog_features = []
for img in test_path:
    img_read = cv2.imread(img)
    resized_img = resize(img_read, (128, 64))
    fd, hog_image = hog(resized_img, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), visualize=True,
                        multichannel=True)
    test_hog_images.append(hog_image)
    test_hog_features.append(fd)


test_hog_features = np.array(test_hog_features)
X_test = test_hog_features[:, :2]
Y_test = []
for i in test_path:
    if 'dog' in i:

        Y_test.append(1)
    else:
        Y_test.append(-1)

X_test = np.array(X_test)
Y_test = np.array(Y_test)


for i, clf in enumerate((svc, lin_svc, rbf_svc, poly_svc)):
    if clf==svc:
        svc_type='svc'
    elif clf==lin_svc:
        svc_type = 'lin_svc'
    elif clf == rbf_svc:
        svc_type = 'rbf_svc'
    elif clf == poly_svc:
        svc_type = 'poly_svc'

    predictions = clf.predict(X_test)
    accuracy = np.mean(predictions == Y_test)
    print('{} accuracy = '.format(svc_type),accuracy)










































