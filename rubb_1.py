import numpy as np
import cv2
import matplotlib.pylab as plt
from skimage import transform

################### 人脸关键点标注 ###################
landmark = np.array([[153, 244, 215, 214, 292, 252, 196, 292, 346, 300]])
landmark = np.reshape(landmark, (2, 5)).T
print(landmark)

img = cv2.imread('rubb/wt.png')
for point in landmark:
    cv2.circle(img, tuple(point), 2, (0, 0, 255))

cv2.imshow('img', img)
cv2.waitKey(10000)

################### 标准脸的关键点 ###################
REFERENCE_FACIAL_POINTS = np.array([
    [30.29459953, 51.69630051],
    [65.53179932, 51.50139999],
    [48.02519989, 71.73660278],
    [33.54930115, 92.3655014],
    [62.72990036, 92.20410156]
], np.float32)

# Lets create a empty image|
empty_img = np.zeros((112, 96, 3), np.uint8)

for point in REFERENCE_FACIAL_POINTS:
    # turn point into int
    x = int(point[0])
    y = int(point[1])
    cv2.circle(empty_img, (x,y), 2, (0, 0, 255))

plt.figure(figsize=(5, 5))
plt.imshow(empty_img)
plt.show()

################### 把人脸1和标准脸对齐 ###################
#### 变换矩阵
trans = transform.SimilarityTransform()
res = trans.estimate(landmark, REFERENCE_FACIAL_POINTS)
M = trans.params
print(res)  # True
print(M)  # 变换矩阵
# [[  0.29581306  -0.16732268  28.26420688]
#  [  0.16732268   0.29581306 -47.51195016]
#  [  0.           0.           1.        ]]

#### 人脸对齐
print(M[:2, :])
new_img = cv2.warpAffine(img, M[:2, :], dsize=(120, 120))
cv2.imshow('new_img', new_img)
cv2.waitKey(1000000)


