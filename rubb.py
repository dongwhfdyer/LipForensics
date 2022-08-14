import math

import cv2
import numpy as np
import matplotlib.pyplot as plt

from skimage import data
from skimage import transform

text = data.text()
height = 70
width = 300
src = np.array([[0, 0], [0, height], [width, height], [width, 0]])
dst = np.array([[155, 15], [65, 40], [260, 130], [360, 95]])

tform3 = transform.ProjectiveTransform()
tform_v4 = transform.estimate_transform("euclidean", dst, src)
# tform_v4 = transform.estimate_transform("similarity", src, dst)

tform3.estimate(src, dst)
warped = transform.warp(text, tform3, output_shape=(height, width))
warped_v4 = transform.warp(text, inverse_map=tform_v4.inverse, output_shape=(400, 300))
# warped_v4 = transform.warp(text, inverse_map=tform_v4.inverse, output_shape=(400,300))

fig, ax = plt.subplots(nrows=2, figsize=(8, 3))
# new_img = cv2.warpAffine(text, tform_v4, (height, width))
# cv2.imshow('new_img', new_img)
# cv2.waitKey(1000000)

ax[0].imshow(text, cmap=plt.cm.gray)
ax[0].plot(dst[:1, 0], dst[:1, 1], '.r')

# ax[0].plot(dst[:, 0], dst[:, 1], '.r')

ax[1].imshow(warped_v4, cmap=plt.cm.gray)
# ax[1].imshow(warped, cmap=plt.cm.gray)

ax[1].plot(src[:1, 0], src[:1, 1], '.r')
for a in ax:
    a.axis('off')

plt.tight_layout()
plt.show()

# import numpy as np
# from skimage import transform
#
# # estimate transformation parameters
# src = np.array([0, 0, 10, 10]).reshape((2, 2))
# dst = np.array([12, 14, 1, -20]).reshape((2, 2))
# tform = transform.estimate_transform('similarity', src, dst)
# all_close_v1 = np.allclose(tform.inverse(tform(src)), src)
#
# # warp image using the estimated transformation
# from skimage import data
#
# image = data.camera()
# transform.warp(image, inverse_map=tform.inverse)
# # create transformation with explicit parameters
# tform2 = transform.SimilarityTransform(scale=1.1, rotation=1, translation=(10, 20))
# # unite transformations, applied in order from left to right
# tform3 = tform + tform2
# all_close_v2 = np.allclose(tform3(src), tform2(tform(src)))
# print("hello")
