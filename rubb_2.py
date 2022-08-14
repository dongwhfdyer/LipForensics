import numpy as np


def read_landmarks():
    landmarks = np.load("preprocessing/20words_mean_face.npy")
    landmarks_2 = np.load("data/datasets/CelebDF/Celeb-real/landmarks/id0_0000/0001.npy")
    print("hello")
if __name__ == '__main__':
    read_landmarks()
