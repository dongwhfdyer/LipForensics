import os
import shutil
from os.path import join
from pathlib import Path

import numpy as np
from tqdm import tqdm

import logging

formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
fh = logging.FileHandler("rubb/face_landmarks_converting.log")
fh.setLevel(logging.DEBUG)
fh.setFormatter(formatter)

logger.addHandler(fh)
logger.info("--------------------------------------------------")


def read_landmarks():
    landmarks = np.load("preprocessing/20words_mean_face.npy")
    print("hello")


def delete_folders(*folder_path):
    for folder in folder_path:
        if os.path.exists(folder):
            shutil.rmtree(folder)


def create_folders(*folders):
    for folder in folders:
        if not os.path.exists(folder):
            os.makedirs(folder)


def detect_face_and_save_landmarks():
    import face_alignment
    from skimage import io
    # ---------kkuhn-block------------------------------ # param settings
    data_path = "data\\datasets\\CelebDF\\Celeb-synthesis\\images"
    out_path = "data\\datasets\\CelebDF\\Celeb-synthesis\\landmarks"
    # data_path = "data\\datasets\\CelebDF\\Celeb-real\\images"
    # out_path = "data\\datasets\\CelebDF\\Celeb-real\\landmarks"
    delete_folders(out_path)
    create_folders(out_path)
    # ---------kkuhn-block------------------------------

    vid_frames_dir = [join(data_path, item) for item in os.listdir(data_path) if os.path.isdir(join(data_path, item))]
    fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, flip_input=False)
    for vid_frames in tqdm(vid_frames_dir):
        # preds = fa.get_landmarks_from_directory(vid_frames)
        create_folders(vid_frames.replace("images", "landmarks"))
        frames = [join(vid_frames, item) for item in os.listdir(vid_frames)]
        for frame in frames:
            img = io.imread(frame)
            # fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, flip_input=True)
            preds = fa.get_landmarks(img)

            if preds is None:
                # print("no face detected")
                logger.info(f"no face detected in image: {frame}")
                continue
            else:
                # print("face detected")
                np.save(join(vid_frames.replace("images", "landmarks"), Path(frame).name.replace(".png", ".npy")), preds[0])

    # fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, flip_input=False)
    # preds = fa.get_landmarks_from_directory('data/datasets/CelebDF/Celeb-real/images/id0_0000')
    # np.save('data/datasets/CelebDF/Celeb-real/landmarks/id0_0000', preds)
    print("hello")


if __name__ == '__main__':
    read_landmarks()
    # detect_face_and_save_landmarks()
