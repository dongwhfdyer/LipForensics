"""
Extracts images from (compressed) videos, used for the FaceForensics++ dataset

Usage: see -h or https://github.com/ondyari/FaceForensics

Author: Andreas Roessler
Date: 25.01.2019
"""
import os
import shutil
from os.path import join
import argparse
import subprocess
import cv2
from tqdm import tqdm

DATASET_PATHS = {
    'original': 'original_sequences',
    'Deepfakes': 'manipulated_sequences/Deepfakes',
    'Face2Face': 'manipulated_sequences/Face2Face',
    'FaceSwap': 'manipulated_sequences/FaceSwap'
}
COMPRESSION = ['c0', 'c23', 'c40']


def extract_frames(data_path, output_path, method='cv2'):
    """Method to extract frames, either with ffmpeg or opencv. FFmpeg won't
    start from 0 so we would have to rename if we want to keep the filenames
    coherent."""
    if method == 'ffmpeg':
        subprocess.check_output(
            'ffmpeg -i {} {}'.format(
                data_path, join(output_path, '%04d.png')),
            shell=True, stderr=subprocess.STDOUT)
    elif method == 'cv2':
        reader = cv2.VideoCapture(data_path)
        frame_num = 0
        while reader.isOpened():
            success, image = reader.read()
            if not success:
                break
            cv2.imwrite(join(output_path, '{:04d}.png'.format(frame_num)), image)
            frame_num += 1
        reader.release()
    else:
        raise Exception('Wrong extract frames method: {}'.format(method))


def delete_folders(*folder_path):
    for folder in folder_path:
        if os.path.exists(folder):
            shutil.rmtree(folder)


def create_folders(*folders):
    for folder in folders:
        if not os.path.exists(folder):
            os.makedirs(folder)


def extract_method_videos(data_path, dataset, compression):
    """
    Extracts all videos of a specified method and compression in the
    """

    # ---------kkuhn-block------------------------------ CelebDF file structure

    out_put_path = r"data/datasets/CelebDF"
    sub_dataset_name = ["Celeb-synthesis"]
    # sub_dataset_name = ["Celeb-real"]
    # sub_dataset_name = ["Celeb-synthesis", "Celeb-real","Youtube-real"]

    # sub_dataset_name = [item for item in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, item))]
    for sub_dataset in sub_dataset_name:  # iter for 3 sub-datasets.
        sub_data_path = join(data_path, sub_dataset)
        sub_output_path = join(out_put_path, sub_dataset, 'images')
        delete_folders(sub_output_path)
        create_folders(sub_output_path)
        video_folders = sorted(os.listdir(sub_data_path))
        for video in tqdm(video_folders):
            video_path = join(sub_data_path, video)
            frames_out_path = join(sub_output_path, video.split('.')[0])
            # if os.path.exists(frames_out_path):
            #     continue
            create_folders(frames_out_path)
            extract_frames(video_path, frames_out_path, method='cv2')

    # ---------kkuhn-block------------------------------

    # # ---------kkuhn-block------------------------------FaceForensics++ file structure
    # videos_path = join(data_path, DATASET_PATHS[dataset], compression, 'videos')
    # images_path = join(data_path, DATASET_PATHS[dataset], compression, 'images')
    # for video in tqdm(os.listdir(videos_path)):
    #     image_folder = video.split('.')[0]
    #     extract_frames(join(videos_path, video),
    #                    join(images_path, image_folder))
    # # ---------kkuhn-block------------------------------


def detect_empty_folder(folder_path):
    for f in os.listdir(folder_path):
        # if empty folder, delete it
        if os.path.isdir(join(folder_path, f)):
            if not os.listdir(join(folder_path, f)):
                print("empty folder: {}".format(join(folder_path, f)))
                # shutil.rmtree(join(folder_path, f))
                # print('empty folder deleted: {}'.format(join(folder_path, f)))


if __name__ == '__main__':

    # out_put_path = r"D:\ANewspace\code\LipForensics\data\datasets\CelebDF\Celeb-real\images"
    # detect_empty_folder(out_put_path)

    # ---------kkuhn-block------------------------------ param setting
    data_path = r"d:\download\Celeb-DF-v2"
    dataset = r""
    compression = r""
    # ---------kkuhn-block------------------------------
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    p.add_argument('--data_path', type=str, default=data_path)
    p.add_argument('--dataset', '-d', type=str,
                   choices=list(DATASET_PATHS.keys()) + ['all'],
                   default=dataset)
    p.add_argument('--compression', '-c', type=str, choices=COMPRESSION,
                   default=compression)
    args = p.parse_args()

    if args.dataset == 'all':
        for dataset in DATASET_PATHS.keys():
            args.dataset = dataset
            extract_method_videos(**vars(args))
    else:
        extract_method_videos(**vars(args))
