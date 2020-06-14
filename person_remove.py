import cv2
import argparse
import os
import matplotlib.pyplot as plt
import numpy as np
from model.unet_model import *
from skimage.transform import resize
from scipy.ndimage.morphology import binary_dilation

IMG_WIDTH = 256
IMG_HEIGHT = 256
IMG_CHANNELS = 3
WEIGHT_FILE = r'weights\unet.h5'
    THRESHOLD = 200
MODEL = unet(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS,pretrained_weights=WEIGHT_FILE)
CODEC = 'avc1'
OUTPUT_PATH = r'output\out.mp4'


def video_to_frames(path):
    vid = cv2.VideoCapture(path)
    fps = vid.get(cv2.CAP_PROP_FPS)
    success, frame = vid.read()
    frames = []
    while success:
        frames.append(frame)
        success, frame = vid.read()
    return frames,fps


def resize_img(img):
    return resize(img, (IMG_HEIGHT, IMG_WIDTH), anti_aliasing=True)


def predict_mask(img):
    mask = MODEL.predict([[img]])[0]
    mask = mask.reshape(mask.shape[0], mask.shape[1])
    return mask > THRESHOLD/255


def pixels_to_replace(current_indexes, replace_indexes):
    return current_indexes - replace_indexes


def replace_pixels(frame, pixel_indexes,replace_frame):
    for x,y in pixel_indexes:
        frame[x,y,0] = replace_frame[x,y,0]
        frame[x,y,1] = replace_frame[x,y,1]
        frame[x,y,2] = replace_frame[x,y,2]
    return frame


def video_inpainting(frames):
    person_indexes = get_person_indexes(frames)
    return replace_frames(frames,person_indexes)


def replace_frames(frames,indexes):
    for i in range(len(frames)):
        j = 0
        while indexes[i]:
            try:
                pixels = pixels_to_replace(indexes[i], indexes[j])
                indexes[i] = indexes[i] - pixels
                frames[i] = replace_pixels(frames[i], pixels, frames[j])
                j+=1
            except IndexError:
                frames[i] = image_inpainting(frames[i],indexes[i])
                break
    return frames


def get_person_pixels(mask):
    try:
        pixels_index = np.where(mask)
        pixels_index = set(zip(pixels_index[0],pixels_index[1]))
        return pixels_index
    except IndexError:
        return set()


def image_inpainting(frame,indexes):
    frame = cv2.normalize(src=frame, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX,
                              dtype=cv2.CV_8U)
    mask = np.zeros((IMG_HEIGHT, IMG_WIDTH), dtype=np.uint8)
    for x, y in indexes:
        mask[x, y] = np.uint8(255)
    frame = cv2.inpaint(frame, mask, 2, cv2.INPAINT_TELEA)
    return cv2.normalize(src=frame, dst=None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX,
                              dtype=cv2.CV_64F)


def frames_to_video(frames,fps):
    output = cv2.VideoWriter(OUTPUT_PATH, cv2.VideoWriter_fourcc(*CODEC), fps, (IMG_WIDTH, IMG_HEIGHT))
    for i in range(len(frames)):
        frame = np.uint8(frames[i]*255)
        plt.imsave(r'saved\frames\{0}.png'.format(i),frame[...,::-1])
        output.write(frame)
    output.release()


def get_person_indexes(frames):
    person_indexes = []
    for i in range(len(frames)):
        mask = predict_mask(frames[i])
        mask = binary_dilation(mask,iterations=3)
        plt.imsave(r'saved\masks\{0}.png'.format(i), mask, cmap='Greys')
        person_indexes.append(get_person_pixels(mask))
    return person_indexes


def get_resized_frames(frames):
    resized_frames = []
    for frame in frames:
        frame = resize_img(frame)
        resized_frames.append(frame)
    return resized_frames


def main():
    frames, fps = video_to_frames(args.src_path)
    frames = get_resized_frames(frames)
    frames = video_inpainting(frames)
    frames_to_video(frames, fps)


if __name__ == '__main__':
    a = argparse.ArgumentParser()
    a.add_argument("--src_path", type=str, required=True, help="path to source video")
    args = a.parse_args()
    if os.path.exists(args.src_path):
        main()
    else:
        print('\n\nThis path does not exist,please try again')