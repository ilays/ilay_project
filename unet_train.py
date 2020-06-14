from keras.callbacks import EarlyStopping, ModelCheckpoint
from model.unet_model import *
from keras.preprocessing.image import ImageDataGenerator
import numpy as np

IMG_WIDTH = 256
IMG_HEIGHT = 256
IMG_CHANNELS = 3
TRAIN_PATH = r'data\train'
MODEL = unet(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS)


def adjustData(img, mask):
    if np.max(img) > 1:
        img = img / 255
        mask = mask / 255
        mask[mask > 0.5] = 1
        mask[mask <= 0.5] = 0
    return img, mask


def trainGenerator(batch_size, train_path, image_folder, mask_folder, aug_dict, image_color_mode="rgb",
                   mask_color_mode="grayscale", image_save_prefix="image", mask_save_prefix="mask", target_size=(256, 256),
                   seed=1):
    '''
    can generate image and mask at the same time
    use the same seed for image_datagen and mask_datagen to ensure the transformation for image and mask is the same
    if you want to visualize the results of generator, set save_to_dir = "your path"
    '''
    image_datagen = ImageDataGenerator(**aug_dict)
    mask_datagen = ImageDataGenerator(**aug_dict)
    image_generator = image_datagen.flow_from_directory(
        train_path,
        classes=[image_folder],
        class_mode=None,
        color_mode=image_color_mode,
        target_size=target_size,
        batch_size=batch_size,
        save_prefix=image_save_prefix,
        seed=seed)
    mask_generator = mask_datagen.flow_from_directory(
        train_path,
        classes=[mask_folder],
        class_mode=None,
        color_mode=mask_color_mode,
        target_size=target_size,
        batch_size=batch_size,
        save_prefix=mask_save_prefix,
        seed=seed)
    train_generator = zip(image_generator, mask_generator)
    for (img, mask) in train_generator:
        img, mask = adjustData(img, mask)
        yield img, mask


def train_model():
    data_gen_args = dict(rotation_range=0.2, width_shift_range=0.05, height_shift_range=0.05, shear_range=0.05,
                         zoom_range=0.05, horizontal_flip=True, fill_mode='nearest')
    train_images = trainGenerator(16, TRAIN_PATH, 'images', 'masks', data_gen_args, save_to_dir=None)
    callbacks = [EarlyStopping(patience=5, verbose=1),
                 ModelCheckpoint(r'weights\unet_weights.h5', monitor='loss', verbose=1, save_best_only=True)]
    MODEL.fit_generator(train_images, steps_per_epoch=17706//16, epochs=50, callbacks=callbacks)


if __name__ == '__main__':
    train_model()
