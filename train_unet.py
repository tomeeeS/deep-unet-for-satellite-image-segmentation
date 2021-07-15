from unet_model import *
from gen_patches import *

import os.path
import numpy as np
import tifffile as tiff
from tensorflow.python.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, \
    TensorBoard, CSVLogger
from keras_unet_collection import models

N_BANDS = 3  # original: 8
N_CLASSES = 5  # buildings, roads, trees, crops and water
CLASS_WEIGHTS = [0.2, 0.3, 0.1, 0.1, 0.3]
N_EPOCHS = 50  # original: 150
UPCONV = True
PATCH_SZ = 64  # original: 160, should divide by 16
BATCH_SIZE = 16  # original: 150
TRAIN_SZ = 1000  # train size, original: 4000
VAL_SZ = 250  # validation size, original: 1000
RED_BAND_INDEX = 4
GREEN_BAND_INDEX = 2
BLUE_BAND_INDEX = 1


def normalize(img):
    min = img.min()
    max = img.max()
    x = 2.0 * (img - min) / (max - min) - 1.0
    return x


def get_model_unet():
    return unet_model(N_CLASSES, PATCH_SZ, n_channels=N_BANDS, upconv=UPCONV,
                      class_weights=CLASS_WEIGHTS)


def weighted_binary_crossentropy(y_true, y_pred):
    class_loglosses = K.mean(K.binary_crossentropy(y_true, y_pred), axis=[0, 1, 2])
    return K.sum(class_loglosses * K.constant(CLASS_WEIGHTS))


def get_model_resunet_a():
    model = models.resunet_a_2d((PATCH_SZ, PATCH_SZ, N_BANDS), [32, 64, 128, 256, 512, 1024],
                            dilation_num=[1, 3, 15, 31],
                            n_labels=N_CLASSES, aspp_num_down=256, aspp_num_up=128,
                            activation='ReLU', output_activation='Sigmoid',
                            batch_norm=True, pool=False, unpool='nearest', name='resunet')
    model.compile(optimizer="Adam", loss=weighted_binary_crossentropy)
    return model


trainIds = [str(i).zfill(2) for i in range(1, 24)]  # most availiable ids: from "01" to "23"
# I use 24 for testing, because it has ground truth for evaluation, unlike test.tif

train_data = dict()
train_label = dict()
val_data = dict()
val_label = dict()

patch_size_text = '{}'.format(PATCH_SZ)
resunet_a_weights_path = 'weights_resunet_a_p' + patch_size_text
unet_weights_path = 'weights_unet_p' + patch_size_text
if not os.path.exists(resunet_a_weights_path):
    os.makedirs(resunet_a_weights_path)
if not os.path.exists(unet_weights_path):
    os.makedirs(unet_weights_path)
resunet_a_weights_path += '/weights.hdf5'
unet_weights_path += '/weights.hdf5'

if __name__ == '__main__':

    print('Reading images')
    for img_id in trainIds:
        img_m = normalize(tiff.imread('./data/mband/{}.tif'.format(img_id)).transpose([1, 2, 0]))
        ground_truth = tiff.imread('./data/gt_mband/{}.tif'.format(img_id))\
                           .transpose([1, 2, 0]) / 255
        train_xsz = int(3 / 4 * img_m.shape[0])  # use 75% of image as train and 25% for validation
        train_data[img_id] = img_m[:train_xsz, :, :]
        train_label[img_id] = ground_truth[:train_xsz, :, :]
        val_data[img_id] = img_m[train_xsz:, :, :]
        val_label[img_id] = ground_truth[train_xsz:, :, :]
        print(img_id + ' read')
    print('Images were read')

    x_train, y_train = get_patches(train_data, train_label, n_patches=TRAIN_SZ, sz=PATCH_SZ)
    x_val, y_val = get_patches(val_data, val_label, n_patches=VAL_SZ, sz=PATCH_SZ)

    def train_net(weights_path, model):
        print("start train net")
        # if os.path.isfile(weights_path):
        #     model.load_weights(weights_path)
        early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=1,
                                       mode='auto')
        # reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.1, patience=5, min_lr=0.00001)
        model_checkpoint = ModelCheckpoint(weights_path, monitor='val_loss', save_best_only=True)
        csv_logger = CSVLogger('log_unet.csv', append=True, separator=';')
        tensorboard = TensorBoard(log_dir='./tensorboard_unet/', write_graph=True,
                                  write_images=True)
        model.fit(x_train, y_train, batch_size=BATCH_SIZE, epochs=N_EPOCHS,
                  verbose=2, shuffle=True,
                  callbacks=[model_checkpoint, csv_logger, tensorboard, early_stopping],
                  validation_data=(x_val, y_val))
        return model


    train_net(resunet_a_weights_path, get_model_resunet_a())
    train_net(unet_weights_path, get_model_unet())
