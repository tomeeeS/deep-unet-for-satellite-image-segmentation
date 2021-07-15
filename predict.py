import math
import numpy as np
import matplotlib.pyplot as plt
import tifffile as tiff
from rasterio.plot import show

from train_unet import resunet_a_weights_path, get_model_resunet_a, normalize, PATCH_SZ, N_CLASSES, \
    val_data, val_label, get_model_unet, unet_weights_path, TRAIN_SZ, VAL_SZ, BATCH_SIZE, \
    RED_BAND_INDEX, GREEN_BAND_INDEX, BLUE_BAND_INDEX

CLASSIFICATION_THRESHOLD = 0.5


def predict(x, model, patch_sz=160, n_classes=5):
    img_height = x.shape[0]
    img_width = x.shape[1]
    n_channels = x.shape[2]
    # make extended img so that it contains integer number of patches
    npatches_vertical = math.ceil(img_height / patch_sz)
    npatches_horizontal = math.ceil(img_width / patch_sz)
    extended_height = patch_sz * npatches_vertical
    extended_width = patch_sz * npatches_horizontal
    ext_x = np.zeros(shape=(extended_height, extended_width, n_channels), dtype=np.float32)
    # fill extended image with mirrors:
    ext_x[:img_height, :img_width, :] = x
    for i in range(img_height, extended_height):
        ext_x[i, :, :] = ext_x[2 * img_height - i - 1, :, :]
    for j in range(img_width, extended_width):
        ext_x[:, j, :] = ext_x[:, 2 * img_width - j - 1, :]

    # now we assemble all patches in one array
    patches_list = []
    for i in range(0, npatches_vertical):
        for j in range(0, npatches_horizontal):
            x0, x1 = i * patch_sz, (i + 1) * patch_sz
            y0, y1 = j * patch_sz, (j + 1) * patch_sz
            patches_list.append(ext_x[x0:x1, y0:y1, :])
    # model.predict() needs numpy array rather than a list
    patches_array = np.asarray(patches_list)
    # predictions:
    patches_predict = model.predict(patches_array, batch_size=4)
    prediction = np.zeros(shape=(extended_height, extended_width, n_classes), dtype=np.float32)
    for k in range(patches_predict.shape[0]):
        i = k // npatches_horizontal
        j = k % npatches_horizontal
        x0, x1 = i * patch_sz, (i + 1) * patch_sz
        y0, y1 = j * patch_sz, (j + 1) * patch_sz
        prediction[x0:x1, y0:y1, :] = patches_predict[k, :, :, :]
    return prediction[:img_height, :img_width, :]


def segmentation_map_from_mask(mask, threshold=0.5):
    colors = {
        0: [150, 150, 150],  # Buildings
        1: [223, 194, 125],  # Roads & Tracks
        2: [27, 120, 55],    # Trees
        3: [166, 219, 160],  # Crops
        4: [116, 173, 209]   # Water
    }
    z_order = {
        1: 3,
        2: 4,
        3: 0,
        4: 1,
        5: 2
    }
    image = 255 * np.ones(shape=(3, mask.shape[1], mask.shape[2]), dtype=np.uint8)  # white by default
    categories = -1 * np.ones(shape=(mask.shape[1], mask.shape[2]), dtype=np.uint8)  # other by default
    for i in z_order:
        category = z_order[i]
        categories[:, :][mask[category, :, :] > threshold] = category
        for ch in range(3):  # R, G, B
            image[ch, :, :][mask[category, :, :] > threshold] = colors[category][ch]
    return categories, image


def save_images():
    tiff.imsave('ground_truth_map.tif', ground_truth_image)
    # tiff.imsave('result.tif', (255 * mymat).astype('uint8'))
    tiff.imsave('map_resunet_a.tif', predicted_image_resunet_a)
    tiff.imsave('map_unet.tif', predicted_image_unet)


def plot():
    total_pixel_count = predicted_categories_unet.size
    correct_pixel_count_unet = np.count_nonzero(
        predicted_categories_unet == ground_truth_categories
    )
    correct_pixel_count_resunet_a = np.count_nonzero(
        predicted_categories_resunet_a == ground_truth_categories
    )
    correct_pixel_percentage_unet = correct_pixel_count_unet / total_pixel_count * 100
    correct_pixel_percentage_resunet_a = correct_pixel_count_resunet_a / total_pixel_count * 100

    fig, ax = plt.subplots(2, 2, figsize=[15, 12])
    fig.suptitle("Predictions when the patch size is {p}x{p},"
                 " training/validation data size: {t}/{v}"
                 .format(p=PATCH_SZ, t=TRAIN_SZ, v=VAL_SZ), fontsize=16)
    show(
        printed_test_img,
        ax=ax[0, 0],
        title='original image'
    )
    show(ground_truth_image, ax=ax[0, 1], title='ground truth')
    show(
        predicted_image_unet,
        ax=ax[1, 0],
        title='U-Net prediction, accuracy: {p:.2f}'.format(p=correct_pixel_percentage_unet)
    )
    show(
        predicted_image_resunet_a,
        ax=ax[1, 1],
        title='ResUNet-a prediction, accuracy: {p:.2f}'.format(p=correct_pixel_percentage_resunet_a)
    )
    plt.show()


def normalize_for_print(img_band):
    min = np.percentile(img_band.flatten(), 0)
    max = np.percentile(img_band.flatten(), 99)
    x = (img_band - min) / (max - min)
    return x


def assemble_print_test_img():
    print_img = test_img
    # tried to see the images in normal colors, but it was just blank every time.
    # Images are from SpaceNet.
    # "The 8-band, multispectral images include the following bands: Coastal Blue, Blue, Green,
    #   Yellow, Red, Red Edge, Near Infrared 1 (NIR1), and Near Infrared 2 (NIR2)"
    red = normalize_for_print(print_img[RED_BAND_INDEX])
    green = normalize_for_print(print_img[GREEN_BAND_INDEX])
    blue = normalize_for_print(print_img[BLUE_BAND_INDEX])
    return [red, green, blue]


if __name__ == '__main__':
    resunet_a_model = get_model_resunet_a()
    resunet_a_model.load_weights(resunet_a_weights_path)
    unet_model = get_model_unet()
    unet_model.load_weights(unet_weights_path)

    test_id = '24'
    test_img_file = 'data/mband/{}.tif'.format(test_id)
    test_img = tiff.imread(test_img_file)

    img = normalize(test_img)

    printed_test_img = assemble_print_test_img()

    img = img.transpose([1, 2, 0])  # make channels last

    predicted_resunet_a = predict(img, resunet_a_model, patch_sz=PATCH_SZ, n_classes=N_CLASSES)\
        .transpose([2, 0, 1])
    predicted_unet = predict(img, unet_model, patch_sz=PATCH_SZ, n_classes=N_CLASSES)\
        .transpose([2, 0, 1])

    (predicted_categories_resunet_a, predicted_image_resunet_a) = \
        segmentation_map_from_mask(predicted_resunet_a, CLASSIFICATION_THRESHOLD)
    (predicted_categories_unet, predicted_image_unet) = \
        segmentation_map_from_mask(predicted_unet, CLASSIFICATION_THRESHOLD)
    ground_truth = tiff.imread('./data/gt_mband/{}.tif'.format(test_id)) / 255
    (ground_truth_categories, ground_truth_image) = segmentation_map_from_mask(ground_truth, CLASSIFICATION_THRESHOLD)

    # save_images()
    plot()
