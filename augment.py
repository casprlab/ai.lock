__author__ = 'Sean Huver'
__email__ = 'huvers@gmail.com'

import sys
import os
import numpy as np
import errno
from PIL import ImageFilter
from PIL import ImageEnhance
from PIL import Image
import random

def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise

def load_np_image(infilename):
    img = Image.open(infilename)
    img.load()
    data = np.asarray(img, dtype="int32")
    return data


def save_image(npdata, outfilename):
    img = Image.fromarray(np.asarray(np.clip(npdata, 0, 255), dtype="uint8"))
    img.save(outfilename)


def make_blur_img(file_name, path_dir):
    file_path = path_dir + '/' + file_name
    img = Image.open('%s' % file_path)
    img.filter(ImageFilter.BLUR).save(path_dir + '/' + '%02s_blur'
                                      % file_name.translate(None, '.jpg') + '.jpg', "JPEG")
    return


def make_bright_img(file_name, path_dir):
    file_path = path_dir + '/' + file_name
    img = Image.open('%s' % file_path)
    ImageEnhance.Brightness(img).enhance(1.5).save(path_dir + '/' + '%02s_dim'
                                                   % file_name.translate(None, '.jpg') + '.jpg', "JPEG")
    return


def make_clr_chg(file_name, path_dir):
    file_path = path_dir + '/' + file_name
    img = Image.open('%s' % file_path)
    ImageEnhance.Color(img).enhance(0.8).save(path_dir + '/' + '%02s_clr'
                                              % file_name.translate(None, '.jpg') + '.jpg', "JPEG")
    return


def make_contrast_chg(file_name, path_dir):
    file_path = path_dir + '/' + file_name
    img = Image.open('%s' % file_path)
    ImageEnhance.Contrast(img).enhance(0.8).save(path_dir + '/' + '%02s_ctrt'
                                                 % file_name.translate(None, '.jpg') + '.jpg', "JPEG")
    return


def make_rot_img(file_name, path_dir):
    file_path = os.path.join(path_dir, file_name)
    img = Image.open('%s' % file_path)
    img.rotate(180).save(path_dir + '/' + '%02s_rot1' % file_name.translate(None, '.jpg') + '.jpg', "JPEG")
    img.rotate(90).save(path_dir + '/' + '%02s_rot2' % file_name.translate(None, '.jpg') + '.jpg', "JPEG")
    img.rotate(-90).save(path_dir + '/' + '%02s_rot3' % file_name.translate(None, '.jpg') + '.jpg', "JPEG")
    img.rotate(-180).save(path_dir + '/' + '%02s_rot4' % file_name.translate(None, '.jpg') + '.jpg', "JPEG")


def make_noise_img(file_name, path_dir):
    file_path = path_dir + '/' + file_name
    img = load_np_image('%s' % file_path)
    array_shape = img.shape
    noise = np.random.randint(1, 100, size=array_shape)
    img += noise
    rand_name = str(np.random.randint(1, 1000))
    file_name = path_dir + '/' + rand_name + '%02s_nz' % file_name.translate(None, '.jpg') + '.jpg'
    save_image(img, file_name)


def main():
    target_path_dir = "Datasets/Nexus"
    for image in [m for m in os.listdir(target_path_dir) if ".jpg" in m]:
        if (image[0] != "."):
            make_rot_img(image, target_path_dir)

    for image in [m for m in os.listdir(target_path_dir) if ".jpg" in m]:
        if (image[0] != "."):
            make_bright_img(image, target_path_dir)
            make_clr_chg(image, target_path_dir)
            make_contrast_chg(image, target_path_dir)

if __name__ == "__main__":
    main()