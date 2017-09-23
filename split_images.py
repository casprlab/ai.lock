import os
import numpy as np
import skimage
from skimage import io
import errno
import math
import sys

def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise
def crop_image_4_sqr(file_name, path_dir, out_path_dir):
    file_path = path_dir + '/' + file_name
    print(file_name)
    img = io.imread(file_path)
    w, h = img.shape[0], img.shape[1]
    w2, h2 = int(math.ceil(w/2)), int(math.ceil(h/2))
    overlap_w = 50
    overlap_h = 51
    folder_name = "Cropped_5overlapping_sqr"
    
    try:
        cropped = img[0:w2+overlap_w,0:h2+overlap_h]
        io.imsave(os.path.join(out_path_dir, folder_name , "part_1" , os.path.splitext(file_name)[0] + "_1.jpg"), cropped)
        cropped = img[0:w2+overlap_w,h2-overlap_h:h]
        io.imsave(os.path.join(out_path_dir, folder_name , "part_2" ,os.path.splitext(file_name)[0] + "_2.jpg"), cropped)
        cropped = img[w2-overlap_w:w,0:int(math.ceil(h/2)+overlap_h)]
        io.imsave(os.path.join(out_path_dir, folder_name , "part_3" ,os.path.splitext(file_name)[0] + "_3.jpg"), cropped)
        cropped = img[w2-overlap_w:w,h2-overlap_h:h]
        io.imsave(os.path.join(out_path_dir, folder_name , "part_4" ,os.path.splitext(file_name)[0] + "_4.jpg"), cropped)
        size_h, size_w = 186, 206
        cropped = img[w2- (size_w/2) :w2 +(size_w/2),h2 - (size_h/2):h2 + (size_h/2)]
        io.imsave(os.path.join(out_path_dir, folder_name, "part_5" ,os.path.splitext(file_name)[0] + "_5.jpg"), cropped)
    except:
        print("error occured for; " + file_name)
        pass

def main():
    target_path_dir = "Image_Datasets/Nexus"# sys.argv[1]
    out_path_dir = 'Image_Datasets/Nexus/Splits'
    num_noise_imgs = 50

    for sub_dir in os.walk(target_path_dir).next()[1]:
        print(sub_dir)
        for image in os.listdir(target_path_dir + '/' + sub_dir):
            if not os.path.isdir(os.path.join(target_path_dir , sub_dir, image)):
                output_path = os.path.join(out_path_dir + "/Cropped_5overlapping_sqr/")
                if not os.path.exists(output_path):
                    os.mkdir(output_path)
                    for part in range(1,6):
                        os.mkdir(os.path.join(output_path, "part_" + str(part)))
                if(image[0] != "." and ".jpg" in image):
                    crop_image_4_sqr(image, target_path_dir + '/' + sub_dir, out_path_dir)
                    

if __name__ == "__main__":
    main()