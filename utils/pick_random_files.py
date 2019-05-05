import random
import os
from utils import random_skimage_trasformation as rskt

# our folder path containing some images
import skimage as sk

def transform(folder_path, num_files_desired, dst_path):
    #folder_path = 'images/cats'
    # the number of file to generate
    #num_files_desired = 1000

    # loop on all files of the folder and build a list of files paths
    images = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]

    num_generated_files = 0
    while num_generated_files <= int(num_files_desired):
        # random image from the folder
        image_path = random.choice(images)
        # read image as an two dimensional array of pixels
        image_to_transform = sk.io.imread(image_path)
        transformed_image = rskt.apply(image_to_transform)
        # define a name for our new file
        new_file_path = '%s/augmented_image_%s.jpg' % (dst_path, num_generated_files)
        # write image to the disk
        sk.io.imsave(new_file_path, transformed_image)
