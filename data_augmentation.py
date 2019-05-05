import os
import random
from scipy import ndarray
import numpy as np

# image processing library
import skimage as sk
from skimage import transform
from skimage import util
from skimage import io



def random_rotation(image_array: ndarray):
    random_degree = random.uniform(-25, 25)
    return sk.transform.rotate(image_array, random_degree)
    # pick a random degree of rotation between 25% on the left and 25% on the right


def random_noise(image_array: ndarray):
    # add random noise to the image
    return sk.util.random_noise(image_array)

def horizontal_flip(image_array: ndarray):
    # horizontal flip doesn't need skimage, it's easy as flipping the image array of pixels !
    return image_array[:, ::-1]

# dictionary of the transformations we defined earlier
available_transformations = {
    'rotate': random_rotation,
    'noise': random_noise,
    'horizontal_flip': horizontal_flip
}
def augment(class_path, num_files_desired, dst):
    #class_path = 'images/cat'
    #num_files_desired = 10

    # find all files paths from the folder
    images = [os.path.join(class_path, f) for f in os.listdir(class_path) if os.path.isfile(os.path.join(class_path, f))]

    num_generated_files = 0
    while num_generated_files <= num_files_desired:
        # random image from the folder
        image_path = random.choice(images)
        # read image as an two dimensional array of pixels

        image_to_transform = sk.io.imread(image_path)
        # random num of transformation to apply
        num_transformations_to_apply = random.randint(1, len(available_transformations))

        num_transformations = 0
        transformed_image = None
        while num_transformations <= num_transformations_to_apply:
            # random transformation to apply for a single image
            key = random.choice(list(available_transformations))
            transformed_image = available_transformations[key](image_to_transform)
            num_transformations += 1
        train_dst = '%s/augmented_image_%s.jpg' % (dst, num_generated_files)
        # write image to the disk
        transformed_image = transformed_image / transformed_image.max()  # normalizes data in range 0 - 255
        transformed_image = 255 * transformed_image
        img = transformed_image.astype(np.uint8)
        io.imsave(train_dst, img)
        num_generated_files += 1