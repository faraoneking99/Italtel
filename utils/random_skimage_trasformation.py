# dictionary of the transformations functions we defined earlier
import random

from utils import skimage_transformation as skt
available_transformations = {
    'rotate': skt.random_rotation,
    'noise': skt.random_noise,
    'horizontal_flip': skt.horizontal_flip
}
def apply(image_to_transform):
    # random num of transformations to apply
    num_transformations_to_apply = random.randint(1, len(available_transformations))

    num_transformations = 0
    transformed_image = None
    while num_transformations <= num_transformations_to_apply:
        # choose a random transformation to apply for a single image
        key = random.choice(list(available_transformations))
        transformed_image = available_transformations[key](image_to_transform)
        num_transformations += 1
    # define a name for our new file
    return transformed_image