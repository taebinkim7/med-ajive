from glob import glob
import os
from cbcs_joint.Paths import Paths


def get_avail_images(image_type='he'):
    """
    Returns a list of the file names of all the images available in the
    image directory.

    Parameters
    ----------
    image_type (str): the type of the image to return. Must be one of
        ['he_raw', 'he', 'er_raw', 'er']
    """
    assert image_type in ['he_raw', 'he', 'er_raw', 'er']

    if image_type == 'he_raw':
        image_list = glob('{}/*_he_*'.format(Paths().raw_image_dir))

    elif image_type == 'he':
        image_list = glob('{}/*_he_*'.format(Paths().pro_image_dir))
    
    elif image_type == 'er_raw':
        image_list = glob('{}/*_er_*'.format(Paths().raw_image_dir))
    
    elif image_type == 'er':
        image_list = glob('{}/*_er_*'.format(Paths().pro_image_dir))

    return [os.path.basename(im) for im in image_list]
