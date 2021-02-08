from glob import glob
import os
from cbcs_joint.Paths import Paths


def get_cbcsid_group(idx):
    """
    Returns the cbcs studyid and group from an image name


    CBCS3_HE_30030_group_1_image_restained.png -> 30030
    """
    stub = idx.split('CBCS3_HE_')[1].split('_image')[0]
    cbcsid = stub.split('_group')[0]
    group = stub.split('group_')[1]
    return cbcsid, group


def get_avail_images(image_type='he_processed'):
    """
    Returns a list of the file names of all the images available in the
    image directory.

    Parameters
    ----------
    image_type (str): the type of the image to return. Must be one of
        ['he_raw', 'he_processed', 'er']
    """
    assert image_type in ['he_raw', 'he_processed', 'er']

    if image_type == 'he_raw':
        image_list = glob('{}/*_he_*'.format(Paths().raw_image_dir))

    elif image_type == 'he':
        image_list = glob('{}/*_he_*_restained*'.format(Paths().pro_image_dir))
     
    elif image_type == 'er':
        image_list = glob('{}/*_er_*'.format(Paths().pro_image_dir))

    return [os.path.basename(im) for im in image_list]
