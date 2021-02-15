import pandas as pd
import os
import json
from sklearn.preprocessing import StandardScaler
import numpy as np

from cbcs_joint.Paths import Paths
from cbcs_joint.utils import retain_pandas, get_mismatches
from cbcs_joint.cbcs_utils import get_cbcsid_group
from cbcs_joint.patches.CBCSPatchGrid import CBCSPatchGrid
from cbcs_joint.patches.utils import get_subj_background, get_subj_background_intensity


def load_analysis_data(load_patch_feats=True):
    
    ##############
    # image data #
    ##############

    # patches dataset
    patch_data_dir = os.path.join(Paths().patches_dir)
    patch_dataset_he = CBCSPatchGrid.load(os.path.join(patch_data_dir, 'patch_dataset_he'))
    patch_dataset_er = CBCSPatchGrid.load(os.path.join(patch_data_dir, 'patch_dataset_er'))

    # image patch features
    subj_img_feats_he = pd.read_csv(os.path.join(patch_data_dir, 'core_centroids_he.csv'),
                                index_col=0)
    subj_img_feats_er = pd.read_csv(os.path.join(patch_data_dir, 'core_centroids_er.csv'),
                                index_col=0)    
    subj_img_feats_he.index = subj_img_feats_he.index.astype(str)
    subj_img_feats_er.index = subj_img_feats_er.index.astype(str)
    subj_img_feats_he.index = [[idx[:-17] for idx in subj_img_feats_he.index]
    subj_img_feats_he.index = [[idx[:-17] for idx in subj_img_feats_he.index]                           

    if load_patch_feats:
        patch_feats_he = \
            pd.read_csv(os.path.join(patch_data_dir, 'patch_features_he.csv'),
                        index_col=['image', 'patch_idx'])
        patch_feats_er = \
            pd.read_csv(os.path.join(patch_data_dir, 'patch_features_er.csv'),
                        index_col=['image', 'patch_idx'])        
    else:
        patch_feats_he, patch_feats_er = None, None

    #############
    # alignment #
    #############
    intersection = list(set(subj_img_feats_he.index).intersection(set(subj_img_feats_er.index)))
    in_he, in_er = get_mismatches(subj_img_feats_he.index, subj_img_feats_er.index)

    print('intersection: {}'.format(len(intersection)))
    print('in HE, not in ER: {}'.format(len(in_he)))
    print('in ER, not in HE: {}'.format(len(in_er)))

    subj_img_feats_he = subj_img_feats_he.loc[intersection]
    subj_img_feats_er = subj_img_feats_er.loc[intersection]

    print(subj_img_feats_he.shape)
    print(subj_img_feats_er.shape)

    # process data
    image_feats_processor = StandardScaler()
    subj_img_feats_he = retain_pandas(subj_img_feats_he, image_feats_processor.fit_transform)
    subj_img_feats_er = retain_pandas(subj_img_feats_er, image_feats_processor.fit_transform)

#     ##################################
#     # add hand crafted image features#
#     ##################################
#     # add proprotion background
#     clinical_data.loc[:, 'background'] = \
#         get_subj_background(patch_dataset, avail_cbcsids=intersection)

#     clinical_data.loc[:, 'background_intensity'] = \
#         get_subj_background_intensity(patch_dataset,
#                                       avail_cbcsids=intersection)

    # make sure subjects are in same order
    idx = subj_img_feats_he.index.sort_values()
    subj_img_feats_he = subj_img_feats_he.loc[idx]
    subj_img_feats_er = subj_img_feats_er.loc[idx]

    return {'subj_img_feats_he': subj_img_feats_he,
            'subj_img_feats_er': subj_img_feats_er,
            'patch_feats_he': patch_feats_he,
            'patch_feats_er': patch_feats_er}


def sphere(X):
    s = 1.0 / np.array(X).std(axis=1)
    return np.array(X) * s[:, None]
