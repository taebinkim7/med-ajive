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

    # clinical
    clinical_data = load_clinical_data()

    var_to_drop = ['morpho_type', 'meno_status']
    clinical_data = clinical_data.drop(columns=var_to_drop)

    # genetic data
    genes = load_genes()

    gene_stats = get_gene_statistics(genes)

    ##############
    # image data #
    ##############

    # patches dataset
    patch_data_dir = os.path.join(Paths().patches_dir)
    patch_dataset = CBCSPatchGrid.load(os.path.join(patch_data_dir,
                                                    'patch_dataset'))

    # image patch features
    img_centroids = pd.read_csv(os.path.join(patch_data_dir,
                                             'core_centroids.csv'),
                                index_col=0)
    img_centroids.index = img_centroids.index.astype(str)

    subj_img_feats = cores_to_subj_mean_feats(img_centroids)

    if load_patch_feats:
        patch_feats = \
            pd.read_csv(os.path.join(patch_data_dir, 'patch_features.csv'),
                        index_col=['image', 'patch_idx'])
    else:
        patch_feats = None

    #############
    # alignment #
    #############
    intersection = list(set(subj_img_feats.index).intersection(genes.index))
    in_image, in_genes = get_mismatches(subj_img_feats.index, genes.index)

    # make sure clincial data up with images/genetic data
    in_clinical_not_intersection, in_intersection_not_clinical = \
        get_mismatches(clinical_data.index, intersection)
    print('in clinical, not intersection: {}'.format(len(in_clinical_not_intersection)))
    print('in intersection, not clinical: {}'.format(len(in_intersection_not_clinical)))

    print('intersection: {}'.format(len(intersection)))
    print('in images, not in genes: {}'.format(len(in_image)))
    print('in genes, not in images: {}'.format(len(in_genes)))

    subj_img_feats = subj_img_feats.loc[intersection]
    genes = genes.loc[intersection]
    clinical_data = clinical_data.loc[intersection]

    print(subj_img_feats.shape)
    print(genes.shape)

    # process data
    image_feats_processor = StandardScaler()
    subj_img_feats = retain_pandas(subj_img_feats,
                                   image_feats_processor.fit_transform)

    genes = process_pam50(genes)

    ##################################
    # add hand crafted image features#
    ##################################
    # add proprotion background
    clinical_data.loc[:, 'background'] = \
        get_subj_background(patch_dataset, avail_cbcsids=intersection)

    clinical_data.loc[:, 'background_intensity'] = \
        get_subj_background_intensity(patch_dataset,
                                      avail_cbcsids=intersection)

    # make sure subjects are in same order
    idx = subj_img_feats.index.sort_values()
    subj_img_feats = subj_img_feats.loc[idx]
    genes = genes.loc[idx]
    clinical_data = clinical_data.loc[idx]

    return {'patch_dataset': patch_dataset,
            'img_centroids': img_centroids,
            'subj_img_feats': subj_img_feats,
            'patch_feats': patch_feats,
            'image_feats_processor': image_feats_processor,
            'genes': genes,
            'gene_stats': gene_stats,
            'clinical_data': clinical_data}


def sphere(X):
    s = 1.0 / np.array(X).std(axis=1)
    return np.array(X) * s[:, None]


def cores_to_subj_mean_feats(image_features):
    """
    Returns the mean subject level features from a set of image features
    """
    id_group = [get_cbcsid_group(i) for i in image_features.index]
    cbcsids, groups = list(zip(*id_group))

    image_features_ = image_features.copy()
    image_features_['cbcsids'] = cbcsids

    subj_mean_feats = image_features_.groupby('cbcsids').mean()

    return subj_mean_feats
