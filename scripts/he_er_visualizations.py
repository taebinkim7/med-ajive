import os
from joblib import load
import numpy as np
import matplotlib as mpl
mpl.use('Agg')

from cbcs_joint.Paths import Paths
from cbcs_joint.load_he_er_feats import load_he_er_feats
from cbcs_joint.make_rpvs_for_component import viz_component
from cbcs_joint.utils import retain_pandas
from cbcs_joint.viz_utils import mpl_noaxis

# load pre-computed data e.g. patch features
data = load_he_er_feats()

#patch_feats = patch_feats.drop(['Unnamed: 0'],axis=1)
# subj_img_feats_he = data['subj_img_feats_he']
image_feats_processor = data['image_feats_processor']

# load precomputed AJIVE
ajive = load(os.path.join(Paths().results_dir, 'data', 'fit_ajive'))

# figure config
mpl_noaxis()

n_extreme_cores = 5
n_extreme_patches = 20

top_dir = Paths().results_dir


####################
# joint components #
####################

def viz_joint_comps(image_type):

    patch_dataset = data['patch_dataset_' + image_type]
    patch_feats = data['patch_feats_' + image_type]

    for comp in range(ajive.common.rank):
        comp_name = image_type + '_joint_comp_{}'.format(comp + 1)
        core_scores = ajive.common.scores(norm=True).iloc[:, comp]
        loading_vec = ajive.blocks[image_type].joint.loadings().iloc[:, comp]

        # transform patch features and project onto loadings vector
        patch_scores = retain_pandas(patch_feats,
                                     image_feats_processor.transform).dot(loading_vec)

#         transform core features and project onto loadings vector
#         core_scores = retain_pandas(core_centroids,
#                                     image_feats_processor.transform).dot(loading_vec)
        viz_component(image_type=image_type,
                      core_scores=core_scores,
                      patch_scores=patch_scores,
                      patch_dataset=patch_dataset,
                      loading_vec=loading_vec,
                      comp_name=comp_name,
                      top_dir=top_dir,
                      signal_kind='common',
                      n_extreme_cores=n_extreme_cores,
                      n_extreme_patches=n_extreme_patches)


####################
# image individual #
####################

n_indiv_comps = 5

def viz_indiv_comps(image_type):

    patch_dataset = data['patch_dataset_' + image_type]
    patch_feats = data['patch_feats_' + image_type]
#     core_centroids =

    for comp in range(n_indiv_comps):
        comp_name = image_type + '_indiv_comp_{}'.format(comp + 1)
        print(comp_name)

        core_scores = ajive.blocks[image_type].\
            individual.scores(norm=True).iloc[:, comp]
        loading_vec = ajive.blocks[image_type].\
            individual.loadings().iloc[:, comp]

        # transform patch features and project onto loadings vector
        patch_scores = \
            retain_pandas(patch_feats, image_feats_processor.transform).dot(loading_vec)

        # transform core features and project onto loadings vector
        core_scores = retain_pandas(core_centroids,
                                    image_feats_processor.transform).dot(loading_vec)

        viz_component(image_type=image_type,
                      core_scores=core_scores,
                      patch_scores=patch_scores,
                      patch_dataset=patch_dataset,
                      loading_vec=loading_vec,
                      comp_name=comp_name,
                      top_dir=top_dir,
                      signal_kind=image_type + '_indiv',
                      n_extreme_cores=n_extreme_cores,
                      n_extreme_patches=n_extreme_patches)


viz_joint_comps('he')
viz_joint_comps('er')

# viz_indiv_comps('he')
# viz_indiv_comps('er')
