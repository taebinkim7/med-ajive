import os
import matplotlib.pyplot as plt
import numpy as np
from shutil import copyfile
from imageio import imsave
from PIL import Image

from cbcs_joint.Paths import Paths
from cbcs_joint.viz_utils import savefig
from cbcs_joint.patches.utils import plot_coord_ticks, get_patch_map
from cbcs_joint.jitter import jitter_hist


def viz_component(image_type, core_scores,
                  patch_scores, patch_dataset,
                  loading_vec, comp_name,
                  top_dir, signal_kind=None,
                  n_extreme_cores=5, n_extreme_patches=20):

    """
    Visualizations for interpreting a component. Looks at core and
    patch level.
    """

    ###########################
    # histogram of all scores #
    ###########################
    scores_dir = get_and_make_dir(top_dir, signal_kind, 'scores')

    jitter_hist(core_scores, hist_kws={'bins': 100})
    plt.xlabel('core scores')
    savefig(os.path.join(scores_dir,
                         '{}_core_scores.png'.format(comp_name)))

    jitter_hist(patch_scores, hist_kws={'bins': 100})
    plt.xlabel('patch scores')
    savefig(os.path.join(scores_dir,
                         '{}_patch_scores.png'.format(comp_name)))

    ##############
    # core level #
    ##############

    # most extreme cores by scores
    extreme_cores = get_most_extreme(core_scores, 4) # n_extreme_cores
    displayed_core_scores = []

    inches = 5
    n_cols = 2
    n_rows = 2

    for extr in extreme_cores.keys():
        folder_list = [signal_kind, 'cores', comp_name, extr]
        save_dir = get_and_make_dir(top_dir, *folder_list)

        plt.figure(figsize=[inches * n_cols, inches * n_rows])
        grid = plt.GridSpec(nrows=n_rows, ncols=n_cols)

        for core_rank, core_id in enumerate(extreme_cores[extr]):
            displayed_core_scores.append(core_scores[core_id])
            old_fpath = os.path.join(Paths().pro_image_dir, core_id)
            new_name = '{}_{}'.format(core_rank, core_id)
            new_fpath = os.path.join(save_dir, new_name)
            copyfile(old_fpath, new_fpath)

            core = np.array(Image.open(old_fpath))
            r_idx = core_rank // n_cols
            c_idx = core_rank % n_cols

            plt.subplot(grid[r_idx, c_idx])
            plt.imshow(core)

        savefig(os.path.join(save_dir, 'top_cores.png'))

    # plot core level scores histogram
    jitter_hist(core_scores.values, hist_kws={'bins': 100})
    plt.xlabel('{}, {} core scores'.format(signal_kind, comp_name))
    for s in displayed_core_scores:
        plt.axvline(s, color='red')
    save_dir = get_and_make_dir(top_dir, signal_kind, 'cores', comp_name)
    savefig(os.path.join(save_dir, 'scores.png'))


    ###############
    # patch level #
    ###############
    # sorts patches by scores (ignoring core grouping)

    # plot extreme patches
    # extreme_patches = get_most_extreme(patch_scores, n_extreme_patches)
    #
    # # plot patches
    # displayed_patch_scores = []
    #
    # inches = 5
    # n_cols = n_extreme_cores
    # n_rows = int(np.ceil(n_extreme_patches / n_cols))
    #
    # for extr in extreme_patches.keys():
    #     folder_list = [signal_kind, 'patches', comp_name, extr]
    #     save_dir = get_and_make_dir(top_dir, *folder_list)
    #
    #     plt.figure(figsize=[inches * n_cols, inches * n_rows])
    #     grid = plt.GridSpec(nrows=n_rows, ncols=n_cols)
    #
    #     for patch_rank, patch_name in enumerate(extreme_patches[extr]):
    #         image_key, patch_idx = patch_name
    #         displayed_patch_scores.append(patch_scores[patch_name])
    #         patch = patch_dataset.load_patches(image_key, patch_idx)
    #         name = '{}_{}_patch_{}'.format(patch_rank, image_key, patch_idx)
    #
    #         # save image (should not repeat this step)
    #         # imsave(os.path.join(save_dir, '{}.png'.format(name)), patch)
    #
    #         # generate subplot
    #         # top_left = patch_dataset.top_lefts_[image_key][patch_idx]
    #
    #         r_idx = patch_rank // n_cols
    #         c_idx = patch_rank % n_cols
    #
    #         plt.subplot(grid[r_idx, c_idx])
    #         plt.imshow(patch)
    #         # plot_coord_ticks(top_left=top_left, size=patch_dataset.patch_size)
    #         # plt.title('({}), {}, patch {} '.format(patch_rank + 1, image_key,
    #         #                                        patch_idx))
    #         # plt.title('{}'.format(patch_rank + 1))
    #
    #     savefig(os.path.join(save_dir, 'top_patches.png'))
    #
    # # plot patch level scores histogram
    # jitter_hist(patch_scores.values, hist_kws={'bins': 100})
    # plt.xlabel('{}, {} scores'.format(signal_kind, comp_name))
    # for s in displayed_patch_scores:
    #     plt.axvline(s, color='red')
    # save_dir = get_and_make_dir(top_dir, signal_kind, 'patches', comp_name)
    # savefig(os.path.join(save_dir, 'scores.png'))
    #
    # ################################
    # # cores containing top patches #
    # ################################
    # # shows the cores that top patches belong to
    #
    # inches = 5
    # n_cols = n_extreme_cores
    # n_rows = 3
    #
    # for extr in extreme_cores.keys():
    #     folder_list = [signal_kind, 'core_patches', comp_name, extr]
    #     save_dir = get_and_make_dir(top_dir, *folder_list)
    #
    #     plt.figure(figsize=[inches * n_cols, inches * n_rows])
    #     grid = plt.GridSpec(nrows=n_rows, ncols=n_cols)
    #
    #     for i in range(n_extreme_cores):
    #         image_key, patch_idx = extreme_patches[extr][i]
    #         core_path = os.path.join(Paths().pro_image_dir, image_key)
    #         core = np.array(Image.open(core_path))
    #         patch = patch_dataset.load_patches(image_key, patch_idx)
    #         patch_map = get_patch_map(patch_dataset, image_key=image_key,
    #                                   patch_idxs=[patch_idx])
    #         masked_core = np.array(overlay_alpha_mask(core, patch_map > 0.0))
    #
    #         plt.subplot(grid[0, i])
    #         plt.imshow(core)
    #         ##################### TODO #####################
    #         # plt.title('{}'.format()) # percentile of core
    #
    #         plt.subplot(grid[1, i])
    #         plt.imshow(masked_core)
    #
    #         plt.subplot(grid[2, i])
    #         plt.imshow(patch)
    #
    #     savefig(os.path.join(save_dir, 'top_core_patches.png'))


def overlay_alpha_mask(image, mask, alpha=.2):
    """
    mask: Mask of pixels to highlight
    """

    alpha_mask = np.zeros(mask.shape)
    # alpha_mask[mask] = alpha_highlight
    alpha_mask[mask] = 1
    alpha_mask[~mask] = alpha
    alpha_mask = np.uint8(255 * alpha_mask)

    x = np.array(image)
    assert x.shape[2] == 3  # make sure image is rgb
    rbga = np.zeros((x.shape[0], x.shape[1], x.shape[2] + 1), dtype=np.uint8)
    rbga[:, :, 0:3] = x
    rbga[:, :, 3] = alpha_mask

    return Image.fromarray(rbga)


def get_most_extreme(values, n=5, ret='dict', middle=False):
    """
    Returns the most extreme positive and negative indices for a pd.Series
    """

    sorted_values = values.sort_values(ascending=False)

    most_pos = np.array(sorted_values.index[0:n])
    most_neg = np.array(sorted_values.index[-n:])[::-1]
    # most_neg = np.array(values.sort_values(ascending=True).index[0:n])

    if middle:
        middle = len(values) // 2
        left = middle - n // 2
        right = left + n
        middle_inds = np.array(sorted_values.index[left:right])

        if ret == 'dict':
            return {'pos': most_pos, 'middle': middle_inds, 'neg': most_neg}
        else:
            return most_pos, middle_inds, most_neg

    else:
        if ret == 'dict':
            return {'pos': most_pos, 'neg': most_neg}
        else:
            return most_pos, most_neg


def get_and_make_dir(*path):
    """
    Returns a path and creates the directory
    """

    paths_ = []
    for p in path:
        if p is not None:
            paths_.append(p)

    path = os.path.join(*paths_)
    os.makedirs(path, exist_ok=True)
    return path
