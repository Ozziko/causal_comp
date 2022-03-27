
from dataclasses import dataclass

import pandas as pd
import numpy as np
import random

from collections import defaultdict
from contextlib import contextmanager

import pickle
import json
import os
from pathlib import Path
from os.path import join as pjoin
from tenacity import retry
from tqdm.auto import tqdm
from copy import deepcopy
import shutil
from time import sleep

from sklearn.model_selection import ParameterGrid
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression

import torch
import torch.nn as nn
import torch.optim as optim

from skorch import NeuralNetClassifier
from skorch.callbacks import EpochScoring, Checkpoint
from skorch.dataset import ValidSplit


@dataclass
class Args:
    outputs_dir: str = 'analysis'

    dataset_name: str = 'ao_clevr'
    data_dir: str = '' # empty = f'data/{args.dataset_name}'
    dataset_variant: str = 'VT'
    # dataset_variant: str = 'OZ'
    
    seen_seed:int = 0

    num_split:int = 5000
    init_seed: int = 0 # for torch seeding
    
    skorch_val_ratio: float = 0.2 # split is done inside skorch training; NOT related to 'val' dataset used for calibration, hypertuning
    batch_size: int = 1024 # also for sampling multi-shifts (multi-shifted train) for MLLS training

    # only for OZ splits
    train_size: int = int(80e3)
    train_unseen_ovr_seen: float = 0
    test_size: int = int(8e3)

    # only for VisProd
    VP_mode: str = 'CE' # multi-class VisProd
    # VP_mode: str = 'BCE' # binary multi-label VisProd
    # VP_calibrate: bool = False
    VP_calibrate: bool = True
    # VP_held_out_from:str = 'train'
    VP_held_out_from:str = 'val'
    VP_held_out_ratio: float = 0.2 # held-out taken from train for calibration (only if VP_calibrate=True)


# %% [markdown]
# Defitions


def display(df):
    pass


def torch_setup(deterministic: bool = True):
    # Setting up torch, seeding (https://pytorch.org/docs/stable/notes/randomness.html)
    # ///////////// Making pytorch deterministic (reproducible)  ////////////////////////////////////////////////////
    if deterministic:
        # read WELL: https://pytorch.org/docs/stable/notes/randomness.html
        # the order of execution is important!

        # https://docs.nvidia.com/cuda/cublas/index.html#cublasApi_reproducibility
        #     os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8" # may limit overall performance
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"  # will increase library footprint in GPU memory by approximately 24MiB
        torch.backends.cudnn.benchmark = False
        print(f"executed: torch.backends.cudnn.benchmark = False")

        torch_set_deterministic = False
        # try:
        #     torch.set_deterministic(True) # beta in torch==1.7
        #     print(f"executed torch.set_deterministic(True)")
        #     torch_set_deterministic = True
        # except Exception as ex:
        #     logger.warning(f"torch.set_deterministic(True) failed: {ex}")
        try:
            torch.use_deterministic_algorithms(True)  # beta in torch==1.8
            print(f"executed: torch.use_deterministic_algorithms(True)")
            torch_set_deterministic = True
        except Exception as ex:
            print(f"torch.use_deterministic_algorithms(True) failed: {ex}")

        if not torch_set_deterministic:  # already contained in torch.use_deterministic_algorithms(True)
            torch.backends.cudnn.deterministic = True
            print(f"executed: torch.backends.cudnn.deterministic = True")

        print(f"torch and cuda will be deterministic (after seeding)")
    else:
        torch.backends.cudnn.benchmark = True
        print(f"executed: torch.backends.cudnn.benchmark = True (torch is not determinisitc!)")
        
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        print('torch is using %s (%s)'%(device, torch.cuda.get_device_name(device=0)))
    else:
        print('torch is using %s'%(device))
    return device

def seed_all(seed: int):
    random.seed(seed)
    np.random.seed(seed + 1000)
    torch.manual_seed(seed + 2000)
    torch.cuda.manual_seed(seed + 3000)
    print(f"completed: seeding with seed={seed} in steps of 1000 for random,np,torch,cuda")


def basic_EM(Y, Y_probs, prior_source, EM_iterations: int = 100):
    # init
    EM_results = pd.DataFrame(index=range(EM_iterations))
    EM_results.index.name = 'iteration'
    shift_probs_0 = Y_probs
    shift_probs = shift_probs_0
    prior_shift_pred = prior_source

    # eval
    shift_preds = shift_probs.argmax(axis=1)
    positives = shift_preds==Y
    # EM_results.loc[0, [f'accuracy ({label})' for label in range(args.n_labels)]] = positives.mean(axis=0)
    EM_results.loc[0, 'accuracy'] = positives.mean()
    Y_true_pos_probs = np.take_along_axis(shift_probs, Y.reshape(-1,1), axis=1)
    EM_results.loc[0, 'log-likelihood'] = np.log(Y_true_pos_probs).sum()

    # EM
    for i_EM in tqdm(range(1, 1+EM_iterations)):        
        # E-step
        shift_probs = shift_probs_0 * prior_shift_pred / prior_source
        shift_probs = (shift_probs.T / shift_probs.sum(axis=1)).T
        
        shift_preds = shift_probs.argmax(axis=1)
        
        # eval
        positives = shift_preds==Y
        EM_results.loc[i_EM, 'accuracy'] = positives.mean()
        Y_true_pos_probs = np.take_along_axis(shift_probs, Y.reshape(-1,1), axis=1)
        EM_results.loc[i_EM, 'log-likelihood'] = np.log(Y_true_pos_probs).sum()

        # M-step
        prior_shift_pred = shift_probs.mean(axis=0)
    assert (~np.isfinite(shift_probs)).sum() == 0
    return EM_results, shift_probs


def EM(Y:dict, Y_probs:dict, EM_iterations: int = 100, prior_source_phase: str = 'est. soft train',
        init_prior: str = 'source prior', n_combs: int = None,
        soft_prior: bool = True, over_update_probs: bool = False):
    """
    Y and Y_probs: dictionaries with 'train' and 'test' keys, EM adaptation is done on 'test'
    soft_prior:
        (for proper EM) True: SOFT prior_shift_pred = shift_probs.mean(axis=0)
        (for improper EM) False: HARD prior_shift_pred (calculated from hist)
    over_update_probs:
        (for proper EM) False: shift_probs = shift_probs_0 * prior_shift_pred / prior_train
        (for improper EM) True: shift_probs = shift_probs * prior_shift_pred / prior_train
    """
    
    if (not soft_prior) or over_update_probs or (prior_source_phase != 'est. soft train'):
        confirm_ui = input("confirm performing an IMPROPER EM for debugging, " \
            "(for a proper EM, set: soft_prior = True, over_update_probs = False, "\
            "prior_source_phase = 'est. soft train'): y/[n] ")
        if confirm_ui != 'y':
            raise RuntimeError("terminated by user")

    if 'est. soft' in prior_source_phase:
        Y_probs_source = Y_probs['test' if 'test' in prior_source_phase else 'train']
        prior_source = Y_probs_source.mean(axis=0)
    elif 'true' in prior_source_phase:
        Y_source = Y['test' if 'test' in prior_source_phase else 'train']
        hist = np.bincount(Y_source, minlength=n_combs)
        prior_source = hist / hist.sum()
    elif prior_source_phase == 'uniform':
        prior_source = 1/n_combs * np.ones(n_combs)
    else:
        raise NotImplementedError
    
    print(f"source prior ('{prior_source_phase}'):")
    display(prior_source)

    # init
    EM_results = pd.DataFrame(index=range(EM_iterations))
    EM_results.index.name = 'iteration'
    shift_labels = Y['test']
    shift_probs_0 = Y_probs['test']
    shift_probs = shift_probs_0

    if init_prior == 'source prior':
        prior_shift_pred = prior_source
    elif init_prior == 'uniform':
        prior_shift_pred = 1/n_combs * np.ones(n_combs)
    else:
        raise NotImplementedError

    # eval
    shift_preds = shift_probs.argmax(axis=1)
    positives = shift_preds==shift_labels
    # EM_results.loc[0, [f'accuracy ({label})' for label in range(args.n_labels)]] = positives.mean(axis=0)
    EM_results.loc[0, 'accuracy'] = positives.mean()
    # likelihood
    Y_true_pos_probs = np.take_along_axis(shift_probs, Y['test'].reshape(-1,1), axis=1)
    EM_results.loc[0, 'log-likelihood'] = np.log(Y_true_pos_probs).sum()

    # EM
    for i_EM in tqdm(range(1, 1+EM_iterations)):        
        # E-step
        if over_update_probs:
            shift_probs = shift_probs * prior_shift_pred / prior_source
        else:
            shift_probs = shift_probs_0 * prior_shift_pred / prior_source
        shift_probs = (shift_probs.T / shift_probs.sum(axis=1)).T
        
        shift_preds = shift_probs.argmax(axis=1)
        
        # eval
        positives = shift_preds==shift_labels
        EM_results.loc[i_EM, 'accuracy'] = positives.mean()
        # likelihood
        Y_true_pos_probs = np.take_along_axis(shift_probs, Y['test'].reshape(-1,1), axis=1)
        EM_results.loc[i_EM, 'log-likelihood'] = np.log(Y_true_pos_probs).sum()

        # M-step
        if soft_prior:
            prior_shift_pred = shift_probs.mean(axis=0)
        else:
            hist_shift_pred = np.bincount(shift_preds, minlength=args.n_classes_per_label * args.n_labels)
            prior_shift_pred = hist_shift_pred / hist_shift_pred.sum()

    assert (~np.isfinite(shift_probs)).sum() == 0
    return EM_results, shift_probs

# %% [markdown]
# ## Yuval's defitions

# %%
from typing import NamedTuple
from pathlib import Path
from ATTOP.data.dataset import sample_negative as ATTOP_sample_negative
from torch.utils import data
from COSMO_utils import temporary_random_numpy_seed


def get_and_update_num_calls(func_ptr):
    try:
        get_and_update_num_calls.num_calls_cnt[func_ptr] += 1
    except AttributeError as e:
        if 'num_calls_cnt' in repr(e):
            get_and_update_num_calls.num_calls_cnt = defaultdict(int)
        else:
            raise

    return get_and_update_num_calls.num_calls_cnt[func_ptr]

def categorical_histogram(data, labels_list, plot=True, frac=True, plt_show=False):
    import matplotlib.pyplot as plt
    s_counts = pd.Series(data).value_counts()
    s_frac = s_counts/s_counts.sum()
    hist_dict = s_counts.to_dict()
    if frac:
        hist_dict = s_frac.to_dict()
    hist = []
    for ix, _ in enumerate(labels_list):
        hist.append(hist_dict.get(ix, 0))

    if plot:
        pd.Series(hist, index=labels_list).plot(kind='bar')
        if frac:
            plt.ylim((0,1))
        if plt_show:
            plt.show()
    else:
        return np.array(hist, dtype='float32')

class DataItem(NamedTuple):
    """ A NamedTuple for returning a Dataset item """
    feat: torch.Tensor
    pos_attr_id: int
    pos_obj_id: int
    neg_attr_id: int
    neg_obj_id: int
    image_fname: str


class CompDataFromDict():
    # noinspection PyMissingConstructor
    def __init__(self, dict_data: dict, data_subset: str, data_dir: str):

        # define instance variables to be retrieved from struct_data_dict
        self.split: str = 'TBD'
        self.phase: str = 'TBD'
        self.feat_dim: int = -1
        self.objs: list = []
        self.attrs: list = []
        self.attr2idx: dict = {}
        self.obj2idx: dict = {}
        self.pair2idx: dict = {}
        self.seen_pairs: list = []
        self.all_open_pairs: list = []
        self.closed_unseen_pairs: list = []
        self.unseen_closed_val_pairs: list = []
        self.unseen_closed_test_pairs: list = []
        self.train_data: tuple = tuple()
        self.val_data: tuple = tuple()
        self.test_data: tuple = tuple()

        self.data_dir: str = data_dir

        # retrieve instance variables from struct_data_dict
        vars(self).update(dict_data)
        self.data = dict_data[data_subset]

        self.activations = {}
        features_dict = torch.load(Path(data_dir) / 'features.t7')
        for i, img_filename in enumerate(features_dict['files']):
            self.activations[img_filename] = features_dict['features'][i]

        self.input_shape = (self.feat_dim,)
        self.num_objs = len(self.objs)
        self.num_attrs = len(self.attrs)
        self.num_seen_pairs = len(self.seen_pairs)
        self.shape_obj_attr = (self.num_objs, self.num_attrs)

        self.flattened_seen_pairs_mask = self.get_flattened_pairs_mask(self.seen_pairs)
        self.flattened_closed_unseen_pairs_mask = self.get_flattened_pairs_mask(self.closed_unseen_pairs)
        self.flattened_all_open_pairs_mask = self.get_flattened_pairs_mask(self.all_open_pairs)
        self.seen_pairs_joint_class_ids = np.where(self.flattened_seen_pairs_mask)

        self.y1_freqs, self.y2_freqs, self.pairs_freqs = self._calc_freqs()
        self._just_load_labels = False

        self.train_pairs = self.seen_pairs

    def sample_negative(self, attr, obj):
        return ATTOP_sample_negative(self, attr, obj)

    def get_flattened_pairs_mask(self, pairs):
        pairs_ids = np.array([(self.obj2idx[obj], self.attr2idx[attr]) for attr, obj in pairs])
        flattened_pairs = np.zeros(self.shape_obj_attr, dtype=bool)  # init an array of False
        flattened_pairs[tuple(zip(*pairs_ids))] = True
        flattened_pairs = flattened_pairs.flatten()
        return flattened_pairs

    def just_load_labels(self, just_load_labels=True):
        self._just_load_labels = just_load_labels

    def get_all_labels(self):
        attrs = []
        objs = []
        joints = []
        self.just_load_labels(True)
        for attrs_batch, objs_batch in self:
            if isinstance(attrs_batch, torch.Tensor):
                attrs_batch = attrs_batch.cpu().numpy()
            if isinstance(objs_batch, torch.Tensor):
                objs_batch = objs_batch.cpu().numpy()
            joint = self.to_joint_label(objs_batch, attrs_batch)

            attrs.append(attrs_batch)
            objs.append(objs_batch)
            joints.append(joint)

        self.just_load_labels(False)
        attrs = np.array(attrs)
        objs = np.array(objs)
        return attrs, objs, joints

    def _calc_freqs(self):
        y2_train, y1_train, ys_joint_train = self.get_all_labels()
        y1_freqs = categorical_histogram(y1_train, range(self.num_objs), plot=False, frac=True)
        y1_freqs[y1_freqs == 0] = np.nan
        y2_freqs = categorical_histogram(y2_train, range(self.num_attrs), plot=False, frac=True)
        y2_freqs[y2_freqs == 0] = np.nan

        pairs_freqs = categorical_histogram(ys_joint_train,
                                            range(self.num_objs * self.num_attrs),
                                            plot=False, frac=True)
        pairs_freqs[pairs_freqs == 0] = np.nan
        return y1_freqs, y2_freqs, pairs_freqs

    def get(self, name):
        return vars(self).get(name)

    def __getitem__(self, idx):
        image_fname, attr, obj = self.data[idx]
        pos_attr_id, pos_obj_id = self.attr2idx[attr], self.obj2idx[obj]
        if self._just_load_labels:
            return pos_attr_id, pos_obj_id

        num_calls_cnt = get_and_update_num_calls(self.__getitem__)

        negative_attr_id, negative_obj_id = -1, -1  # default values
        if self.phase == 'train':
            # we set a temp np seed to override a weird issue with
            # sample_negative() at __getitem__, where the sampled pairs
            # could not be deterministically reproduced:
            # Now at each call to _getitem_ we set the seed to a 834276 (chosen randomly) + the number of calls to _getitem_
            with temporary_random_numpy_seed(834276 + num_calls_cnt):
                # draw a negative pair
                negative_attr_id, negative_obj_id = self.sample_negative(attr, obj)

        item = DataItem(
            feat=self.activations[image_fname],
            pos_attr_id=pos_attr_id,
            pos_obj_id=pos_obj_id,
            neg_attr_id=negative_attr_id,
            neg_obj_id=negative_obj_id,
            image_fname=image_fname,
        )
        return item

    def __len__(self):
        return len(self.data)

    def to_joint_label(self, y1_batch, y2_batch):
        return (y1_batch * self.num_attrs + y2_batch)


def get_data_loaders(train_dataset, valid_dataset, test_dataset, batch_size,
                    num_workers=10, test_batchsize=None, shuffle_eval_set=True):
    if test_batchsize is None:
        test_batchsize = batch_size

    pin_memory = True
    if num_workers == 0:
        pin_memory = False
    print('num_workers = ', num_workers)
    print('pin_memory = ', pin_memory)
    train_loader = data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers,
                                pin_memory=pin_memory)
    valid_loader = None
    if valid_dataset is not None and len(valid_dataset) > 0:
        valid_loader = data.DataLoader(valid_dataset, batch_size=test_batchsize, shuffle=shuffle_eval_set,
                                    num_workers=num_workers, pin_memory=pin_memory)
    test_loader = data.DataLoader(test_dataset, batch_size=test_batchsize, shuffle=shuffle_eval_set,
                                num_workers=num_workers, pin_memory=pin_memory)
    return test_loader, train_loader, valid_loader




def run_experiment(args: Args):
    if args.data_dir == '':
        args.data_dir = f'data/{args.dataset_name}'

    if args.VP_calibrate and args.VP_held_out_from == 'val' and args.dataset_variant=='OZ':
        raise NotImplementedError("for calibration on 'val' in 'OZ' variant - implement 'val' dataset (should split out of 'train')")

    num_split_str = str(args.num_split)
    unseen_ovr_tot = int(num_split_str[:2])/100
    data_seed = int(num_split_str[3:])


    # %% [markdown]
    # # Data processing

    # %% [markdown]
    # ## Reading global metadata

    # %%
    meta_df = pd.read_csv(pjoin(args.data_dir, 'objects_metadata.csv'))
    print(f"available samples: len(meta_df) = {len(meta_df)}")
    meta_df

    # %% [markdown]
    # ## Building global combinations distribution (global_label_combs)

    # %%
    label_cols = ['shape', 'color']
    # --------------------------------

    global_label_combs = meta_df[label_cols].groupby(label_cols).size()
    global_label_combs.name = 'global samples'
    global_label_combs = global_label_combs.reset_index()
    global_label_combs['global freq'] = global_label_combs['global samples']/global_label_combs['global samples'].sum()
    global_label_combs.index.name = 'comb idx'
    global_label_combs = global_label_combs.reset_index()

    label_maps = {label:{} for label in label_cols}
    for label in label_cols:
        label_map = label_maps[label]
        for val in global_label_combs[label].unique():
            label_map[val] = len(label_map)
    # display(label_maps)

    for label in label_cols:
        global_label_combs[f'{label} idx'] = global_label_combs[label].map(label_maps[label])
    display(global_label_combs)

    assert len(label_cols) == 2
    global_label_combs_pivot = global_label_combs.pivot(index=label_cols[0], columns=label_cols[1], values='global samples')
    display(global_label_combs_pivot)

    distinct_label_n_vals = {label: len(global_label_combs[label].unique()) for label in label_cols}
    display(distinct_label_n_vals)


    # %% [markdown]
    # ## (A) Oz's splits, data

    # %% [markdown]
    # ### Sampling distributions
    # Verifying compositional sampling is valid - each label has at least one shared combination with any other label, unseen pivot shape is the same as global pivot - no empty columns or rows.

    # %%
    if args.dataset_variant == 'OZ':
        assert len(label_cols) == 2 # for the pivot below used to decide if the sampling is valid

        invalid_sampling = True
        max_i_sampling = 100
        i_sampling = 0

        while invalid_sampling:
            seen_label_dist = global_label_combs.sample(frac=unseen_ovr_tot,
                random_state=max_i_sampling*data_seed+i_sampling)
            seen_label_dist_pivot = seen_label_dist.pivot(index=label_cols[0], columns=label_cols[1], values='global samples')
            
            if seen_label_dist_pivot.shape == global_label_combs_pivot.shape:
                invalid_sampling = False
                print(f"succeeded sampling with all unique label values in iteration {i_sampling}")
            i_sampling += 1
            if i_sampling >= max_i_sampling:
                print(f"reached max iterations ({max_i_sampling}) without success in sampling with all unique label values ")
                break

    # %%
    if args.dataset_variant == 'OZ':
        unseen_combs_idx = set(global_label_combs.index) - set(seen_label_dist.index)
        unseen_label_dist = global_label_combs.loc[list(unseen_combs_idx)]
        assert len(set(seen_label_dist.index) & set(unseen_label_dist.index)) == 0

        label_dists_df = global_label_combs.copy()
        label_dists_df.loc[seen_label_dist.index, 'train prob'] = 1/len(seen_label_dist)
        label_dists_df.loc[seen_label_dist.index, 'state'] = 'seen'
        label_dists_df.loc[unseen_label_dist.index, 'train prob'] =\
            args.train_unseen_ovr_seen / len(seen_label_dist)
        label_dists_df.loc[unseen_label_dist.index, 'state'] = 'unseen'
        label_dists_df['train prob'] = label_dists_df['train prob'] / label_dists_df['train prob'].sum()
        label_dists_df['test prob'] =  1/len(label_dists_df)
        display(label_dists_df)

    # %% [markdown]
    # ### Sampling samples

    # %%
    if args.dataset_variant == 'OZ':
        meta_df_ = meta_df.merge(label_dists_df, on=label_cols)
        meta_df_ = meta_df_.sample(frac=1, random_state=data_seed) # shuffling

        for i_row in range(len(label_dists_df)):
            row = label_dists_df.iloc[i_row]
            comb_idx = row['comb idx']
            comb_samples = meta_df_.query(f"`comb idx` == {comb_idx}")

            # selecting train
            n_samples_train = round(args.train_size * row['train prob'])
            assert n_samples_train <= len(comb_samples)
            meta_df_.loc[comb_samples.index[:n_samples_train], 'phase'] = 'train'

            # selecting test
            n_samples_test = round(args.test_size * row['test prob'])
            assert n_samples_test <= len(comb_samples) - n_samples_train
            meta_df_.loc[comb_samples.index[
                n_samples_train: n_samples_train+n_samples_test], 'phase'] = 'test'

    # %% [markdown]
    # ### Verifying

    # %%
    if args.dataset_variant == 'OZ':
        for phase in ['train', 'test']:
            label_dists_df[f'{phase} samples'] = meta_df_.query(f"phase == '{phase}'")['comb idx'].value_counts()
            label_dists_df[f'{phase} freq'] = label_dists_df[f'{phase} samples']/label_dists_df[f'{phase} samples'].sum()
        display(label_dists_df)

    # %% [markdown]
    # ### Preparing data

    # %%
    if args.dataset_variant == 'OZ':
        features_dict = torch.load(pjoin(args.data_dir, 'features.t7'))
        features_dict_ = {file:tensor for file, tensor in zip(
            features_dict['files'], features_dict['features'])}

        idx = 5
        assert torch.equal(features_dict_[features_dict['files'][idx]], features_dict['features'][idx])
        del features_dict

    # %%
    if args.dataset_variant == 'OZ':
        X, Y_comb = {}, {}
        Y_shape, Y_color = {}, {}
        Y_shape_onehot, Y_color_onehot = {}, {}
        for phase in ['train', 'test']:
            data_df = meta_df_.query(f"phase == '{phase}'")
            print(f"{phase} contains {len(data_df)} samples")

            Y_comb[phase] = data_df['comb idx'].values.astype('int64')

            Y_shape[phase] = data_df['shape idx'].values.astype('int64')
            one_hot_encoder = OneHotEncoder(sparse=False, categories=[range(distinct_label_n_vals['shape'])])
            Y_shape_onehot[phase] = one_hot_encoder.fit_transform(data_df['shape idx'].values.reshape(-1, 1)).astype('float32') # float32 required in skorch for multi-label learning
            
            Y_color[phase] = data_df['color idx'].values.astype('int64')
            one_hot_encoder = OneHotEncoder(sparse=False, categories=[range(distinct_label_n_vals['color'])])
            Y_color_onehot[phase] = one_hot_encoder.fit_transform(data_df['color idx'].values.reshape(-1, 1)).astype('float32') # float32 required in skorch for multi-label learning     

            X[phase] = torch.cat([features_dict_[filename].unsqueeze(0)
                for filename in data_df['image_filename']], dim=0)
        del features_dict_

    # %% [markdown]
    # ## (B) Yuval's splits, dataloaders

    # %%
    if args.dataset_variant in ['VT', 'UV']:
        meta_path = Path(f"{args.data_dir}/metadata_pickles")
        random_state_path = Path(f"{args.data_dir}/np_random_state_pickles")
        # meta_path = meta_path.expanduser()

        dict_data = dict()

        for subset in ['train', 'valid', 'test']:
            metadata_full_filename = meta_path / f"metadata_{args.dataset_name}__{args.dataset_variant}_random__comp_seed_{args.num_split}__seen_seed_{args.seen_seed}__{subset}.pkl"
            dict_data[f'{subset}'] = deepcopy(pickle.load(open(metadata_full_filename, 'rb')))

        np_rnd_state_fname = random_state_path / f"np_random_state_{args.dataset_name}__{args.dataset_variant}_random__comp_seed_{args.num_split}__seen_seed_{args.seen_seed}.pkl"
        np_seed_state = pickle.load(open(np_rnd_state_fname, 'rb'))
        np.random.set_state(np_seed_state)

        datasets = {}
        for phase in ['train', 'val', 'test']:
            datasets[phase] = CompDataFromDict(dict_data[phase if phase!='val' else 'valid'],
                data_subset=f'{phase}_data', data_dir=args.data_dir)

    # %% [markdown]
    # ### Preparing data

    # %%
    if args.dataset_variant in ['VT', 'UV']:
        X, Y_comb = {}, {}
        Y_shape, Y_color = {}, {}
        Y_shape_onehot, Y_color_onehot = {}, {}
        label_dists_df = global_label_combs.copy()
        for phase, dataset in datasets.items():
            dataset = datasets[phase]
            data_df = pd.DataFrame(dataset.data, columns=['filename', 'color', 'shape'])
            data_df = data_df.merge(label_dists_df, on=label_cols)
            print(f"{phase} contains {len(data_df)} samples")

            Y_comb[phase] = data_df['comb idx'].values.astype('int64')

            Y_shape[phase] = data_df['shape idx'].values.astype('int64')
            one_hot_encoder = OneHotEncoder(sparse=False, categories=[range(distinct_label_n_vals['shape'])])
            Y_shape_onehot[phase] = one_hot_encoder.fit_transform(data_df['shape idx'].values.reshape(-1, 1)).astype('float32') # float32 required in skorch for multi-label learning
            
            Y_color[phase] = data_df['color idx'].values.astype('int64')
            one_hot_encoder = OneHotEncoder(sparse=False, categories=[range(distinct_label_n_vals['color'])])
            Y_color_onehot[phase] = one_hot_encoder.fit_transform(data_df['color idx'].values.reshape(-1, 1)).astype('float32') # float32 required in skorch for multi-label learning

            features = dataset.activations
            X[phase] = torch.cat([features[filename].unsqueeze(0)
                for filename in data_df['filename']], dim=0)

            dist_df = data_df.groupby(label_cols).size()
            dist_df.name = f'{phase} samples'
            dist_df = dist_df.reset_index()
            dist_df[f'{phase} freq'] = dist_df[f'{phase} samples']/dist_df[f'{phase} samples'].sum()
            label_dists_df = label_dists_df.merge(dist_df, on=label_cols, how='outer')

        label_dists_df = label_dists_df.set_index('comb idx')
        label_dists_df.loc[label_dists_df['train freq'].isna(), 'state'] = 'unseen'
        label_dists_df.loc[~label_dists_df['train freq'].isna(), 'state'] = 'seen'
        
        display(label_dists_df)

    # %% [markdown]
    # ## Analyzing

    # %%
    n_unseen = len(label_dists_df.query(f"state == 'seen'"))
    n_seen = len(label_dists_df.query(f"state == 'unseen'"))
    assert len(label_dists_df) == n_unseen + n_seen
    print(f"n_unseen / n_total = {n_unseen} / {len(label_dists_df)} = {n_unseen/len(label_dists_df)}")

    # %%
    for phase in X:
        print(f"{phase} samples:")
        # with pd.option_context('display.float_format',"{:,.4f}".format):
        #     display(label_dists_df.pivot(index=label_cols[0], columns=label_cols[1], values=f'{phase} freq'))
        with pd.option_context('display.float_format',"{:,.0f}".format):
            display(label_dists_df.pivot(index=label_cols[0], columns=label_cols[1], values=f'{phase} samples'))

    # %% [markdown]
    # # VisProd

    # %% [markdown]
    # ## Preparing

    # %%
    device = torch_setup(deterministic=True)

    in_dim = X['train'].shape[1]

    if args.VP_mode == 'CE':
        VP_criterion = nn.CrossEntropyLoss()
        Y_shape_VP = Y_shape
        Y_color_VP = Y_color
    elif args.VP_mode == 'BCE':
        VP_criterion = nn.BCEWithLogitsLoss()
        Y_shape_VP = Y_shape_onehot
        Y_color_VP = Y_color_onehot
    else:
        raise NotImplemented

    X_VP = X.copy()
    Y_comb_VP = Y_comb.copy()
    if args.VP_calibrate:
        if args.VP_held_out_from == 'train':
            held_out_size = round(len(X['train']) * args.VP_held_out_ratio)
            available_indices = np.arange(len(X['train']))
            with temporary_random_numpy_seed(seed=data_seed):
                np.random.shuffle(available_indices)
            
            train_indices = available_indices[held_out_size:]
            held_out_indices = available_indices[:held_out_size]

            X_VP['train'] = X['train'][train_indices]
            X_VP['held-out'] = X['train'][held_out_indices]

            Y_comb_VP['train'] = Y_comb['train'][train_indices]
            Y_comb_VP['held-out'] = Y_comb['train'][held_out_indices]

            Y_shape_VP_ = Y_shape_VP.copy()
            Y_shape_VP_['train'] = Y_shape_VP['train'][train_indices]
            Y_shape_VP_['held-out'] = Y_shape_VP['train'][held_out_indices]
            Y_shape_VP = Y_shape_VP_

            Y_color_VP_ = Y_color_VP.copy()
            Y_color_VP_['train'] = Y_color_VP['train'][train_indices]
            Y_color_VP_['held-out'] = Y_color_VP['train'][held_out_indices]
            Y_color_VP = Y_color_VP_
        
        elif args.VP_held_out_from == 'val':
            held_out_size = round(len(X['val']) * args.VP_held_out_ratio)
            available_indices = np.arange(len(X['val']))
            with temporary_random_numpy_seed(seed=data_seed):
                np.random.shuffle(available_indices)
            
            held_out_indices = available_indices[:held_out_size]
            
            X_VP['held-out'] = X['val'][held_out_indices]

            Y_comb_VP['held-out'] = Y_comb['val'][held_out_indices]

            Y_shape_VP_ = Y_shape_VP.copy()
            Y_shape_VP_['held-out'] = Y_shape_VP['val'][held_out_indices]
            Y_shape_VP = Y_shape_VP_

            Y_color_VP_ = Y_color_VP.copy()
            Y_color_VP_['held-out'] = Y_color_VP['val'][held_out_indices]
            Y_color_VP = Y_color_VP_
        else:
            raise NotImplemented

    # %% [markdown]
    # ## Shape net training

    # %%
    seed_all(args.init_seed)

    out_dim = distinct_label_n_vals['shape']
    assert in_dim == 512
    Net_shape = nn.Sequential( # leave net init here, otherwise cp continues training from last Net init
        nn.Linear(in_dim, 128),
        nn.ELU(inplace=True),
        nn.Linear(128, 64),
        nn.ELU(inplace=True),
        nn.Linear(64, out_dim),
    )
    cp = Checkpoint(dirname='checkpoints\shape_net', monitor='valid_acc_best', load_best=True)
    net_shape = NeuralNetClassifier(
        Net_shape,
        max_epochs=20,
        # max_epochs=10,
        criterion=VP_criterion,
        optimizer=optim.SGD,
        lr=2e-1,
        batch_size=args.batch_size,
        iterator_train__shuffle=True,
        train_split=ValidSplit(cv=args.skorch_val_ratio, stratified=True),
        device=device,
        callbacks=[EpochScoring(scoring='accuracy', lower_is_better=False, on_train=True), cp],
    )
    net_shape.fit(X_VP['train'], Y_shape_VP['train'])

    # %% [markdown]
    # ## Color net training

    # %%
    seed_all(args.init_seed)

    out_dim = distinct_label_n_vals['color']
    assert in_dim == 512
    Net_color = nn.Sequential( # leave net init here, otherwise cp continues training from last Net init
        nn.Linear(in_dim, 128),
        nn.ELU(inplace=True),
        nn.Linear(128, 64),
        nn.ELU(inplace=True),
        nn.Linear(64, out_dim),
    )
    cp = Checkpoint(dirname='checkpoints\color_net', monitor='valid_acc_best', load_best=True)
    net_color = NeuralNetClassifier(
        Net_color,
        max_epochs=5,
        criterion=VP_criterion,
        optimizer=optim.SGD,
        lr=1e-1,
        batch_size=args.batch_size,
        iterator_train__shuffle=True,
        train_split=ValidSplit(cv=args.skorch_val_ratio, stratified=True),
        device=device,
        callbacks=[EpochScoring(scoring='accuracy', lower_is_better=False, on_train=True), cp],
    )
    net_color.fit(X_VP['train'], Y_color_VP['train'])

    # %% [markdown]
    # ## VisProd

    # %%
    if args.VP_calibrate:
        phases = ['train', 'held-out', 'test']
    else:
        phases = ['train', 'test']

    if args.VP_mode == 'CE':
        Y_shape_probs = {phase: net_shape.predict_proba(X_VP[phase]) for phase in phases}
        Y_color_probs = {phase: net_color.predict_proba(X_VP[phase]) for phase in phases}
    elif args.VP_mode == 'BCE':
        Y_shape_probs = {phase: net_shape.predict_proba(X_VP[phase])[:,1,:] for phase in phases}
        Y_color_probs = {phase: net_color.predict_proba(X_VP[phase])[:,1,:] for phase in phases}
    else:
        raise NotImplemented

    # %%
    Y_shape_probs

    # %%
    Y_shape_probs['test'][0]

    # %%
    shape_acc = {phase: (probs.argmax(1) == Y_shape_VP[phase]).sum() / len(X_VP[phase]) for phase, probs in Y_shape_probs.items()}
    # print("shape_acc:", shape_acc)

    color_acc = {phase: (probs.argmax(1) == Y_color_VP[phase]).sum() / len(X_VP[phase]) for phase, probs in Y_color_probs.items()}
    # print("color_acc:", color_acc)

    # %%
    a = Y_shape_probs['test'][0]
    b = Y_color_probs['test'][0]
    c = []
    for a_ in a:
        for b_ in b:
            c.append(a_*b_)
    np.array_equal(np.array(c), np.tensordot(a, b, axes=0).flatten())

    # %%
    Y_VisProd_probs = {phase: np.zeros((len(X_VP[phase]), distinct_label_n_vals['color']*distinct_label_n_vals['shape'])) for phase in phases}
    for phase, probs in Y_VisProd_probs.items():
        for i in tqdm(range(len(X_VP[phase]))):
            outer_product = np.tensordot(Y_shape_probs[phase][i], Y_color_probs[phase][i], axes=0).flatten()
            probs[i,:] = outer_product / outer_product.sum()

    # %%
    # print("EM requires probs, verify normalization:")
    # for phase in phases:
    #     print(f"(Y_VisProd_probs[{phase}].sum(axis=1) - 1).mean(): %.12f"%(Y_VisProd_probs[phase].sum(axis=1) - 1).mean())

    # %%
    Y_VisProd_preds = {phase: probs.argmax(1) for phase, probs in Y_VisProd_probs.items()}
    Y_VisProd_preds

    # %%
    for phase in ['train', 'test']:
        y = Y_comb_VP[phase]
        y_pred = Y_VisProd_preds[phase]
        true_pos = pd.Series(y[y==y_pred]).value_counts()
        label_dists_df[f'VisProd: {phase} soft pred prior'] = pd.Series(Y_VisProd_probs[phase].mean(0))
        label_dists_df[f'VisProd: {phase} true pos'] = true_pos
        label_dists_df[f'VisProd: {phase} acc'] = label_dists_df[f'VisProd: {phase} true pos'] / label_dists_df[f'{phase} samples']
    label_dists_df

    # %%
    unseen_combs = label_dists_df.query("state == 'unseen'").index
    unseen_combs_idx_to_comb_idx = {i: comb_idx for i, comb_idx in enumerate(unseen_combs)}
    closed_preds = pd.DataFrame(Y_VisProd_probs['test'][:, unseen_combs].argmax(1), columns=['unseen comb idx'])
    closed_preds['comb idx'] = closed_preds['unseen comb idx'].map(unseen_combs_idx_to_comb_idx)
    closed_preds['closed true pos'] = closed_preds['comb idx'] == Y_comb_VP['test']
    closed_true_pos = closed_preds.query("`closed true pos`")['comb idx'].value_counts()
    label_dists_df[f'VisProd: test CLOSED true pos'] = closed_true_pos
    label_dists_df

    # %%
    # label_dists_df[f'VisProd: test CLOSED true pos'] = true_pos

    # %% [markdown]
    # ## VisProd-EM

    # %%
    # EM_iterations = 100
    # # EM_iterations = 10
    # # -------------------------------------
    # prior_source = Y_VisProd_probs['train'].mean(0)

    # VisProd_EM_evolution, VisProd_EM_shift_probs = basic_EM(Y=Y_comb['test'],
    #                                                     Y_probs=Y_VisProd_probs['test'],
    #                                                     prior_source = prior_source,
    #                                                     EM_iterations=EM_iterations)

    # %%
    EM_iterations = 100
    # EM_iterations = 2

    init_prior = 'source prior'
    # init_prior = 'uniform'

    prior_source_phase = 'est. soft train'
    # prior_source_phase = 'est. soft test'
    # prior_source_phase = 'true train'
    # prior_source_phase = 'true test'
    # prior_source_phase = 'uniform'

    soft_prior = True # proper EM: SOFT prior_shift_pred = shift_probs.mean(axis=0)
    # soft_prior = False # improper EM: HARD prior_shift_pred (calculated from hist)
    over_update_probs = False # proper EM shift_probs = shift_probs_0 * prior_shift_pred / prior_train
    # over_update_probs = True # improper EM shift_probs = shift_probs * prior_shift_pred / prior_train
    # -------------------------------------

    VisProd_EM_evolution, VisProd_EM_shift_probs = EM(Y=Y_comb, Y_probs=Y_VisProd_probs, EM_iterations=EM_iterations,
        prior_source_phase=prior_source_phase, init_prior=init_prior, n_combs=len(label_dists_df),
        soft_prior=soft_prior, over_update_probs=over_update_probs)

    # %%
    # metric = 'accuracy'
    metric = 'log-likelihood'
    # -----------------------------
    # plt.figure(figsize=(4,3))
    # VisProd_EM_evolution[metric].plot(ax=plt.gca(), linewidth=2)
    # plt.ylabel(metric)
    # plt.title("VisProd-EM evolution");

    # %%
    Y_VisProdEM_preds = VisProd_EM_shift_probs.argmax(1)
    Y_VisProdEM_preds

    # %%
    phase = 'test'

    y = Y_comb[phase]
    y_pred = Y_VisProdEM_preds
    true_pos = pd.Series(y[y==y_pred]).value_counts()
    label_dists_df[f'VisProd-EM: {phase} soft pred prior'] = pd.Series(VisProd_EM_shift_probs.mean(0))
    label_dists_df[f'VisProd-EM: {phase} true pos'] = true_pos
    label_dists_df[f'VisProd-EM: {phase} acc'] = label_dists_df[f'VisProd-EM: {phase} true pos'] / label_dists_df[f'{phase} samples']

    closed_preds = pd.DataFrame(VisProd_EM_shift_probs[:, unseen_combs].argmax(1), columns=['unseen comb idx'])
    closed_preds['comb idx'] = closed_preds['unseen comb idx'].map(unseen_combs_idx_to_comb_idx)
    closed_preds['closed true pos'] = closed_preds['comb idx'] == Y_comb_VP['test']
    closed_true_pos = closed_preds.query("`closed true pos`")['comb idx'].value_counts()
    label_dists_df[f'VisProd-EM: test CLOSED true pos'] = closed_true_pos
    label_dists_df

    # %% [markdown]
    # ## Calibration

    # %%
    # multi_class = 'ovr'
    # multi_class = 'multinomial'
    multi_class = 'none'

    max_iter = 1000 # default: 100
    # ------------------------------

    if args.VP_calibrate:
        if multi_class == 'none':
            for phase in phases:
                Y_VisProd_probs[f'{phase}_calib'] = np.zeros_like(Y_VisProd_probs[phase])
                Y_comb_VP[f'{phase}_calib'] = Y_comb_VP[phase]

            for i_comb in tqdm(range(Y_VisProd_probs['held-out'].shape[1])):
                if (Y_comb_VP['train'] == i_comb).sum() > 0:
                    comb_seen = True
                else:
                    comb_seen = False

                y = (Y_comb_VP['held-out'] == i_comb)
                if comb_seen: # can calibrate only if i_comb is seen - contains samples
                    calibrator = LogisticRegression(penalty='none', max_iter=max_iter)
                    calibrator.fit(X=Y_VisProd_probs['held-out'][:,i_comb].reshape(-1, 1), y=y)
                
                for phase in phases:
                    if comb_seen:
                        Y_VisProd_probs[f'{phase}_calib'][:,i_comb] = calibrator.predict_proba(
                            Y_VisProd_probs[phase][:,i_comb].reshape(-1, 1))[:,1]
                    else:
                        Y_VisProd_probs[f'{phase}_calib'][:,i_comb] = Y_VisProd_probs[phase][:,i_comb]
        else:
            raise NotImplemented
            # for phase in phases:
            #     Y_VisProd_probs[f'{phase}_calib'] = Y_VisProd_probs[phase].copy()
            #     Y_comb_VP[f'{phase}_calib'] = Y_comb_VP[phase]
            
            # hard_seen_combs = label_dists_df.query('`train samples` > 0').index
            # calibrator = LogisticRegression(penalty='none', max_iter=max_iter, multi_class=multi_class)
            # calibrator.fit(X=Y_VisProd_probs['held-out'][:,hard_seen_combs], y=Y_comb_VP['held-out'])
            # for phase in phases:
            #         Y_VisProd_probs[f'{phase}_calib'][:,hard_seen_combs] = calibrator.predict_proba(
            #             Y_VisProd_probs[phase][:,hard_seen_combs])
            
        
        for phase in ['train', 'test']:
            y = Y_comb_VP[f'{phase}_calib']
            y_pred = Y_VisProd_probs[f'{phase}_calib'].argmax(1)
            true_pos = pd.Series(y[y==y_pred]).value_counts()
            label_dists_df[f'calib VisProd: {phase} acc'] = label_dists_df[f'VisProd: {phase} true pos'] / label_dists_df[f'{phase} samples']
        # display(label_dists_df)

    # %% [markdown]
    # ## Calib. VisProd-EM

    # %%
    EM_iterations = 100
    # EM_iterations = 10
    # -------------------------------------

    if args.VP_calibrate:
        prior_source = Y_VisProd_probs['train_calib'].mean(0)

        calib_VisProd_EM_evolution, calib_VisProd_EM_shift_probs = basic_EM(Y=Y_comb_VP['test_calib'],
                                                            Y_probs=Y_VisProd_probs['test_calib'],
                                                            prior_source = prior_source,
                                                            EM_iterations=EM_iterations)

    # %%
    metric = 'accuracy'
    # metric = 'log-likelihood'
    # -----------------------------

    # if args.VP_calibrate:
        # plt.figure(figsize=(4,3))
        # calib_VisProd_EM_evolution[metric].plot(ax=plt.gca(), linewidth=2)
        # plt.ylabel(metric)
        # plt.title("Calib. VisProd-EM evolution")

    # %%
    if args.VP_calibrate:
        phase = 'test'

        y = Y_comb_VP[phase]
        y_pred = calib_VisProd_EM_shift_probs.argmax(1)
        true_pos = pd.Series(y[y==y_pred]).value_counts()
        label_dists_df[f'calib VisProd-EM: {phase} soft pred prior'] = pd.Series(VisProd_EM_shift_probs.mean(0))
        label_dists_df[f'calib VisProd-EM: {phase} true pos'] = true_pos
        label_dists_df[f'calib VisProd-EM: {phase} acc'] = label_dists_df[f'calib VisProd-EM: {phase} true pos'] / label_dists_df[f'{phase} samples']

        closed_preds = pd.DataFrame(calib_VisProd_EM_shift_probs[:, unseen_combs].argmax(1), columns=['unseen comb idx'])
        closed_preds['comb idx'] = closed_preds['unseen comb idx'].map(unseen_combs_idx_to_comb_idx)
        closed_preds['closed true pos'] = closed_preds['comb idx'] == Y_comb_VP['test']
        closed_true_pos = closed_preds.query("`closed true pos`")['comb idx'].value_counts()
        label_dists_df[f'calib VisProd-EM: test CLOSED true pos'] = closed_true_pos
        label_dists_df

    # %% [markdown]
    # # Summary

    # %% [markdown]
    # ## Macro (per-combination) results

    # %%
    acc_result_cols = [col for col in label_dists_df.columns if 'acc' in col]
    macro_result_cols = label_cols + ['state', 'train freq', 'test freq'] + acc_result_cols

    macro_results_df = label_dists_df[macro_result_cols]
    # results_df = results_df.sort_values(by='train freq', ascending=False)
    with pd.option_context('display.float_format',"{:,.3%}".format):
        display(macro_results_df)

    # %%
    with pd.option_context('display.float_format',"{:,.3%}".format):
        # display(results_df.groupby(by=label_cols).mean())
        display(macro_results_df.groupby(by='color').mean())

    # %%
    macro_result_summary_dict = {}
    for state in ['seen', 'unseen']:
        macro_result_summary_dict[state] = macro_results_df.query(f"state == '{state}'")[acc_result_cols].fillna(0).mean(0)
    macro_result_summary_dict['all'] = macro_results_df[acc_result_cols].fillna(0).mean(0)
    macro_result_summary_df = pd.DataFrame.from_dict(macro_result_summary_dict, orient='index')
    macro_result_summary_df.columns = macro_result_summary_df.columns.str.replace('acc', '')

    acc_seen = macro_result_summary_df.loc['seen']
    acc_unseen = macro_result_summary_df.loc['unseen']
    macro_result_summary_df.loc['harmonic'] = 2*acc_seen*acc_unseen/(acc_seen+acc_unseen)

    print("macro accuracy results:")
    with pd.option_context('display.float_format',"{:,.1%}".format):
        display(macro_result_summary_df)

    # %% [markdown]
    # ## MAIN: Micro metrics

    # %%
    pos_result_cols = [col for col in label_dists_df.columns if 'true pos' in col]
    open_pos_result_cols = [col for col in pos_result_cols if 'CLOSED' not in col]
    closed_pos_result_cols = [col for col in pos_result_cols if 'CLOSED' in col]
    micro_result_cols = label_cols + ['state', 'train samples', 'test samples'] + pos_result_cols

    micro_results_df = label_dists_df[micro_result_cols]
    with pd.option_context('display.float_format',"{:,.0f}".format):
        display(micro_results_df)

    # %%
    models = set([name.split(':')[0] for name in pos_result_cols])
    models

    # %%
    micro_result_summary_dict = {}
    for state in ['seen', 'unseen']:
        micro_result_summary_dict[state] = micro_results_df.query(f"state == '{state}'")[open_pos_result_cols].sum(0)
    micro_result_summary_dict['all'] = micro_results_df[open_pos_result_cols].sum(0)
    micro_result_summary_df = pd.DataFrame.from_dict(micro_result_summary_dict, orient='index')

    # from positives to accuracy
    for col in micro_result_summary_df.columns:
        phase = col.split(': ')[1].split(' true')[0]
        micro_result_summary_df.loc['seen', col] /= micro_results_df.query("state == 'seen'")[f'{phase} samples'].sum()
        micro_result_summary_df.loc['unseen', col] /= micro_results_df.query("state == 'unseen'")[f'{phase} samples'].sum()
        micro_result_summary_df.loc['all', col] /= micro_results_df[f'{phase} samples'].sum()
    micro_result_summary_df.columns = micro_result_summary_df.columns.str.replace("true pos", '')

    acc_seen = micro_result_summary_df.loc['seen']
    acc_unseen = micro_result_summary_df.loc['unseen']
    micro_result_summary_df.loc['harmonic'] = 2*acc_seen*acc_unseen/(acc_seen+acc_unseen)

    for col in closed_pos_result_cols:
        micro_result_summary_df.loc['closed', col.replace('CLOSED true pos', '')] = micro_results_df[col].sum(0) / micro_results_df.query("state == 'unseen'")['test samples'].sum()

    print("micro accuracy results:")
    with pd.option_context('display.float_format',"{:,.1%}".format):
        display(micro_result_summary_df)



    # %%
    soft_prior_cols = [col for col in label_dists_df.columns if 'soft' in col]
    with pd.option_context('display.float_format',"{:,.1%}".format):
        display(label_dists_df[label_cols + ['state', 'train freq', 'test freq'] + soft_prior_cols].query("state == 'unseen'"))

    return micro_result_summary_df


#%% experiment control
exp_dir = 'analysis/MLLS/basic exp'
if not os.path.isdir(exp_dir):
    os.makedirs(exp_dir)

args = Args()
json.dump(vars(args), open(pjoin(exp_dir, 'args.json'), 'w'))

for num_split in tqdm([5000], desc='split'):
    args.num_split = num_split
    
    for init_seed in tqdm([0], desc='seed', leave=False):
        args.init_seed = init_seed
        exp_name = f"VT={num_split} init_seed={init_seed}.xlsx"
        exp_path = pjoin(exp_dir, exp_name)

        if not os.path.isfile(exp_path):
            micro_result_summary_df = run_experiment(args=args)
            micro_result_summary_df.to_excel(exp_path)