from MLLS_VisProd_params import ProjectArgs

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
from tqdm.auto import tqdm
from copy import deepcopy

from time import sleep

import torch
import torch.nn as nn
import torch.optim as optim

from pprint import pprint
import logging
logging.basicConfig(format='%(asctime)s (%(levelname)s): %(message)s',
                    # datefmt='%Y-%m-%d %H:%M:%S',
                    )
logger_name = 'MLLS'
logger = logging.getLogger(logger_name)
logger.setLevel(logging.INFO)


def set_logger_level(log_level):
    logger.setLevel(log_level)


# torch.cuda.set_device(args.gpu)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    logger.info(f"torch is using {device}, {torch.cuda.get_device_name()}")
else:
    logger.info(f"torch is using {device}")


def dict2df(dict_) -> pd.DataFrame:
    return pd.DataFrame.from_dict(dict_, orient='index')


def flatten_dict(dict_) -> dict:
    """flattens nested dicts in json format (normal python dict with str keys) into flat dicts with dots to mark levels:
        dict_ = {'a':{'1':'a1', '2':'a2'}, 'b':{'bb':1, 'bbb':'c'}}
        ->  flatten_dict(dict_) = {'a.1': 'a1', 'a.2': 'a2', 'b.bb': 1, 'b.bbb': 'c'}
    """
    return pd.json_normalize(dict_).iloc[0].to_dict()


def slice_dict(d: dict, keys: iter) -> dict:
    return {key: d[key] for key in keys}


@contextmanager
def temp_np_random_seed(seed):
    # when entering context:
    state = np.random.get_state()
    np.random.seed(seed)
    yield None
    # when exiting context:
    np.random.set_state(state)


def torch_setup(deterministic: bool = True):
    # Setting up torch, seeding (https://pytorch.org/docs/stable/notes/randomness.html)
    # ///////////// Making pytorch deterministic (reproducible)  ////////////////////////////////////////////////////
    if deterministic:
        # read WELL: https://pytorch.org/docs/stable/notes/randomness.html
        # the order of execution is important!

        # https://docs.nvidia.com/cuda/cublas/index.html#cublasApi_reproducibility
        #     os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8" # may limit overall performance
        os.environ[
            "CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"  # will increase library footprint in GPU memory by approximately 24MiB
        torch.backends.cudnn.benchmark = False
        logger.info(f"executed: torch.backends.cudnn.benchmark = False")

        torch_set_deterministic = False
        # try:
        #     torch.set_deterministic(True) # beta in torch==1.7
        #     logger.info(f"executed torch.set_deterministic(True)")
        #     torch_set_deterministic = True
        # except Exception as ex:
        #     logger.warning(f"torch.set_deterministic(True) failed: {ex}")
        try:
            torch.use_deterministic_algorithms(True)  # beta in torch==1.8
            logger.info(f"executed: torch.use_deterministic_algorithms(True)")
            torch_set_deterministic = True
        except Exception as ex:
            logger.warning(f"torch.use_deterministic_algorithms(True) failed: {ex}")

        if not torch_set_deterministic:  # already contained in torch.use_deterministic_algorithms(True)
            torch.backends.cudnn.deterministic = True
            logger.info(f"executed: torch.backends.cudnn.deterministic = True")

        logger.info(f"torch and cuda will be deterministic (after seeding)")
    else:
        torch.backends.cudnn.benchmark = True
        logger.info(f"executed: torch.backends.cudnn.benchmark = True (torch is not determinisitc!)")


def seed_all(seed: int):
    random.seed(seed)
    np.random.seed(seed + 1000)
    torch.manual_seed(seed + 2000)
    torch.cuda.manual_seed(seed + 3000)
    logger.info(f"completed: seeding with seed={seed} in steps of 1000 for random,np,torch,cuda")


def naming_run(args: ProjectArgs) -> str:
    """Returns run_name named according to:
    * if args.wandb.is_sweep and args.wandb.run_naming_args != '': adding arg value for each arg in
        args.wandb.run_naming_args, e.g.:
            args.wandb.run_naming_args = 'cfg.seed, data.val.label_dist, data.seed' and args.wandb.sweep_name = 'dev sweep':
            -> run_name = 'cfg.seed=0|data.val.label_dist=Dirichlet|data.seed=5000__SWEEP__dev sweep'
    """
    args_dict_flat = flatten_dict(args.to_dict())
    run_name = args.cfg.run_name
    if args.wandb.is_sweep and args.wandb.run_naming_args != '':
        run_naming_args = [arg.strip() for arg in args.wandb.run_naming_args.split(',')]
        run_naming_args_dict = {arg: args_dict_flat[arg] for arg in run_naming_args}
        run_name = str(run_naming_args_dict)

        run_name = run_name.replace('{', '').replace('}', '').replace("'", "").replace(": ", '=').replace(' ', '')
        if args.wandb.sweep_name is not None:
            run_name = run_name + '__SWEEP__' + args.wandb.sweep_name

    # getting rid of spaces, post-training touch command doesn't deal with them well
    # run_name = run_name.replace(' ','_')
    return run_name


def renaming_paths_inplace(args: ProjectArgs):
    if args.data.dir == '':
        args.data.dir = pjoin(args.cfg.project_path, 'data', args.data.dataset)
    else:
        args.data.dir = pjoin(args.cfg.project_path, Path(args.data.dir))

    if args.cfg.output_dir == '':
        args.cfg.output_dir = pjoin(args.cfg.project_path, 'outputs')
    else:
        args.cfg.output_dir = pjoin(args.cfg.project_path, Path(args.cfg.output_dir))

    # if args.cache.dir == '':
    #     args.cache.dir = pjoin(args.cfg.project_path, 'cache')
    # else:
    #     args.cache.dir = pjoin(args.cfg.project_path, Path(args.cache.dir))


def creating_outputs_dir(args: ProjectArgs):
    base_output_dir = pjoin(args.cfg.project_path, args.cfg.output_dir)
    output_dir = pjoin(base_output_dir, args.cfg.run_name)

    counter = 1
    while os.path.isdir(output_dir):
        output_dir = os.path.join(base_output_dir, args.cfg.run_name + f"_{counter + 1}")
        counter += 1
        if counter == 100:
            raise RuntimeError(f"suspicious that 100 renames all exist up to '{args.cfg.output_dir}', delete all manually")

    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
        logger.info(f"created output_dir: '{output_dir}'")
    args.cfg.output_dir = output_dir


def load_data(args: ProjectArgs):
    if args.data.dataset == 'ao_clevr':
        label_cols = ['shape', 'color']
    else:
        raise NotImplementedError

    meta_df = pd.read_csv(pjoin(args.data.dir, 'objects_metadata.csv'))
    global_label_combs = meta_df[label_cols].groupby(label_cols).size()
    global_label_combs.name = 'global samples'
    global_label_combs = global_label_combs.reset_index()
    global_label_combs['global freq'] = global_label_combs['global samples'] / global_label_combs[
        'global samples'].sum()
    global_label_combs.index.name = 'comb idx'
    global_label_combs = global_label_combs.reset_index()

    label_maps = {label: {} for label in label_cols}
    for label in label_cols:
        label_map = label_maps[label]
        for val in global_label_combs[label].unique():
            label_map[val] = len(label_map)

    for label in label_cols:
        global_label_combs[f'{label} idx'] = global_label_combs[label].map(label_maps[label])

    assert len(label_cols) == 2
    global_label_combs_pivot = global_label_combs.pivot(index=label_cols[0], columns=label_cols[1],
                                                        values='global samples')

    label_n_vals = {label: len(global_label_combs[label].unique()) for label in label_cols}
    args.data.n_colors = label_n_vals['color']
    args.data.n_shapes = label_n_vals['shape']

    if args.data.variant == 'OZ':
        assert len(label_cols) == 2  # for the pivot below used to decide if the sampling is valid

        invalid_sampling = True
        max_i_sampling = 100
        i_sampling = 0

        while invalid_sampling:
            seen_label_dist = global_label_combs.sample(frac=args.data.unseen_ovr_tot,
                                                        random_state=max_i_sampling * args.data.seed + i_sampling)
            seen_label_dist_pivot = seen_label_dist.pivot(index=label_cols[0], columns=label_cols[1],
                                                          values='global samples')

            if seen_label_dist_pivot.shape == global_label_combs_pivot.shape:
                invalid_sampling = False
                logger.info(f"succeeded sampling with all unique label values in iteration {i_sampling}")
            i_sampling += 1
            if i_sampling >= max_i_sampling:
                logger.warning(f"reached max iterations ({max_i_sampling}) without success in sampling with all unique label values ")
                break

        unseen_combs_idx = set(global_label_combs.index) - set(seen_label_dist.index)
        unseen_label_dist = global_label_combs.loc[list(unseen_combs_idx)]
        assert len(set(seen_label_dist.index) & set(unseen_label_dist.index)) == 0

        label_dists_df = global_label_combs.copy()
        label_dists_df.loc[seen_label_dist.index, 'train prob'] = 1 / len(seen_label_dist)
        label_dists_df.loc[seen_label_dist.index, 'state'] = 'seen'
        label_dists_df.loc[unseen_label_dist.index, 'train prob'] = \
            args.data.train_unseen_ovr_seen / len(seen_label_dist)
        label_dists_df.loc[unseen_label_dist.index, 'state'] = 'unseen'
        label_dists_df['train prob'] = label_dists_df['train prob'] / label_dists_df['train prob'].sum()
        label_dists_df['test prob'] = 1 / len(label_dists_df)

        meta_df_ = meta_df.merge(label_dists_df, on=label_cols)
        meta_df_ = meta_df_.sample(frac=1, random_state=args.data.seed)  # shuffling

        for i_row in range(len(label_dists_df)):
            row = label_dists_df.iloc[i_row]
            comb_idx = row['comb idx']
            comb_samples = meta_df_.query(f"`comb idx` == {comb_idx}")

            # selecting train
            n_samples_train = round(args.data.train_size * row['train prob'])
            assert n_samples_train <= len(comb_samples)
            meta_df_.loc[comb_samples.index[:n_samples_train], 'phase'] = 'train'

            # selecting test
            n_samples_test = round(args.data.test_size * row['test prob'])
            assert n_samples_test <= len(comb_samples) - n_samples_train
            meta_df_.loc[comb_samples.index[
                         n_samples_train: n_samples_train + n_samples_test], 'phase'] = 'test'

        for phase in ['train', 'test']:
            label_dists_df[f'{phase} samples'] = meta_df_.query(f"phase == '{phase}'")['comb idx'].value_counts()
            label_dists_df[f'{phase} freq'] = label_dists_df[f'{phase} samples'] / label_dists_df[
                f'{phase} samples'].sum()

        features_dict = torch.load(pjoin(args.data.dir, 'features.t7'))
        features_dict_ = {file: tensor for file, tensor in zip(
            features_dict['files'], features_dict['features'])}

        idx = 5
        assert torch.equal(features_dict_[features_dict['files'][idx]], features_dict['features'][idx])

        X, Y_comb = {}, {}
        Y_shape, Y_color = {}, {}
        # Y_shape_onehot, Y_color_onehot = {}, {}
        for phase in ['train', 'test']:
            data_df = meta_df_.query(f"phase == '{phase}'")
            print(f"{phase} contains {len(data_df)} samples")

            Y_comb[phase] = data_df['comb idx'].values.astype('int64')

            Y_shape[phase] = data_df['shape idx'].values.astype('int64')
            # one_hot_encoder = OneHotEncoder(sparse=False, categories=[range(label_n_vals['shape'])])
            # Y_shape_onehot[phase] = one_hot_encoder.fit_transform(data_df['shape idx'].values.reshape(-1, 1)).astype(
            #     'float32')  # float32 required in skorch for multi-label learning

            Y_color[phase] = data_df['color idx'].values.astype('int64')
            # one_hot_encoder = OneHotEncoder(sparse=False, categories=[range(label_n_vals['color'])])
            # Y_color_onehot[phase] = one_hot_encoder.fit_transform(data_df['color idx'].values.reshape(-1, 1)).astype(
            #     'float32')  # float32 required in skorch for multi-label learning

            X[phase] = torch.cat([features_dict_[filename].unsqueeze(0)
                                  for filename in data_df['image_filename']], dim=0)

    elif args.data.variant in ['VT', 'UV']:
        meta_path = Path(f"{args.data.dir}/metadata_pickles")
        # random_state_path = Path(f"{args.data.dir}/np_random_state_pickles")
        # meta_path = meta_path.expanduser()

        dict_data = dict()

        for subset in ['train', 'valid', 'test']:
            metadata_full_filename = meta_path / f"metadata_{args.data.dataset}__{args.data.variant}_random__comp_seed_{args.data.num_split}__seen_seed_{args.data.seen_seed}__{subset}.pkl"
            dict_data[f'{subset}'] = deepcopy(pickle.load(open(metadata_full_filename, 'rb')))

        # np_rnd_state_fname = random_state_path / f"np_random_state_{args.dataset_name}__{args.dataset_variant}_random__comp_seed_{args.num_split}__seen_seed_{args.seen_seed}.pkl"
        # np_seed_state = pickle.load(open(np_rnd_state_fname, 'rb'))
        # np.random.set_state(np_seed_state)

        datasets = {}
        for phase in ['train', 'val', 'test']:
            datasets[phase] = CompDataFromDict(dict_data[phase if phase != 'val' else 'valid'],
                                               data_subset=f'{phase}_data', data_dir=args.data.dir)

        X, Y_comb = {}, {}
        Y_shape, Y_color = {}, {}
        Y_shape_onehot, Y_color_onehot = {}, {}
        label_dists_df = global_label_combs.copy()
        meta_df_ = meta_df.copy()

        for phase in ['train', 'test', 'val']:  # val is last for args.val_unseen_mode == 'complete'
            dataset = datasets[phase]
            data_df = pd.DataFrame(dataset.data, columns=['filename', 'color', 'shape'])
            data_df = data_df.merge(label_dists_df, on=label_cols)

            data_df[phase] = 1
            meta_df_ = meta_df_.merge(data_df.set_index('filename')[phase], left_on='image_filename',
                                      right_on='filename', how='outer')

            if phase == 'val':
                if args.data.val_unseen_mode == 'drop':
                    seen_combs = label_dists_df.index[~label_dists_df['train freq'].isna()]
                    data_df = data_df[data_df['comb idx'].isin(seen_combs)]
                    data_df[phase] = 1
                    meta_df_[phase] = data_df[phase]
                elif args.data.val_unseen_mode == 'complete':
                    phases_sum = meta_df_[['train', 'val', 'test']].sum(1)
                    unallocated_samples = meta_df_[phases_sum == 0]

                    unseen_combs = label_dists_df.index[label_dists_df['train freq'].isna()]
                    seen_val_comb_counts = data_df[~data_df['comb idx'].isin(unseen_combs)]['comb idx'].value_counts()
                    mean_seen_val_samples = round(seen_val_comb_counts.mean())

                    unallocated_samples_ = unallocated_samples.merge(
                        label_dists_df.reset_index()[['shape', 'color', 'shape idx', 'color idx', 'comb idx']],
                        on=label_cols)
                    for comb in unseen_combs:
                        unallocated_samples_in = unallocated_samples_.query(f"`comb idx` == {comb}").rename(
                            columns={'image': 'filename'})
                        n_samples_to_add = mean_seen_val_samples - len(data_df.query(f"`comb idx` == {comb}"))
                        if len(unallocated_samples_in) > n_samples_to_add:
                            unallocated_samples_in = unallocated_samples_in[:n_samples_to_add]
                        data_df = pd.concat([data_df, unallocated_samples_in[
                            ['shape', 'color', 'shape idx', 'color idx', 'comb idx', 'filename']]], axis=0)

                    data_df[phase] = 1
                    del meta_df_[phase]
                    meta_df_ = meta_df_.merge(data_df.set_index('filename')[phase], left_on='image_filename',
                                              right_on='filename', how='outer')

                    data_df = data_df.sample(frac=1, random_state=args.data.seed)  # shuffling

            logger.info(f"{phase} contains {len(data_df)} samples")

            Y_comb[phase] = data_df['comb idx'].values.astype('int64')

            Y_shape[phase] = data_df['shape idx'].values.astype('int64')
            # one_hot_encoder = OneHotEncoder(sparse=False, categories=[range(label_n_vals['shape'])])
            # Y_shape_onehot[phase] = one_hot_encoder.fit_transform(data_df['shape idx'].values.reshape(-1, 1)).astype(
            #     'float32')  # float32 required in skorch for multi-label learning

            Y_color[phase] = data_df['color idx'].values.astype('int64')
            # one_hot_encoder = OneHotEncoder(sparse=False, categories=[range(label_n_vals['color'])])
            # Y_color_onehot[phase] = one_hot_encoder.fit_transform(data_df['color idx'].values.reshape(-1, 1)).astype(
            #     'float32')  # float32 required in skorch for multi-label learning

            features = dataset.activations
            X[phase] = torch.cat([features[filename].unsqueeze(0)
                                  for filename in data_df['filename']], dim=0)

            dist_df = data_df.groupby(label_cols).size()
            dist_df.name = f'{phase} samples'
            dist_df = dist_df.reset_index()
            dist_df[f'{phase} freq'] = dist_df[f'{phase} samples'] / dist_df[f'{phase} samples'].sum()
            label_dists_df = label_dists_df.merge(dist_df, on=label_cols, how='outer')
    else:
        raise NotImplementedError

    label_dists_df = label_dists_df.set_index('comb idx')
    label_dists_df.loc[label_dists_df['train freq'].isna(), 'state'] = 'unseen'
    label_dists_df.loc[~label_dists_df['train freq'].isna(), 'state'] = 'seen'

    phases_sum = meta_df_[['train', 'val', 'test']].sum(1)
    assert (phases_sum > 1).sum() == 0  # no phase overlap (train/val/test)

    unseen_combs = label_dists_df.query("state == 'unseen'").index
    unseen_combs_idx_to_comb_idx = {i: comb_idx for i, comb_idx in enumerate(unseen_combs)}
    seen_combs = label_dists_df.query("state == 'seen'").index

    n_unseen = len(label_dists_df.query(f"state == 'seen'"))
    n_seen = len(label_dists_df.query(f"state == 'unseen'"))
    assert len(label_dists_df) == n_unseen + n_seen
    logger.info(f"n_unseen / n_total = {n_unseen} / {len(label_dists_df)} = {n_unseen / len(label_dists_df)}")

    if args.cfg.log_level <= 20:
        # show_total = True
        show_total = False
        vals = 'samples'
        # vals = 'freq'
        # ----------------------
        if vals == 'freq':
            format = "{:,.1%}".format
        elif vals == 'samples':
            format = "{:,}".format
        else:
            raise NotImplementedError

        for phase in X:
            # for phase in ['train', 'test']:
            print(f"{phase} freq:")
            # with pd.option_context('display.float_format',"{:,.4f}".format):
            #     display(label_dists_df.pivot(index=label_cols[0], columns=label_cols[1], values=f'{phase} freq'))
            with pd.option_context('display.float_format', format):
                df = label_dists_df.pivot(index=label_cols[0], columns=label_cols[1], values=f'{phase} samples')
                df_ = df.copy()
                if show_total:
                    df_['total'] = df.sum(1)
                    df_.loc['total'] = df.sum(0)
                if vals == 'freq':
                    df_ /= label_dists_df[f'{phase} samples'].sum()
                pprint(df_)

    return X, Y_comb, Y_shape, Y_color, label_dists_df, unseen_combs, seen_combs, unseen_combs_idx_to_comb_idx


class VisProdDataset(torch.utils.data.Dataset):
    def __init__(self, X, y_color, y_shape, y_comb, y_seen):
        assert len(X) == len(y_color) == len(y_shape) == len(y_comb) == len(y_seen)

        self.X = X
        self.y_color = y_color
        self.y_shape = y_shape
        self.y_comb = y_comb
        self.y_seen = y_seen

    def __len__(self):
        return len(self.y_color)

    def __getitem__(self, i):
        return self.X[i], self.y_color[i], self.y_shape[i], self.y_comb[i], self.y_seen[i]


def build_training(X: dict, Y_comb: dict, Y_shape: dict, Y_color: dict, seen_combs, args: ProjectArgs):
    datasets = {phase: VisProdDataset(X=X[phase], y_color=Y_color[phase], y_shape=Y_shape[phase],
                               y_comb=Y_comb[phase], y_seen=np.isin(Y_comb[phase], seen_combs)) for phase in X}
    dataloaders = {phase: torch.utils.data.DataLoader(datasets[phase],
                                                      batch_size=args.training.batch_size, shuffle=True,
                                                      num_workers=args.training.workers,
                                                      pin_memory=args.training.pin_memory) for phase in datasets}

    in_dim = X['train'].shape[1]
    assert in_dim == 512

    models = {space: nn.Sequential(
        nn.Linear(in_dim, 128),
        nn.ELU(inplace=True),
        nn.Linear(128, 64),
        nn.ELU(inplace=True),
        nn.Linear(64, args.data.n_colors if space == 'color' else args.data.n_shapes),
        nn.LogSoftmax(dim=1)).to(device)
              for space in ['color', 'shape']}
    if args.VP.decoupling_loss == 'MI est':
        models['joint'] = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.ELU(inplace=True),
            nn.Linear(128, 64),
            nn.ELU(inplace=True),
            nn.Linear(64, args.data.n_colors * args.data.n_shapes),
            nn.LogSoftmax(dim=1)).to(device)

    for space, model in models.items():
        net_params_tot = sum(p.numel() for p in model.parameters())
        net_params_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        if args.cfg.log_level <= 20:
            logger.info(f"{space} model (trainable/total params: %.2e/%.2e):" % (net_params_trainable, net_params_tot))
            pprint(model)

    clss_loss = nn.NLLLoss(reduction='sum').to(device)

    if args.VP.optimizer == 'SGD':
        optimizers = {space: optim.SGD(model.parameters(), lr=args.VP.lr) for space, model in models.items()}
    elif args.VP.optimizer == 'Adam':
        optimizers = {space: optim.Adam(model.parameters(), lr=args.VP.lr) for space, model in models.items()}
    else:
        raise NotImplementedError(f"args.VP.optimizer = '{args.VP.optimizer}' not implemented")

    schedulers = None
    if args.VP.plateau_mode != 'off':
        schedulers = {space: optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode=args.VP.plateau_mode,
                                                                  factor=args.VP.plateau_factor, verbose=True,
                                                                  patience=args.VP.plateau_patience,
                                                                  threshold=args.VP.plateau_threshold,
                                                                  threshold_mode=args.VP.plateau_threshold_mode)
                      for space, optimizer in optimizers.items()}

    return datasets, dataloaders, models, clss_loss, optimizers, schedulers


def run_epoch(training: bool, models: dict, dataloader, clss_loss, args: ProjectArgs, optimizers: dict = {},
              seen_combs=None):
    if training:
        for model in models.values():
            model.train()
    else:
        for model in models.values():
            model.eval()

    epoch_true_pos = defaultdict(float)
    epoch_losses = defaultdict(float)
    n_samples, n_seen = 0, 0  # n_unseen = n_samples- n_seen
    epoch_log_probs = defaultdict(list)
    loss_was_finite = True
    with torch.set_grad_enabled(training):
        for i_batch, batch in enumerate(dataloader):
            x, y_color, y_shape, y_comb, y_seen = batch
            batch_n_samples = x.shape[0]
            n_samples += batch_n_samples
            x = x.to(device)
            y_s = {}
            y_s['color'] = y_color.to(device)
            y_s['shape'] = y_shape.to(device)
            if args.VP.decoupling_loss == 'MI est':
                y_s['joint'] = y_comb.to(device)

            if training:
                for optimizer in optimizers.values():
                    optimizer.zero_grad()

            # forward
            log_probs = {space: model(x) for space, model in models.items()}
            probs = {space: torch.exp(space_log_probs) for space, space_log_probs in log_probs.items()}
            for space in log_probs:
                epoch_log_probs[space].append(log_probs[space])

            # eval
            preds = {space: log_probs.argmax(1) for space, log_probs in log_probs.items()}
            for space in models:
                epoch_true_pos[space] += (preds[space] == y_s[space]).sum().item()
            join_true_pos = (preds['color'] == y_s['color']) & (preds['shape'] == y_s['shape'])
            epoch_true_pos['joint'] += (join_true_pos).sum().item()
            epoch_true_pos['seen'] += (join_true_pos[y_seen]).sum().item()
            n_seen += y_seen.sum().item()
            epoch_true_pos['unseen'] += (join_true_pos[~y_seen]).sum().item()

            # losses
            losses = {space: clss_loss(log_probs[space], y_s[space]) for space in models}
            # probs_on_true = {space: torch.exp(torch.take_along_dim(input=log_probs[space], indices=y_s[space].unsqueeze(-1), dim=1)) for space in y_s}
            probs_on_true = {
                space:  # moving to cpu and back since it doens't have a deterministic implementation for backprop
                    torch.take_along_dim(input=probs[space].cpu(), indices=y_s[space].unsqueeze(-1).cpu(), dim=1).to(
                        device) for space in models}

            losses['MI g.t.'] = torch.log(1 / (probs_on_true['color'] * probs_on_true['shape'])).sum()
            if args.VP.decoupling_loss == 'MI est':
                # VisProd_probs = []
                # for i_sample in range(batch_n_samples):
                #     sample_VisProd_probs = torch.outer(probs['shape'][i_sample], probs['color'][i_sample]).reshape(1,-1)
                #     sample_VisProd_probs /= sample_VisProd_probs.sum()
                #     VisProd_probs.append(sample_VisProd_probs)
                # VisProd_probs = torch.cat(VisProd_probs, dim=0)
                VisProd_probs = torch.bmm(probs['shape'].unsqueeze(2), probs['color'].unsqueeze(1)).reshape(
                    batch_n_samples, -1)  # https://discuss.pytorch.org/t/batch-outer-product/4025
                # VisProd_probs = (VisProd_probs.T / VisProd_probs.sum(1)).T
                VisProd_probs = (VisProd_probs.T / VisProd_probs[:, seen_combs].sum(1)).T

                MI_per_comb = probs['joint'] * torch.log(probs['joint'] / VisProd_probs)
                losses['MI est'] = MI_per_comb[:, seen_combs].sum()
            elif 'HSIC' in args.VP.decoupling_loss:
                if args.VP.decoupling_loss in ['HSIC,L', 'HSIC,G']:
                    if args.VP.decoupling_loss == 'HSIC,L':
                        Hkernel = 'L'
                        Hkernel_sigma_obj = None
                        Hkernel_sigma_attr = None
                    else:
                        Hkernel = 'G'
                        Hkernel_sigma_obj = 1
                        Hkernel_sigma_attr = 1

                    losses['HSIC'], HSIC_rep_loss_terms, HSIC_mean_of_median_pairwise_dist_terms = \
                        conditional_indep_losses(probs['shape'], probs['color'], ys=[y_s['shape'], y_s['color']],
                                                 indep_coeff=1, indep_coeff2=None,
                                                 num_classes1=args.data.n_shapes,
                                                 num_classes2=args.data.n_colors,
                                                 Hkernel=Hkernel, Hkernel_sigma_obj=Hkernel_sigma_obj,
                                                 Hkernel_sigma_attr=Hkernel_sigma_attr,
                                                 log_median_pairwise_distance=False, device=device)
                elif args.VP.decoupling_loss == 'HSIC,N':
                    losses['HSIC'] = hsic_normalized(x=probs['shape'], y=probs['color'], sigma=None, use_cuda=True)
                else:
                    raise NotImplemented

            loss = 0
            if args.VP.lambda_color > 0:
                loss += args.VP.lambda_color * losses['color']
            if args.VP.lambda_shape > 0:
                loss += args.VP.lambda_shape * losses['shape']
            if args.VP.decoupling_loss == 'MI est' and args.VP.lambda_joint > 0:
                loss += args.VP.lambda_joint * losses['joint']
            if args.VP.lambda_decoupling > 0:
                if 'HSIC' in args.VP.decoupling_loss:
                    loss += args.VP.lambda_decoupling * losses['HSIC']
                else:
                    loss += args.VP.lambda_decoupling * losses[args.VP.decoupling_loss]
            losses['total'] = loss

            # backward
            if training:
                if torch.isfinite(loss):
                    for space in losses:
                        if isinstance(losses[space], float): # to treat the case when conditional_indep_losses returns HSIC = 0.0
                            epoch_losses[space] += losses[space]
                        else:
                            epoch_losses[space] += losses[space].item()

                    loss.backward()
                    for optimizer in optimizers.values():
                        optimizer.step()
                else:
                    loss_was_finite = False
                    if training:
                        logger.warning(f"batch {i_batch} reached non-finite loss in training -> breaking")
                        break
                    else:
                        logger.warning(f"batch {i_batch} reached non-finite loss in eval (only skipping on adding batch to epoch loss")

        for space in epoch_log_probs:
            epoch_log_probs[space] = torch.cat(epoch_log_probs[space], dim=0)
        results = {}
        results.update({f'{space} acc': epoch_true_pos[space] / n_samples for space in ['color', 'shape', 'joint']})
        results['seen acc'] = epoch_true_pos['seen'] / n_seen
        n_unseen = n_samples - n_seen
        if n_unseen > 0:
            results['unseen acc'] = epoch_true_pos['unseen'] / n_unseen
            results['harmonic acc'] = 2 * results['seen acc'] * results['unseen acc'] / (
                    results['seen acc'] + results['unseen acc'])
        else:
            results['unseen acc'] = np.nan
            results['harmonic acc'] = np.nan
        results.update({f'{space} loss per sample': loss / n_samples for space, loss in epoch_losses.items()})
    return results, epoch_log_probs, loss_was_finite


def basic_EM(Y, Y_probs, prior_source, EM_iterations: int = 100):
    # init
    EM_results = pd.DataFrame(index=range(EM_iterations))
    EM_results.index.name = 'iteration'
    shift_probs_0 = Y_probs
    shift_probs = shift_probs_0
    prior_shift_pred = prior_source

    # eval
    shift_preds = shift_probs.argmax(axis=1)
    positives = shift_preds == Y
    # EM_results.loc[0, [f'accuracy ({label})' for label in range(args.n_labels)]] = positives.mean(axis=0)
    EM_results.loc[0, 'accuracy'] = positives.mean()
    Y_true_pos_probs = np.take_along_axis(shift_probs, Y.reshape(-1, 1), axis=1)
    EM_results.loc[0, 'log-likelihood'] = np.log(Y_true_pos_probs).sum()

    # EM
    for i_EM in tqdm(range(1, 1 + EM_iterations)):
        # E-step
        shift_probs = shift_probs_0 * prior_shift_pred / prior_source
        shift_probs = (shift_probs.T / shift_probs.sum(axis=1)).T

        shift_preds = shift_probs.argmax(axis=1)

        # eval
        positives = shift_preds == Y
        EM_results.loc[i_EM, 'accuracy'] = positives.mean()
        Y_true_pos_probs = np.take_along_axis(shift_probs, Y.reshape(-1, 1), axis=1)
        EM_results.loc[i_EM, 'log-likelihood'] = np.log(Y_true_pos_probs).sum()

        # M-step
        prior_shift_pred = shift_probs.mean(axis=0)
    assert (~np.isfinite(shift_probs)).sum() == 0
    return EM_results, shift_probs


def VisProd_EM(datasets: dict, models: dict, clss_loss, Y_comb: dict, label_dists_df, seen_combs, unseen_combs,
            unseen_combs_idx_to_comb_idx, args: ProjectArgs):
    # inference
    unshuffled_dataloaders = {phase: torch.utils.data.DataLoader(datasets[phase],
                                                                 batch_size=args.training.batch_size, shuffle=False,
                                                                 num_workers=args.training.workers,
                                                                 pin_memory=args.training.pin_memory) for phase in datasets}
    Y_color_probs, Y_shape_probs = {}, {}

    for phase in unshuffled_dataloaders:
        _, log_probs, _ = run_epoch(training=False, models=models, dataloader=unshuffled_dataloaders[phase],
                                 clss_loss=clss_loss, args=args, seen_combs=seen_combs)
        Y_color_probs[phase] = torch.exp(log_probs['color']).cpu().numpy()
        Y_shape_probs[phase] = torch.exp(log_probs['shape']).cpu().numpy()

    # calculating VisProd
    Y_VisProd_probs = {phase: np.zeros((len(Y_comb[phase]), args.data.n_colors * args.data.n_shapes)) for phase in
                       Y_comb}
    for phase, probs in Y_VisProd_probs.items():
        for i in tqdm(range(len(Y_comb[phase])), desc=f'{phase} VisProd'):
            outer_product = np.tensordot(Y_shape_probs[phase][i], Y_color_probs[phase][i], axes=0).flatten()
            probs[i, :] = outer_product / outer_product.sum()
    logger.info("EM requires probs, verify normalization:")
    for phase in Y_VisProd_probs:
        logger.info(f"(Y_VisProd_probs[{phase}].sum(axis=1) - 1).mean(): %.12f" % (
                    Y_VisProd_probs[phase].sum(axis=1) - 1).mean())

    # VisProd eval
    Y_VisProd_preds = {phase: probs.argmax(1) for phase, probs in Y_VisProd_probs.items()}
    for phase in Y_VisProd_probs:
        y = Y_comb[phase]
        y_pred = Y_VisProd_preds[phase]
        true_pos = pd.Series(y[y == y_pred]).value_counts()
        label_dists_df[f'VisProd: {phase} soft pred prior'] = pd.Series(Y_VisProd_probs[phase].mean(0))
        label_dists_df[f'VisProd: {phase} true pos'] = true_pos
        label_dists_df[f'VisProd: {phase} acc'] = label_dists_df[f'VisProd: {phase} true pos'] / label_dists_df[
            f'{phase} samples']
    closed_preds = pd.DataFrame(Y_VisProd_probs['test'][:, unseen_combs].argmax(1), columns=['unseen comb idx'])
    closed_preds['comb idx'] = closed_preds['unseen comb idx'].map(unseen_combs_idx_to_comb_idx)
    closed_preds['closed true pos'] = closed_preds['comb idx'] == Y_comb['test']
    closed_true_pos = closed_preds.query("`closed true pos`")['comb idx'].value_counts()
    label_dists_df[f'VisProd: test CLOSED true pos'] = closed_true_pos

    # adapted VisProd
    prior_source = Y_VisProd_probs['train'].mean(0)
    prior_shift_pred = 1 / len(label_dists_df) * np.ones((len(label_dists_df)))

    adapted_VisProd_probs = Y_VisProd_probs['test'] * prior_shift_pred / prior_source
    adapted_VisProd_probs = (adapted_VisProd_probs.T / adapted_VisProd_probs.sum(axis=1)).T

    phase = 'test'

    y = Y_comb[phase]
    y_pred = adapted_VisProd_probs.argmax(1)
    true_pos = pd.Series(y[y == y_pred]).value_counts()
    label_dists_df[f'adapted VisProd: {phase} soft pred prior'] = pd.Series(adapted_VisProd_probs.mean(0))
    label_dists_df[f'adapted VisProd: {phase} true pos'] = true_pos
    label_dists_df[f'adapted VisProd: {phase} acc'] = label_dists_df[f'adapted VisProd: {phase} true pos'] / \
                                                      label_dists_df[f'{phase} samples']

    closed_preds = pd.DataFrame(adapted_VisProd_probs[:, unseen_combs].argmax(1), columns=['unseen comb idx'])
    closed_preds['comb idx'] = closed_preds['unseen comb idx'].map(unseen_combs_idx_to_comb_idx)
    closed_preds['closed true pos'] = closed_preds['comb idx'] == Y_comb['test']
    closed_true_pos = closed_preds.query("`closed true pos`")['comb idx'].value_counts()
    label_dists_df[f'adapted VisProd: test CLOSED true pos'] = closed_true_pos

    # VisProd + EM
    VisProd_EM_probs = {}
    EM_metric_to_plot = 'accuracy'
    # EM_metric_to_plot = 'log-likelihood'
    # -------------------------------------
    prior_source = Y_VisProd_probs['train'].mean(0)

    for phase in set(Y_VisProd_probs.keys()) - set(['train']):
        VisProd_EM_evolution, VisProd_EM_probs[phase] = basic_EM(Y=Y_comb[phase],
                                                                 Y_probs=Y_VisProd_probs[phase],
                                                                 prior_source=prior_source,
                                                                 EM_iterations=args.VP.EM_iters)

        # plotting EM evolution
        # plt.figure(figsize=(4, 3))
        # VisProd_EM_evolution[EM_metric_to_plot].plot(ax=plt.gca(), linewidth=2)
        # plt.ylabel(EM_metric_to_plot)
        # plt.title(f"VisProd-EM evolution: {phase}")

        # eval
        y = Y_comb[phase]
        y_probs = VisProd_EM_probs[phase]
        # ----------------------------------

        y_pred = y_probs.argmax(1)
        true_pos = pd.Series(y[y == y_pred]).value_counts()
        label_dists_df[f'VisProd+EM: {phase} soft pred prior'] = pd.Series(y_probs.mean(0))
        label_dists_df[f'VisProd+EM: {phase} true pos'] = true_pos
        label_dists_df[f'VisProd+EM: {phase} acc'] = label_dists_df[f'VisProd+EM: {phase} true pos'] / label_dists_df[
            f'{phase} samples']

        closed_preds = pd.DataFrame(y_probs[:, unseen_combs].argmax(1), columns=['unseen comb idx'])
        closed_preds['comb idx'] = closed_preds['unseen comb idx'].map(unseen_combs_idx_to_comb_idx)
        closed_preds['closed true pos'] = closed_preds['comb idx'] == Y_comb[phase]
        closed_true_pos = closed_preds.query("`closed true pos`")['comb idx'].value_counts()
        label_dists_df[f'VisProd+EM: {phase} CLOSED true pos'] = closed_true_pos


    # VisProd + unseen-EM
    for phase in ['val', 'test']:
        VisProd_unseen_EM_probs = VisProd_EM_probs[phase].copy()
        VisProd_unseen_EM_probs[:, seen_combs] = Y_VisProd_probs[phase][:, seen_combs]
        VisProd_unseen_EM_probs = (VisProd_unseen_EM_probs.T / VisProd_unseen_EM_probs.sum(1)).T

        y = Y_comb[phase]
        y_probs = VisProd_unseen_EM_probs
        # ----------------------------------------

        y_pred = y_probs.argmax(1)
        true_pos = pd.Series(y[y == y_pred]).value_counts()
        label_dists_df[f'VisProd+unseen-EM: {phase} soft pred prior'] = pd.Series(y_probs.mean(0))
        label_dists_df[f'VisProd+unseen-EM: {phase} true pos'] = true_pos
        label_dists_df[f'VisProd+unseen-EM: {phase} acc'] = label_dists_df[f'VisProd+unseen-EM: {phase} true pos'] / \
                                                            label_dists_df[f'{phase} samples']

        closed_preds = pd.DataFrame(y_probs[:, unseen_combs].argmax(1), columns=['unseen comb idx'])
        closed_preds['comb idx'] = closed_preds['unseen comb idx'].map(unseen_combs_idx_to_comb_idx)
        closed_preds['closed true pos'] = closed_preds['comb idx'] == Y_comb[phase]
        closed_true_pos = closed_preds.query("`closed true pos`")['comb idx'].value_counts()
        label_dists_df[f'VisProd+unseen-EM: {phase} CLOSED true pos'] = closed_true_pos


def summarize(label_dists_df, args: ProjectArgs):
    pos_result_cols = [col for col in label_dists_df.columns if 'true pos' in col]
    open_pos_result_cols = [col for col in pos_result_cols if 'CLOSED' not in col]
    closed_pos_result_cols = [col for col in pos_result_cols if 'CLOSED' in col]
    samples_cols = [col for col in label_dists_df.columns if 'samples' in col]

    label_cols = ['shape', 'color']
    micro_result_cols = label_cols + ['state'] + samples_cols + pos_result_cols

    micro_results_df = label_dists_df[micro_result_cols]
    # with pd.option_context('display.float_format', "{:,.0f}".format):
    #     pprint(micro_results_df)

    micro_result_summary_dict = {}
    for state in ['seen', 'unseen']:
        micro_result_summary_dict[state] = micro_results_df.query(f"state == '{state}'")[open_pos_result_cols].sum(0)
    micro_result_summary_dict['total acc'] = micro_results_df[open_pos_result_cols].sum(0)
    micro_result_summary_df = pd.DataFrame.from_dict(micro_result_summary_dict, orient='index')

    # from positives to accuracy
    for col in micro_result_summary_df.columns:
        phase = col.split(': ')[1].split(' true')[0]
        micro_result_summary_df.loc['seen', col] /= micro_results_df.query("state == 'seen'")[f'{phase} samples'].sum()
        micro_result_summary_df.loc['unseen', col] /= micro_results_df.query("state == 'unseen'")[
            f'{phase} samples'].sum()
        micro_result_summary_df.loc['total acc', col] /= micro_results_df[f'{phase} samples'].sum()
    micro_result_summary_df.columns = micro_result_summary_df.columns.str.replace(" true pos", '')

    acc_seen = micro_result_summary_df.loc['seen']
    acc_unseen = micro_result_summary_df.loc['unseen']
    micro_result_summary_df.loc['harmonic'] = 2 * acc_seen * acc_unseen / (acc_seen + acc_unseen)

    for col in closed_pos_result_cols:
        micro_result_summary_df.loc['closed', col.replace('CLOSED true pos', '')] = micro_results_df[col].sum(0) / \
                                                                                    micro_results_df.query(
                                                                                        "state == 'unseen'")[
                                                                                        'test samples'].sum()
    if args.cfg.log_level <= 20:
        with pd.option_context('display.float_format', "{:,.1%}".format):
            pprint(micro_result_summary_df)

    return micro_result_summary_df


# %% Yuval's definitions
from typing import NamedTuple
from pathlib import Path
from ATTOP.data.dataset import sample_negative as ATTOP_sample_negative
from torch.utils import data


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
    s_frac = s_counts / s_counts.sum()
    hist_dict = s_counts.to_dict()
    if frac:
        hist_dict = s_frac.to_dict()
    hist = []
    for ix, _ in enumerate(labels_list):
        hist.append(hist_dict.get(ix, 0))

    if plot:
        pd.Series(hist, index=labels_list).plot(kind='bar')
        if frac:
            plt.ylim((0, 1))
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


@contextmanager
def ns_profiling_label(label):
    """
    A do nothing version of ns_profiling_label()

    """
    try:
        yield None
    finally:
        pass


def pairwise_distances(x):
    x_distances = torch.sum(x ** 2, -1).reshape((-1, 1))
    return -2 * torch.mm(x, x.t()) + x_distances + x_distances.t()


def kernelMatrixGaussian(x, sigma=1):
    pairwise_distances_ = pairwise_distances(x)
    gamma = -1.0 / (sigma ** 2)
    return torch.exp(gamma * pairwise_distances_)


def kernelMatrixLinear(x):
    return torch.matmul(x, x.t())


def median_pairwise_distance(X):
    t = pairwise_distances(X).detach()
    triu_indices = t.triu(diagonal=1).nonzero().T

    if triu_indices[0].shape[0] == 0 or triu_indices[1].shape[0] == 0:
        return 0.
    else:
        return torch.median(t[triu_indices[0], triu_indices[1]]).item()


def HSIC(X, Y, kernelX="Gaussian", kernelY="Gaussian", sigmaX=1, sigmaY=1,
         log_median_pairwise_distance=False):
    m, _ = X.shape
    assert (m > 1)

    median_pairwise_distanceX, median_pairwise_distanceY = np.nan, np.nan
    if log_median_pairwise_distance:
        # This calc takes a long time. It is used for debugging and disabled by default.
        with ns_profiling_label('dist'):
            median_pairwise_distanceX = median_pairwise_distance(X)
            median_pairwise_distanceY = median_pairwise_distance(Y)

    with ns_profiling_label('Hkernel'):
        K = kernelMatrixGaussian(X, sigmaX) if kernelX == "Gaussian" else kernelMatrixLinear(X)
        L = kernelMatrixGaussian(Y, sigmaY) if kernelY == "Gaussian" else kernelMatrixLinear(Y)

    with ns_profiling_label('Hfinal'):
        H = torch.eye(m, device='cuda') - 1.0 / m * torch.ones((m, m), device='cuda')
        H = H.float().cuda()

        Kc = torch.mm(H, torch.mm(K, H))

        HSIC = torch.trace(torch.mm(L, Kc)) / ((m - 1) ** 2)
    return HSIC, median_pairwise_distanceX, median_pairwise_distanceY


def conditional_indep_losses(repr1, repr2, ys, indep_coeff, indep_coeff2=None, num_classes1=None, num_classes2=None,
                             Hkernel='L', Hkernel_sigma_obj=None, Hkernel_sigma_attr=None,
                             log_median_pairwise_distance=False, device=None):
    # check readability

    normalize_to_mean = (num_classes1, num_classes2)

    if indep_coeff2 is None:
        indep_coeff2 = indep_coeff

    HSIC_loss_terms = []
    HSIC_mean_of_median_pairwise_dist_terms = []
    with ns_profiling_label('HSIC/d loss calc'):
        # iterate on both heads
        for m, num_class in enumerate((num_classes1, num_classes2)):
            with ns_profiling_label(f'iter m={m}'):
                HSIC_tmp_loss = 0.
                HSIC_median_pw_y1 = []
                HSIC_median_pw_y2 = []

                labels_in_batch_sorted, indices = torch.sort(ys[m])
                unique_ixs = 1 + (labels_in_batch_sorted[1:] - labels_in_batch_sorted[:-1]).nonzero()
                unique_ixs = [0] + unique_ixs.flatten().cpu().numpy().tolist() + [len(ys[m])]

                for j in range(len(unique_ixs) - 1):
                    current_class_indices = unique_ixs[j], unique_ixs[j + 1]
                    count = current_class_indices[1] - current_class_indices[0]
                    if count < 2:
                        continue
                    curr_class_slice = slice(*current_class_indices)
                    curr_class_indices = indices[curr_class_slice].sort()[0]

                    with ns_profiling_label(f'iter j={j}'):
                        HSIC_kernel = dict(G='Gaussian', L='Linear')[Hkernel]
                        with ns_profiling_label('HSIC call'):
                            hsic_loss_i, median_pairwise_distance_y1, median_pairwise_distance_y2 = \
                                HSIC(repr1[curr_class_indices, :].float(), repr2[curr_class_indices, :].float(),
                                     kernelX=HSIC_kernel, kernelY=HSIC_kernel,
                                     sigmaX=Hkernel_sigma_obj, sigmaY=Hkernel_sigma_attr,
                                     log_median_pairwise_distance=log_median_pairwise_distance)
                        HSIC_tmp_loss += hsic_loss_i
                        HSIC_median_pw_y1.append(median_pairwise_distance_y1)
                        HSIC_median_pw_y2.append(median_pairwise_distance_y2)

                HSIC_tmp_loss = HSIC_tmp_loss / normalize_to_mean[m]
                HSIC_loss_terms.append(HSIC_tmp_loss)
                HSIC_mean_of_median_pairwise_dist_terms.append([np.mean(HSIC_median_pw_y1), np.mean(HSIC_median_pw_y2)])

    indep_loss = torch.tensor(0.).to(device)
    if indep_coeff > 0:
        indep_loss = (indep_coeff * HSIC_loss_terms[0] + indep_coeff2 * HSIC_loss_terms[1]) / 2
    return indep_loss, HSIC_loss_terms, HSIC_mean_of_median_pairwise_dist_terms


# %% ProtoProp HSIC
# https://github.com/FrankRuis/ProtoProp/blob/0b60866bab619d39072f97bb96eb3ec713b6f51b/model/hsic.py

def sigma_estimation(X, Y):
    """ sigma from median distance
    """
    D = distmat(torch.cat([X, Y]))
    D = D.detach().cpu().numpy()
    Itri = np.tril_indices(D.shape[0], -1)
    Tri = D[Itri]
    med = np.median(Tri)
    if med <= 0:
        med = np.mean(Tri)
    if med < 1E-2:
        med = 1E-2
    return med


def distmat(X):
    """ distance matrix
    """
    r = torch.sum(X * X, 1)
    r = r.view([-1, 1])
    a = torch.mm(X, torch.transpose(X, 0, 1))
    D = r.expand_as(a) - 2 * a + torch.transpose(r, 0, 1).expand_as(a)
    D = torch.abs(D)
    return D


def kernelmat(X, sigma):
    """ kernel matrix baker
    """
    m = int(X.size()[0])
    H = torch.eye(m) - (1. / m) * torch.ones([m, m])
    Dxx = distmat(X)

    if sigma:
        variance = 2. * sigma * sigma * X.size()[1]
        Kx = torch.exp(-Dxx / variance).type(torch.FloatTensor)  # kernel matrices
    else:
        try:
            sx = sigma_estimation(X, X)
            Kx = torch.exp(-Dxx / (2. * sx * sx)).type(torch.FloatTensor)
        except RuntimeError as e:
            raise RuntimeError("Unstable sigma {} with maximum/minimum input ({},{})".format(
                sx, torch.max(X), torch.min(X)))

    Kxc = torch.mm(Kx, H)

    return Kxc


def distcorr(X, sigma=1.0):
    X = distmat(X)
    X = torch.exp(-X / (2. * sigma * sigma))
    return torch.mean(X)


def compute_kernel(x, y):
    x_size = x.size(0)
    y_size = y.size(0)
    dim = x.size(1)
    x = x.unsqueeze(1)  # (x_size, 1, dim)
    y = y.unsqueeze(0)  # (1, y_size, dim)
    tiled_x = x.expand(x_size, y_size, dim)
    tiled_y = y.expand(x_size, y_size, dim)
    kernel_input = (tiled_x - tiled_y).pow(2).mean(2) / float(dim)
    return torch.exp(-kernel_input)  # (x_size, y_size)


def mmd(x, y, sigma=None, use_cuda=True, to_numpy=False):
    Dxx = distmat(x)
    Dyy = distmat(y)

    if sigma:
        Kx = torch.exp(-Dxx / (2. * sigma * sigma))  # kernel matrices
        Ky = torch.exp(-Dyy / (2. * sigma * sigma))
        sxy = sigma
    else:
        sx = sigma_estimation(x, x)
        sy = sigma_estimation(y, y)
        sxy = sigma_estimation(x, y)
        Kx = torch.exp(-Dxx / (2. * sx * sx))
        Ky = torch.exp(-Dyy / (2. * sy * sy))

    Dxy = distmat(torch.cat([x, y]))
    Dxy = Dxy[:x.size()[0], x.size()[0]:]
    Kxy = torch.exp(-Dxy / (1. * sxy * sxy))

    mmdval = torch.mean(Kx) + torch.mean(Ky) - 2 * torch.mean(Kxy)

    return mmdval


def mmd_pxpy_pxy(x, y, sigma=None, use_cuda=True, to_numpy=False):
    """
    """
    if use_cuda:
        x = x.cuda()
        y = y.cuda()

    Dxx = distmat(x)
    Dyy = distmat(y)
    if sigma:
        Kx = torch.exp(-Dxx / (2. * sigma * sigma))  # kernel matrices
        Ky = torch.exp(-Dyy / (2. * sigma * sigma))
    else:
        sx = sigma_estimation(x, x)
        sy = sigma_estimation(y, y)
        Kx = torch.exp(-Dxx / (2. * sx * sx))
        Ky = torch.exp(-Dyy / (2. * sy * sy))
    A = torch.mean(Kx * Ky)
    B = torch.mean(torch.mean(Kx, dim=0) * torch.mean(Ky, dim=0))
    C = torch.mean(Kx) * torch.mean(Ky)
    mmd_pxpy_pxy_val = A - 2 * B + C
    return mmd_pxpy_pxy_val


def hsic_regular(x, y, sigma=None, use_cuda=True):
    """
    """
    Kxc = kernelmat(x, sigma)
    Kyc = kernelmat(y, sigma)
    KtK = torch.mul(Kxc, Kyc.t())
    Pxy = torch.mean(KtK)
    return Pxy


def hsic_normalized(x, y, sigma=None, use_cuda=True):
    """
    """
    Pxy = hsic_regular(x, y, sigma, use_cuda)
    Px = torch.sqrt(hsic_regular(x, x, sigma, use_cuda))
    Py = torch.sqrt(hsic_regular(y, y, sigma, use_cuda))
    if Py == 0:
        print(y)
        exit()
    thehsic = Pxy / (Px * Py)
    return thehsic
