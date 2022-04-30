from dataclasses import dataclass, fields
from simple_parsing import ArgumentParser, ConflictResolution, Serializable
from typing import Sequence

"""Type verification is done automatically in simple_parsing, BUT here it is added also directly to the 
    dataclasses (by inheriting SimpleArgsTemplate or adding verification in __post_init__) to force
    verification also if simple_parsing is not used - e.g. in notebooks!!!
"""


class SimpleArgsTemplate(Serializable):
    """Inherit this class by classes intended for (decorated by) dataclass, for added functionality:
        * Serialization (to/from dict, to/from json etc)
        * Input type verification in __post_init__
    """
    def __post_init__(self):
        # Input type verification
        for field in fields(self):
            value = getattr(self, field.name)
            if (value is not None) and \
                    not (isinstance(value, field.type) or (isinstance(value, int) and field.type is float)):
                raise ValueError(f'{self.__class__} expects {field.name} to be {field.type}, '
                                 f'got {repr(value)}')


@dataclass
class VisProd(SimpleArgsTemplate):
    EM_iters: int = 30
    # decoupling_loss: str = 'MI g.t.' # calculating MI loss with ground-truth labels for joint dist
    # lambda_decoupling: float = 0

    # MI_mode: str = 'MI est' # training an additional net for joint classification, to calculate MI loss with estimated probability for joint dist
    # lambda_decoupling: float = 10
    lambda_joint: float = 1

    # decoupling_loss: str = 'HSIC,L' # Yuval's HSIC with Linear kernel
    decoupling_loss: str = 'HSIC,G'  # Yuval's HSIC with Gaussian kernel and std=1
    # decoupling_loss: str = 'HSIC,N' # ProtoProp's normalized Gaussian HSIC
    lambda_decoupling: float = 3e5

    epochs: int = 50
    lambda_color: float = 1
    lambda_shape: float = 1

    optimizer: str = 'Adam'
    lr: float = 1e-3
    # optimizer: str = 'SGD'
    # lr: float = 1e-4

    early_stop_phase: str = 'val'  # also for plateau scheduler
    early_stop_on: str = 'harmonic acc'  # also for plateau scheduler
    # early_stop_mode: str = 'off'
    early_stop_mode: str = 'max'
    early_stop_patience: int = 8
    early_stop_threshold: float = 1e-3  # abs diff

    # plateau_mode: str = 'off'
    plateau_mode: str = 'max'
    plateau_factor: float = 0.1
    plateau_patience: int = 5
    plateau_threshold: float = 1e-3
    plateau_threshold_mode: str = 'abs'
    # plateau_threshold_mode: str = 'rel'

    def __post_init__(self):
        # Input type verification
        for field in fields(self):
            value = getattr(self, field.name)
            if not (isinstance(value, field.type) or (isinstance(value, int) and field.type is float)):
                raise ValueError(f'{self.__class__} expects {field.name} to be {field.type}, '
                                 f'got {repr(value)}')

        # Input value verifications
        assert self.decoupling_loss in ['MI g.t.', 'MI est', 'HSIC,L', 'HSIC,G', 'HSIC,N']
        assert self.optimizer in ['Adam', 'SGD']
        assert self.early_stop_mode in ['off', 'max', 'min']
        assert self.plateau_mode in ['off', 'max', 'min']
        assert self.plateau_threshold_mode in ['rel', 'abs']

@dataclass
class WandbArgs(SimpleArgsTemplate):
    use: bool = True  # Whether to use wandb for experiment control, logging, saving...
    dir: str = ''  # Wandb run cache path, relative to project_path
    save_results: bool = True  # Wether to save results (.torch) in addition to logging, which is enabled if wandb.use=True
    save_weights: bool = False
    run_naming_args: str = ''  #
    is_sweep: bool = False  # Making the script aware/assume it's sweeping, good also for sweep debugging
    sweep_name: str = ''  # Sweep name (before adding run_naming_args)
    run_notes: str = ''  # Run notes
    delete_synced_dir: bool = True  # Deleting wandb run dir after sync by wandb.finish()

    def __post_init__(self):
        # Input type verification
        for field in fields(self):
            value = getattr(self, field.name)
            if not (isinstance(value, field.type) or (isinstance(value, int) and field.type is float)):
                raise ValueError(f'{self.__class__} expects {field.name} to be {field.type}, '
                                 f'got {repr(value)}')

        # Input value verifications
        assert r'/' not in self.sweep_name


@dataclass
class Configuration(SimpleArgsTemplate):
    seed: int = -1  # For general seeding in steps of 1000: random,np,torch,cuda; = -1 -> cfg.seed = data.seed (in ProjectArgs.__post_init__)
    project_name: str = 'MLLS_VisProd'  # For logging etc
    project_path: str = ''  # Relative path from execution dir to project; default ('')
    output_dir: str = ''  # Default ('') = pjoin(project_path, 'outputs'); relative path from project_path to outputs dir, where each run outputs will be saved under a unique run_name dir
    run_name: str = 'unnamed run'  # Run name; output will be saved under a unique run_name dir, adding a suffix if necessary, e.g. 'dev', 'dev_2', 'dev_3'...
    post_delete_outputs: bool = True  # Whether to delete outputs at execution completion
    log_level: int = 20  # 10 (DEBUG) | 20 (INFO) | 30 (WARNING) | 40 (ERROR) | 50 (CRITICAL)
    log_to_file: bool = False  # if True: log to f'{project_name} log.txt' in outputs dir
    last_git_commit: str = ''  # Enter manually if .git is not in the project's dir on the executing machine or not using wandb; else wandb saves it automatically

    def __post_init__(self):
        # Input type verification
        for field in fields(self):
            value = getattr(self, field.name)
            if not (isinstance(value, field.type) or (isinstance(value, int) and field.type is float)):
                raise ValueError(f'{self.__class__} expects {field.name} to be {field.type}, '
                                 f'got {repr(value)}')

        # Input value verifications
        assert r'/' not in self.project_name
        assert r'/' not in self.run_name
        assert self.log_level in [10,20,30,40,50]
        if self.log_level == 10:
            print("WARNING: log_level=10 (DEBUG) set, may not reproduce results with higher log levels (no debugging)!")


@dataclass
class Data(SimpleArgsTemplate):
    """Args for raw data, and datasets
    """
    dataset: str = 'ao_clevr'
    dir: str = ''  # empty = f'data/{args.dataset_name}'
    variant: str = 'VT' # 'VT', 'OZ'
    seen_seed: int = 0
    num_split: int = 5000
    seed: int = -1 # seed = leave to be set in ProjectArgs.__post_init__: int(num_split_str[2:])
    unseen_ovr_tot: float = 0  # leave to be set in ProjectArgs.__post_init__: int(num_split_str[:2]) / 100

    # val_unseen_mode: str = 'leave' # leave Yuval's val as is (with 5 images for unseen combinations)
    val_unseen_mode: str = 'complete'  # add images to val unseen combinations to reach the mean val size in seen combinations
    # val_unseen_mode: str = 'drop' # drop any val images from unseen combinations

    # only for OZ splits
    train_size: int = int(80e3)
    train_unseen_ovr_seen: float = 0
    test_size: int = int(8e3)

    n_shapes: int = 0 # leave to be set according to dataset in MLLS_utils.load_data
    n_colors: int = 0 # leave to be set according to dataset in MLLS_utils.load_data

    def __post_init__(self):
        # Input type verification
        for field in fields(self):
            value = getattr(self, field.name)
            if not (isinstance(value, field.type) or (isinstance(value, int) and field.type is float)):
                raise ValueError(f'{self.__class__} expects {field.name} to be {field.type}, '
                                 f'got {repr(value)}')

        # Input value verifications
        assert self.dataset in ['ao_clevr']
        assert self.variant in ['VT', 'UV', 'OZ']
        assert self.val_unseen_mode in ['leave', 'complete', 'drop']

@dataclass
class Training(SimpleArgsTemplate):
    deterministic: bool = True
    batch_size: int = 512
    pin_memory: bool = True
    workers: int = 0


@dataclass
class ProjectArgs(SimpleArgsTemplate):
    cfg: Configuration = Configuration()
    data: Data = Data()
    training: Training = Training()
    VP: VisProd = VisProd()
    wandb: WandbArgs = WandbArgs()

    def __post_init__(self):
        # Input type verification
        for field in fields(self):
            value = getattr(self, field.name)
            if not (isinstance(value, field.type) or (isinstance(value, int) and field.type is float)):
                raise ValueError(f'{self.__class__} expects {field.name} to be {field.type}, '
                                 f'got {repr(value)}')

        # assertions
        if self.cfg.log_level > 10:
            # assert self.data.test.shuffle_sampled_indices, "args.data.test.shuffle_sampled_indices=False set but not allowed outside of debugging (args.cfg.log_level > 10) to avoid critical mistakes, for non-debugging runs comment this line"
            assert self.training.deterministic, "args.training.deterministic=False set but not allowed outside of debugging (args.cfg.log_level > 10) to avoid critical mistakes"

        # auxiliary params
        num_split_str = str(self.data.num_split)
        self.data.unseen_ovr_tot = int(num_split_str[:2]) / 100
        self.data.seed = int(num_split_str[2:])
        if self.cfg.seed == -1:
            self.cfg.seed = self.data.seed

    @classmethod
    def argparse(cls):
        """Passing the class through simple_parsing to make the class arguments accessible in commandline"""
        parser = ArgumentParser(conflict_resolution=ConflictResolution.EXPLICIT, add_dest_to_option_strings=True)
        parser.add_arguments(cls, dest='args')
        args: cls = parser.parse_args().args
        return args
