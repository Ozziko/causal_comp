#%% analyzing sweep combinations
sweep_path = 'sweep_MLLS_VisProd.yaml'
# -----------------------------

import numpy as np
from pprint import pprint
import yaml

with open(sweep_path, 'r') as f:
    yaml_dict = yaml.safe_load(f)
args_dict = yaml_dict['parameters']
swept_args_dict = {}
for arg in args_dict:
    if 'values' in args_dict[arg]:
        swept_args_dict[arg] = args_dict[arg]['values']

print('swept_args_dict:')
pprint(swept_args_dict)
combinations = [len(vals) for vals in swept_args_dict.values()]
print('\ncombinations:', np.prod(combinations))


#%% making commandlines
command = r'wandb agent ozziko/MLLS_VisProd/o7m3le74'
# command = "python simple_train.py " + """--run_name="DGX cifar100 SGD CrossEntropyLoss bs=256 lr=1e-3" --wandb=True --lr=1e-3
# command = "python DLS_run.py " + """--args.cache.trained_weights.save=True --args.cache.training_results.save=True --args.cfg.seed=0 --args.data.TRAIN_size=200000 --args.data.calibration=none "--args.data.dataset=synthetic 1D uniform train-CLIPPED sigma=1.5" "--args.data.input_source=raw x" --args.data.normalize_by_softmax=False --args.data.seed=-1 --args.data.split_train_val_no_rand=True --args.data.test.Dirichlet_alpha=10 --args.data.test.discretizing_mode=multinomial --args.data.test.label_dist=Dirichlet --args.data.test.shuffle_sampled_indices=True --args.data.test.size=10000 --args.data.test_splits_to_load=10 --args.data.val_over_TRAIN=0.5 --args.dataloading.train.label_dist=Dirichlet --args.dataloading.train.n_batches=0 --args.dataloading.val.Dirichlet_alpha=log10U~[-1,1] --args.dataloading.val.freeze_batches=True --args.dataloading.val.label_dist=Dirichlet --args.dataloading.val.n_batches=0 --args.eval.average_tests=True --args.eval.calib_metrics=True --args.eval.prior_err_metrics_hard=False --args.eval.prior_err_metrics_soft=False --args.eval.prior_loss_hard=True --args.eval.prior_loss_soft=True --args.eval.test.eval_with_epoch_prior_input=True --args.eval.test.reorder_by=none --args.eval.test.skip_unadapted=True --args.eval.train.eval_with_epoch_prior_input=True --args.eval.train_val_diff=True --args.eval.val.eval_with_epoch_prior_input=True --args.experiment_alpha=log10U~[-1,1] --args.experiment_alpha_on=train --args.model.architecture=LayerParsed --args.model.recurrence=0 --args.model.variant=in-8MS2-E-4M2-E-2M2-E-outM2 --args.training.batch_size=10000 --args.training.early_stop.limit_mode=off "--args.training.early_stop.metric=accuracy - micro" --args.training.early_stop.min_delta=0.001 --args.training.early_stop.mode=max --args.training.early_stop.patience=20 --args.training.epochs=250 "--args.training.loss_labels=CrossEntropyLoss (sum)" --args.training.loss_labels_lambda=1 --args.training.loss_labels_weights=none --args.training.loss_prior_lambda=0 --args.training.lr=0.001 --args.training.mixup_alpha=100 --args.training.optimizer=Adam --args.training.plateau_scheduler.mode=off --args.training.rand_transforms=off --args.training.teacher_forcing_ratio=0 --args.training.wd=0 --args.training.workers=4 --args.wandb.is_sweep=True "--args.wandb.run_naming_args=experiment_alpha, model.variant, training.batch_size, training.lr, cfg.seed" "--args.wandb.sweep_name=synthetic CLIPPED LayerParsed mixup bs=1e4 n_samples=1e5 exp_alpha=log10U[-1,1]" --args.wandb.use=True
#      """.strip()
# command = "python DLS_run.py " + command_multiline # run relevant cell from commandlines.py, then run this cell

commands_per_gpu = 5
# commands_per_gpu = 1

agent = 0

machine, gpus = 'dsictal01', range(4)
# machine, gpus = 'dsictx01', range(3)

# machine, gpus = 'dsictal01', [0]
# machine, gpus = 'dgx02', [7]

# machine, gpus = 'dsictal01', range(3)
# machine, gpus = 'dgx02', set(range(8))-set([4])
# machine, gpus = 'dgx02', range(6, 8)
# machine, gpus = 'dgx01', [2,4,5,6,7]
# machine, gpus = 'ctx20', range(2)

# gpus = [0,1,3]
# machine = input("machine: ")

tail_last = True
# -------------------------------------------
# agent = int(input("agent: "))

for gpu in gpus:
    print(f"export CUDA_VISIBLE_DEVICES={gpu}")
    for _ in range(commands_per_gpu):
        print(f"nohup {command} > nohup_{machine}_{agent}.out 2>&1 &")
        agent += 1
if tail_last:
    print(f"tail -f nohup_{machine}_{agent-1}.out\n")

