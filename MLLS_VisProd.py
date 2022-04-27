# Imports
from MLLS_VisProd_params import ProjectArgs
import MLLS_utils

import numpy as np
import pandas as pd
import torch

import sys
import os
from os.path import join as pjoin
# from pathlib import Path
import shutil
from copy import deepcopy

from pprint import pprint
import logging
from tqdm.auto import tqdm
from time import time

from collections import defaultdict, Counter


if __name__ == '__main__':
    start_time = time()

    # //////////// Argparsing //////////////////////////////////////////////////////////
    commandline = sys.argv
    commandline_args = [elem.split('=')[0].replace('-', '').replace('"', '') for elem in commandline[1:]]
    duplicated_commandline_args_counter = {arg: counts for arg, counts in Counter(commandline_args).items() if
                                           counts > 1}
    if len(duplicated_commandline_args_counter) > 0:
        raise RuntimeError(f"Remove duplicated commandline args! Duplicated args counter: {duplicated_commandline_args_counter}")

    args = ProjectArgs().argparse()

    # //////////// # Configuration //////////////////////////////////////////////////////////
    args.cfg.run_name = MLLS_utils.naming_run(args)
    MLLS_utils.renaming_paths_inplace(args)
    MLLS_utils.creating_outputs_dir(args)

    args_dict = args.to_dict()

    # //////////// Initializing wandb //////////////////////////////////////////////////////////
    # also in sweeps - although informing "ignoring wandb.init()" it only means it doesn't change args -
    # but logs args, outputs since wandb.init()
    if args.wandb.use:
        import wandb

        if args.wandb.dir == '':
            args.wandb.dir = args.cfg.output_dir
        #         args.wandb.dir = pjoin('outputs', os.path.split(output_dir)[-1])
        wandb_run = wandb.init(config=args_dict, project=args.cfg.project_name, name=args.cfg.run_name,
                               notes=args.wandb.run_notes, dir=args.wandb.dir, reinit=True, save_code=False)

    # //////////// Starting logging //////////////////////////////////////////////////////////
    logging.basicConfig(format='%(asctime)s (%(levelname)s): %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    logger_name = args.cfg.project_name + ' training'
    logger = logging.getLogger(logger_name)
    logger.setLevel(args.cfg.log_level)
    MLLS_utils.set_logger_level(args.cfg.log_level)
    if args.cfg.log_to_file:
        logger.addHandler(logging.FileHandler(pjoin(args.cfg.output_dir, f'{args.cfg.project_name} log.txt')))

    if args.cfg.log_level <= 20:
        print('/////// commandline: ///////')
        pprint(commandline)

        print('/////// args: ///////')
        pprint(args_dict)

    # //////////// Assertions //////////////////////////////////////////////////////////
    if os.name == 'nt' and args.training.workers > 0 and args.training.pin_memory:
        logger.warning("Running on Windows with args.training.workers > 0 and args.training.pin_memory might lead to memory issues")

    # //////////// Torch setup, general seeding //////////////////////////////////////////////////////////
    MLLS_utils.torch_setup(deterministic=args.training.deterministic)
    MLLS_utils.seed_all(seed=args.cfg.seed)

    # //////////// Data loading //////////////////////////////////////////////////////////
    X, Y_comb, Y_shape, Y_color, label_dists_df, unseen_combs, seen_combs, unseen_combs_idx_to_comb_idx = MLLS_utils.load_data(args=args)

    # //////////// Preparing training //////////////////////////////////////////////////////////
    datasets, dataloaders, models, clss_loss, optimizers, schedulers = MLLS_utils.build_training(
        X=X, Y_comb=Y_comb, Y_shape=Y_shape, Y_color=Y_color, seen_combs=seen_combs, args=args)
    results = defaultdict(dict)

    # /////// VisProd training & evaluation ///////////////
    time_before_train = time()
    results = {phase: {} for phase in ['train', 'val', 'test']}
    early_stopped = False
    for epoch in tqdm(range(args.VP.epochs + 1)):
        # training
        if epoch > 0:
            results['train'][epoch], _ = MLLS_utils.run_epoch(training=True, models=models, dataloader=dataloaders['train'],
                                                   clss_loss=clss_loss, args=args, optimizers=optimizers,
                                                   seen_combs=seen_combs)

        # eval
        for phase in ['train', 'val']:
            results[phase][epoch], _ = MLLS_utils.run_epoch(training=False, models=models, dataloader=dataloaders[phase],
                                                 clss_loss=clss_loss, args=args, seen_combs=seen_combs)
            if args.wandb.use:
                results4wandb = results[phase][epoch]
                results4wandb = {key + f' <{phase}>': val for key, val in results4wandb.items()}
                results4wandb.update({'epoch': epoch, 'phase': phase})
                if phase == 'train' and epoch > 0:
                    for space, optimizer in optimizers.items():
                        results4wandb[f'{space} lr <{phase}>'] = optimizer.param_groups[0]['lr']
                wandb.log(results4wandb)

            # early stopping
            if args.VP.early_stop_mode != 'off' and args.VP.early_stop_phase == phase:
                if epoch == 0:
                    best_epoch = 0
                    best_metric = results[phase][epoch][args.VP.early_stop_on]
                    best_model_states = {space: deepcopy(model.state_dict()) for space, model in models.items()}
                    no_improv_epochs = 0
                else:
                    current_metric = results[phase][epoch][args.VP.early_stop_on]
                    delta = current_metric - best_metric
                    if (args.VP.early_stop_mode == 'max' and delta > args.VP.early_stop_threshold) or (
                            args.VP.early_stop_mode == 'min' and delta < args.VP.early_stop_threshold):
                        best_epoch = epoch
                        best_metric = current_metric
                        best_model_states = {space: deepcopy(model.state_dict()) for space, model in models.items()}
                    else:
                        no_improv_epochs += 1
                        if no_improv_epochs >= args.VP.early_stop_patience:
                            print(f"early stopping on epoch {epoch}: '{args.VP.early_stop_on}' has stopped improving "
                                  f"(patience = {args.VP.early_stop_patience}),"
                                  f" delta ({delta}) is {'less' if args.VP.early_stop_mode == 'max' else 'more'} "
                                  f"than min_delta ({args.VP.early_stop_threshold}) for more than {args.VP.early_stop_patience} epochs")
                            early_stopped = True
                            break
        if early_stopped:
            break

        # schedulers
        if args.VP.plateau_mode != 'off':
            for scheduler in schedulers.values():
                scheduler.step(results[args.VP.early_stop_phase][epoch][args.VP.early_stop_on])

    train_val_time = time() - time_before_train
    logger.info("train-val time: %.2e sec = %dm:%.1fs" % (train_val_time, train_val_time // 60, train_val_time % 60))

    # returning to best epoch
    if args.VP.early_stop_mode != 'off':
        print(
            f"loading best model states to have results['{args.VP.early_stop_phase}'][epoch={best_epoch}]['{args.VP.early_stop_on}'] = {best_metric}")
        for space, model in models.items():
            model.load_state_dict(best_model_states[space])
            print(f"loaded '{space}' state")

    # /////// VisProd-EM ///////////////
    # results are saved by VisProd_EM inplace in label_dists_df:
    MLLS_utils.VisProd_EM(datasets=datasets, models=models, clss_loss=clss_loss, Y_comb=Y_comb,
                          label_dists_df=label_dists_df, seen_combs=seen_combs, unseen_combs=unseen_combs,
                          unseen_combs_idx_to_comb_idx=unseen_combs_idx_to_comb_idx, args=args)
    micro_result_summary_df = MLLS_utils.summarize(label_dists_df=label_dists_df, args=args)

    # //////////// Run summary //////////////////////////////////////////////////////////
    results['summary'] = {
        'train-val time (s)': train_val_time,
        'last epoch': epoch,
        'early_stopped': early_stopped,
    }
    if args.VP.early_stop_mode != 'off':
        results['summary']['best epoch'] = best_epoch

    summary_dict = {}
    for metric, row in micro_result_summary_df.iterrows():
        for phase, val in row.items():
            summary_dict[f"{metric} <{phase}>"] = val
    results['summary'].update(summary_dict)
    if args.wandb.use:
        wandb.run.summary.update(results['summary'])

    # //////////// Saving final outputs (including test) //////////////////////////////////////////////////////////
    # saving to wandb
    if args.wandb.use:
        args.save_json(pjoin(wandb.run.dir, 'args.json'))
        if args.wandb.save_results:
            torch.save(results, pjoin(wandb.run.dir, 'results.torch'))
        # if args.wandb.save_weights:
        #     torch.save(model.state_dict(), pjoin(wandb.run.dir, 'model_state_path.torch'))

    # saving in run dir
    if args.cfg.run_name != '' and not args.cfg.post_delete_outputs:
        args_json_path = pjoin(args.cfg.output_dir, 'args.json')
        args.save_json(args_json_path)
        logger.info(f"saved '{args_json_path}'")

        results_path = pjoin(args.cfg.output_dir, 'results.torch')
        torch.save(results, results_path)
        logger.info(f"saved '{results_path}'")

        # model_state_path = pjoin(args.cfg.output_dir, 'model_state.torch')
        # torch.save(model.state_dict(), model_state_path)
        # logger.info(f"saved '{model_state_path}'")

    # //////////// Completing run //////////////////////////////////////////////////////////
    if args.wandb.use:
        wandb_run.finish()
        logger.info("synced wandb")
    if args.cfg.post_delete_outputs:
        shutil.rmtree(args.cfg.output_dir)
        logger.info(f"removed '{args.cfg.output_dir}'")
    if not args.cfg.post_delete_outputs and args.wandb.use and args.wandb.delete_synced_dir:
        wandb_dir = pjoin(args.wandb.dir, 'wandb')
        shutil.rmtree(wandb_dir)
        logger.info(f"removed '{wandb_dir}'")

    run_time = time() - start_time
    logger.info("total run time: %.2e sec = %dm:%.1fs" % (run_time, run_time // 60, run_time % 60))
    logger.info(f"completed :-)")