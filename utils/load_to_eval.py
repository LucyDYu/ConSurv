# Copyright 2022-present, Lorenzo Bonicelli, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import gc
import numpy as np
import copy
import math
import os
import sys
from argparse import Namespace
from time import sleep, time
from typing import Iterable, Tuple
import logging
import pandas as pd
import torch
from tqdm import tqdm

from datasets import get_dataset
from datasets.utils.continual_dataset import ContinualDataset, MammothDatasetWrapper
from datasets.utils.gcl_dataset import GCLDataset
from models.utils.continual_model import ContinualModel
from models.utils.future_model import FutureModel

from utils import disable_logging
from utils.checkpoints import mammoth_load_checkpoint, save_mammoth_checkpoint
from utils.core_utils import EarlyStopping, Monitor_CIndex
from utils.loggers import log_extra_metrics, log_accs, Logger
from utils.schedulers import get_scheduler
from utils.stats import track_system_stats

from sksurv.metrics import concordance_index_censored

from utils.training_utils import initialize_survival_metrics, wait_for_memory
from utils.training_wsi import get_run_name, initialize_wandb, train_single_task_setup, train_single_epoch
from utils.wsi_metrics import df_add_column, df_add_columns, get_last_value_from_metric_dict_tuple, matrix_remove_keys, update_metric_matrix_tuple
# from pytorch_memlab import MemReporter
# from pytorch_memlab import LineProfiler



def load_to_eval(model: ContinualModel, dataset: ContinualDataset,
          args: Namespace) -> None:
    """
    The training process, including evaluations and loggers.

    Args:
        model: the module to be trained
        dataset: the continual dataset at hand
        args: the arguments of the current execution
    """
    logging.info(f"load_to_eval on task level")
    assert args.loadcheck_base_name is not None
    assert args.inference_only == 1

    # The dataset should be seq-survival
    assert dataset.NAME == 'seq-survival'

    fold = args.fold

    # Initialize flags related to forward transfer evaluation
    is_fwd_enabled = True
    can_compute_fwd_beforetask = True
    random_results_class, random_results_task = [], []
    random_results_class_c_index_ipcw, random_results_task_c_index_ipcw = [], []


    # If logging is enabled, create a logger instance
    if not args.disable_log:
        logger = Logger(args, dataset.SETTING, dataset.NAME, model.NAME)

    # Move the model to the specified device and clear GPU cache
    model.net.to(model.device)
    torch.cuda.empty_cache()

    # Copy the model for random baseline
    model_random = copy.deepcopy(model)

    # Proactively clean up memory before evaluation
    gc.collect()
    torch.cuda.empty_cache()

    # Use system resource tracker
    with track_system_stats(logger) as system_tracker:
        # Initialize result storage lists
        results, results_mask_classes = [], []
        results_c_index_ipcw, results_mask_classes_c_index_ipcw = [], []

        # If evaluation on future tasks is required, initialize corresponding result lists
        if args.eval_future:
            results_transf, results_mask_classes_transf = [], []

        # If a starting task is specified, skip earlier tasks
        if args.start_from is not None:
            for i in range(args.start_from):
                train_loader, _ = dataset.get_data_loaders()
                model.meta_begin_task(dataset)
                model.meta_end_task(dataset)

        print(file=sys.stderr)
        # Determine the index of the starting and ending tasks
        start_task = 0 if args.start_from is None else args.start_from
        end_task = dataset.N_TASKS if args.stop_after is None else args.stop_after

        # If evaluation on future tasks is required, prepare evaluation dataset
        if args.eval_future:
            assert isinstance(model, FutureModel), "Model must be an instance of FutureModel to evaluate on future tasks"
            eval_dataset = get_dataset(args)

            # disable logging for this loop
            with disable_logging(logging.WARNING):
                for _ in range(dataset.N_TASKS):
                    eval_dataset.get_data_loaders()
                    model.change_transform(eval_dataset)
                    del eval_dataset.train_loader
        else:
            eval_dataset = dataset

        # Clear GPU cache again
        torch.cuda.empty_cache()

        task_name_order = dataset.task_name_order
        # Initialize a dictionary to store epochs for each task at the start of the function
        c_index_val_matrix = {}
        c_index_ipcw_val_matrix = {}

        # Start the main training loop, iterate through each task
        for t in range(start_task, end_task):
            # empty_cache before each task
            gc.collect()
            torch.cuda.empty_cache()
            # Get data loaders for the current task and fold
            if args.joint_training:
                task_name = args.joint_task_name
                train_loader, val_loader = dataset.get_joint_loaders(fold=fold)
                # use dataset 0 as placeholder, will update for each data in this joint case
                times_coordinate = np.append(0, dataset.datasets[args.task_name_order[0]].bins[1:])

            else:
                task_name = task_name_order[t]
                train_loader, val_loader = dataset.get_data_loaders(task_name=task_name, fold=fold)
                times_coordinate = np.append(0, dataset.datasets[task_name].bins[1:])

            logging.info(f"load_to_eval task {t} {task_name}")

            start_idx, end_idx = dataset.get_offsets(t)

            # Set model to training mode
            model.net.train()

            writer, loss_fn, reg_fn, early_stopping, monitor_cindex = train_single_task_setup(model, train_loader, val_loader, args, task_name, fold)

            # set kwargs
            kwargs = {
                'task_name': task_name,
                'task_label': t,
                'writer': writer,
                'loss_fn': loss_fn,
                'reg_fn': reg_fn,
                'args': args,
                'start_idx': start_idx,
                'end_idx': end_idx,
                'early_stopping': early_stopping,
                'monitor_cindex': monitor_cindex,
                'times_coordinate': times_coordinate,
                # 'dataset': dataset
            }

            if can_compute_fwd_beforetask and is_fwd_enabled and args.enable_other_metrics:
                # try to compute accuracy at the beginning of the task
                try:
                    logging.info("Evaluating model before task (for Forward Transfer metric)...")
                    kwargs['epoch'] = 0
                    random_res_class, random_res_task, val_latest_dict_random, metric_dict_tuple_random = dataset.evaluate(model_random, dataset, last=True, **kwargs)  # the ugliness of this line is for backward compatibility
                    random_results_class.append(random_res_class)
                    random_results_task.append(random_res_task)

                    # Other metrics
                    c_index_random, c_index_ipcw_random = get_last_value_from_metric_dict_tuple(metric_dict_tuple_random)
                    random_results_class_c_index_ipcw.append([c_index_ipcw_random])
                    random_results_task_c_index_ipcw.append([c_index_ipcw_random])

                except Exception as e:
                    logging.info(f"Could not evaluate before `begin_task`, will try after")
                    # Will try after the begin_task in case the model needs to setup something
                    can_compute_fwd_beforetask = False

            if not can_compute_fwd_beforetask and is_fwd_enabled and args.enable_other_metrics:
                if train_loader.dataset.num_times_iterated == 0:  # compute only if the model has not been trained yet
                    try:
                        logging.info("Evaluating model before task (for Forward Transfer metric)...")
                        random_res_class, random_res_task, *_ = dataset.evaluate(model, dataset, last=True, **kwargs)
                        random_results_class.append(random_res_class)
                        random_results_task.append(random_res_task)
                    except Exception as e:
                        logging.error(f"Model `{model.NAME}` does not support pre-evaluation, will not compute Forward Transfer metric\n{e}")
                        # If the model does not support pre-evaluation, disable forward transfer computation
                        is_fwd_enabled = False
                else:
                    logging.info("Model used the training data, skipping Forward Transfer metric compute")
                    # If the model has already used training data, disable forward transfer computation
                    is_fwd_enabled = False

            epoch = 0
            kwargs['epoch'] = epoch

            if t and args.enable_other_metrics:
                logging.info("Evaluating the current task with model trained on the previous task (before training the current task) (for Forward Transfer metric)...")
                accs, accs_mask_classes, val_latest_dict_next, metric_dict_tuple_next = eval_dataset.evaluate(model, eval_dataset, last=True, **kwargs)
                accs = accs, accs_mask_classes
                results[t - 1] = results[t - 1] + accs[0]

                # Other metrics
                c_index_next, c_index_ipcw_next = get_last_value_from_metric_dict_tuple(metric_dict_tuple_next)
                results_c_index_ipcw[t - 1] = results_c_index_ipcw[t - 1] + [c_index_ipcw_next]

                if dataset.SETTING == 'class-il':
                    results_mask_classes[t - 1] = results_mask_classes[t - 1] + accs[1]

            # Evaluate the performance of the model
            # If checkpoint loading is specified, recover model state from checkpoint
            if args.loadcheck_base_name is not None:
                args.loadcheck = f'{args.loadcheck_base_name}_{t}.pt'
                model, past_res = mammoth_load_checkpoint(args, model)

                # if not args.disable_log and past_res is not None:
                #     (results, results_mask_classes, csvdump) = past_res
                #     logger.load(csvdump)

                print(f'Checkpoint Loaded! Path: {args.loadcheck}')
                if 'ms_moe' in args.model:
                    model.net.classifier.expert_usage_counts = torch.zeros(dataset.N_TASKS, args.num_experts, dtype=torch.int32, device=model.device)
                    model.net.classifier.expert_weight_sums = torch.zeros(dataset.N_TASKS, args.num_experts, dtype=torch.float32, device=model.device)
                    model.net.MoME_patho2.moe.expert_usage_counts = torch.zeros(dataset.N_TASKS, args.num_experts, dtype=torch.int32, device=model.device)
                    model.net.MoME_patho2.moe.expert_weight_sums = torch.zeros(dataset.N_TASKS, args.num_experts, dtype=torch.float32, device=model.device)
                    model.net.MoME_genom2.moe.expert_usage_counts = torch.zeros(dataset.N_TASKS, args.num_experts, dtype=torch.int32, device=model.device)
                    model.net.MoME_genom2.moe.expert_weight_sums = torch.zeros(dataset.N_TASKS, args.num_experts, dtype=torch.float32, device=model.device)

            accs_eval, accs_mask_classes_eval, val_latest_dict, metric_dict_tuple  = eval_dataset.evaluate(model, eval_dataset, **kwargs)
            c_index_val_dict, c_index_ipcw_val_dict = metric_dict_tuple
            # Compute the C-index for the current task
            c_index_val_current = accs_eval[-1]

            accs = accs_eval, accs_mask_classes_eval
            metric_dict_tuple = c_index_val_dict, c_index_ipcw_val_dict
            metric_matrix_tuple = c_index_val_matrix, c_index_ipcw_val_matrix
            metric_matrix_tuple = update_metric_matrix_tuple(task_name, metric_dict_tuple, metric_matrix_tuple)
            c_index_val_matrix, c_index_ipcw_val_matrix = metric_matrix_tuple

            logging.info(f"accs_eval: {accs_eval}")
            logging.info(f"accs_mask_classes_eval: {accs_mask_classes_eval}")
            # logging.info(f"val_latest_dict: {val_latest_dict}")
            logging.info(f"c_index_val_dict: {c_index_val_dict}")

            # If performance on future tasks needs to be evaluated
            if args.eval_future and t < dataset.N_TASKS - 1:
                transf_accs = accs[0][t + 1:], accs[1][t + 1:]
                accs = accs[0][:t + 1], accs[1][:t + 1]
                results_transf.append(transf_accs[0])
                results_mask_classes_transf.append(transf_accs[1])

            # Record evaluation results
            logged_accs = eval_dataset.log(args, logger, accs, t, dataset.SETTING)

            # Save results according to different settings
            if dataset.SETTING != 'biased-class-il':
                results.append(accs[0])
                results_mask_classes.append(accs[1])

                # Other metrics
                results_c_index_ipcw.append(list(c_index_ipcw_val_dict.values()))
            else:
                results.append(logged_accs[0])  # avg
                results_mask_classes.append(logged_accs[1])  # worst

            # If evaluation on future tasks is required, print transfer metrics
            if args.eval_future:
                avg_transf = np.mean([np.mean(task_) for task_ in results_transf])
                print(f"Transfer Metrics  -  AVG Transfer {avg_transf:.2f}")
                if t < dataset.N_TASKS - 1:
                    eval_dataset.log(args, logger, transf_accs, t, dataset.SETTING, future=True)

            # Clean up memory after each task evaluation
            gc.collect()
            torch.cuda.empty_cache()

        # If in validation mode, perform final evaluation
        if args.validation:
            # Final evaluation on the real test set
            print("Starting final evaluation on the real test set...", file=sys.stderr)
            del dataset
            args.validation = None
            args.validation_mode = 'current'

            # Get final evaluation dataset
            final_dataset = get_dataset(args)
            for _ in range(final_dataset.N_TASKS):
                final_dataset.get_data_loaders()
            accs, accs_mask_classes, *_ = final_dataset.evaluate(model, final_dataset, **kwargs)
            accs = accs, accs_mask_classes

            # Record final evaluation results
            final_dataset.log(args, logger, accs, 'final', final_dataset.SETTING, prefix="FINAL")

        # If other metrics are enabled, calculate and record them
        bwt, bwt_mask_class, bwt_c_index_ipcw, bwt_mask_class_c_index_ipcw = None, None, None, None
        forgetting, forgetting_mask_class, forgetting_c_index_ipcw, forgetting_mask_class_c_index_ipcw = None, None, None, None
        fwt, fwt_mask_class, fwt_c_index_ipcw, fwt_mask_class_c_index_ipcw = None, None, None, None
        if args.enable_other_metrics:
            try:
                # Calculate and record backward transfer
                bwt, bwt_mask_class = logger.add_bwt(results, results_mask_classes)
                # bwt_mask_class not used, copy for compatibility
                bwt_mask_class = bwt
                log_extra_metrics(args, bwt, bwt_mask_class, 'Backward Transfer', t)
                # Calculate and record forgetting
                forgetting, forgetting_mask_class = logger.add_forgetting(results, results_mask_classes)
                # forgetting_mask_class not used, copy for compatibility
                forgetting_mask_class = forgetting
                log_extra_metrics(args, forgetting, forgetting_mask_class, 'Forgetting', t)
                # If forward transfer is enabled, calculate and record it
                if is_fwd_enabled:
                    fwt, fwt_mask_class = logger.add_fwt(results, random_results_class,
                                                        results_mask_classes, random_results_task)
                    # fwt_mask_class not used, copy for compatibility
                    fwt_mask_class = fwt
                    log_extra_metrics(args, fwt, fwt_mask_class, 'Forward Transfer', t)
                else:
                    logging.warning("Forward Transfer metric incompatible with the current model, skipped.")

                # Other metrics
                # bwt
                bwt_c_index_ipcw, bwt_mask_class_c_index_ipcw = logger.add_bwt(results_c_index_ipcw, results_mask_classes_c_index_ipcw)
                # forgetting
                forgetting_c_index_ipcw, forgetting_mask_class_c_index_ipcw = logger.add_forgetting(results_c_index_ipcw, results_mask_classes_c_index_ipcw)
                # fwt
                if is_fwd_enabled:
                    fwt_c_index_ipcw, fwt_mask_class_c_index_ipcw = logger.add_fwt(results_c_index_ipcw, random_results_class_c_index_ipcw, results_mask_classes_c_index_ipcw, random_results_task_c_index_ipcw)

            except Exception as e:
                logging.error(f"Error in calculating metrics: {e}")

    if not args.disable_log:
        logger.write(vars(args))
        # Step 3: Create a DataFrame from the C-index matrix
        pd_index = list(c_index_val_matrix.keys())
        pd_columns = list(c_index_val_matrix[pd_index[-1]].keys())
        c_index_df = pd.DataFrame(results, index=pd_index, columns=pd_columns)
        c_index_df = df_add_columns(c_index_df, forgetting, bwt, fwt, random_results_class)

        # Other metrics
        # add columns
        c_index_ipcw_df = pd.DataFrame(results_c_index_ipcw, index=pd_index, columns=pd_columns)
        c_index_ipcw_df = df_add_columns(c_index_ipcw_df, forgetting_c_index_ipcw, bwt_c_index_ipcw, fwt_c_index_ipcw, random_results_class_c_index_ipcw)

        # Step 4: Save the DataFrame as a CSV file
        eval_str = "eval_" + args.loadcheck_base_name.split("/")[-1]
        file_name = f'c_index_val_matrix_fold_{args.fold}_{eval_str}.csv'
        c_index_results_csv_path = logger.get_target_path(file_name)
        print(f"Results saved to {c_index_results_csv_path}")

        # Concatenate all DataFrames, with a blank line in between
        dfs = [c_index_df, c_index_ipcw_df]
        with open(c_index_results_csv_path, 'w') as f:
            for i, df in enumerate(dfs):
                df.to_csv(f, index=True, header=True)
                if i < len(dfs) - 1:  # Write a blank line if not the last DataFrame
                    f.write('\n')
