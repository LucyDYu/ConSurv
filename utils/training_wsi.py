# Copyright 2022-present, Lorenzo Bonicelli, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import gc
import traceback
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
from utils.wsi_metrics import df_add_column, df_add_columns, get_last_value_from_metric_dict_tuple, matrix_remove_keys, replace_none_with_zero, update_metric_matrix_tuple
# from pytorch_memlab import MemReporter
# from pytorch_memlab import LineProfiler
from copy import deepcopy

try:
    import wandb
except ImportError:
    wandb = None

# Initialize LineProfiler
# profiler = LineProfiler()

def initialize_wandb(args: Namespace) -> None:
    """
    Initializes wandb, if installed.

    Args:
        args: the arguments of the current execution
    """
    assert wandb is not None, "Wandb not installed, please install it or run without wandb"

    name = get_run_name(args)
    mode = 'disabled' if os.getenv('MAMMOTH_TEST', '0') == '1' else os.getenv('WANDB_MODE', 'online')
    wandb.init(project=args.wandb_project, entity=args.wandb_entity, config=vars(args), name=name, mode=mode)
    args.wandb_url = wandb.run.get_url()

def get_run_name(args: Namespace) -> str:
    run_name = args.wandb_name if args.wandb_name is not None else args.model
    run_id = args.conf_jobnum.split('-')[0]
    buffer_str = '_buffer_'+ str(args.buffer_size) if 'buffer_size' in args else ''
    joint_str = '_joint' if args.joint_training else ''
    return f'{run_name}_fold_{args.fold}{buffer_str}{joint_str}_{run_id}'

def _to_device(name: str, x, device):
    if isinstance(x, torch.Tensor):
        if 'label' in name.lower() or 'target' in name.lower():
            return x.to(device, dtype=torch.long)
        return x.to(device)
    return x

def train_single_epoch(model: ContinualModel,
                       train_loader: Iterable,
                       dataset: ContinualDataset,
                    #    args: Namespace, # put in kwargs
                    #    epoch: int,
                       pbar: tqdm,
                       system_tracker=None,
                       scheduler=None,
                       **kwargs) -> int:
    """
    Trains the model for a single epoch.

    Args:
        model: the model to be trained
        train_loader: the data loader for the training set
        args: the arguments from the command line
        epoch: the current epoch
        system_tracker: the system tracker to monitor the system stats
        scheduler: the scheduler for the current epoch

    Returns:
        the number of iterations performed in the current epoch
    """
    model.train()
    train_loss_CL = 0.

    args = kwargs['args']
    train_loss_surv, train_loss, all_risk_scores, all_censorships, all_event_times = initialize_survival_metrics(len(train_loader), args)
    
    # Get training data iterator
    train_iter = iter(train_loader)
    # Get total epoch length (if available)
    epoch_len = len(train_loader) if hasattr(train_loader, "__len__") else None

    i = 0
    previous_time = time()

    batch_idx = 0
    epoch = kwargs['epoch']
    task_label = kwargs['task_label']
    task_name = kwargs['task_name']

    writer = kwargs['writer']

    # logging.info(f"Training iteration on batch level")
        
    while True:
        try:
            # Get next batch of data
            data = next(train_iter)
        except StopIteration:
            break
        # Exit if in debug mode and exceeded debug iterations
        if args.debug_mode and i > model.get_debug_iters():
            break
        # Exit if training by iterations and target reached
        if args.fitting_mode == 'iters' and model.task_iteration >= model.args.n_iters:
            break

        # logging.info(f"Training task {task_name}, epoch {epoch}, batch {batch_idx}")
        
        # # Placeholder
        empty_data = torch.zeros((1,1))
        empty_not_aug_data = torch.zeros((1,1))
        label = data[7]
        # Process extra data fields
        default_tuple_size = args.data_tuple_size
        # extra_fields = {}
        # if len(data) > default_tuple_size:
        #     if hasattr(train_loader.dataset.dataset, 'logits'):
        #         extra_fields['logits']= _to_device('logits', data[default_tuple_size], model.device)
        extra_fields = {
            train_loader.dataset.extra_return_fields[k]: _to_device(train_loader.dataset.extra_return_fields[k], data[default_tuple_size + k], model.device)
            for k in range(len(data) - default_tuple_size)
        }
        # hazards, S, Y_hat, A  = model(x_path=wsi_mb, x_omic1=data_omic1, x_omic2=data_omic2, x_omic3=data_omic3, x_omic4=data_omic4, x_omic5=data_omic5, x_omic6=data_omic6)

        
        # add more kwargs
        # data
        kwargs['data'] = data
        kwargs['task_label'] = task_label
        data_task_label = extra_fields['data_task_label']
        start_idx, end_idx = dataset.get_offsets(data_task_label)
        kwargs['start_idx'] = start_idx
        kwargs['end_idx'] = end_idx 

        # update information based on data for joint training
        if args.joint_training:
            times_coordinate = np.append(0, dataset.datasets[args.task_name_order[data_task_label.item()]].bins[1:])
            kwargs['times_coordinate'] = times_coordinate
        # statistics
        kwargs['batch_idx'] = batch_idx
        kwargs['all_risk_scores'] = all_risk_scores
        kwargs['all_censorships'] = all_censorships
        kwargs['all_event_times'] = all_event_times
        kwargs['train_loss_surv'] = train_loss_surv
        kwargs['train_loss'] = train_loss
        
        # Merge dictionaries
        if len(extra_fields) > 0:
            kwargs.update(extra_fields)

        # Execute meta-observe step to get loss and other metrics
        # observe: This method is called at each training iteration and is used to update the model parameters according to the current training batch. 
        
        ret_dict = model.meta_observe(empty_data, label, empty_not_aug_data, **kwargs)

        batch_idx += 1
        loss = ret_dict['loss']
        train_loss_CL += loss.item()

        # Ensure loss is not NaN
        assert not math.isnan(loss)

        # Update statistics / accumulators
        train_loss_surv = ret_dict['train_loss_surv']
        train_loss = ret_dict['train_loss']
        all_risk_scores = ret_dict['all_risk_scores']
        all_censorships = ret_dict['all_censorships']
        all_event_times = ret_dict['all_event_times']
        # Update learning rate if using iteration-level scheduling
        if scheduler is not None and args.scheduler_mode == 'iter':
            scheduler.step()

        # Synchronize device in CUDA mode
        if args.code_optimization == 0 and 'cuda' in str(args.device):
            torch.cuda.synchronize()
        system_tracker()
        i += 1

        # Calculate time difference and update progress bar
        time_diff = time() - previous_time
        previous_time = time()
        bar_log = {'loss': loss, 'lr': model.opt.param_groups[0]['lr']}
        # bar_log.update({k: v for k, v in ret_dict.items() if k != 'loss'})  # display other metrics
        if epoch_len:
            ep_h = 3600 / (epoch_len * time_diff)
            bar_log['ep/h'] = ep_h
        pbar.set_postfix(bar_log, refresh=False)
        pbar.update()



    # Update learning rate if using epoch-level scheduling
    if scheduler is not None and args.scheduler_mode == 'epoch':
        scheduler.step()

    # calculate loss and error for epoch
    train_loss_surv /= len(train_loader)
    train_loss /= len(train_loader)
    train_loss_CL /= len(train_loader) # After gradient accumulation, this is loss/gc then averaged
    
    c_index_train = concordance_index_censored((1-all_censorships).astype(bool), all_event_times, all_risk_scores, tied_tol=1e-08)[0]
    train_epoch_str = 'Epoch: {}, train_loss_surv: {:.4f}, train_loss: {:.4f}, train_c_index: {:.4f}'.format(
        epoch, train_loss_surv, train_loss, c_index_train)
    print(train_epoch_str)
    with open(os.path.join(args.writer_dir, 'log.txt'), 'a') as f:
        f.write(train_epoch_str+'\n')
    f.close()
    if wandb.run:
        wandb.log({
            f"train_c_index_{task_label}": c_index_train,
            f"train_loss_surv_{task_label}": train_loss_surv, 
            f"train_loss_{task_label}": train_loss,
            f"train_loss_CL_{task_label}": train_loss_CL, # CL methods may alter loss, so record separately
            f"train_loss_CL": train_loss_CL, # Track loss changes throughout the CL process
        })

    if writer:
        writer.add_scalar('train/loss_surv', train_loss_surv, epoch)
        writer.add_scalar('train/loss', train_loss, epoch)
        writer.add_scalar('train/c_index', c_index_train, epoch)

    # Print line-by-line memory profiling results
    # profiler.print_stats()

def train(model: ContinualModel, dataset: ContinualDataset,
          args: Namespace) -> None:
    """
    The training process, including evaluations and loggers.

    Args:
        model: the module to be trained
        dataset: the continual dataset at hand
        args: the arguments of the current execution
    """
    logging.info(f"Training iteration on task level")

    # dataset should be seq-survival
    assert dataset.NAME == 'seq-survival'

    fold = args.fold
    task_name_order = dataset.task_name_order



    # Initialize forward transfer evaluation flags
    is_fwd_enabled = True
    can_compute_fwd_beforetask = True
    random_results_class, random_results_task = [], []
    random_results_class_c_index_ipcw, random_results_task_c_index_ipcw = [], []

    # Initialize wandb if enabled
    if not args.nowand:
        initialize_wandb(args)

    # Create logger instance if logging is enabled
    if not args.disable_log:
        logger = Logger(args, dataset.SETTING, dataset.NAME, model.NAME)

    # Move model to device and clear GPU cache
    model.net.to(model.device)
    torch.cuda.empty_cache()

    # Copy the model for random baseline
    model_random = copy.deepcopy(model)

    # Use system resource tracker
    with track_system_stats(logger) as system_tracker:
        # Initialize result storage lists
        results, results_mask_classes = [], []
        results_c_index_ipcw, results_mask_classes_c_index_ipcw = [], []

        # Initialize future task result lists if needed
        if args.eval_future:
            results_transf, results_mask_classes_transf = [], []

        # Skip previous tasks if start_from is specified
        if args.start_from is not None:
            for i in range(args.start_from):
                task_name = task_name_order[i]
                train_loader, _ = dataset.get_data_loaders(task_name=task_name, fold=fold)
                model.meta_begin_task(dataset)
                model.meta_end_task(dataset)

        # Restore model state from checkpoint if specified
        if args.loadcheck is not None:
            model, past_res = mammoth_load_checkpoint(args, model)

            if not args.disable_log and past_res is not None:
                (results, results_mask_classes, csvdump) = past_res
                results = replace_none_with_zero(results)
                results_mask_classes = replace_none_with_zero(results_mask_classes)
                logger.load(csvdump)
                # copy fpr placeholder, will use load_to_eval to get results
                results_c_index_ipcw, results_c_index_ipcw_mask_classes = deepcopy(results), deepcopy(results_mask_classes)

            print('Checkpoint Loaded!')

        print(file=sys.stderr)
        # Determine start and end task indices
        start_task = 0 if args.start_from is None else args.start_from
        end_task = dataset.N_TASKS if args.stop_after is None else args.stop_after

        # Prepare evaluation dataset if future task evaluation is needed
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
        
        # Initialize a dictionary to store completed epochs per task
        completed_epochs = {}
        c_index_val_matrix = {}
        c_index_ipcw_val_matrix = {}
        # Check memory distribution after initialization
        # reporter = MemReporter(model)
        # reporter.report()
        if torch.cuda.is_available():
            pass
            # print("torch.cuda.memory_summary()")
            # print(torch.cuda.memory_summary())
        # Start main training loop over tasks
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

            logging.info(f"Training task {t} {task_name}")
            if wandb.run:
                wandb.log({
                    'task_name': task_name,
                    'task_label': t,
                })
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

            # Verify dataset type
            if not issubclass(dataset.__class__, GCLDataset):
                assert issubclass(train_loader.dataset.__class__, MammothDatasetWrapper), "Dataset must be an instance of MammothDatasetWrapper (did you forget to call the `store_masked_loaders`?)"

            if can_compute_fwd_beforetask and is_fwd_enabled and args.enable_other_metrics:
                # try to compute accuracy at the beginning of the task
                try:
                    logging.info("Evaluating model before task (for Forward Transfer metric)...")
                    kwargs['epoch'] = 0
                    random_res_class, random_res_task, val_latest_dict_random, metric_dict_tuple_random = dataset.evaluate(model_random, dataset, last=True, **kwargs)  # the ugliness of this line is for backward compatibility
                    # if last task, delete the model_random
                    if t == end_task - 1:
                        del model_random
                        gc.collect()
                        torch.cuda.empty_cache()

                    random_results_class.append(random_res_class)
                    random_results_task.append(random_res_task)

                    # Other metrics
                    c_index_random, c_index_ipcw_random = get_last_value_from_metric_dict_tuple(metric_dict_tuple_random)
                    random_results_class_c_index_ipcw.append([c_index_ipcw_random])
                    random_results_task_c_index_ipcw.append([c_index_ipcw_random])

                except Exception as e:
                    logging.info(f"Could not evaluate before `begin_task`, will try after.")
                    logging.info(f"Error: {e}. Stack trace: {traceback.format_exc()}")
                    # will try after the begin_task in case the model needs to setup something
                    can_compute_fwd_beforetask = False

            model.meta_begin_task(dataset, **kwargs)   

            if not can_compute_fwd_beforetask and is_fwd_enabled and args.enable_other_metrics:
                if train_loader.dataset.num_times_iterated == 0:  # compute only if the model has not been trained yet
                    try:
                        logging.info("Evaluating model before task (for Forward Transfer metric)...")
                        random_res_class, random_res_task, *_ = dataset.evaluate(model, dataset, last=True, **kwargs)
                        random_results_class.append(random_res_class)
                        random_results_task.append(random_res_task)
                    except Exception as e:
                        logging.error(f"Model `{model.NAME}` does not support pre-evaluation, will not compute Forward Transfer metric\n{e}")
                        # Disable forward transfer if model doesn't support pre-evaluation
                        is_fwd_enabled = False
                else:
                    logging.info("Model used the training data, skipping Forward Transfer metric compute")
                    # Disable forward transfer if model has already used training data
                    is_fwd_enabled = False

            if not args.inference_only and args.n_epochs > 0:
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

                # Scheduler is automatically reloaded after each task if defined in the dataset.
                # If the model defines it, it becomes the job of the model to reload it.
                scheduler = get_scheduler(model, args, reload_optim=True) if not hasattr(model, 'scheduler') else model.scheduler

                # Initialize training variables
                best_ea_metric = None  # Best early stopping metric
                best_ea_model = None   # Best model state
                cur_stopping_patience = args.early_stopping_patience  # Current early stopping patience

                max_c_index = 0.
                epoch_max_c_index = 0
                max_eval_output = None
                best_val_model = None

                # Calculate total iterations
                n_iterations = None
                if not isinstance(dataset, GCLDataset):
                    n_iterations = model.args.n_epochs * len(train_loader) if model.args.fitting_mode == 'epochs' else model.args.n_iters
                
                # Set progress bar update interval
                mininterval = 0.2 if n_iterations is not None and n_iterations > 1000 else 0.1
                train_pbar = tqdm(train_loader, total=n_iterations,  # train_loader is actually ignored, will update the progress bar manually
                                  disable=args.non_verbose, mininterval=mininterval)
                
                # Print at least the task number in non-verbose mode
                if args.non_verbose:
                    logging.info(f"Task {t + 1}")  # at least print the task number

                logging.info(f"Training iteration on epoch level")
                

                while True:
                    logging.info(f"Training task {t} {task_name}, epoch {epoch}")
                    if wandb.run:
                        wandb.log({
                            'epoch': epoch,
                            'task_epoch': f'{task_name}_{epoch}'
                        })
                    # Begin new epoch
                    model.begin_epoch(epoch, dataset)

                    # Update progress bar description
                    train_pbar.set_description(f"Task {t } - Epoch {epoch }")
                    kwargs['task_label'] = t
                    kwargs['epoch'] = epoch

                    # profiler.enable()
        
                    # Train a single epoch
                    train_single_epoch(model, train_loader, dataset, pbar=train_pbar, 
                                       system_tracker=system_tracker, scheduler=scheduler, **kwargs)
                    # profiler.disable()
                    # profiler.display()
                    # End current epoch
                    model.end_epoch(epoch, dataset)

                    epoch += 1
                    # Determine whether to end training based on fitting mode
                    if args.fitting_mode == 'epochs' and epoch >= model.args.n_epochs:
                        completed_epochs[t] = {'task_name': task_name, 'epochs': epoch}
                        break
                    elif args.fitting_mode == 'iters' and model.task_iteration >= model.args.n_iters:
                        completed_epochs[t] = {'task_name': task_name, 'epochs': epoch}
                        break
                    elif args.fitting_mode == 'early_stopping' and epoch % args.early_stopping_freq == 0 and epoch > 0:
                        # Evaluate current model performance
                        epoch_accs, _, epoch_loss, *_ = eval_dataset.evaluate(model, eval_dataset, return_loss=True, last=True, **kwargs)

                        # Calculate performance based on specified early stopping metric
                        if args.early_stopping_metric == 'accuracy':
                            ea_metric = np.mean(epoch_accs)  # Higher accuracy is better
                        elif args.early_stopping_metric == 'loss':
                            ea_metric = -epoch_loss  # Lower loss is better
                        else:
                            raise ValueError(f'Unknown early stopping metric {args.early_stopping_metric}')

                        # Higher accuracy is better
                        if best_ea_metric is not None and ea_metric - best_ea_metric < args.early_stopping_epsilon:
                            # Decrease patience
                            cur_stopping_patience -= args.early_stopping_freq
                            if cur_stopping_patience <= 0:
                                # Load best model and stop training if patience exhausted
                                print(f"\nEarly stopping at epoch {epoch} with metric {abs(ea_metric)}", file=sys.stderr)
                                model.load_state_dict({k: v.to(model.device) for k, v in best_ea_model.items()})
                                completed_epochs[t] = {'task_name': task_name, 'epochs': epoch}
                                break
                            print(f"\nNo improvement at epoch {epoch} (best {abs(best_ea_metric)} | current {abs(ea_metric)}). "
                                  f"Waiting for {cur_stopping_patience} epochs to stop.", file=sys.stderr)
                        else:
                            # Update best model if a better one is found
                            print(f"\nFound better model with metric {abs(ea_metric)} at epoch {epoch}. "
                                  f"Previous value was {abs(best_ea_metric) if best_ea_metric is not None else 'None'}", file=sys.stderr)
                            best_ea_metric = ea_metric
                            best_ea_model = copy.deepcopy({k: v.cpu() for k, v in model.state_dict().items()})
                            cur_stopping_patience = args.early_stopping_patience

                    # Periodically evaluate model performance
                    if args.eval_epochs is not None and (epoch > 0 or args.eval_epochs) and epoch % args.eval_epochs == 0 and epoch < model.args.n_epochs:
                        # Evaluate model performance

                        accs, accs_mask_classes, val_latest_dict, metric_dict_tuple = eval_dataset.evaluate(model, eval_dataset, **kwargs)
                        c_index_val_dict, c_index_ipcw_val_dict = metric_dict_tuple
                        epoch_accs = accs, accs_mask_classes
                        eval_dataset.log(args, logger, epoch_accs, t, dataset.SETTING, epoch=epoch)
                        
                        # Calculate current task c-index
                        c_index_val_current = accs[-1]
                        
                        # save checkpoint
                        if c_index_val_current > max_c_index:
                            logging.info(f"task_name {task_name}: Found better model with c-index {c_index_val_current} at epoch {kwargs['epoch']}. \n")
                            max_c_index = c_index_val_current
                            epoch_max_c_index = kwargs['epoch']
                            max_eval_output = accs, accs_mask_classes, val_latest_dict, metric_dict_tuple
                            best_val_model = copy.deepcopy({k: v.cpu() for k, v in model.state_dict().items()})

                # Close progress bar
                train_pbar.close()

            print(completed_epochs)
            kwargs['epoch'] = completed_epochs[t]['epochs']

            # End training for the current task
            model.meta_end_task(dataset, **kwargs)

            # Evaluate model performance
            
            accs_eval, accs_mask_classes_eval, val_latest_dict, metric_dict_tuple  = eval_dataset.evaluate(model, eval_dataset, **kwargs)
            c_index_val_dict, c_index_ipcw_val_dict = metric_dict_tuple
            # Calculate current task c-index
            c_index_val_current = accs_eval[-1]

            if c_index_val_current < max_c_index:
                logging.info(f"task_name {task_name}: Use best model at epoch {epoch_max_c_index} with c-index {max_c_index}, instead of current model at epoch {kwargs['epoch']} with c-index {c_index_val_current}")
                model.load_state_dict({k: v.to(model.device) for k, v in best_val_model.items()})
                accs_eval, accs_mask_classes_eval, val_latest_dict, metric_dict_tuple = max_eval_output
                c_index_val_dict, c_index_ipcw_val_dict = metric_dict_tuple

            accs = accs_eval, accs_mask_classes_eval
            metric_dict_tuple = c_index_val_dict, c_index_ipcw_val_dict
            metric_matrix_tuple = c_index_val_matrix, c_index_ipcw_val_matrix
            metric_matrix_tuple = update_metric_matrix_tuple(task_name, metric_dict_tuple, metric_matrix_tuple)
            c_index_val_matrix, c_index_ipcw_val_matrix = metric_matrix_tuple

            # Print training summary after all tasks
            print("\n===== Training Summary =====")
            for task_id, info_dict in completed_epochs.items():
                print(f"Task {task_id}: task name {info_dict['task_name']}, total epochs {info_dict['epochs']}.")
            print("===========================\n")


            logging.info(f"accs_eval: {accs_eval}")
            logging.info(f"accs_mask_classes_eval: {accs_mask_classes_eval}")
            # logging.info(f"val_latest_dict: {val_latest_dict}")
            logging.info(f"c_index_val_dict: {c_index_val_dict}")
            
            # Evaluate future task performance if needed
            if args.eval_future and t < dataset.N_TASKS - 1:
                transf_accs = accs[0][t + 1:], accs[1][t + 1:]
                accs = accs[0][:t + 1], accs[1][:t + 1]
                results_transf.append(transf_accs[0])
                results_mask_classes_transf.append(transf_accs[1])

            # Log evaluation results
            logged_accs = eval_dataset.log(args, logger, accs, t, dataset.SETTING)

            # Save results based on different settings
            if dataset.SETTING != 'biased-class-il':
                results.append(accs[0])
                results_mask_classes.append(accs[1])

                # Other metrics
                results_c_index_ipcw.append(list(c_index_ipcw_val_dict.values()))
            else:
                results.append(logged_accs[0])  # avg
                results_mask_classes.append(logged_accs[1])  # worst

            # Print transfer metrics if evaluating future tasks
            if args.eval_future:
                avg_transf = np.mean([np.mean(task_) for task_ in results_transf])
                print(f"Transfer Metrics  -  AVG Transfer {avg_transf:.2f}")
                if t < dataset.N_TASKS - 1:
                    eval_dataset.log(args, logger, transf_accs, t, dataset.SETTING, future=True)

            # Save checkpoint if enabled and not in debug mode
            if args.savecheck and not args.debug_mode:
                save_mammoth_checkpoint(t, end_task, args,
                                        model,
                                        results=[results, results_mask_classes, logger.dump()],
                                        optimizer_st=model.opt.state_dict() if hasattr(model, 'opt') else None,
                                        scheduler_st=scheduler.state_dict() if scheduler is not None else None)

            if torch.cuda.is_available():
                pass
                # print("torch.cuda.memory_summary()")
                # print(torch.cuda.memory_summary())

        # Perform final evaluation in validation mode
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

            # Log final evaluation results
            final_dataset.log(args, logger, accs, 'final', final_dataset.SETTING, prefix="FINAL")

        # Calculate and log other metrics if enabled
        bwt, bwt_mask_class, bwt_c_index_ipcw, bwt_mask_class_c_index_ipcw = None, None, None, None
        forgetting, forgetting_mask_class, forgetting_c_index_ipcw, forgetting_mask_class_c_index_ipcw = None, None, None, None
        fwt, fwt_mask_class, fwt_c_index_ipcw, fwt_mask_class_c_index_ipcw = None, None, None, None
        if args.enable_other_metrics:
            try:
                # Calculate and log backward transfer
                bwt, bwt_mask_class = logger.add_bwt(results, results_mask_classes)
                # bwt_mask_class not used, copy for compatibility
                bwt_mask_class = bwt
                log_extra_metrics(args, bwt, bwt_mask_class, 'Backward Transfer', t)
                # Calculate and log forgetting
                forgetting, forgetting_mask_class = logger.add_forgetting(results, results_mask_classes)
                # forgetting_mask_class not used, copy for compatibility
                forgetting_mask_class = forgetting
                log_extra_metrics(args, forgetting, forgetting_mask_class, 'Forgetting', t)
                # Calculate and log forward transfer if enabled
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

        # Print system resource usage statistics
        system_tracker.print_stats()

    if not args.disable_log:
        logger.write(vars(args))
        # Step 3: Create a DataFrame from the C-index matrix
        # pd_index = list(c_index_val_matrix.keys())
        # pd_columns = list(c_index_val_matrix[pd_index[-1]].keys())
        if args.dataset_config == 'joint':
            # val_tcga_joint
            pd_index = ["train_tcga_joint"]
            pd_columns = ["val_tcga_joint"]
        else:
            pd_index = [f"train_{task_name}" for task_name in dataset.task_name_order]
            pd_columns = [f"val_{task_name}" for task_name in dataset.task_name_order]
        c_index_df = pd.DataFrame(results, index=pd_index, columns=pd_columns)
        c_index_df = df_add_columns(c_index_df, forgetting, bwt, fwt, random_results_class)

        # Other metrics
        # add columns
        c_index_ipcw_df = pd.DataFrame(results_c_index_ipcw, index=pd_index, columns=pd_columns)
        c_index_ipcw_df = df_add_columns(c_index_ipcw_df, forgetting_c_index_ipcw, bwt_c_index_ipcw, fwt_c_index_ipcw, random_results_class_c_index_ipcw)

        # Step 4: Save the DataFrame as a CSV file
        file_name = f'c_index_val_matrix_fold_{args.fold}_{get_run_name(args)}.csv'
        c_index_results_csv_path = logger.get_target_path(file_name)
        print(f"Results saved to {c_index_results_csv_path}")

        # Concatenate all DataFrames with blank lines between them
        dfs = [c_index_df, c_index_ipcw_df]
        with open(c_index_results_csv_path, 'w') as f:
            for i, df in enumerate(dfs):
                df.to_csv(f, index=True, header=True)
                if i < len(dfs) - 1:  # Write blank line between DataFrames
                    f.write('\n')

        # log to wandb
        if not args.nowand:
            d = logger.dump()
            d['wandb_url'] = wandb.run.get_url()
            wandb.log(d)
            wandb.log({'c_index_val_matrix': c_index_df,
                      'c_index_ipcw_val_matrix': c_index_ipcw_df})
            sleep(10)

    if not args.nowand:
        # Wait for all async tasks to complete
        wandb.join()
        wandb.finish()
        sleep(10)

    # Print line-by-line memory profiling results
    # profiler.print_stats()
        


def train_single_task_setup(model, train_loader, val_loader, args, task_name, fold):
    # Appends to the results_dir path: 1) which splits were used for training (e.g. - 5foldcv), and then 2) the parameter code and 3) experiment code
    
    if not os.path.isdir(args.results_dir):
        os.makedirs(args.results_dir)
    print("logs saved at ", args.results_dir)
    
    args.writer_dir = os.path.join(args.results_dir, task_name, str(fold))
    if not os.path.isdir(args.writer_dir):
        os.makedirs(args.writer_dir, exist_ok=False)
    if args.log_data:   
        from tensorboardX import SummaryWriter
        writer = SummaryWriter(args.writer_dir, flush_secs=15)
    else:
        writer = None

    # print('\nInit train/val/test splits...', end=' ')
    # train_split, val_split = datasets
    # save_splits(datasets, ['train', 'val'], os.path.join(args.results_dir, 'splits_{}.csv'.format(cur)))
    # print('Done!')
    print("Training on {} samples".format(len(train_loader.dataset)))
    print("Validating on {} samples".format(len(val_loader.dataset)))

    print('\nInit loss function...', end=' ')
    if args.task_type == 'survival':
        if args.bag_loss == 'ce_surv':
            from utils.utils import CrossEntropySurvLoss
            loss_fn = CrossEntropySurvLoss(alpha=args.alpha_surv)
        elif args.bag_loss == 'nll_surv':
            from utils.utils import NLLSurvLoss
            loss_fn = NLLSurvLoss(alpha=args.alpha_surv)
        elif args.bag_loss == 'cox_surv':
            from utils.utils import CoxSurvLoss
            loss_fn = CoxSurvLoss()
        else:
            raise NotImplementedError
    else:
        raise NotImplementedError

    if args.reg_type == 'omic':
        from utils.utils import l1_reg_all
        reg_fn = l1_reg_all
    elif args.reg_type == 'pathomic':
        from utils.utils import l1_reg_modules
        reg_fn = l1_reg_modules
    else:
        reg_fn = None

    print('Done!')
    
    print('\nSetup EarlyStopping...', end=' ')
    if args.early_stopping:
        early_stopping = EarlyStopping(warmup=0, patience=10, stop_epoch=20, verbose = True)
    else:
        early_stopping = None
    
    print('\nSetup Validation C-Index Monitor...', end=' ')
    monitor_cindex = Monitor_CIndex()
    print('Done!')

    return writer, loss_fn, reg_fn, early_stopping, monitor_cindex