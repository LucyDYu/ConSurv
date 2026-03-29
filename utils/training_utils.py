# Copyright 2022-present, Lorenzo Bonicelli, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import random
import numpy as np
import copy
import math
import os
import sys
from argparse import Namespace
from time import sleep, time
from typing import Iterable, Tuple
import logging
import torch
from tqdm import tqdm



try:
    import wandb
except ImportError:
    wandb = None

def data_to_device(data_omic1, data_omic2, data_omic3, data_omic4, data_omic5, data_omic6, label, c, device='cuda', data_WSI=None, event_time=None, move_data_WSI=False):
    """
    Move multi-modal data and labels to specified device with type casting
    
    Args:
        data_omic1-6: Input omic data tensors
        label: Ground truth labels
        c: Censorship indicators
        device: Target device (e.g. 'cuda', 'cpu')
    
    Returns:
        Tuple: Data tensors moved to target device
    """
    data_omic1 = data_omic1.type(torch.FloatTensor).to(device)
    data_omic2 = data_omic2.type(torch.FloatTensor).to(device)
    data_omic3 = data_omic3.type(torch.FloatTensor).to(device)
    data_omic4 = data_omic4.type(torch.FloatTensor).to(device)
    data_omic5 = data_omic5.type(torch.FloatTensor).to(device)
    data_omic6 = data_omic6.type(torch.FloatTensor).to(device)
    label = label.type(torch.LongTensor).to(device)
    c = c.type(torch.FloatTensor).to(device)

    if data_WSI is not None or event_time is not None:
        if move_data_WSI and data_WSI is not None:
            data_WSI = data_WSI.to(device)
        return data_WSI, data_omic1, data_omic2, data_omic3, data_omic4, data_omic5, data_omic6, label, event_time, c
    else:
        return data_omic1, data_omic2, data_omic3, data_omic4, data_omic5, data_omic6, label, c

def full_data_to_device(data, device='cuda', move_data_WSI=False):
    if data is None:
        return None  
    data_WSI, data_omic1, data_omic2, data_omic3, data_omic4, data_omic5, data_omic6, label, event_time, c, *_ = data
    result = data_to_device(data_omic1, data_omic2, data_omic3, data_omic4, data_omic5, data_omic6, label, c, device, data_WSI, event_time, move_data_WSI)
    return (*result, *_)

def train_loop_survival_coattn_mb_batch(bs_micro, model, **kwargs):
    """
    Process a single batch of data using micro-batches and update statistics
    
    Args:
        epoch: current epoch number
        bs_micro: micro batch size
        model: the model to train
        data_WSI, data_omic1-6: input data
        label: ground truth labels
        event_time: event times
        c: censorship indicator
        batch_idx: current batch index
        all_risk_scores: array to store risk scores
        all_censorships: array to store censorship info
        all_event_times: array to store event times
        train_loss_surv: accumulator for survival loss
        train_loss: accumulator for total loss
        writer: writer for logging
        loss_fn: loss function
        reg_fn: regularization function
        lambda_reg: regularization strength
        gc: gradient accumulation steps
        args: additional arguments

    Returns:
        dict: Dictionary containing updated statistics
    """
    # unpack kwargs
    data = kwargs['data']
    
    batch_idx = kwargs['batch_idx']
    data_index = kwargs['indexes'][0].item()

    all_risk_scores = kwargs['all_risk_scores']
    all_censorships = kwargs['all_censorships']
    all_event_times = kwargs['all_event_times']
    train_loss_surv = kwargs['train_loss_surv']
    train_loss = kwargs['train_loss']
    # writer = kwargs['writer']
    loss_fn = kwargs['loss_fn']
    reg_fn = kwargs['reg_fn']
    args = kwargs['args']
    lambda_reg = args.lambda_reg
    gc = args.gc
    
    start_idx = kwargs['start_idx']
    end_idx = kwargs['end_idx']
    task_label = kwargs['task_label']
    device = args.device
    

    # Move data to GPU
    data_WSI, data_omic1, data_omic2, data_omic3, data_omic4, data_omic5, data_omic6, label, event_time, c, *_ = full_data_to_device(data, model.device)

    loss = 0.
    all_risk = 0.
    cnt = 0
    
    # Initialize lists to store outputs for each micro-batch
    model_outputs = []

    # Split data into micro-batches
    index_chunk_list = split_chunk_list(data_WSI, bs_micro, args.bs_micro_fix_shuffle)
    
    if 'no_grad_feature' in kwargs:
        no_grad_feature = kwargs['no_grad_feature']
    else:
        no_grad_feature = False
    # Process each micro-batch
    for tindex in index_chunk_list:
        # Select data for current micro-batch
        wsi_mb = torch.index_select(data_WSI, dim=0, index=torch.LongTensor(tindex).to(data_WSI.device)).to(device) 
        # Process micro-batches of different sizes with the same model
        # Decide whether to disable gradient computation based on no_grad parameter
        kwargs['x_path'] = wsi_mb
        kwargs['x_omic1'] = data_omic1
        kwargs['x_omic2'] = data_omic2
        kwargs['x_omic3'] = data_omic3
        kwargs['x_omic4'] = data_omic4
        kwargs['x_omic5'] = data_omic5
        kwargs['x_omic6'] = data_omic6
        kwargs['patch_index'] = tindex
        

        if no_grad_feature:
            with torch.no_grad():  # Disable gradient computation
                kwargs['returnt'] = 'features'
                feats, attention_scores = model(**kwargs)  # Extract features

            kwargs['returnt'] = 'out'
            kwargs['attention_scores'] = attention_scores
            output = model.forward_features(feats, **kwargs)
        else:
            output = model(**kwargs)

        # Store outputs from each micro-batch
        model_outputs.append(output)
        if 'returnt' in kwargs and kwargs['returnt'] == 'full':
            hazards, S, Y_hat, A, *_ = output['origin_output']
        else:
            hazards, S, Y_hat, A, *_ = output

        if args.bag_loss == 'nll_surv':
            loss_micro = loss_fn(hazards=hazards, S=S, Y=label, c=c)
        elif args.bag_loss == 'cox_surv':
            loss_micro = loss_fn(hazards=hazards.squeeze(), S=S, c=c)
        else:
            raise NotImplementedError
            
        loss += loss_micro
        all_risk += -torch.sum(S, dim=1).detach().cpu().numpy().item()
        cnt += 1

    # Average the loss and risk over micro-batches
    loss = loss / cnt
    loss_value = loss.item()

    # Calculate regularization loss if needed
    if reg_fn is None:
        loss_reg = 0
    else:
        loss_reg = reg_fn(model) * lambda_reg

    risk = all_risk / cnt # averaged risk
    
    # Update global statistics
    all_risk_scores[batch_idx] = risk
    all_censorships[batch_idx] = c.item()
    all_event_times[batch_idx] = event_time

    train_loss_surv += loss_value
    train_loss += loss_value + loss_reg

    if (batch_idx + 1) % 50 == 0:
        train_batch_str = 'batch {}, loss: {:.4f}, label: {}, event_time: {:.4f}, risk: {:.4f}'.format(
            batch_idx, loss_value, label.item(), float(event_time), float(risk))
        with open(os.path.join(args.writer_dir, 'log.txt'), 'a') as f:
            f.write(train_batch_str+'\n')
        f.close()
        # print(train_batch_str)
        
    loss = loss / gc + loss_reg

    return {
        'train_loss_surv': train_loss_surv,
        'train_loss': train_loss,
        'all_risk_scores': all_risk_scores,
        'all_censorships': all_censorships, 
        'all_event_times': all_event_times,
        'loss': loss,
        'model_outputs': model_outputs
    }

def validate_loop_survival_coattn_mb_batch(bs_micro, model, **kwargs):
    """
    Process a single validation batch using micro-batches and update statistics
    
    Args:
        epoch: current epoch number
        bs_micro: micro batch size
        model: the model to validate
        **kwargs: Dictionary containing:
            data_WSI, data_omic1-6: input data
            label: ground truth labels
            event_time: event times
            c: censorship indicator
            batch_idx: current batch index
            all_risk_scores: array to store risk scores
            all_censorships: array to store censorship info
            all_event_times: array to store event times
            val_loss_surv: accumulator for survival loss
            val_loss: accumulator for total loss
            loss_fn: loss function
            reg_fn: regularization function
            lambda_reg: regularization strength
            args: additional arguments
            
    Returns:
        dict: Dictionary containing:
            - val_loss_surv: updated survival loss
            - val_loss: updated total loss
            - all_risk_scores: updated risk scores
            - all_censorships: updated censorships
            - all_event_times: updated event times
            - patient_result: dictionary with patient-specific results
    """
    # Unpack kwargs
    data = kwargs['data']

    batch_idx = kwargs['batch_idx']
    data_index = kwargs['indexes'][0].item()
    
    all_risk_scores = kwargs['all_risk_scores']
    all_censorships = kwargs['all_censorships']
    all_event_times = kwargs['all_event_times']
    val_loss_surv = kwargs['val_loss_surv']
    val_loss = kwargs['val_loss']
    patient_results = kwargs['patient_results']

    early_stopping = kwargs['early_stopping']
    monitor_cindex = kwargs['monitor_cindex']
    # writer = kwargs['writer']
    loss_fn = kwargs['loss_fn']
    reg_fn = kwargs['reg_fn']
    args = kwargs['args']
    lambda_reg = args.lambda_reg
    results_dir = args.results_dir
    start_idx = kwargs['start_idx']
    end_idx = kwargs['end_idx']
    task_label = kwargs['task_label']
    device = args.device

    slide_ids = kwargs['slide_ids']
    # Get slide_ids for current batch
    slide_id = slide_ids.iloc[batch_idx]

    # Move data to GPU
    data_WSI, data_omic1, data_omic2, data_omic3, data_omic4, data_omic5, data_omic6, label, event_time, c, *_ = full_data_to_device(data, model.device)

    # Initialize batch accumulators
    loss = 0.
    all_risk = 0.
    cnt = 0

    # Initialize lists to store outputs for each micro-batch
    model_outputs = []

    # Disable gradient computation for validation
    with torch.no_grad():
        # Split WSI data into micro-batches
        index_chunk_list = split_chunk_list(data_WSI, bs_micro, args.bs_micro_fix_shuffle)
        for tindex in index_chunk_list:
            # Select WSI data for current micro-batch
            wsi_mb = torch.index_select(data_WSI, dim=0, index=torch.LongTensor(tindex).to(data_WSI.device)).to(device)
            kwargs['x_path'] = wsi_mb
            kwargs['x_omic1'] = data_omic1
            kwargs['x_omic2'] = data_omic2
            kwargs['x_omic3'] = data_omic3
            kwargs['x_omic4'] = data_omic4
            kwargs['x_omic5'] = data_omic5
            kwargs['x_omic6'] = data_omic6

            # Model forward pass
            output = model(**kwargs)
            
            # Store outputs from each micro-batch
            model_outputs.append(output)
            hazards, S, Y_hat, A, *_ = output
            
            # Calculate micro-batch loss
            loss_micro = loss_fn(hazards=hazards, S=S, Y=label, c=c, alpha=0)
            loss += loss_micro
            # Accumulate risk scores
            all_risk += -torch.sum(S, dim=1).detach().cpu().numpy().item()
            cnt+=1

    # Calculate average loss
    loss = loss / cnt
    loss_value = loss.item()
    # Calculate regularization loss
    if reg_fn is None:
        loss_reg = 0
    else:
        loss_reg = reg_fn(model) * lambda_reg

    # Calculate average risk score
    risk = all_risk / cnt

    # Update evaluation metrics
    all_risk_scores[batch_idx] = risk
    all_censorships[batch_idx] = c.cpu().numpy()
    all_event_times[batch_idx] = event_time

    # Update patient results dictionary
    patient_results.update({slide_id: {'slide_id': np.array(slide_id), 
                                       'risk': risk, 
                                       'disc_label': label.item(), 
                                       'survival': event_time.item(), 
                                       'censorship': c.item()}})
    # Accumulate loss
    val_loss_surv += loss_value
    val_loss += loss_value + loss_reg
    

    # Return updated statistics
    return {
        'val_loss_surv': val_loss_surv,
        'val_loss': val_loss,
        'all_risk_scores': all_risk_scores,
        'all_censorships': all_censorships,
        'all_event_times': all_event_times,
        'patient_results': patient_results,
        'model_outputs': model_outputs
    }

def split_chunk_list(data, batch_size, fixed_shuffle=False):
    # Save current random state
    random_state = random.getstate()

    numGroup = data.shape[0] // batch_size + 1
    feat_index = list(range(data.shape[0]))

    if fixed_shuffle:
        # Use hash of feat_index as random seed
        random.seed(hash(tuple(feat_index)))

    # Shuffle feat_index
    random.shuffle(feat_index)

    index_chunk_list = np.array_split(np.array(feat_index), numGroup)
    index_chunk_list = [sst.tolist() for sst in index_chunk_list]
    # Restore previous random state
    random.setstate(random_state)
    
    return index_chunk_list



def initialize_survival_metrics(loader_length, args):
    """
    Initialize arrays for storing survival analysis metrics
    
    Args:
        loader_length (int): Length of the data loader
        
    Returns:
        tuple: Contains:
            - loss_surv: Initial loss_surv
            - loss: Initial loss 
            - all_risk_scores (np.ndarray): Array for risk scores
            - all_censorships (np.ndarray): Array for censorship data
            - all_event_times (np.ndarray): Array for event times
    """
    loss_surv, loss = (0., 0.)  # train_loss_surv, train_loss,. or val_loss_surv, val_loss
    all_risk_scores = np.zeros((loader_length))
    all_censorships = np.zeros((loader_length))
    all_event_times = np.zeros((loader_length))

    return loss_surv, loss, all_risk_scores, all_censorships, all_event_times



def data_to_output(full_data, model, returnt='out', **kwargs):
    if len(full_data) == 1:
        # data is a list containing a single data sample
        full_data = full_data[0]
    # Default: data is already a single sample with 10 elements
    data_WSI, data_omic1, data_omic2, data_omic3, data_omic4, data_omic5, data_omic6, label, event_time, c, *_ = full_data_to_device(full_data, model.device)
    
    args = kwargs['args']
    device = args.device
    # Split data into micro-batches
    index_chunk_list = split_chunk_list(data_WSI, args.bs_micro, args.bs_micro_fix_shuffle)
    return_pair_list = []
    # Process each micro-batch
    for tindex in index_chunk_list:
        # Select data for current micro-batch
        wsi_mb = torch.index_select(data_WSI, dim=0, index=torch.LongTensor(tindex).to(data_WSI.device)).to(device)
        # Process micro-batches of different sizes with the same model
        kwargs['x_path'] = wsi_mb
        kwargs['x_omic1'] = data_omic1
        kwargs['x_omic2'] = data_omic2
        kwargs['x_omic3'] = data_omic3
        kwargs['x_omic4'] = data_omic4
        kwargs['x_omic5'] = data_omic5
        kwargs['x_omic6'] = data_omic6
        kwargs['returnt'] = returnt
        kwargs['patch_index'] = tindex

        output = model(**kwargs)
        return_pair_list.append((output, label))

    return return_pair_list

# micro-batch loss, outputs is a list, label is one number
def output_to_loss(outputs, full_data, model, loss_fn, bag_loss = 'nll_surv'):
    if len(full_data) == 1:
        # data is a list containing a single data sample
        full_data = full_data[0]
    # Default: data is already a single sample with 10 elements
    data_WSI, data_omic1, data_omic2, data_omic3, data_omic4, data_omic5, data_omic6, label, event_time, c, *_ = full_data_to_device(full_data, model.device)

    cnt = 0
    loss = 0.
    for output in outputs:
        if isinstance(output, dict):
            output = output['origin_output']
        hazards, S, Y_hat, A, *_ = output
        
        if bag_loss == 'nll_surv':
            loss_micro = loss_fn(hazards=hazards, S=S, Y=label, c=c)
        elif bag_loss == 'cox_surv':
            loss_micro = loss_fn(hazards=hazards.squeeze(), S=S, c=c)
        else:
            raise NotImplementedError
        loss += loss_micro
        cnt+=1
    loss = loss / cnt
    return loss

def wait_for_memory(min_memory_gb=1, max_retries=None, wait_interval=1):
    """
    Wait for GPU memory to be freed. Waits indefinitely by default; max retries can be set.
    
    Args:
        min_memory_gb: Minimum required memory (GB).
        max_retries: Maximum retry count. Default None means infinite waiting.
        wait_interval: Wait interval between retries (seconds).
    """
    min_memory_bytes = min_memory_gb * 1024 * 1024 * 1024  # Convert GB to bytes
    retries = 0
    if not torch.cuda.is_available():
        return True
    
    while max_retries is None or retries < max_retries:
        # Get current available memory
        free_memory, total_memory = torch.cuda.mem_get_info()
        
        # Check fragmentation
        allocated = torch.cuda.memory_allocated()
        reserved = torch.cuda.memory_reserved()
        fragmentation = reserved - allocated

        # if retries == 0:
            # print(f"GPU memory status: available {free_memory / 1024**2:.2f} MB, "
            #     f"allocated {allocated / 1024**2:.2f} MB, "
            #     f"reserved {reserved / 1024**2:.2f} MB, "
            #     f"fragmentation {fragmentation / 1024**2:.2f} MB")
        
        if free_memory >= min_memory_bytes:
            if retries >= 2:
                print(f"GPU memory sufficient, continuing: available {free_memory / 1024**2:.2f} MB, "
                      f"required {min_memory_bytes / 1024**2:.2f} MB, "
                      f"allocated {allocated / 1024**2:.2f} MB, "
                      f"reserved {reserved / 1024**2:.2f} MB, "
                      f"fragmentation {fragmentation / 1024**2:.2f} MB")
                print(f"Retries: {retries}, waited {wait_interval*retries} seconds")
            return True  # Memory sufficient, continue
        
        if retries >= 1:
            torch.cuda.empty_cache() 
        # print(f"GPU memory insufficient: available {free_memory / 1024**2:.2f} MB, "
        #       f"required {min_memory_bytes / 1024**2:.2f} MB, "
        #       f"fragmentation {fragmentation / 1024**2:.2f} MB")
        # print(f"Waiting {wait_interval} seconds... (retry: {retries + 1}/{max_retries})")
        
        # If fragmentation is severe, try more aggressive memory cleanup
        # if fragmentation > 0.5 * reserved:  # If fragmentation exceeds 50% of reserved memory
        #     print("Severe memory fragmentation detected.")
        
        sleep(wait_interval)
        retries += 1
    
    print(f"After {max_retries} retries, still insufficient memory. Continuing, but may OOM.")
    return False  # Memory insufficient, retries exhausted