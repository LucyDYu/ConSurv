import numpy as np
import torch

import os
from sksurv.metrics import concordance_index_censored
import random
from timeit import default_timer as timer

from utils.training_utils import initialize_survival_metrics, split_chunk_list, train_loop_survival_coattn_mb_batch, validate_loop_survival_coattn_mb_batch

# Limit the number of threads used by torch
torch.set_num_threads(2)

def train_loop_survival_coattn_mb(epoch, bs_micro, model, loader, optimizer, n_classes, writer=None, loss_fn=None, reg_fn=None, lambda_reg=0., gc=32, args=None):
    # Set model to training mode
    model.train()

    print('\n')
    # initialize survival metrics
    train_loss_surv, train_loss, all_risk_scores, all_censorships, all_event_times = initialize_survival_metrics(len(loader))

    for batch_idx, data in enumerate(loader):
        # Process batch and update statistics
        data_WSI, data_omic1, data_omic2, data_omic3, data_omic4, data_omic5, data_omic6, label, event_time, c, *_ = data
        # set kwargs
        kwargs = {
            # Data-related parameters
            'data_WSI': data_WSI,      # WSI image data
            'data_omic1': data_omic1,  # Omics data 1
            'data_omic2': data_omic2,  # Omics data 2
            'data_omic3': data_omic3,  # Omics data 3
            'data_omic4': data_omic4,  # Omics data 4
            'data_omic5': data_omic5,  # Omics data 5
            'data_omic6': data_omic6,  # Omics data 6
            'label': label,            # Discretized survival months category
            'event_time': event_time,  # survival_months
            'c': c,                    # censorship

            # Statistics-related parameters
            'batch_idx': batch_idx,          # Batch index
            'all_risk_scores': all_risk_scores,  # Risk scores
            'all_censorships': all_censorships,  # Censorship collection
            'all_event_times': all_event_times,  # Survival months collection
            'train_loss_surv': train_loss_surv,  # Survival loss
            'train_loss': train_loss,            # Total loss

            # Other parameters
            'writer': writer,          # TensorBoard writer
            'loss_fn': loss_fn,        # Loss function
            'reg_fn': reg_fn,          # Regularization function
            'lambda_reg': lambda_reg,  # Regularization coefficient
            'gc': gc,                  # Gradient accumulation steps
            'args': args,               # Other arguments
            'start_idx': 0,
            'end_idx': 4
        }

        # Execute single batch training
        ret_dict = train_loop_survival_coattn_mb_batch(
            bs_micro, model, **kwargs
        )

        # Update accumulators
        train_loss_surv = ret_dict['train_loss_surv']    # Update survival loss
        train_loss = ret_dict['train_loss']              # Update total loss
        all_risk_scores = ret_dict['all_risk_scores']    # Update risk scores
        all_censorships = ret_dict['all_censorships']    # Update censorship status
        all_event_times = ret_dict['all_event_times']    # Update survival months

        # Backward propagation
        loss = ret_dict['loss']
        loss.backward()
        
        # Update parameters every gc steps
        if (batch_idx + 1) % gc == 0:
            optimizer.step()
            optimizer.zero_grad()

    # Compute epoch-level loss and evaluation metrics
    train_loss_surv /= len(loader)  # Compute average survival loss
    train_loss /= len(loader)       # Compute average total loss
    # Compute concordance index
    c_index_train = concordance_index_censored((1-all_censorships).astype(bool), all_event_times, all_risk_scores, tied_tol=1e-08)[0]
    # Generate training log string
    train_epoch_str = 'Epoch: {}, train_loss_surv: {:.4f}, train_loss: {:.4f}, train_c_index: {:.4f}'.format(
        epoch, train_loss_surv, train_loss, c_index_train)
    print(train_epoch_str)
    # Save training log
    with open(os.path.join(args.writer_dir, 'log.txt'), 'a') as f:
        f.write(train_epoch_str+'\n')
    f.close()

    # Log training metrics to TensorBoard if available
    if writer:
        writer.add_scalar('train/loss_surv', train_loss_surv, epoch)
        writer.add_scalar('train/loss', train_loss, epoch)
        writer.add_scalar('train/c_index', c_index_train, epoch)

def validate_survival_coattn_mb(cur, epoch, bs_micro, model, loader, n_classes, early_stopping=None, monitor_cindex=None, writer=None, loss_fn=None, reg_fn=None, lambda_reg=0., results_dir=None, args=None):
    # Set model to evaluation mode
    model.eval()

    # initialize survival metrics
    val_loss_surv, val_loss, all_risk_scores, all_censorships, all_event_times = initialize_survival_metrics(len(loader))
    
    # Get slide_ids
    slide_ids = loader.dataset.slide_data['slide_id']
    # Initialize patient results dictionary
    patient_results = {}

    for batch_idx, data in enumerate(loader):
        
        # Process batch and update statistics
        data_WSI, data_omic1, data_omic2, data_omic3, data_omic4, data_omic5, data_omic6, label, event_time, c, *_ = data
        # set kwargs
        kwargs = {
            # Data-related parameters
            'data_WSI': data_WSI,      # WSI image data
            'data_omic1': data_omic1,  # Omics data 1
            'data_omic2': data_omic2,  # Omics data 2
            'data_omic3': data_omic3,  # Omics data 3
            'data_omic4': data_omic4,  # Omics data 4
            'data_omic5': data_omic5,  # Omics data 5
            'data_omic6': data_omic6,  # Omics data 6
            'label': label,            # Discretized survival months category
            'event_time': event_time,  # survival_months
            'c': c,                    # censorship
            'slide_ids': slide_ids,    # slide_ids

            # Statistics-related parameters
            'batch_idx': batch_idx,          # Batch index
            'all_risk_scores': all_risk_scores,  # Risk scores
            'all_censorships': all_censorships,  # Censorship collection
            'all_event_times': all_event_times,  # Survival months collection
            'val_loss_surv': val_loss_surv,  # Survival loss
            'val_loss': val_loss,            # Total loss
            'patient_results': patient_results,  # Patient results dictionary

            # Other parameters
            'early_stopping': early_stopping,  # Early stopping
            'monitor_cindex': monitor_cindex,  # Monitor c-index
            'writer': writer,          # TensorBoard writer
            'loss_fn': loss_fn,        # Loss function
            'reg_fn': reg_fn,          # Regularization function
            'lambda_reg': lambda_reg,  # Regularization coefficient
            'args': args,               # Other arguments

            'start_idx': 0,
            'end_idx': 4
        }

        # Execute single batch validation
        ret_dict = validate_loop_survival_coattn_mb_batch(
            bs_micro, model, **kwargs
        )

        # Update patient results
        val_loss_surv = ret_dict['val_loss_surv']
        val_loss = ret_dict['val_loss']
        all_risk_scores = ret_dict['all_risk_scores']
        all_censorships = ret_dict['all_censorships']
        all_event_times = ret_dict['all_event_times']
        patient_results = ret_dict['patient_results']


    # Compute average loss
    val_loss_surv /= len(loader)
    val_loss /= len(loader)
    # Compute concordance index
    c_index = concordance_index_censored((1-all_censorships).astype(bool), all_event_times, all_risk_scores, tied_tol=1e-08)[0]
    # Generate validation log string
    val_epoch_str = "val c-index: {:.4f}".format(c_index)
    # Save validation log
    with open(os.path.join(args.writer_dir, 'log.txt'), 'a') as f:
        f.write(val_epoch_str+'\n')
    print(val_epoch_str)
    
    # Log validation metrics to TensorBoard if available
    if writer:
        writer.add_scalar('val/loss_surv', val_loss_surv, epoch)
        writer.add_scalar('val/loss', val_loss, epoch)
        writer.add_scalar('val/c-index', c_index, epoch)

    # Early stopping check
    if early_stopping:
        assert results_dir
        early_stopping(epoch, val_loss_surv, model, ckpt_name=os.path.join(results_dir, "s_{}_minloss_checkpoint.pt".format(cur)))
        
        # Return results and terminate training if early stopping is triggered
        if early_stopping.early_stop:
            print("Early stopping")
            return patient_results, c_index, True

    # Return validation results
    return patient_results, c_index, False

