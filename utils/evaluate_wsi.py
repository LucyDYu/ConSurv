import gc
import logging
import os
from typing import TYPE_CHECKING, Any, Callable, Tuple
import numpy as np
import torch
from tqdm import tqdm

from utils.evaluate_wsi_joint import evaluate_wsi_joint
from utils.evaluate import mask_classes
from utils.training_utils import initialize_survival_metrics, validate_loop_survival_coattn_mb_batch
from sksurv.metrics import concordance_index_censored

from utils.wsi_metrics import _calculate_metrics, _extract_survival_metadata, update_metric_dict_tuple

# Import type hints for type checking
if TYPE_CHECKING:
    from models.utils.continual_model import ContinualModel
    from datasets.utils.continual_dataset import ContinualDataset

try:
    import wandb
except ImportError:
    wandb = None

@torch.no_grad()
def evaluate_wsi(model: 'ContinualModel', dataset: 'ContinualDataset', last=False, return_loss=False, **kwargs) -> Tuple[list, list]:
    """
    Evaluates the single-class top-1 accuracy of the model for each past task.

    The accuracy is evaluated for all the tasks up to the current one, only for the total number of classes seen so far.

    Args:
        model: the model to be evaluated
        dataset: the continual dataset at hand
        last: a boolean indicating whether to evaluate only the last task
        return_loss: a boolean indicating whether to return the loss in addition to the accuracy

    Returns:
        a tuple of lists, containing the class-il and task-il accuracy for each task. If return_loss is True, the loss is also returned as a third element.
    """
    args = kwargs['args']
    if args.joint_training:
        # Delegate to joint training evaluation
        return evaluate_wsi_joint(model, dataset, last, return_loss, **kwargs)

    logging.info(f"Validation iteration on task level")

    # dataset should be seq-survival
    assert dataset.NAME == 'seq-survival'
    # Print validation parameters
    writer = kwargs['writer']
    epoch = kwargs['epoch']
    early_stopping = kwargs['early_stopping']

    lambda_reg = args.lambda_reg
    results_dir = args.results_dir
    fold = args.fold

    # Save current training status and set model to evaluation mode
    status = model.net.training
    model.net.eval()
    
    # Initialize accuracy lists and other variables
    accs, accs_mask_classes = [], []
    n_classes = dataset.get_offsets()[1]
    # loss_fn = dataset.get_loss()
    avg_loss = 0
    tot_seen_samples = 0
    
    # Calculate total length of test data if possible
    total_len = sum(len(x) for x in dataset.test_loaders) if hasattr(dataset.test_loaders[0], '__len__') else None

    # Create progress bar for evaluation
    pbar = tqdm(dataset.test_loaders, total=total_len, desc='Evaluating', disable=model.args.non_verbose)
    
    val_latest_dict = {}
    c_index_val_dict = {}
    c_index_ipcw_val_dict = {}
    metric_dict_tuple = c_index_val_dict, c_index_ipcw_val_dict


    # Iterate through each test loader (representing different tasks)
    # for k, test_loader in enumerate(dataset.test_loaders):
    for k, (task_name, test_loader) in enumerate(dataset.test_loaders_dict.items()):

        # Skip if only evaluating last task
        if last and k < len(dataset.test_loaders) - 1:
            continue
            
        logging.info(f"Validation task {k} {task_name}")
        # empty_cache before validation of each task
        gc.collect() 
        torch.cuda.empty_cache()

        # Initialize metrics for current task
        correct, correct_mask_classes, total = 0.0, 0.0, 0.0
        test_iter = iter(test_loader)
        i = 0
        # initialize survival metrics
        val_loss_surv, val_loss, all_risk_scores, all_censorships, all_event_times = initialize_survival_metrics(len(test_loader), args)

        times_coordinate = np.append(0, dataset.datasets[task_name].bins[1:])
        kwargs['times_coordinate'] = times_coordinate
        logging.info(f"times_coordinate for {task_name}: {times_coordinate}")

        # Get slide_ids
        slide_ids = test_loader.dataset.slide_data['slide_id']
        # Initialize patient results dictionary
        patient_results = {}

        batch_idx = 0

        # logging.info(f"Validation iteration on batch level")
        # Process each batch in the current task
        while True:
            try:
                # Get next batch of data
                data = next(test_iter)
            except StopIteration:
                break
                
            # Break if in debug mode and exceeded debug iterations
            if model.args.debug_mode and i > model.get_debug_iters():
                break
                
            label = data[7]
            # Get model outputs based on compatibility mode
            # add more kwargs
            # data
            kwargs['data'] = data
            kwargs['slide_ids'] = slide_ids
            # statistics
            kwargs['batch_idx'] = batch_idx
            kwargs['indexes'] = data[10]
            data_task_label = data[11]
            kwargs['data_task_label'] = data_task_label
            start_idx, end_idx = dataset.get_offsets(data_task_label)
            kwargs['start_idx'] = start_idx
            kwargs['end_idx'] = end_idx
            kwargs['times_coordinate'] = times_coordinate
            
            kwargs['all_risk_scores'] = all_risk_scores
            kwargs['all_censorships'] = all_censorships
            kwargs['all_event_times'] = all_event_times
            kwargs['val_loss_surv'] = val_loss_surv
            kwargs['val_loss'] = val_loss
            kwargs['patient_results'] = patient_results

            args = kwargs['args']
            bs_micro = args.bs_micro
            # # Placeholder
            empty_data = torch.zeros((1,1))

            if 'class-il' not in model.COMPATIBILITY and 'general-continual' not in model.COMPATIBILITY:
                outputs = model(empty_data, k, **kwargs)
            else:
                if model.args.eval_future and k >= model.current_task:
                    outputs = model.future_forward(empty_data, **kwargs)
                else:
                    # outputs = model(empty_data, **kwargs)
                    # Execute single batch validation
                    ret_dict = validate_loop_survival_coattn_mb_batch(
                        bs_micro, model.net, **kwargs
                    )

            batch_idx += 1

            # Update patient results
            val_loss_surv = ret_dict['val_loss_surv']
            val_loss = ret_dict['val_loss']
            all_risk_scores = ret_dict['all_risk_scores']
            all_censorships = ret_dict['all_censorships']
            all_event_times = ret_dict['all_event_times']

            patient_results = ret_dict['patient_results']
            model_outputs = ret_dict['model_outputs']

            
            # Calculate loss if required
            if return_loss:
                # loss = loss_fn(outputs, labels)
                # avg_loss += loss.item()
                avg_loss += val_loss.item()

            # Calculate accuracy for current batch
            # Calculate evaluation metrics for current batch
            # outputs[:, :n_classes] selects prediction scores for the first n_classes classes
            # torch.max returns max value and index per row; _ ignores the max, keeping only the index pred
            # (hazards, S, Y_hat, A) = output
            Y_hat_all = torch.cat([output[2] for output in model_outputs], dim=0)

            # Get micro_batch_size
            micro_batch_size = Y_hat_all.shape[0] 
            labels = label.repeat(micro_batch_size)  # Use repeat to duplicate
            # move all tensors to the same device
            Y_hat_all = Y_hat_all.to(model.device)
            labels = labels.to(model.device)

            # _, pred = torch.max(outputs[:, :n_classes].data, 1)

            # Count correctly predicted samples:
            # pred == labels compares predictions with ground truth labels, returns boolean tensor
            # torch.sum counts the number of correct predictions
            # .item() converts tensor to Python scalar
            correct += torch.sum(Y_hat_all == labels).item()

            # Accumulate total sample count for current batch
            total += labels.shape[0]

            # Increment batch counter
            i += 1
            if i % 50 == 0:
                torch.cuda.empty_cache()
            # Update progress bar display
            # f'acc_task_{k+1}' shows accuracy of current task
            # correct / total * 100 calculates accuracy percentage
            # refresh=False avoids frequent refreshing
            # This is just for monitoring; c-index is the final metric
            # pbar.set_postfix({f'acc_task_{k+1}': max(0, correct / total * 100)}, refresh=False)

            # Set progress bar description showing current evaluation task
            pbar.set_description(f"Evaluating Task {k}", refresh=False)

            # Update progress bar
            pbar.update(1)

            # # Handle class-il specific evaluation
            # if dataset.SETTING == 'class-il':
            #     # Apply class mask to outputs, keeping only current task classes
            #     mask_classes(outputs, dataset, k)
            #     # Recalculate predictions after masking
            #     _, pred = torch.max(outputs.data, 1)
            #     # Accumulate correctly predicted samples after masking
            #     correct_mask_classes += torch.sum(pred == labels).item()

        # Update total samples seen
        tot_seen_samples += total

        # Calculate average loss
        val_loss_surv /= len(test_loader)
        val_loss /= len(test_loader)
        # Calculate concordance index
        c_index_old = concordance_index_censored((1-all_censorships).astype(bool), all_event_times, all_risk_scores, tied_tol=1e-08)[0]

        # Multi-metric evaluation, verify c_index consistency
        # only use columns event_time and censorship, no privacy issue
        train_loader = dataset.train_loaders_dict[task_name]
        val_loader = test_loader
        all_data_survival = _extract_survival_metadata(train_loader, val_loader)
        c_index, val_cindex_ipcw = _calculate_metrics(dataset, task_name, all_data_survival, all_risk_scores, all_censorships, all_event_times)
        assert c_index == c_index_old

        print(
            'Epoch:{} Val c-index: {:.4f} | Final Val c-index-ipcw: {:.4f}'.format(
                epoch,
                c_index,
                val_cindex_ipcw,
            ))

        # Generate validation log string
        val_epoch_str = "val c-index on task {}: {:.4f}".format(task_name, c_index)
        # Save validation log
        with open(os.path.join(args.writer_dir, 'log.txt'), 'a') as f:
            f.write(val_epoch_str+'\n')
        print(val_epoch_str)
        if wandb.run:
            task_label = k
            wandb.log({
                f"val_c_index_{task_label}": c_index,
                f"val_loss_surv_{task_label}": val_loss_surv, 
                f"val_loss_{task_label}": val_loss, # Track loss changes throughout the CL process
                f"val_c_index_ipcw_{task_label}": val_cindex_ipcw,
            })

        # Log validation metrics to tensorboard if available
        if writer:
            writer.add_scalar('val/loss_surv', val_loss_surv, epoch)
            writer.add_scalar('val/loss', val_loss, epoch)
            writer.add_scalar('val/c-index', c_index, epoch)

        # Early stopping check
        if early_stopping:
            assert results_dir
            early_stopping(epoch, val_loss_surv, model, ckpt_name=os.path.join(results_dir, "s_{}_minloss_checkpoint.pt".format(fold)))
            
            # If early stopping triggered, return results and terminate training
            if early_stopping.early_stop:
                print("Early stopping")
                # return patient_results, c_index, True

        # Validation results
        val_latest_dict[task_name] = patient_results
        c_index_val_dict[f'val_{task_name}'] = c_index
        metric_output_tuple = (c_index, val_cindex_ipcw)
        metric_dict_tuple = (c_index_val_dict, c_index_ipcw_val_dict)
        metric_dict_tuple = update_metric_dict_tuple(task_name, metric_output_tuple, metric_dict_tuple)

        # Calculate and store accuracies for current task.
        # Only accs is used. Here is task-il.
        accs.append(c_index)
        accs_mask_classes.append(c_index)
    
    # Close progress bar
    pbar.close()

    # Restore model's training status
    model.net.train(status)
    
    # Return results based on return_loss flag
    # if return_loss:
    #     return accs, accs_mask_classes, avg_loss / tot_seen_samples
    # return accs, accs_mask_classes
    return accs, accs_mask_classes, val_latest_dict, metric_dict_tuple