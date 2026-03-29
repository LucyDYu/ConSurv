# Copyright 2022-present, Lorenzo Bonicelli, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from copy import deepcopy
import torch  # Import PyTorch
import numpy as np  # Import NumPy for numerical computations
from torch.optim import SGD  # Import SGD optimizer from PyTorch
import logging  # Import logging module for recording training progress

from models.utils.continual_model import ContinualModel  # Import base class for continual learning models
from utils.args import ArgumentParser  # Import argument parser
from utils.training_utils import data_to_output, initialize_survival_metrics, train_loop_survival_coattn_mb_batch, validate_loop_survival_coattn_mb_batch  # Import training utility functions
import torch.optim as optim

from utils.utils import collate_MIL_survival, collate_MIL_survival_cluster, collate_MIL_survival_sig, select_collate

def smooth(logits, temp, dim):
    """
    Smooths the logits using temperature scaling
    Args:
        logits: prediction logits
        temp: temperature parameter to control smoothing (higher = more smooth)
        dim: dimension to normalize over
    Returns:
        Smoothed logits normalized to sum to 1
    """
    log = logits ** (1 / temp)  # Temperature scaling: higher temperature produces smoother distribution
    return log / torch.sum(log, dim).unsqueeze(1)  # Normalization to ensure output probabilities sum to 1


def modified_kl_div(old, new):
    """
    Calculates modified KL divergence between old and new predictions
    Args:
        old: predictions from old model
        new: predictions from current model
    Returns:
        KL divergence loss between distributions
    """
    return -torch.mean(torch.sum(old * torch.log(new), 1))  # Compute modified KL divergence measuring distribution difference between old and new model predictions


class Lwf(ContinualModel):
    """
    Learning without Forgetting (LwF) implementation for continual learning.
    
    LwF preserves knowledge of previous tasks by:
    1. Keeping predictions of old model on new data
    2. Using these to regularize training on new tasks
    3. Balancing between learning new tasks and preserving old knowledge
    """
    NAME = 'lwf'
    COMPATIBILITY = ['class-il', 'task-il']

    @staticmethod
    def get_parser(parser) -> ArgumentParser:
        """
        Adds LwF-specific command line arguments
        Args:
            parser: argument parser
        Returns:
            parser with added arguments
        """
        parser.add_argument('--alpha', type=float, default=0.5,
                            help='Penalty weight.')  # Added penalty weight parameter to control importance of preserving old task knowledge
        parser.add_argument('--softmax_temp', type=float, default=2,
                            help='Temperature of the softmax function.')  # Added softmax temperature parameter to control soft label smoothing for knowledge distillation
        return parser  # Return parser with LwF-specific arguments added

    def __init__(self, backbone, loss, args, transform, dataset=None):
        """
        Initialize LwF model
        Args:
            backbone: neural network architecture
            loss: loss function
            args: command line arguments
            transform: data transformations
            dataset: training dataset
        """
        super(Lwf, self).__init__(backbone, loss, args, transform, dataset=dataset)  # Call parent class initialization
        self.old_net = None  # Initialize old network as None, used to store previous task model
        self.soft = torch.nn.Softmax(dim=1)  # Create softmax function for generating probability distributions
        self.logsoft = torch.nn.LogSoftmax(dim=1)  # Create log-softmax function for computing loss

    def begin_task(self, dataset, **kwargs):
        """
        Preparation before starting a new task:
        1. For tasks after first, do warm-up training
        2. Store model predictions on training data
        Args:
            dataset: training dataset for current task
        """
        kwarg_temp=deepcopy(kwargs)
        self.net.eval()  # Set network to eval mode
        if self.current_task > 0:  # If not the first task
            # Warm-up phase: Train only the classifier on new task data
            # opt = SGD(self.net.classifier.parameters(), lr=self.args.lr)  
            logging.info(f"LwF: Warm-up training for task {kwargs['task_label']} {kwargs['task_name']}")
            opt = optim.Adam(self.net.classifier.parameters(), lr=self.args.lr)  # Create optimizer for classifier-only warm-up training
            opt.zero_grad()
            task_iteration = 0
            for epoch in range(self.args.n_epochs):  # Iterate over training epochs
                train_loss_surv, train_loss, all_risk_scores, all_censorships, all_event_times = initialize_survival_metrics(len(dataset.train_loader), self.args)

                for i, data in enumerate(dataset.train_loader):  # Iterate over training data
                    kwarg_temp['indexes'] = data[10]
                    data_task_label = data[11]
                    kwarg_temp['data_task_label'] = data_task_label

                    # Exit if in debug mode and exceeded debug iterations
                    if self.args.debug_mode and task_iteration > self.get_debug_iters():
                        break
                    # add more kwargs
                    # data
                    kwarg_temp['data'] = data
                    
                    # statistics
                    kwarg_temp['batch_idx'] = i
                    kwarg_temp['all_risk_scores'] = all_risk_scores
                    kwarg_temp['all_censorships'] = all_censorships
                    kwarg_temp['all_event_times'] = all_event_times
                    kwarg_temp['train_loss_surv'] = train_loss_surv
                    kwarg_temp['train_loss'] = train_loss
                    # no grad for network before feature 
                    kwarg_temp['no_grad_feature'] = True
                    # Process batch and update statistics
                    stats = train_loop_survival_coattn_mb_batch(
                        self.args.bs_micro, self.net, **kwarg_temp
                    )
                    task_iteration += 1
                    # Extract loss
                    loss = stats['loss']

                    loss.backward()

                    # Update model parameters using optimizer
                    if (task_iteration + 1) % self.args.gc == 0:
                        opt.step()
                        opt.zero_grad()
            logging.info(f"LwF: Store logits")
            # Store current model's predictions on training data
            logits = []  # Initialize logits list to store model predictions on training data
            with torch.no_grad():  # Disable gradient computation
                for i in range(0, len(dataset.train_loader.dataset), self.args.batch_size):
                    inputs = [dataset.train_loader.dataset.__getitem__(j)
                                          for j in range(i, min(i + self.args.batch_size,
                                                         len(dataset.train_loader.dataset)))]
                    
                    collate = select_collate(self.args.mode)
                    data = collate(inputs)
                    assert i == data[10] 
                    kwargs['indexes'] = data[10]
                    data_task_label = data[11]
                    kwargs['data_task_label'] = data_task_label
                    return_pair_list = data_to_output(data, self.net, returnt='logits', **kwargs)
                    batch_logits = [output.cpu() for output, label in return_pair_list]  # Move tensors to CPU
                    batch_logits = torch.cat(batch_logits, dim=0)
                    logits.append(batch_logits)  # Append to logits list
                    
            dataset.train_loader.dataset.logits = logits  # Merge all logits and store in dataset
            dataset.train_loader.dataset.extra_return_fields += ('logits',)  # Add logits to extra return fields
        self.net.train()  # Set network back to train mode

    def observe(self, inputs, labels, not_aug_inputs, logits=None, epoch=None, **kwargs):
        """
        Training step for LwF:
        1. Compute loss on current task
        2. Add distillation loss to preserve old task knowledge
        Args:
            inputs: training samples
            labels: ground truth labels
            not_aug_inputs: non-augmented inputs (unused)
            logits: old model predictions for distillation
            epoch: current epoch number
            **kwargs: additional arguments for WSI and multi-omics data
        Returns:
            Total loss value
        """
        # Zero out parameter gradients
        if self.task_iteration == 0:
            # Zero out optimizer parameters at task start
            self.opt.zero_grad()

        # Process batch and update statistics 
        stats = train_loop_survival_coattn_mb_batch(
            self.args.bs_micro, self.net, **kwargs
        )

        # Classification loss on all seen classes
        # loss = self.loss(outputs[:, :self.n_seen_classes], labels)  # Compute classification loss for current task
        # Extract loss
        loss = stats['loss']
        model_outputs = stats['model_outputs']
        outputs = [logits for hazards, S, Y_hat, attention_scores, logits in model_outputs]
        outputs = torch.cat(outputs, dim=0)

        # Distillation loss to preserve old task knowledge
        if logits is not None:  # If old model predictions are available
            kd_loss = self.args.alpha * modified_kl_div(  # Add knowledge distillation loss, weighted by alpha
                smooth(self.soft(logits[:, :self.n_past_classes]).to(self.device), self.args.softmax_temp, 1),  # Softened predictions from old model
                smooth(self.soft(outputs[:, :self.n_past_classes]), self.args.softmax_temp, 1)  # Softened predictions from current model on old classes
            )
            loss += kd_loss / self.args.gc
            stats['loss'] = loss

        loss.backward()  # Backward pass
        # Update model parameters using optimizer
        if (self.task_iteration + 1) % self.args.gc == 0:
            self.opt.step()
            self.opt.zero_grad()

            # empty_cache after model update
            torch.cuda.empty_cache()

        return stats
