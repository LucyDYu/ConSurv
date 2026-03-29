"""
This module implements the simplest form of incremental training, i.e., finetuning.
"""

# Copyright 2020-present, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Davide Abati, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
from models.utils.continual_model import ContinualModel
from utils.training_utils import train_loop_survival_coattn_mb_batch


class Sgd(ContinualModel):
    """
    Finetuning baseline - simple incremental training.
    """

    # Name identifier for the model
    NAME = 'sgd'
    # List of compatible continual learning settings
    COMPATIBILITY = ['class-il', 'domain-il', 'task-il', 'general-continual']

    def __init__(self, backbone, loss, args, transform, dataset=None):
        # Initialize parent class with model components
        super(Sgd, self).__init__(backbone, loss, args, transform, dataset=dataset)

    def observe(self, inputs, labels, not_aug_inputs, epoch=None, **kwargs):
        """
        SGD trains on the current task using the data provided, with no 
        countermeasures to avoid forgetting.
        
        Args:
            inputs: Input data batch
            labels: Ground truth labels
            not_aug_inputs: Non-augmented inputs (unused)
            epoch: Current epoch number
            kwargs: Additional arguments passed to training loop
            
        Returns:
            Dictionary containing training statistics
        """
        # Zero out parameter gradients
        if self.task_iteration == 0:
            # Zero out optimizer parameters at task start
            self.opt.zero_grad()
        # # Forward pass through the network
        # outputs = self.net(inputs)
        # # Calculate loss between predictions and ground truth
        # loss = self.loss(outputs, labels)

        # Process batch and update statistics 
        stats = train_loop_survival_coattn_mb_batch(
            self.args.bs_micro, self.net, **kwargs
        )
        
        # Extract loss
        loss = stats['loss']
        
        # # Normalize loss by gradient accumulation steps
        # loss = loss / self.args.gc


        # Backward pass to compute gradients
        loss.backward()
        # Update model parameters using optimizer
        if (self.task_iteration + 1) % self.args.gc == 0:
            self.opt.step()
            self.opt.zero_grad()
            # empty_cache after model update
            torch.cuda.empty_cache()
        # Return all statistics
        return stats

