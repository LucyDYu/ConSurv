"""
This module implements the simplest form of rehearsal training: Experience Replay. It maintains a buffer
of previously seen examples and uses them to augment the current batch during training.

Example usage:
    model = Er(backbone, loss, args, transform, dataset)
    loss = model.observe(inputs, labels, not_aug_inputs, epoch)

"""

# Copyright 2020-present, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Davide Abati, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from copy import deepcopy
import torch

from models.utils.continual_model import ContinualModel
from utils.args import add_rehearsal_args, ArgumentParser
from utils.buffer import Buffer
from utils.buffer_wsi import Buffer_WSI
from utils.training_utils import data_to_output, output_to_loss, train_loop_survival_coattn_mb_batch


class Er(ContinualModel):
    """Continual learning via Experience Replay."""
    NAME = 'er'
    COMPATIBILITY = ['class-il', 'domain-il', 'task-il', 'general-continual']

    @staticmethod
    def get_parser(parser) -> ArgumentParser:
        """
        Returns an ArgumentParser object with predefined arguments for the Er model.

        This model requires the `add_rehearsal_args` to include the buffer-related arguments.
        """
        add_rehearsal_args(parser)
        return parser

    def __init__(self, backbone, loss, args, transform, dataset=None):
        """
        The ER model maintains a buffer of previously seen examples and uses them to augment the current batch during training.
        """
        super(Er, self).__init__(backbone, loss, args, transform, dataset=dataset)
        # Use Buffer_WSI instead of Buffer for WSI dataset
        self.buffer = Buffer_WSI(self.args.buffer_size, dataset, args)

    def observe(self, inputs, labels, not_aug_inputs, epoch=None, **kwargs):
        """
        ER trains on the current task using the data provided, but also augments the batch with data from the buffer.
        
        Args:
            inputs: Input data for the current batch
            labels: Labels for the current batch
            not_aug_inputs: Non-augmented inputs (used for buffer storage)
            indexes: Indexes of the current batch (required for WSI dataset)
            epoch: Current epoch number
            **kwargs: Additional arguments for WSI dataset
        
        Returns:
            Dictionary containing training statistics
        """
        args = kwargs['args']
        task_label = torch.tensor(kwargs['data_task_label'].item())
        task_labels = torch.full((args.batch_size,), task_label, dtype=torch.long)
        start_idx, end_idx = self.dataset.get_offsets(task_label)
        kwargs['start_idx'] = start_idx
        kwargs['end_idx'] = end_idx
        
        # Zero out parameter gradients at the beginning of each task iteration
        if self.task_iteration == 0:
            self.opt.zero_grad()

        # Process batch and update statistics using the training utility function
        stats = train_loop_survival_coattn_mb_batch(
            self.args.bs_micro, self.net, **kwargs
        )

        loss = stats['loss']

        # If buffer is not empty, get samples from buffer and train on them
        if not self.buffer.is_empty():  
            buf_inputs, buf_labels, buf_task_labels = self.buffer.get_data(
                self.args.minibatch_size, transform=self.transform, device=self.device)
            
            # Get offsets for the buffer task
            start_idx, end_idx = self.dataset.get_offsets(buf_task_labels[0])
            kwargs_buf = deepcopy(kwargs)
            kwargs_buf['data_task_label'] = buf_task_labels[0]
            kwargs_buf['start_idx'] = start_idx
            kwargs_buf['end_idx'] = end_idx
            
            # Process buffer data
            return_pair_list = data_to_output(buf_inputs, self.net, returnt='out', **kwargs_buf)
            outputs = [output for output, label in return_pair_list]
            
            # Calculate loss on buffer data
            loss_buf = output_to_loss(outputs, buf_inputs, self.net, self.loss, self.args.bag_loss)
            loss += loss_buf / self.args.gc

        # Backward pass
        loss.backward()

        # Add current batch to buffer
        self.buffer.add_data(examples=[kwargs['data']],
                             labels=labels,
                             task_labels=task_labels)
        
        # Update model parameters using optimizer with gradient accumulation
        if (self.task_iteration + 1) % self.args.gc == 0:
            self.opt.step()
            self.opt.zero_grad()

            # empty_cache after model update
            torch.cuda.empty_cache()


        return stats
