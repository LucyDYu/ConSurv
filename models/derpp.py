# Copyright 2020-present, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Davide Abati, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from copy import deepcopy
import numpy as np
import torch
from torch.nn import functional as F

from models.utils.continual_model import ContinualModel
from utils.args import add_rehearsal_args, ArgumentParser
from utils.buffer import Buffer
from utils.buffer_wsi import Buffer_WSI
from utils.training_utils import data_to_output, output_to_loss, train_loop_survival_coattn_mb_batch


class Derpp(ContinualModel):
    """Continual learning via Dark Experience Replay++."""
    NAME = 'derpp'
    COMPATIBILITY = ['class-il', 'domain-il', 'task-il', 'general-continual']

    @staticmethod
    def get_parser(parser) -> ArgumentParser:
        add_rehearsal_args(parser)
        parser.add_argument('--alpha', type=float, required=True,
                            help='Penalty weight.')  # Added alpha parameter to control knowledge distillation loss weight
        parser.add_argument('--beta', type=float, required=True,
                            help='Penalty weight.')  # Added beta parameter to control replay sample cross-entropy loss weight
        return parser  # Return updated argument parser

    def __init__(self, backbone, loss, args, transform, dataset=None):
        super().__init__(backbone, loss, args, transform, dataset=dataset)

        # self.buffer = Buffer(self.args.buffer_size)  # Create experience replay buffer, size specified by args.buffer_size
        self.buffer = Buffer_WSI(self.args.buffer_size, dataset, args)  # Create experience replay buffer, size specified by args.buffer_size

    def observe(self, inputs, labels, not_aug_inputs, epoch=None, **kwargs):
        args = kwargs['args']
        task_label = torch.tensor(kwargs['data_task_label'].item())
        task_labels = torch.full((args.batch_size,), task_label, dtype=torch.long)
        start_idx, end_idx = self.dataset.get_offsets(task_label)
        kwargs['start_idx'] = start_idx
        kwargs['end_idx'] = end_idx
        # Zero out parameter gradients
        if self.task_iteration == 0:
            # Zero out optimizer parameters at task start
            self.opt.zero_grad()

        # Process batch and update statistics 
        stats = train_loop_survival_coattn_mb_batch(
            self.args.bs_micro, self.net, **kwargs
        )

        loss = stats['loss']
        model_outputs = stats['model_outputs']
        output_logits = [logits for hazards, S, Y_hat, attention_scores, logits in model_outputs]
        output_logits = torch.cat(output_logits, dim=0)

        if not self.buffer.is_empty():  # If buffer is not empty
            buf_inputs, _, buf_logits, _ = self.buffer.get_data(self.args.minibatch_size, transform=self.transform, device=self.device)  # Get samples, labels and previous outputs from buffer
            # minibatch_size=1
            buf_logits = buf_logits[0]
            # buf_outputs = self.net(buf_inputs)  # Forward pass on buffer samples
            return_pair_list = data_to_output(buf_inputs, self.net, returnt='logits', **kwargs)
            batch_logits = [output for output, label in return_pair_list] 
            buf_outputs = torch.cat(batch_logits, dim=0)   

            loss_mse = self.args.alpha * F.mse_loss(buf_outputs, buf_logits)  # Calculate knowledge distillation loss (MSE)
            loss += loss_mse / self.args.gc # Add knowledge distillation loss to total loss

            buf_inputs, buf_labels, _, buf_task_labels = self.buffer.get_data(self.args.minibatch_size, transform=self.transform, device=self.device)  # Get samples and labels from buffer again
            start_idx, end_idx = self.dataset.get_offsets(buf_task_labels[0]) # may not be the current task
            kwargs_buf = deepcopy(kwargs)
            kwargs_buf['data_task_label'] = buf_task_labels[0]
            kwargs_buf['start_idx'] = start_idx
            kwargs_buf['end_idx'] = end_idx
            # buf_outputs = self.net(buf_inputs)  # Forward pass on buffer samples again
            return_pair_list = data_to_output(buf_inputs, self.net, returnt='out', **kwargs_buf)
            outputs = [output for output, label in return_pair_list]
            loss_buf = self.args.beta * output_to_loss(outputs, buf_inputs, self.net, self.loss, self.args.bag_loss) 
            loss_buf_gc = loss_buf / self.args.gc
            loss += loss_buf_gc # Add loss to total loss

        loss.backward()  # Backward pass to compute gradients


        # Add current batch samples, labels and outputs to buffer, examples=indexes,
        self.buffer.add_data(examples=[kwargs['data']],
                             labels=labels,
                             logits=[output_logits.data],
                             task_labels=task_labels)  # Add current batch samples, labels and outputs to buffer
        # Update model parameters using optimizer
        if (self.task_iteration + 1) % self.args.gc == 0:
            self.opt.step()
            self.opt.zero_grad()

            # empty_cache after model update
            torch.cuda.empty_cache()


        return stats
