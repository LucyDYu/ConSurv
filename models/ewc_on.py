# Copyright 2020-present, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Davide Abati, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.utils.continual_model import ContinualModel
from trainer.mb_trainer import split_chunk_list
from utils.args import ArgumentParser
from utils.training_utils import data_to_output, train_loop_survival_coattn_mb_batch


class EwcOn(ContinualModel):
    """
    Online Elastic Weight Consolidation (EWC-Online) for Continual Learning.
    
    EWC-Online prevents catastrophic forgetting by:
    1. Keeping track of important parameters for previous tasks
    2. Penalizing changes to these important parameters
    3. Using an online update rule to merge importance between tasks
    """
    NAME = 'ewc_on'
    COMPATIBILITY = ['class-il', 'domain-il', 'task-il']

    @staticmethod
    def get_parser(parser) -> ArgumentParser:
        # e_lambda: Controls how strongly we want to prevent changes to important parameters
        parser.add_argument('--e_lambda', type=float, required=True,
                            help='lambda weight for EWC')
        # gamma: Controls how much we want to keep old task information vs new task information
        parser.add_argument('--gamma', type=float, required=True,
                            help='gamma parameter for EWC online')
        return parser

    def __init__(self, backbone, loss, args, transform, dataset=None):
        super(EwcOn, self).__init__(backbone, loss, args, transform, dataset=dataset)

        self.logsoft = nn.LogSoftmax(dim=1)
        # checkpoint: Stores the model parameters after each task
        self.checkpoint = None
        # fish: Stores the Fisher Information Matrix (importance of each parameter)
        self.fish = None

    def penalty(self):
        """
        Compute the EWC penalty term.
        This penalizes changes to parameters that were important for previous tasks.
        """
        if self.checkpoint is None:
            return torch.tensor(0.0).to(self.device)
        else:
            # Penalty = λ * Fisher * (current_params - old_params)^2
            penalty = self.args.e_lambda * (self.fish * ((self.net.get_params() - self.checkpoint) ** 2)).sum()
            return penalty

    def end_task(self, dataset, **kwargs):
        """
        Update the Fisher Information Matrix at the end of each task.
        The Fisher Matrix represents how important each parameter is for the current task.
        """
        args = kwargs['args']
        task_label = kwargs['task_label']
        # Initialize Fisher matrix for current task
        fish = torch.zeros_like(self.net.get_params())
        total_samples = 0
        
        # Compute Fisher Information Matrix
        for j, data in enumerate(dataset.train_loader):
            # Use data_to_output to handle various data formats
            kwargs['indexes'] = data[10]
            data_task_label = data[11]
            kwargs['data_task_label'] = data_task_label

            return_pair_list = data_to_output(data, self.net, returnt='logits', **kwargs)
            
            for output, label in return_pair_list:
                self.opt.zero_grad()
                # Compute loss (negative log-likelihood)
                start_idx = kwargs['start_idx']
                end_idx = kwargs['end_idx']
                current_logits = output[:, start_idx: end_idx]
                loss = - F.nll_loss(self.logsoft(current_logits), label,
                                    reduction='none')
                # Get probability of correct class
                exp_cond_prob = torch.mean(torch.exp(loss.detach().clone()))
                loss = torch.mean(loss)
                loss.backward()
                # Update Fisher with squared gradients weighted by probability
                fish += exp_cond_prob * self.net.get_grads() ** 2

                total_samples += 1


        # Normalize Fisher by dataset size
        # data size depend on mini batch
        # fish /= (len(dataset.train_loader) * self.args.batch_size)
        fish /= total_samples

        # Online update of Fisher Information Matrix
        if self.fish is None:
            self.fish = fish
        else:
            # Decay old Fisher and add new Fisher (online EWC update rule)
            self.fish *= self.args.gamma
            self.fish += fish

        # Store current model parameters
        self.checkpoint = self.net.get_params().data.clone()

    def get_penalty_grads(self):
        """
        Compute gradients of the EWC penalty term.
        These gradients will be added to the task loss gradients.
        """
        return self.args.e_lambda * 2 * self.fish * (self.net.get_params().data - self.checkpoint)

    def observe(self, inputs, labels, not_aug_inputs, epoch=None, **kwargs):
        """
        Training step that includes both task loss and EWC penalty.
        """

        # Zero out parameter gradients
        if self.task_iteration == 0:
            # Zero out optimizer parameters at task start
            self.opt.zero_grad()

        # Process batch and update statistics
        stats = train_loop_survival_coattn_mb_batch(
            self.args.bs_micro, self.net, **kwargs
        )

        # Extract loss
        loss = stats['loss']


        assert not torch.isnan(loss)
        loss.backward()
        
        # Update model parameters using optimizer
        if (self.task_iteration + 1) % self.args.gc == 0:
            # Add EWC penalty gradients if we have previous task information
            if self.checkpoint is not None:
                # Accumulate EWC penalty gradients into current gradients
                current_grads = self.net.get_grads()
                # Get EWC penalty gradients
                penalty_grads = self.get_penalty_grads()
                assert current_grads.shape == penalty_grads.shape, "Gradient shapes do not match!"
                
                # Add EWC penalty gradients to current gradients
                self.net.set_grads(current_grads + penalty_grads)

            # Optimizer step
            self.opt.step()
            self.opt.zero_grad()
            # empty_cache after model update
            torch.cuda.empty_cache()

        return stats