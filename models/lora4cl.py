"""
This module implements the simplest form of incremental training, i.e., finetuning.
"""

# Copyright 2020-present, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Davide Abati, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from copy import deepcopy
import logging
import torch
from models.utils.continual_model import ContinualModel
from utils.args import add_management_args, add_experiment_args, ArgumentParser
from utils.training_utils import train_loop_survival_coattn_mb_batch

from models.lora4cl_utils.dynamic_lora_factory import DynamicLoRAFactory
from models.lora4cl_utils.lora4cl_base import LoRALayer, LoRALinear


def get_lora_backbone(backbone, args):
    """
    Convert base backbone network to LoRA version.
    
    Args:
        backbone: Base backbone network
        args: Arguments
        
    Returns:
        Backbone network with LoRA applied
    """
    # Create LoRA configuration list
    lora_configs = []
    
    # Add multi_layer1 LoRA configuration
    if args.lora_multi_layer1:
        lora_configs.append({
            'layer_name': 'multi_layer1',
            'mode': 'append',
            'input_dim': 512,
            'output_dim': 512,
        })

    # Add MoME_patho2 LoRA configuration
    if args.lora_MoME_patho2:
        lora_configs.append({
            'layer_name': 'MoME_patho2',
            'mode': 'append',
            'input_dim': 512,
            'output_dim': 512,
        })
    
    # Add MoME_genom2 LoRA configuration
    if args.lora_MoME_genom2:
        lora_configs.append({
            'layer_name': 'MoME_genom2',
            'mode': 'append',
            'input_dim': 512,
            'output_dim': 512,
        })
    
    # If no configurations specified, return original backbone
    if not lora_configs:
        logging.warning("No LoRA configurations specified, using original backbone")
        return backbone
    
    # Apply LoRA to model
    return DynamicLoRAFactory.convert_backbone(backbone, lora_configs, args)


class LoRA4CL(ContinualModel):
    """
    LoRA for Continual Learning.

    This model applies LoRA to specified layers of a backbone network using
    a configuration-driven approach. It supports 'replace' and 'append' modes.
    The logic for parameter freezing and optimizer setup is handled in
    `meta_begin_task` and `begin_task` for clean task transitions.
    """

    NAME = 'lora4cl'
    COMPATIBILITY = ['class-il', 'domain-il', 'task-il']

    @staticmethod
    def get_parser(parser) -> ArgumentParser:
        
        parser.add_argument('--lora_rank', type=int, default=8, help='Rank of the LoRA adapters.')
        parser.add_argument('--lora_alpha', type=int, default=16, help='Alpha scaling factor for LoRA adapters.')
        
        # LoRA application positions
        # Can be flexibly configured for different backbones
        parser.add_argument('--lora_MoME_patho2', action='store_true', help='Apply LoRA on pathology feature processing')
        parser.add_argument('--lora_MoME_genom2', action='store_true', help='Apply LoRA on genomic feature processing')
        return parser

    def __init__(self, backbone, loss, args, transform, dataset=None):
        """
        Initialize LoRA4CL model.
        
        Args:
            backbone: Backbone network
            loss: Loss function
            args: Arguments
            transform: Data transforms
            dataset: Dataset
        """
        logging.info("LoRA4CL will override the backbone model.")
        # Create LoRA4CL network
        lora_backbone = get_lora_backbone(deepcopy(backbone), args)
        
        # Initialize parent class with model components
        super().__init__(lora_backbone, loss, args, transform, dataset=dataset)

    def meta_begin_task(self, dataset, **kwargs):
        """
        Overrides the parent method to ensure the optimizer is created *after*
        the parameters have been configured for the current task in `begin_task`.
        """
        # update internal counters
        self._task_iteration = 0
        self._epoch_iteration = 0
        self._past_epoch = 0
        self._n_classes_current_task = self._cpt if isinstance(self._cpt, int) else self._cpt[self._current_task]
        self._n_past_classes, self._n_seen_classes = self.compute_offsets(self._current_task)
        self._n_remaining_classes = self.N_CLASSES - self._n_seen_classes

        # First, call our custom logic to set `requires_grad` flags for the new task.
        self.begin_task(dataset, **kwargs)

       # reload optimizer if the model has no scheduler
        if not hasattr(self, 'scheduler') or self.scheduler is None:
            if hasattr(self, 'opt') and self.opt is not None:
                self.opt.zero_grad(set_to_none=True)
            self.opt = self.get_optimizer()
        else:
            logging.warning("Model defines a custom scheduler. The optimizer will not be reloaded.")

    def begin_task(self, dataset, **kwargs):
        """
        Prepares the model for a new task by setting which parameters are trainable.
        """
        task_label = self._current_task

        # Configure parameters based on the task_id.
        if task_label == 0:
            # For task 0 (pre-training), train the whole model EXCEPT for any LoRA adapters.
            for param in self.net.parameters():
                param.requires_grad = True

            # Specifically find and freeze all LoRALayer modules.
            for module in self.net.modules():
                if isinstance(module, LoRALayer):
                    for param in module.parameters():
                        param.requires_grad = False
            
        else:
            # For subsequent tasks, freeze everything...
            for param in self.net.parameters():
                param.requires_grad = False

            # ...and then unfreeze only the LoRA adapter for the current task.
            task_id_str = str(task_label)
            for module in self.net.modules():
                if isinstance(module, LoRALinear):
                    if task_id_str in module.lora_adapters:
                        for param in module.lora_adapters[task_id_str].parameters():
                            param.requires_grad = True
                    else:
                        logging.warning(f"No LoRA adapter found for task {task_id_str} in a LoRALinear module.")

        # Always keep normalization layers trainable to adapt to new data distributions.
        for module in self.net.modules():
            if isinstance(module, (torch.nn.modules.batchnorm._BatchNorm, torch.nn.LayerNorm)):
                for param in module.parameters():
                    param.requires_grad = True
        
        # Log the number of trainable parameters
        trainable_params = sum(p.numel() for p in self.net.parameters() if p.requires_grad)
        print(f"INFO: Task {task_label} configured. Trainable parameters: {trainable_params / 1e6:.2f}M")

    def observe(self, inputs, labels, not_aug_inputs, epoch=None, **kwargs):
        """
        Trains on the current task using the data provided, with no 
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

        # Process batch and update statistics 
        stats = train_loop_survival_coattn_mb_batch(
            self.args.bs_micro, self.net, **kwargs
        )
        
        # Extract loss
        loss = stats['loss']

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

