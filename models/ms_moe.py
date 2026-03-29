"""
Mixture of Experts for Multimodal Survival Analysis
"""

# Copyright 2020-present, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Davide Abati, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import logging
import torch
from copy import deepcopy
from models.utils.continual_model import ContinualModel
from utils.training_utils import train_loop_survival_coattn_mb_batch
from models.ms_moe_utils.dynamic_moe_factory import DynamicMoEFactory
from utils.args import ArgumentParser
from backbone import get_backbone


def get_moe_backbone(backbone, args):
    """
    Convert base backbone network to MoE version.
    
    Args:
        backbone: Base backbone network
        args: Arguments
        
    Returns:
        Backbone network with MoE applied
    """
    # Create MoE configuration list
    moe_configs = []
    
    # Add classifier MoE configuration
    if args.moe_MoME_classifier:
        moe_configs.append({
            'layer_name': 'classifier',
            'mode': 'replace',
        })
    
    # Add MoME_patho2 MoE configuration
    if args.moe_MoME_patho2:
        moe_configs.append({
            'layer_name': 'MoME_patho2',
            'mode': 'append',
            'output_dim': 512,
        })
    
    # Add MoME_genom2 MoE configuration
    if args.moe_MoME_genom2:
        moe_configs.append({
            'layer_name': 'MoME_genom2',
            'mode': 'append',
            'output_dim': 512,
        })
    
    # If no configurations specified, return original backbone
    if not moe_configs:
        logging.warning("No MoE configurations specified, using original backbone")
        return backbone
    
    # Apply MoE to model
    return DynamicMoEFactory.convert_backbone(backbone, moe_configs, args)


class MS_MoE(ContinualModel):
    """
    Mixture of Experts for Multimodal Survival Analysis
    """

    # Name identifier for the model
    NAME = 'ms_moe'
    # List of compatible continual learning settings
    COMPATIBILITY = ['class-il', 'task-il']

    @staticmethod
    def get_parser(parser) -> ArgumentParser:
        """
        Add model-specific command line arguments.
        
        Args:
            parser: Command line argument parser
            
        Returns:
            Updated argument parser
        """
        # MoE-related parameters
        parser.add_argument('--num_experts', type=int, default=8, help='Number of experts')
        parser.add_argument('--top_k', type=int, default=3, help='Number of selected experts')
        parser.add_argument('--num_shared_experts', type=int, default=1, help='Number of shared experts')
        parser.add_argument('--num_tasks', type=int, default=4, help='Number of tasks')
        
        # MoE application positions
        # Can be flexibly configured for different backbones
        parser.add_argument('--moe_MoME_classifier', action='store_true', help='Apply MoE on classifier')
        parser.add_argument('--moe_MoME_patho2', action='store_true', help='Apply MoE on pathology feature processing')
        parser.add_argument('--moe_MoME_genom2', action='store_true', help='Apply MoE on genomic feature processing')
        
        return parser

    def __init__(self, backbone, loss, args, transform, dataset=None):
        """
        Initialize MS_MoE model.
        
        Args:
            backbone: Backbone network
            loss: Loss function
            args: Arguments
            transform: Data transforms
            dataset: Dataset
        """
        logging.info("MS MoE will override the backbone model.")
        # Create MS MoE network
        moe_backbone = get_moe_backbone(deepcopy(backbone), args)
        
        # Initialize parent class with model components
        super().__init__(moe_backbone, loss, args, transform, dataset=dataset)


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

