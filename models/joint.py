"""
This module contains the implementation of the Joint CL model for WSI survival analysis.

The Joint model is the upper bound of the CL scenario, as it has access to all the data from all the tasks.
This model is required for the `domain-il` scenario, while `class-il` and `task-il` scenarios can use the `--joint=1` flag.
"""

import math
import torch
import logging
from models.utils.continual_model import ContinualModel
from tqdm import tqdm
import numpy as np

from utils.conf import create_seeded_dataloader
from utils.schedulers import get_scheduler
from utils.training_utils import data_to_output, validate_loop_survival_coattn_mb_batch


class Joint(ContinualModel):
    NAME = 'joint'
    COMPATIBILITY = ['class-il', 'domain-il', 'task-il']

    def __init__(self, backbone, loss, args, transform, dataset=None):
        super(Joint, self).__init__(backbone, loss, args, transform, dataset=dataset)
        self.old_data = []
        self.old_labels = []

    def end_task(self, dataset):
        """
        This version of joint training for WSI tasks saves all data from previous tasks 
        and then trains on all data at the end of the last one.
        """
        # Store current task's data loader
        self.old_data.append(dataset.train_loader)
        self.task_names.append(dataset.CURRENT_TASK)
        
        # Only train after all tasks have been seen
        if len(dataset.test_loaders) != dataset.N_TASKS:
            return

        logging.info("Starting joint training on all tasks...")
        
        # Get joint loaders from dataset
        train_loader, _ = dataset.get_joint_loaders(fold=self.args.fold)
        
        bs = self.args.batch_size
        scheduler = get_scheduler(self.net, self.args, reload_optim=True)

        # Train on joint dataset
        with tqdm(total=self.args.n_epochs * len(train_loader)) as pbar:
            for e in range(self.args.n_epochs):
                pbar.set_description(f"Joint - Epoch {e}", refresh=False)
                train_iter = iter(train_loader)
                i = 0
                
                while True:
                    try:
                        # Get next batch
                        data = next(train_iter)
                    except StopIteration:
                        break
                    
                    # Debug mode check
                    if self.args.debug_mode and i > self.get_debug_iters():
                        break
                    
                    # Process batch with micro-batching
                    kwargs = {
                        'data': data,
                        'args': self.args,
                        'epoch': e,
                    }
                    
                    # Zero gradients
                    self.opt.zero_grad()
                    
                    # Process batch and get loss
                    stats = validate_loop_survival_coattn_mb_batch(
                        self.args.bs_micro, self.net, **kwargs
                    )
                    
                    loss = stats['val_loss']
                    
                    # Backward pass and optimization
                    loss.backward()
                    self.opt.step()
                    
                    # Update progress bar
                    pbar.update(1)
                    pbar.set_postfix({'loss': loss.item()}, refresh=False)
                    
                    i += 1
                
                # Step scheduler at the end of each epoch
                if scheduler is not None:
                    scheduler.step()
        
        logging.info("Joint training completed.")

    def observe(self, *args, **kwargs):
        # ignore training on task
        return 0
