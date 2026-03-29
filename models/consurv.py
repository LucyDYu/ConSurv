
from copy import deepcopy
import numpy as np
import torch
from torch.nn import functional as F
import logging

from models.utils.continual_model import ContinualModel
from utils.args import add_rehearsal_args, ArgumentParser
from utils.buffer import Buffer
from utils.buffer_wsi import Buffer_WSI
from utils.training_utils import data_to_output, output_to_loss, train_loop_survival_coattn_mb_batch
from models.ms_moe_utils.dynamic_moe_factory import DynamicMoEFactory
from utils.args import ArgumentParser, add_rehearsal_args
from backbone import get_backbone
from copy import deepcopy
from models.ms_moe import get_moe_backbone

class ConSurv(ContinualModel):
    """Continual learning via MS_MoE with FCR."""
    NAME = 'consurv'
    COMPATIBILITY = ['class-il', 'domain-il', 'task-il']

    @staticmethod
    def get_parser(parser) -> ArgumentParser:
        add_rehearsal_args(parser)
        parser.add_argument('--alpha', type=float, required=True,
                            help='Penalty weight.')  # Add alpha parameter for controlling the weight of knowledge distillation loss
        parser.add_argument('--alpha_wsi', type=float, required=True,
                            help='Penalty weight.')  # Add alpha parameter for controlling the weight of knowledge distillation loss
        parser.add_argument('--alpha_omic', type=float, required=True,
                            help='Penalty weight.')  # Add alpha parameter for controlling the weight of knowledge distillation loss
        parser.add_argument('--beta', type=float, required=True,
                            help='Penalty weight.')  # Add beta parameter for controlling the weight of replay sample loss
        
        # MS-MoE related parameters
        parser.add_argument('--num_experts', type=int, default=8, help='Number of experts')
        parser.add_argument('--top_k', type=int, default=3, help='Number of selected experts')
        parser.add_argument('--num_shared_experts', type=int, default=1, help='Number of shared experts')
        parser.add_argument('--num_tasks', type=int, default=4, help='Number of tasks')
        
        # MS-MoE application points
        # Can be flexibly configured for different backbones
        parser.add_argument('--moe_MoME_classifier', action='store_true', help='Apply MoE on the classifier')
        parser.add_argument('--moe_MoME_patho2', action='store_true', help='Apply MoE on pathology feature processing')
        parser.add_argument('--moe_MoME_genom2', action='store_true', help='Apply MoE on genome feature processing')
        
        return parser  # Return the updated argument parser

    def __init__(self, backbone, loss, args, transform, dataset=None):
        """
        Initializes the ConSurv model.
        
        Args:
            backbone: The backbone network.
            loss: The loss function.
            args: Arguments.
            transform: Data transformations.
            dataset: The dataset.
        """
        logging.info("MS MoE will override the backbone model.")
        # Create the MS MoE network
        moe_backbone = get_moe_backbone(deepcopy(backbone), args)
        
        # Initialize parent class with model components
        super().__init__(moe_backbone, loss, args, transform, dataset=dataset)

        # Create an experience replay buffer with size specified by args.buffer_size
        self.buffer = Buffer_WSI(self.args.buffer_size, dataset, args)

    def observe(self, inputs, labels, not_aug_inputs, epoch=None, **kwargs):
        args = kwargs['args']
        task_label = torch.tensor(kwargs['data_task_label'].item())
        task_labels = torch.full((args.batch_size,), task_label, dtype=torch.long)
        start_idx, end_idx = self.dataset.get_offsets(task_label)
        kwargs['start_idx'] = start_idx
        kwargs['end_idx'] = end_idx
        # Zero out parameter gradients
        if self.task_iteration == 0:
            # At the beginning of a task, zero out the optimizer's parameters
            self.opt.zero_grad()

        kwargs['returnt'] = 'full'
        # Process batch and update statistics 
        stats = train_loop_survival_coattn_mb_batch(
            self.args.bs_micro, self.net, **kwargs
        )

        loss = stats['loss']


        if not self.buffer.is_empty():  # If the buffer is not empty
            buf_inputs, _, _, buf_final_features, buf_wsi_features, buf_omic_features = self.buffer.get_data(self.args.minibatch_size, transform=self.transform, device=self.device)  # Get samples, labels, and previous outputs from the buffer
            # minibatch_size=1
            buf_final_features = buf_final_features[0]
            buf_wsi_features = buf_wsi_features[0]
            buf_omic_features = buf_omic_features[0]
            # buf_outputs = self.net(buf_inputs)  # Perform a forward pass on the buffer samples
            kwargs['returnt'] = 'full'
            return_pair_list = data_to_output(buf_inputs, self.net, **kwargs)
            batch_final_features = [output_dict['fusion'] for output_dict, _ in return_pair_list] 
            buf_outputs_final_features = torch.cat(batch_final_features, dim=0)   
            loss_mse_final = self.args.alpha * F.mse_loss(buf_outputs_final_features, buf_final_features)  # Calculate knowledge distillation loss
            batch_wsi_features = [output_dict['path'] for output_dict, _ in return_pair_list] 
            buf_outputs_wsi_features = torch.cat(batch_wsi_features, dim=0)
            loss_mse_wsi = self.args.alpha_wsi * F.mse_loss(buf_outputs_wsi_features, buf_wsi_features)  # Calculate knowledge distillation loss
            batch_omic_features = [output_dict['omic'] for output_dict, _ in return_pair_list] 
            buf_outputs_omic_features = torch.cat(batch_omic_features, dim=0)
            loss_mse_omic = self.args.alpha_omic * F.mse_loss(buf_outputs_omic_features, buf_omic_features)  # Calculate knowledge distillation loss
            loss_mse = loss_mse_final + loss_mse_wsi + loss_mse_omic

            loss_mse_gc = loss_mse / self.args.gc
            loss += loss_mse_gc # Add knowledge distillation loss to the total loss

            buf_inputs, buf_labels, buf_task_labels, *_ = self.buffer.get_data(self.args.minibatch_size, transform=self.transform, device=self.device)  # Get samples and labels from the buffer again
            start_idx, end_idx = self.dataset.get_offsets(buf_task_labels[0]) # may not be the current task
            kwargs_buf = deepcopy(kwargs)
            kwargs_buf['data_task_label'] = buf_task_labels[0]
            kwargs_buf['start_idx'] = start_idx
            kwargs_buf['end_idx'] = end_idx
            # buf_outputs = self.net(buf_inputs)  # Perform another forward pass on the buffer samples
            kwargs_buf['returnt'] = 'out'
            return_pair_list = data_to_output(buf_inputs, self.net, **kwargs_buf)
            outputs = [output for output, label in return_pair_list]
            loss_buf = self.args.beta * output_to_loss(outputs, buf_inputs, self.net, self.loss, self.args.bag_loss) 
            loss_buf_gc = loss_buf / self.args.gc
            loss += loss_buf_gc # Add the loss to the total loss

        loss.backward()  # Backpropagate to calculate gradients

        if epoch >= self.args.warm_up_epochs:
            # Add data to buffer after several epochs (consider them as warm-up)
            model_outputs = stats['model_outputs']
            final_features = [output_dict['fusion'] for output_dict in model_outputs]
            final_features = torch.cat(final_features, dim=0)
            wsi_features = [output_dict['path'] for output_dict in model_outputs]
            wsi_features = torch.cat(wsi_features, dim=0)
            omic_features = [output_dict['omic'] for output_dict in model_outputs]
            omic_features = torch.cat(omic_features, dim=0)
            # Add current batch's samples, labels, and outputs to the buffer, 
            self.buffer.add_data(examples=[kwargs['data']],
                                 labels=labels,
                                 task_labels=task_labels,
                                 final_features=[final_features.data],
                                 wsi_features=[wsi_features.data],
                                 omic_features=[omic_features.data]) # Add current batch's samples, labels, and outputs to the buffer
         
        # Update model parameters using optimizer
        if (self.task_iteration + 1) % self.args.gc == 0:
            self.opt.step()
            self.opt.zero_grad()

            # empty_cache after model update
            torch.cuda.empty_cache()


        return stats
