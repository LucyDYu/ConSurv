import torch
from self_supervised.augmentations.feature_transform import FeatureTransform
from self_supervised.augmentations.simclr_transform import SimCLRTransform
from self_supervised.criterion.supcontrast import SupConLoss
from utils.buffer import Buffer
from utils.args import *
from models.utils.continual_model import ContinualModel
from utils.buffer_wsi import Buffer_WSI
from utils.ecr import ECR
from copy import deepcopy
import torch.nn.functional as F

from utils.training_utils import data_to_output, output_to_loss, train_loop_survival_coattn_mb_batch




class ImexReg(ContinualModel):
    NAME = 'imex_reg'
    COMPATIBILITY = ['class-il', 'domain-il', 'task-il', 'general-continual']

    @staticmethod
    def get_parser(parser) -> ArgumentParser:
        add_rehearsal_args(parser)
        parser.add_argument('--ecr_weight', type=float, default=0.1,
                            help='multitask weight for rotation')
        parser.add_argument('--crl_weight', type=float, default=1,
                            help='multitask weight for rotation')
        parser.add_argument('--reg_weight', type=float, default=0.2,
                            help='EMA regularization weight')
        parser.add_argument('--ema_update_freq', type=float, default=0.5,
                            help='EMA update frequency')
        parser.add_argument('--ema_alpha', type=float, default=0.999,
                        help='EMA alpha')
        return parser  # Return updated argument parser
    
    def __init__(self, backbone, loss, args, transform, dataset=None):
        super().__init__(backbone, loss, args, transform, dataset=dataset)
        self.buffer = Buffer_WSI(self.args.buffer_size, dataset, args)  # Create experience replay buffer, size specified by args.buffer_size

        self.ema_model = deepcopy(self.net).to(self.device)
        # set regularization weight
        self.reg_weight = self.args.reg_weight
        # set parameters for ema model
        self.ema_update_freq = self.args.ema_update_freq
        self.ema_alpha = self.args.ema_alpha
        self.consistency_loss = torch.nn.MSELoss(reduction='none')
        self.global_step = 0
        # Additional models
        self.addit_models = ['ema_model']

        self.ecr = ECR()
        self.CRL_loss = SupConLoss()
        self.CRL_transform = FeatureTransform()
        

    def update_ema_model_variables(self):
        alpha = min(1 - 1 / (self.global_step + 1), self.ema_alpha)
        for ema_param, param in zip(self.ema_model.parameters(), self.net.parameters()):
            ema_param.data.mul_(alpha).add_(1 - alpha, param.data)

    def compute_loss(self, inputs, not_aug_inputs, origin_loss, **kwargs):
        kwargs_inputs=kwargs.copy()

        return_pair_list = data_to_output(inputs, self.net, returnt='imex', **kwargs_inputs)
        # Calculate loss on inputs data
        if origin_loss:
            outputs = [output for output, label in return_pair_list]
            loss_origin = output_to_loss(outputs, inputs, self.net, self.loss, self.args.bag_loss)
        else:
            loss_origin = 0
        
        # Create the augmented view for SupCon and run through the model
        input_data_aug = self.CRL_transform.transform_wsi(not_aug_inputs)
        kwargs_inputs_aug=kwargs.copy()
        kwargs_inputs_aug['data'] = input_data_aug
        return_pair_list_aug = data_to_output(input_data_aug, self.net, returnt='imex', **kwargs_inputs_aug)

        cnt = 0
        loss = 0.
        for i in range(len(return_pair_list)):
            output, label = return_pair_list[i]
            _,_,_,_, outputs_m, zm, output_proj = output
            output_aug, label_aug = return_pair_list_aug[i]
            _,_,_,_, outputs_x, zx, _ = output_aug
            # Compute the losses
            start_idx = kwargs['start_idx']
            label_actual = label + start_idx
            labels=torch.tensor([label_actual])
            loss_proj_m = self.CRL_loss(zm, zx, labels=labels)
            # loss_ce_m = self.loss(outputs_m, labels) # Already computed earlier
            loss_ecr = self.ecr.update(output_proj, zm.detach())

            loss += self.args.crl_weight * loss_proj_m + self.args.ecr_weight * loss_ecr
            cnt += 1

        # Average the loss 
        loss = loss / cnt + loss_origin

        return loss.squeeze(), return_pair_list, return_pair_list_aug

    # not need in wsi
    # def transform_inputs(self, inputs):
    #     return torch.stack([self.transform(ee.cpu()) for ee in inputs]).to(self.device)

    def observe(self, inputs, labels, not_aug_inputs, epoch=None, **kwargs):
        args = kwargs['args']
        task_label = torch.tensor(kwargs['data_task_label'].item())
        task_labels = torch.full((args.batch_size,), task_label, dtype=torch.long)
        start_idx, end_idx = self.dataset.get_offsets(task_label)
        kwargs['start_idx'] = start_idx
        kwargs['end_idx'] = end_idx
        inputs_data = kwargs['data']

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
        loss_imex, _, _ = self.compute_loss(inputs_data, inputs_data, origin_loss=False, **kwargs)

        loss += loss_imex / self.args.gc

        model_outputs = stats['model_outputs']
        output_logits = [logits for hazards, S, Y_hat, attention_scores, logits in model_outputs]
        output_logits = torch.cat(output_logits, dim=0)

        if not self.buffer.is_empty():
            # Load buffer data without any augmentation
            buf_inputs, buf_labels, buf_task_labels = self.buffer.get_data(
                self.args.minibatch_size, transform=None, device=self.device) # transform is None for features
            # Get offsets for the buffer task
            start_idx, end_idx = self.dataset.get_offsets(buf_task_labels[0])
            kwargs_buf = kwargs.copy()
            kwargs_buf['data_task_label'] = buf_task_labels[0]
            kwargs_buf['start_idx'] = start_idx
            kwargs_buf['end_idx'] = end_idx
            kwargs_buf['data'] = buf_inputs
            # Process buffer data
            # The first argument to compute_loss is the 'weakly' augmented view, which for features is just the original data.
            # The second argument is the source for the 'strongly' augmented view.
            loss_m, return_pair_list, return_pair_list_aug = self.compute_loss(buf_inputs, buf_inputs, origin_loss=True, **kwargs_buf)
            loss += loss_m / self.args.gc

            # Apply consistency regularization w.r.t to EMA output
            kwargs_ema = kwargs_buf.copy()
            kwargs_ema['data'] = buf_inputs # EMA model also processes original buffer data
            return_pair_list_ema = data_to_output(buf_inputs, self.ema_model, returnt='imex', **kwargs_ema)

            loss_cr=0; cnt=0;
            for i in range(len(return_pair_list)):
                output, label = return_pair_list[i]
                _,_,_,_, outputs_m, zm, output_proj = output
                output_ema, label_ema = return_pair_list_ema[i]
                _,_,_,_, outputs_ema, z_ema, _ = output_ema
                # Compute the losses
                loss_cr += F.mse_loss(outputs_m, outputs_ema.detach()) + F.mse_loss(zm, z_ema.detach())
                cnt += 1
            loss_cr = loss_cr / cnt
            loss += self.args.reg_weight * loss_cr / self.args.gc
            # --- Cleanup: delete after use and force garbage collection ---
            # 1. Explicitly dereference large variables
            try:
                del buf_inputs, buf_labels, buf_task_labels, kwargs_buf, kwargs_ema
                del loss_m, return_pair_list, return_pair_list_aug, loss_cr
                del output, label, outputs_m, zm, output_proj, output_ema, label_ema, outputs_ema, z_ema
            except NameError:
                pass # Ignore NameError from variables not defined due to conditional branches

            # 2. Force Python garbage collection
            import gc
            gc.collect()

            # 3. After Python GC, clear PyTorch GPU cache
            torch.cuda.empty_cache()
            # --- End of cleanup code ---

        loss.backward()

        # Add current batch to buffer
        self.buffer.add_data(examples=[kwargs['data']],
                             labels=labels,
                             task_labels=task_labels)

        # Update the ema model
        self.global_step += 1
        if torch.rand(1) < self.ema_update_freq:
            self.update_ema_model_variables()

        # Update model parameters using optimizer with gradient accumulation
        if (self.task_iteration + 1) % self.args.gc == 0:
            self.opt.step()
            self.opt.zero_grad()

            # empty_cache after model update
            torch.cuda.empty_cache()

        return stats
