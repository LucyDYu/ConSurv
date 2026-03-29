import torch
import torch.nn.functional as F
from torch.amp import GradScaler, autocast

from losses.loss import sup_con_loss  # Use MOSE original contrastive loss function
from self_supervised.augmentations.feature_transform import FeatureTransform
from utils.buffer_wsi import Buffer_WSI
from utils.args import *
from models.utils.continual_model import ContinualModel
from utils.training_utils import data_to_output, output_to_loss, train_loop_survival_coattn_mb_batch
from backbone.model_mome_mose import align_wsi_features


class MOSE_WSI(ContinualModel):
    """MOSE adapted for WSI multimodal survival analysis continual learning"""
    NAME = 'mose_wsi'
    COMPATIBILITY = ['class-il', 'domain-il', 'task-il', 'general-continual']

    @staticmethod
    def get_parser(parser) -> ArgumentParser:
        add_rehearsal_args(parser)
        # MOSE core parameters
        parser.add_argument('--ins_t', type=float, default=0.07,
                            help='Temperature for instance-wise contrastive loss')
        parser.add_argument('--expert', type=int, default=2,
                            help='Which expert to use for student in RSD (0-2 for 3 experts)')
        parser.add_argument('--moe_mode', type=str, default='single', choices=['single', 'ensemble'],
                            help='MOSE inference mode: single (MOSE) or ensemble (MOE-MOSE)')
        # Multi-Level Supervision related
        parser.add_argument('--mls_weight', type=float, default=1.0,
                            help='Weight for multi-level supervision loss')
        parser.add_argument('--rsd_weight', type=float, default=0.1,
                            help='Weight for reverse self-distillation loss')
        return parser

    def __init__(self, backbone, loss, args, transform, dataset=None):
        super().__init__(backbone, loss, args, transform, dataset=dataset)

        # WSI-specific buffer
        self.buffer = Buffer_WSI(self.args.buffer_size, dataset, args)

        # MOSE core parameters
        self.ins_t = args.ins_t
        self.expert = int(args.expert)  # Expert for RSD (0=path, 1=omic, 2=fusion)
        self.moe_mode = args.moe_mode  # 'single' (MOSE) or 'ensemble' (MOE-MOSE)

        # Data augmentation component
        self.feature_transform = FeatureTransform()

        # Training state tracking
        self.total_step = 0
        self.class_holder = []
        self.new_class_holder = []

        # Mixed precision training
        self.scaler = GradScaler('cuda')

    def compute_mose_supervision(self, model_outputs, labels, inputs_data, kwargs):
        """
        Compute MOSE Multi-Level Supervision and Reverse Self-Distillation losses

        Args:
            model_outputs: Model's multi-layer output feature list
            labels: Corresponding label tensor
            inputs_data: Original input data (used for data augmentation)
            kwargs: Configuration parameter dictionary

        Returns:
            torch.Tensor: MOSE supervision loss (MLS + RSD)
        """
        # Extract multi-layer features from train_loop outputs (path, omic, fusion)
        kwargs_mose = kwargs.copy()
        if labels.dim() == 1:
            current_label = labels[0].item()  # WSI is per-patient level, entire batch shares one label
        else:
            current_label = labels.item()
        return_pair_list = [(output, current_label) for output in model_outputs]

        # Data augmentation
        inputs_data_aug = self.feature_transform.transform_wsi(inputs_data)
        kwargs_aug = kwargs_mose.copy()
        kwargs_aug['data'] = inputs_data_aug
        return_pair_list_aug = data_to_output(inputs_data_aug, self.net, **kwargs_aug)

        # Multi-Level Supervision: Compute instance-wise contrastive loss for each layer features
        # Follow MOSE original mode: Concatenate original + augmented samples in batch dimension
        expert_names = ['path', 'omic', 'fusion']

        # Collect aligned features and labels from all micro-batches
        all_features = {exp: [] for exp in expert_names}
        all_features_aug = {exp: [] for exp in expert_names}
        all_labels = []

        for pair, pair_aug in zip(return_pair_list, return_pair_list_aug):
            output_dict, pair_label = pair
            output_dict_aug, _ = pair_aug

            # Align features to same dimension [512]
            aligned_feat = align_wsi_features(output_dict)
            aligned_feat_aug = align_wsi_features(output_dict_aug)

            # Collect features from each layer
            for exp_name in expert_names:
                all_features[exp_name].append(aligned_feat[exp_name])
                all_features_aug[exp_name].append(aligned_feat_aug[exp_name])

            all_labels.append(pair_label)

        # Compute Multi-Level Supervision loss for each expert layer
        # MLS = Contrastive Loss (projection space) + Classification Loss (prediction space)
        mls_loss = 0.
        for exp_name in expert_names:
            # Stack features from all micro-batches [n_micro, 512]
            feat_orig = torch.stack(all_features[exp_name], dim=0)
            feat_aug = torch.stack(all_features_aug[exp_name], dim=0)

            # ========== Part 1: Contrastive Learning (projection space) ==========
            # ✅ Project first then contrast (512 → 128), corresponding to MOSE original proj_list
            proj_orig = self.net.projection_heads[exp_name](feat_orig)  # [n_micro, 128]
            proj_aug = self.net.projection_heads[exp_name](feat_aug)    # [n_micro, 128]

            # Concatenate projection features in batch dimension [2*n_micro, 128]
            # Pattern: [orig_1, orig_2, ..., aug_1, aug_2, ...]
            all_proj = torch.cat([proj_orig, proj_aug], dim=0)

            # Prepare labels: [label_1, label_2, ..., label_1, label_2, ...]
            labels_tensor = torch.tensor(all_labels, device=all_proj.device)
            all_labels_tensor = torch.cat([labels_tensor, labels_tensor], dim=0)

            # ✅ Call MOSE original sup_con_loss: computed in projection space (128-dim)
            ins_loss = sup_con_loss(all_proj, self.ins_t, all_labels_tensor)
            mls_loss += ins_loss

            # ========== Part 2: Survival Loss (prediction space) ==========
            # ⚠️ fusion loss already computed by train_loop, only compute path and omic survival losses
            if exp_name != 'fusion':
                # Get prediction logits through classifiers [n_micro, n_classes]
                pred_orig = self.net.classifiers[exp_name](feat_orig)
                pred_aug = self.net.classifiers[exp_name](feat_aug)

                # Concatenate predictions [2*n_micro, n_classes]
                all_pred = torch.cat([pred_orig, pred_aug], dim=0)

                # Compute survival analysis outputs from logits (consistent with forward_features logic)
                hazards = torch.sigmoid(all_pred)
                S = torch.cumprod(1 - hazards, dim=1)
                Y_hat = torch.topk(all_pred, 1, dim=1)[1]

                # Construct outputs list for survival loss calculation
                # Separate each data point in batch into independent tuples
                batch_size = hazards.shape[0]
                outputs = [(hazards[i:i+1], S[i:i+1], Y_hat[i:i+1], {}, all_pred[i:i+1]) for i in range(batch_size)]

                # Use survival loss (replacing original cross-entropy)
                survival_loss = output_to_loss(outputs, inputs_data, self.net, self.loss, self.args.bag_loss)
                mls_loss += survival_loss

        # Weighted MLS loss
        mose_loss = self.args.mls_weight * mls_loss / self.args.gc

        # ============= Reverse Self-Distillation (RSD) =============
        # MOE-MOSE mode: skip RSD distillation, only use MLS
        if self.moe_mode == 'ensemble':
            # MOE-MOSE: do not use RSD distillation
            pass
        else:
            # MOSE: use RSD distillation
            # ✅ Reverse Self-Distillation: deep experts learn from shallow experts
            # Teachers: path and omic (shallow experts, provide diverse expertise, detached and fixed)
            # Student: fusion (deep expert, learns shallow advantages, receives gradients)
            #
            # Corresponding to MOSE original Eq.9: ĥ'i (shallow, detached) and h'n (deep, receives gradients)
            # "RSD takes latent sequential experts as teachers and treats the largest expert F as the student"
            #
            # Collect aligned features from all micro-batches
            aligned_features = {
                'path': [],
                'omic': [],
                'fusion': []
            }

            for pair in return_pair_list:
                output_dict, _ = pair
                aligned_feat = align_wsi_features(output_dict)
                for exp_name in expert_names:
                    aligned_features[exp_name].append(aligned_feat[exp_name])

            # Convert list to tensor
            for exp_name in expert_names:
                aligned_features[exp_name] = torch.stack(aligned_features[exp_name], dim=0)

            # Compute RSD loss: path and omic as teachers (shallow, detached), fusion as student (deep, receives gradients)
            stu_features = aligned_features['fusion']  # [n_micro_batch, 512] deep student
            rsd_loss = 0.

            for teacher_name in ['path', 'omic']:
                teacher_features = aligned_features[teacher_name]  # [n_micro_batch, 512] shallow teacher

                # L2 distillation loss: deep learns to approach shallow (shallow detached and fixed, deep receives gradients)
                distill_loss = torch.dist(
                    F.normalize(stu_features, dim=1),  # deep student fusion (receives gradients, being optimized)
                    F.normalize(teacher_features.detach(), dim=1),  # shallow teacher path/omic (fixed target)
                    p=2
                )
                rsd_loss += distill_loss

            # Weighted RSD loss
            mose_loss += self.args.rsd_weight * rsd_loss / self.args.gc

        return mose_loss

    def observe(self, inputs, labels, not_aug_inputs=None, epoch=None, **kwargs):
        """
        MOSE-WSI core training function
        1. Basic WSI training process (train_loop_survival_coattn_mb_batch)
        2. Multi-Level Supervision (MLS): multi-layer feature contrastive learning
        3. Reverse Self-Distillation (RSD): shallow to deep knowledge distillation
        4. Buffer replay with MOSE losses
        """
        args = kwargs['args']
        task_label = torch.tensor(kwargs['data_task_label'].item())
        task_labels = torch.full((args.batch_size,), task_label, dtype=torch.long)
        start_idx, end_idx = self.dataset.get_offsets(task_label)
        kwargs['start_idx'] = start_idx
        kwargs['end_idx'] = end_idx
        kwargs['returnt'] = 'full'  # Set to return full feature dictionary, reuse train_loop outputs
        kwargs['moe_mode'] = self.moe_mode  # Pass MOE mode parameter
        inputs_data = kwargs['data']

        # Update class holder
        for label in labels:
            if label.item() not in self.class_holder:
                self.class_holder.append(label.item())
                self.new_class_holder.append(label.item())

        # Zero gradients
        if self.task_iteration == 0:
            self.opt.zero_grad()

        # ============= Step 1: Basic WSI training process =============
        stats = train_loop_survival_coattn_mb_batch(
            self.args.bs_micro, self.net, **kwargs
        )
        loss = stats['loss']

        # ============= Step 2: MOSE Multi-Level Supervision (MLS) + Reverse Self-Distillation (RSD) =============
        # Use independent method to compute MOSE supervision loss, avoid duplicate forward
        # with autocast('cuda'):
        # Basic training labels need to add task offset, convert to global class indices (16-dim range)
        # labels_global = labels + start_idx  # Add task offset
        mose_loss = self.compute_mose_supervision(
            stats['model_outputs'], labels, inputs_data, kwargs
        )
        loss += mose_loss

        # ============= Step 4: Buffer Replay =============
        if not self.buffer.is_empty() and task_label > 0:
            buf_inputs, buf_labels, buf_task_labels = self.buffer.get_data(
                self.args.minibatch_size, transform=None, device=self.device
            )
            buf_start_idx, buf_end_idx = self.dataset.get_offsets(buf_task_labels[0])
            kwargs_buf = kwargs.copy()
            kwargs_buf['data_task_label'] = buf_task_labels[0]
            kwargs_buf['start_idx'] = buf_start_idx
            kwargs_buf['end_idx'] = buf_end_idx
            kwargs_buf['data'] = buf_inputs
            kwargs_buf['returnt'] = 'full'  # Set to return full feature dictionary for MOSE supervision
            kwargs_buf['moe_mode'] = self.moe_mode  # Pass MOE mode parameter

            # Basic loss for buffer data
            buf_return_pair_list = data_to_output(buf_inputs, self.net, **kwargs_buf)
            buf_outputs = [output for output, _ in buf_return_pair_list]
            buf_loss = output_to_loss(buf_outputs, buf_inputs, self.net, self.loss, self.args.bag_loss)
            loss += (buf_loss / self.args.gc)

            # MOSE supervision loss for buffer data (MLS + RSD)
            # with autocast('cuda'):
            buf_model_outputs = [output for output, _ in buf_return_pair_list]
            # Buffer labels need to add task offset, convert to global class indices
            # buf_labels_global = buf_labels[0] + buf_start_idx  # Add task offset
            buf_mose_loss = self.compute_mose_supervision(
                buf_model_outputs, buf_labels[0], buf_inputs, kwargs_buf
            )
            loss += buf_mose_loss

            # 2. Force Python garbage collection
            import gc
            gc.collect()

            # 3. After Python GC, clear PyTorch GPU cache
            torch.cuda.empty_cache()
            # --- End of new code ---

        # ============= Step 5: Backward and Optimizer Step =============
        loss.backward()

        if (self.task_iteration + 1) % self.args.gc == 0:
            self.opt.step()
            self.opt.zero_grad()
            torch.cuda.empty_cache()

        # ============= Step 6: Buffer Update =============
        if epoch ==0:
            self.buffer.add_data(
                examples=[inputs_data],
                labels=labels,
                task_labels=task_labels
            )

        return stats

