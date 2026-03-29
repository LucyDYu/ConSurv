import torch
from torch import linalg as LA
import torch.nn.functional as F
import torch.nn as nn

from backbone import MammothBackbone, register_backbone
from backbone.model_utils import *
from nystrom_attention import NystromAttention
import admin_torch
import logging

from models.ms_moe_utils.moe_base import MoEBase


def align_wsi_features(output_dict):
    """
    Align WSI multi-level features to unified dimension [512]

    Args:
        output_dict: Dictionary containing path, omic, fusion features
            - path: [micro_batch_size, 512] variable length
            - omic: [6, 512] fixed 6 modalities
            - fusion: [1, 512] single CLS token

    Returns:
        aligned_dict: Aligned feature dictionary, all features shaped as [512]
    """
    aligned = {}

    # Path features: [micro_batch_size, 512] → [512]
    path_feat = output_dict['path']
    if len(path_feat.shape) > 1 and path_feat.shape[0] > 1:
        aligned['path'] = path_feat.mean(0)  # Average multiple patches
    else:
        aligned['path'] = path_feat.squeeze()

    # Omic features: [6, 512] → [512]
    omic_feat = output_dict['omic']
    if len(omic_feat.shape) > 1:
        aligned['omic'] = omic_feat.mean(0)  # Average 6 modalities
    else:
        aligned['omic'] = omic_feat.squeeze()

    # Fusion features: [1, 512] → [512]
    fusion_feat = output_dict['fusion']
    aligned['fusion'] = fusion_feat.squeeze()

    return aligned


class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        """
        Initialize the RMSNorm normalization layer.

        Args:
            dim (int): The dimension of the input tensor.
            eps (float, optional): A small value added to the denominator for numerical stability. Default is 1e-6.

        Attributes:
            eps (float): A small value added to the denominator for numerical stability.
            weight (nn.Parameter): Learnable scaling parameter.

        """
        super().__init__()
        # Set eps parameter for numerical stability
        self.eps = eps
        # Create learnable weight parameter
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        """
        Apply the RMSNorm normalization to the input tensor.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The normalized tensor.

        """
        # Compute RMSNorm normalization
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        """
        Forward pass through the RMSNorm layer.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor after applying RMSNorm.

        """
        # Perform normalization and convert data type
        output = self._norm(x.float()).type_as(x)
        # Apply learnable weight and return
        return output * self.weight

class TransFusion(nn.Module):
    def __init__(self, norm_layer=RMSNorm, dim=512):
        # Initialize TransFusion module
        super().__init__()
        # Create TransLayer
        self.translayer = TransLayer(norm_layer, dim)

    def forward(self, x1, x2):
        # Concatenate two input tensors along dimension 1
        x = torch.cat([x1, x2], dim=1)
        # Process through TransLayer
        x = self.translayer(x)
        # Return the first half after processing
        return x[:, :x1.shape[1], :]

class BottleneckTransFusion(nn.Module):
    def __init__(self, n_bottlenecks, norm_layer=RMSNorm, dim=512):
        # Initialize BottleneckTransFusion module
        super().__init__()
        # Create normalization layer
        self.norm = norm_layer(dim)
        # Set number of bottlenecks
        self.n_bottlenecks = n_bottlenecks
        # Create two TransLayers
        self.attn1 = TransLayer(nn.LayerNorm, dim=dim)
        self.attn2 = TransLayer(nn.LayerNorm, dim=dim)
        # Create randomly initialized bottleneck parameters
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.bottleneck = torch.rand((1,n_bottlenecks,dim)).to(self.device)

    def forward(self, x1, x2):
        # Get input tensor shapes
        b, seq, dim_len = x1.shape
        # Concatenate bottleneck and x2
        bottleneck = torch.cat([self.bottleneck, x2], dim=1)
        # Process through second attention layer
        bottleneck = self.attn2(bottleneck)[:,:self.n_bottlenecks, :]
        # Concatenate x1 and processed bottleneck
        x = torch.cat([x1, bottleneck], dim=1)
        # Process through first attention layer
        x = self.attn1(x)
        # Return the first seq sequences after processing
        return x[:, :seq, :]

class AddFusion(nn.Module):
    def __init__(self, norm_layer=RMSNorm, dim=512):
        # Initialize AddFusion module
        super().__init__()
        # Create two SNN blocks
        self.snn1 = SNN_Block(dim1=dim, dim2=dim)
        self.snn2 = SNN_Block(dim1=dim, dim2=dim)
        # Create two normalization layers
        self.norm1 = norm_layer(dim)
        self.norm2 = norm_layer(dim)

    def forward(self, x1, x2):
        # Process both inputs and add them together
        return self.snn1(self.norm1(x1)) + self.snn2(self.norm2(x2)).mean(dim=1).unsqueeze(1)

class DropX2Fusion(nn.Module):
    def __init__(self, norm_layer=RMSNorm, dim=512):
        # Initialize DropX2Fusion module
        super().__init__()

    def forward(self, x1, x2):
        # Return x1 directly, ignore x2
        return x1

def DiffSoftmax(logits, tau=1.0, hard=False, dim=-1):
    # Compute softened softmax values
    y_soft = (logits / tau).softmax(dim)
    if hard:
        # If using hard mode, perform straight-through estimation
        index = y_soft.max(dim, keepdim=True)[1]
        y_hard = torch.zeros_like(logits, memory_format=torch.legacy_contiguous_format).scatter_(dim, index, 1.0)
        ret = y_hard - y_soft.detach() + y_soft
    else:
        # Otherwise use reparameterization trick
        ret = y_soft
    return ret

class RoutingNetwork(nn.Module):
    def __init__(self, branch_num, norm_layer=RMSNorm, dim=256):
        # Initialize routing network
        super(RoutingNetwork, self).__init__()
        # Set number of branches
        self.bnum = branch_num
        # Create first fully connected layer sequence
        self.fc1 = nn.Sequential(
            *[
                nn.Linear(dim, dim),
                norm_layer(dim),
                nn.GELU(),
            ]
        )
        # Create second fully connected layer sequence
        self.fc2 = nn.Sequential(
            *[
                nn.Linear(dim, dim),
                norm_layer(dim),
                nn.GELU(),
            ]
        )
        # Create classifier layer
        self.clsfer = nn.Linear(dim, branch_num)

    def forward(self, x1, x2, temp=1.0, hard=False):
        # Process inputs through two fully connected layers
        x1, x2 = self.fc1(x1), self.fc2(x2)
        # Compute mean and merge
        x = x1.mean(dim=1) + x2.mean(dim=1)
        # Apply differential softmax
        logits = DiffSoftmax(self.clsfer(x), tau=temp, hard=hard, dim=1)
        return logits

class MoME(nn.Module):
    def __init__(self, n_bottlenecks, norm_layer=RMSNorm, dim=256,):
        # Initialize MoME module
        super().__init__()
        # Create various fusion modules
        self.TransFusion = TransFusion(norm_layer, dim)
        self.BottleneckTransFusion = BottleneckTransFusion(n_bottlenecks, norm_layer, dim)
        self.AddFusion = AddFusion(norm_layer, dim)
        self.DropX2Fusion = DropX2Fusion(norm_layer, dim)
        # Create routing network
        self.routing_network = RoutingNetwork(4, dim=dim)
        # Create routing dictionary
        self.routing_dict = {
            0: self.TransFusion,
            1: self.BottleneckTransFusion,
            2: self.AddFusion,
            3: self.DropX2Fusion,
        }

    def forward(self, x1, x2, hard=False, **kwargs):
        # Get routing logits
        logits = self.routing_network(x1, x2, hard)
        if hard:
            # If hard mode, select the branch with maximum probability
            corresponding_net_id = torch.argmax(logits, dim=1).item()
            x = self.routing_dict[corresponding_net_id](x1, x2)
        else:
            # Otherwise perform weighted average over all branches
            x = torch.zeros_like(x1)
            for branch_id, branch in self.routing_dict.items():
                x += branch(x1, x2)
        return x

class TransLayer(nn.Module):
    def __init__(self, norm_layer=nn.LayerNorm, dim=512):
        # Initialize TransLayer module
        super().__init__()
        # Create normalization layer
        self.norm = norm_layer(dim)
        # Create residual attention module
        self.residual_attn = admin_torch.as_module(8)
        # Create Nystrom attention layer
        self.attn = NystromAttention(
            dim = dim,
            dim_head = dim//8,
            heads = 8,
            num_landmarks = dim//2,    # number of landmarks
            pinv_iterations = 6,    # number of moore-penrose iterations for approximating pinverse. 6 was recommended by the paper
            residual = True,         # whether to do an extra residual with the value or not. supposedly faster convergence if turned on
            dropout = 0.1
        )

    def forward(self, x, return_attn=False, **kwargs):
        # Apply normalization
        norm_x = self.norm(x)
        # Get attention output and weights
        attn_output, *attn_weights = self.attn(norm_x, return_attn=return_attn)
        x = self.residual_attn(x, attn_output)
        return (x, *attn_weights) if return_attn else x

class MoMETransformer(MammothBackbone):
    # n_classes=16, 4 classes for each task
    def __init__(self, n_bottlenecks, n_classes, omic_sizes=[100, 200, 300, 400, 500, 600],
                 model_size_wsi: str='small', model_size_omic: str='small', dropout=0.25):
        super(MoMETransformer, self).__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.omic_sizes = omic_sizes
        self.n_classes = n_classes
        self.size_dict_WSI = {"small": [1024, 512, 512], "big": [1024, 512, 384]}
        self.size_dict_omic = {'small': [512, 512], 'big': [1024, 1024, 1024, 256]}

        ### Constructing Genomic SNN
        hidden = self.size_dict_omic[model_size_omic]
        sig_networks = []
        for input_dim in omic_sizes:
            fc_omic = [SNN_Block(dim1=input_dim, dim2=hidden[0])]
            for i, _ in enumerate(hidden[1:]):
                fc_omic.append(SNN_Block(dim1=hidden[i], dim2=hidden[i+1], dropout=dropout))
            sig_networks.append(nn.Sequential(*fc_omic))
        self.sig_networks = nn.ModuleList(sig_networks)

        ### FC Layer over WSI bag
        size = self.size_dict_WSI[model_size_wsi]
        fc = [nn.Linear(size[0], size[1]), nn.ReLU()]
        fc.append(nn.Dropout(dropout))
        self.wsi_net = nn.Sequential(*fc)

        ### MoMEs
        self.MoME_genom1 = MoME(n_bottlenecks=n_bottlenecks, dim=size[2])
        self.MoME_patho1 = MoME(n_bottlenecks=n_bottlenecks, dim=size[2])
        self.MoME_genom2 = MoME(n_bottlenecks=n_bottlenecks, dim=size[2])
        self.MoME_patho2 = MoME(n_bottlenecks=n_bottlenecks, dim=size[2])

        ### Classifier
        self.multi_layer1 = TransLayer(dim=size[2])
        self.cls_multimodal = torch.rand((1, size[2])).to(self.device)

        # ✅ Multi-level classifiers for MOSE-WSI (corresponding to MOSE's self.linear)
        # Add classifiers for path, omic, fusion modalities for multi-level supervision
        self.classifiers = nn.ModuleDict({
            'path': nn.Linear(size[2], n_classes),   # path modality classifier
            'omic': nn.Linear(size[2], n_classes),   # omic modality classifier
            'fusion': nn.Linear(size[2], n_classes)  # fusion modality classifier (main classifier)
        })

        # ✅ Projection heads for MOSE-WSI contrastive learning (corresponding to MOSE's simclr layer)
        # Add projection heads for path, omic, fusion modalities: 512→128
        self.projection_heads = nn.ModuleDict({
            'path': nn.Linear(size[2], 128),   # path modality projection
            'omic': nn.Linear(size[2], 128),   # omic modality projection
            'fusion': nn.Linear(size[2], 128)  # fusion modality projection
        })

    def forward(self, returnt='out', **kwargs):
        # Get input data
        task_label = kwargs['data_task_label']
        patch_index = kwargs['patch_index'] if 'patch_index' in kwargs else None

        x_path = kwargs['x_path']  # Pathology image features, shape [batch_size, micro_batch_size, 1024]. micro_batch_size is number of patches, may vary per input
        # Get 6 different omic data features, each with different dimensions
        x_omic = [kwargs['x_omic%d' % i] for i in range(1,7)]  # Includes genomic, transcriptomic and other multi-omic data

        # Process pathology image features through FC layer, convert dimension to [batch_size, 512], batch_size is usually 1
        h_path_bag = self.wsi_net(x_path) ### path embeddings are fed through a FC layer

        # Process multi-omic data
        # Each omic data goes through its own FC network for feature extraction and dimension conversion
        h_omic = [self.sig_networks[idx].forward(sig_feat) for idx, sig_feat in enumerate(x_omic)] ### each omic signature goes through it's own FC layer
        # Stack 6 omic features along dimension 0, forming tensor [6, batch_size, 512]
        h_omic_bag = torch.stack(h_omic) ### omic embeddings are stacked (to be used in co-attention)

        # Add batch dimension for MoME layer processing
        # Convert shape from [micro_batch_size, 512] to [1, micro_batch_size, 512]
        h_path_bag = h_path_bag.unsqueeze(0)
        h_omic_bag = h_omic_bag.unsqueeze(0)

        # First layer MoME multimodal fusion
        # MoME_patho1: Fuse pathology features with omic features to enhance pathology feature representation
        h_path_bag = self.MoME_patho1(h_path_bag, h_omic_bag, hard=True)
        # MoME_genom1: Fuse omic features with pathology features to enhance omic feature representation
        h_omic_bag = self.MoME_genom1(h_omic_bag, h_path_bag, hard=True)

        # Second layer MoME multimodal fusion for higher-level feature extraction
        h_path_bag = self.MoME_patho2(h_path_bag, h_omic_bag, hard=True, **kwargs)
        h_omic_bag = self.MoME_genom2(h_omic_bag, h_path_bag, hard=True, **kwargs)

        # Remove extra batch dimension
        # Convert shape from [1, micro_batch_size, 512] back to [micro_batch_size, 512]
        h_path_bag = h_path_bag.squeeze()
        h_omic_bag = h_omic_bag.squeeze()

        # Multimodal feature fusion
        # 1. Concatenate CLS token, pathology features and omic features along dimension 0
        # 2. Add batch dimension for transformer processing
        # h_multi.shape before unsqueeze = [1+micro_batch_size+6, 512]
        h_multi = torch.cat([self.cls_multimodal, h_path_bag, h_omic_bag], dim=0).unsqueeze(0)
        # Process through transformer layer and extract CLS token position features as final representation
        h_trans, attn_weights = self.multi_layer1(h_multi, return_attn=True, **kwargs)
        h = h_trans[:,0,:]

        # Store attention weights for subsequent use
        attn_weights = attn_weights[:, :, :h_multi.shape[1], :h_multi.shape[1]]
        attention_scores = {
            'attention_map': attn_weights,
        }
        kwargs['attention_scores'] = attention_scores

        # Return final layer features
        if returnt == 'features':
            return h, attention_scores

        # Get logits through classifier
        # Prepare all expert features for MOE-MOSE mode
        all_features = {
            'path': h_path_bag,
            'omic': h_omic_bag,
            'fusion': h
        }
        hazards, S, Y_hat, attention_scores, logits = self.forward_features(h, all_features=all_features, **kwargs)

        if returnt == 'logits':
            return logits

        # If full features are needed, return all intermediate results
        if returnt == 'full':
        # Return all intermediate layer features
            return {
                'x_path_input_index': patch_index,
                'x_path_input': x_path,
                'x_omic_input': x_omic,
                'path': h_path_bag,  # Pathology image feature representation
                'omic': h_omic_bag,  # Multi-omic data feature representation
                'fusion': h,         # Fused multimodal features
                'logits': logits,     # Model's raw prediction scores
                'attention_scores': attention_scores,
                'origin_output': (hazards, S, Y_hat, attention_scores, logits)
            }

        # default returnt == 'out'
        return hazards, S, Y_hat, attention_scores, logits

    def forward_features(self, features, all_features=None, expert_name='fusion', **kwargs):
        """
        Forward propagation features, support selecting different expert outputs

        Args:
            features: Input features (usually fusion features)
            all_features: All expert features dictionary {'path': ..., 'omic': ..., 'fusion': ...}
            expert_name: Select which expert to use ('path', 'omic', 'fusion'), default 'fusion'
            **kwargs: Other parameters

        Returns:
            hazards, S, Y_hat, attention_scores, full_logits
        """
        h = features
        task_label = kwargs['data_task_label']
        attention_scores = kwargs.get('attention_scores', {})

        ### Survival analysis prediction layer
        # Check if MOE-MOSE mode (average all expert outputs)
        # MOE-MOSE only used during inference, normal supervision signals used during training
        # Prioritize getting moe_mode directly from kwargs
        moe_mode = kwargs.get('moe_mode')
        # If not in kwargs, get from kwargs['args'] (Namespace object)
        if moe_mode is None and 'args' in kwargs:
            # If no 'args' in kwargs, or no moe_mode in args, will raise error directly (avoid silent failure)
            moe_mode = kwargs['args'].moe_mode

        if moe_mode == 'ensemble' and not self.training and all_features is not None:
            # MOE-MOSE mode: Average outputs from path, omic, fusion three experts
            # First align feature dimensions to ensure all are [512] shape
            aligned_features = align_wsi_features(all_features)

            expert_logits = {}
            for exp_name in ['path', 'omic', 'fusion']:
                classifier = self.classifiers[exp_name]
                feat = aligned_features[exp_name]  # Use aligned features
                if isinstance(classifier, MoEBase):
                    expert_logits[exp_name] = classifier(feat, **kwargs)
                else:
                    expert_logits[exp_name] = classifier(feat)

            # Average logits from three experts
            full_logits = torch.stack([expert_logits['path'], expert_logits['omic'], expert_logits['fusion']], dim=0).mean(dim=0).unsqueeze(0)

        elif expert_name in ['path', 'omic'] and all_features is not None:
            # Use specified path or omic expert
            aligned_features = align_wsi_features(all_features)
            classifier = self.classifiers[expert_name]
            feat = aligned_features[expert_name]  # [512]

            # Ensure feature has batch dimension
            if feat.dim() == 1:
                feat = feat.unsqueeze(0)  # [1, 512]

            if isinstance(classifier, MoEBase):
                full_logits = classifier(feat, **kwargs)
            else:
                full_logits = classifier(feat)

        else:
            # MOSE mode: Use fusion classifier (default behavior)
            fusion_classifier = self.classifiers['fusion']
            if isinstance(fusion_classifier, MoEBase):
                full_logits = fusion_classifier(h, **kwargs)
            else:
                full_logits = fusion_classifier(h)

        args = kwargs['args']
        if args.scenario == 'task-il':

            start_idx = kwargs['start_idx']
            end_idx = kwargs['end_idx']

            # manual calculation for checking
            data_task_label = kwargs['data_task_label'].item()
            start_idx_manual = data_task_label* args.n_bins
            end_idx_manual = (data_task_label+1)* args.n_bins
            assert start_idx_manual == start_idx and end_idx_manual == end_idx, f"start_idx {start_idx}, start_idx_manual {start_idx_manual} and end_idx {end_idx}, end_idx_manual {end_idx_manual}"

            current_logits = full_logits[:, start_idx: end_idx]
        else:
            current_logits = full_logits  # If not task-il, use complete logits directly

        # 2. Predict most likely class
        Y_hat = torch.topk(current_logits, 1, dim = 1)[1]
        # 3. Calculate risk probability at each time point
        hazards = torch.sigmoid(current_logits)
        # 4. Calculate survival probability (cumulative probability)
        S = torch.cumprod(1 - hazards, dim=1)

        # For storing attention scores
        # attention_scores = {}

        # default returnt == 'out'
        return hazards, S, Y_hat, attention_scores, full_logits

    def head(self, features_dict, use_proj=False):
        """
        MOSE-WSI head method, support projection and classification predictions
        Corresponding to MOSE original head(self, x: List[torch.Tensor], use_proj=False)

        Args:
            features_dict: Feature dictionary {'path': [n, 512], 'omic': [n, 512], 'fusion': [n, 512]}
            use_proj: Whether to use projection head
                     True: Return 128-dim projection for contrastive learning
                     False: Return n_classes-dim classification logits for CE loss

        Returns:
            output_dict: Output dictionary {'path': [...], 'omic': [...], 'fusion': [...]}
                        If use_proj=True, output 128-dim projection
                        If use_proj=False, output n_classes-dim classification logits
        """
        if use_proj:
            # Use projection heads: 512 → 128 (for contrastive learning)
            output_dict = {
                'path': self.projection_heads['path'](features_dict['path']),
                'omic': self.projection_heads['omic'](features_dict['omic']),
                'fusion': self.projection_heads['fusion'](features_dict['fusion'])
            }
        else:
            # Use classifiers: 512 → n_classes (for classification loss)
            output_dict = {
                'path': self.classifiers['path'](features_dict['path']),
                'omic': self.classifiers['omic'](features_dict['omic']),
                'fusion': self.classifiers['fusion'](features_dict['fusion'])
            }

        return output_dict



# n_classes=16, 4 classes for each task, depend on config
@register_backbone("mome_mose")
def mome(n_bottlenecks=2, n_classes=16, omic_sizes=[100, 200, 300, 400, 500, 600],
         model_size_wsi: str='small', model_size_omic: str='small', dropout=0.25) -> MoMETransformer:
    """Register MoME model as a backbone

    Args:
        omic_sizes (list): List of omic feature sizes
        n_classes (int): Number of output classes
        n_bottlenecks (int): Number of bottleneck tokens
        model_size_wsi (str): Size of WSI branch ('small' or 'big')
        model_size_omic (str): Size of omic branch ('small' or 'big')
        dropout (float): Dropout rate

    Returns:
        MoMETransformer: Initialized MoME model
    """

    model_dict = {
        'omic_sizes': omic_sizes,
        'n_classes': n_classes,
        'n_bottlenecks': n_bottlenecks,
        'model_size_wsi': model_size_wsi,
        'model_size_omic': model_size_omic,
        'dropout': dropout
    }
    logging.info(f"Initializing MoME model with the following parameters:")
    logging.info(f"Model dict: {model_dict}")
    return MoMETransformer(**model_dict)
