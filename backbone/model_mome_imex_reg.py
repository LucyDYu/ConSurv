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
        # Set eps for numerical stability
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
        # Apply normalization and cast data type
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
        # Return the first half of the processed output
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
        # Get input tensor shape
        b, seq, dim_len = x1.shape
        # Concatenate bottleneck with x2
        bottleneck = torch.cat([self.bottleneck, x2], dim=1)
        # Process through second attention layer
        bottleneck = self.attn2(bottleneck)[:,:self.n_bottlenecks, :]
        # Concatenate x1 with processed bottleneck
        x = torch.cat([x1, bottleneck], dim=1)
        # Process through first attention layer
        x = self.attn1(x)
        # Return first seq sequences of processed output
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
        # Process both inputs and add them
        return self.snn1(self.norm1(x1)) + self.snn2(self.norm2(x2)).mean(dim=1).unsqueeze(1)

class DropX2Fusion(nn.Module):
    def __init__(self, norm_layer=RMSNorm, dim=512):
        # Initialize DropX2Fusion module
        super().__init__()

    def forward(self, x1, x2):
        # Return x1 directly, ignoring x2
        return x1

def DiffSoftmax(logits, tau=1.0, hard=False, dim=-1):
    # Compute softened softmax values
    y_soft = (logits / tau).softmax(dim)
    if hard:
        # If in hard mode, apply straight-through estimation
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
        # Create first FC layer sequence
        self.fc1 = nn.Sequential(
            *[
                nn.Linear(dim, dim),
                norm_layer(dim),
                nn.GELU(),
            ]
        )
        # Create second FC layer sequence
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
        # Process inputs through two FC layers
        x1, x2 = self.fc1(x1), self.fc2(x2)
        # Compute mean and combine
        x = x1.mean(dim=1) + x2.mean(dim=1)
        # Apply differentiable softmax
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
            # If in hard mode, select the branch with highest probability
            corresponding_net_id = torch.argmax(logits, dim=1).item()
            x = self.routing_dict[corresponding_net_id](x1, x2)
        else:
            # Otherwise compute weighted average over all branches
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

    def forward(self, x, return_attn=False):
        # Apply normalization
        norm_x = self.norm(x)
        # Get attention output and weights
        attn_output, *attn_weights = self.attn(norm_x, return_attn=return_attn)
        x = self.residual_attn(x, attn_output)
        return (x, *attn_weights) if return_attn else x

class MoMETransformer_imex_reg(MammothBackbone):
    # n_classes=16，4 classes for each task
    def __init__(self, n_bottlenecks, n_classes, omic_sizes=[100, 200, 300, 400, 500, 600], 
                 model_size_wsi: str='small', model_size_omic: str='small', dropout=0.25):
        super(MoMETransformer_imex_reg, self).__init__()
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
        self.classifier = nn.Linear(size[2], n_classes)

        self.classifier_projection = nn.Sequential(nn.Linear(n_classes, n_classes, bias=False),
                                                   nn.LayerNorm(n_classes),
                                                   nn.ReLU(inplace=True),
                                                   nn.Linear(n_classes, n_classes, bias=False))
        # --- Self-Supervised Learning / IMEX Modules (inspired by ResNet18.py) ---
        # A linear layer before the projection head
        self.ssl_linear = nn.Linear(size[2], 1000)
        
        # A projection head 'g' to project features into a latent space for consistency loss
        self.g = nn.Sequential(
            nn.Linear(1000, 512, bias=False),
            nn.LayerNorm(512),  # Replaced BN with LayerNorm since batch size is 1
            nn.ReLU(inplace=True),
            nn.Linear(512, 128, bias=True),
            nn.LayerNorm(128)
        ).to(self.device)

        # self.classifiers = nn.ModuleList()  # Create independent classifier for each task
        # for _ in range(n_tasks):  # n_tasks is the total number of tasks
        #     self.classifiers.append(nn.Linear(size[2], n_classes))

    def forward(self, returnt='out', **kwargs):
        # Get input data
        task_label = kwargs['data_task_label']
        patch_index = kwargs['patch_index'] if 'patch_index' in kwargs else None
        
        x_path = kwargs['x_path']  # Pathology image features, shape [batch_size, micro_batch_size, 1024]. micro_batch_size is the number of patches, may vary per input
        # Get 6 different omics data features, each with different dimensions
        x_omic = [kwargs['x_omic%d' % i] for i in range(1,7)]  # Contains genomic, transcriptomic and other multi-omics data
        
        # Process pathology features through FC layer, convert to [batch_size, 512], batch_size is typically 1
        h_path_bag = self.wsi_net(x_path) ### path embeddings are fed through a FC layer

        # Process multi-omics data
        # Each omics data goes through independent FC network for feature extraction and dimension conversion
        h_omic = [self.sig_networks[idx].forward(sig_feat) for idx, sig_feat in enumerate(x_omic)] ### each omic signature goes through it's own FC layer
        # Stack 6 omics features along dimension 0, forming [6, batch_size, 512] tensor
        h_omic_bag = torch.stack(h_omic) ### omic embeddings are stacked (to be used in co-attention)

        # Add batch dimension for MoME layer processing
        # Convert shape from [micro_batch_size, 512] to [1, micro_batch_size, 512]
        h_path_bag = h_path_bag.unsqueeze(0)
        h_omic_bag = h_omic_bag.unsqueeze(0)

        # First layer MoME multimodal fusion
        # MoME_patho1: Fuse pathology features with omics features, enhancing pathology representation
        h_path_bag = self.MoME_patho1(h_path_bag, h_omic_bag, hard=True)
        # MoME_genom1: Fuse omics features with pathology features, enhancing omics representation
        h_omic_bag = self.MoME_genom1(h_omic_bag, h_path_bag, hard=True)

        # Second layer MoME multimodal fusion, further extracting high-level features
        h_path_bag = self.MoME_patho2(h_path_bag, h_omic_bag, hard=True, **kwargs)
        h_omic_bag = self.MoME_genom2(h_omic_bag, h_path_bag, hard=True, **kwargs)

        # Remove extra batch dimension
        # Convert shape from [1, micro_batch_size, 512] back to [micro_batch_size, 512]
        h_path_bag = h_path_bag.squeeze()
        h_omic_bag = h_omic_bag.squeeze()
        
        # Multimodal feature fusion
        # 1. Concatenate CLS token, pathology features and omics features along dimension 0
        # 2. Add batch dimension for transformer processing
        # h_multi.shape before unsqueeze = [1+micro_batch_size+6, 512]
        h_multi = torch.cat([self.cls_multimodal, h_path_bag, h_omic_bag], dim=0).unsqueeze(0)
        # Process through transformer layer and extract CLS token features as final representation 
        h_trans, attn_weights = self.multi_layer1(h_multi, return_attn=True)
        h = h_trans[:,0,:]
        
        # Store attention weights for later use
        attn_weights = attn_weights[:, :, :h_multi.shape[1], :h_multi.shape[1]]
        attention_scores = {
            'attention_map': attn_weights,
        }
        kwargs['attention_scores'] = attention_scores
        
        # Return last layer features
        if returnt == 'features':
            return h, attention_scores

        # Get logits through classifier
        hazards, S, Y_hat, attention_scores, logits = self.forward_features(h, **kwargs)

        if returnt == 'logits':
            return logits

        # If full features needed, return all intermediate layer results
        if returnt == 'full':
        # Return all intermediate layer features
            return {
                'x_path_input_index': patch_index,
                'x_path_input': x_path,
                'x_omic_input': x_omic,
                'path': h_path_bag,  # Pathology image feature representation
                'omic': h_omic_bag,  # Multi-omics data feature representation
                'fusion': h,         # Fused multimodal features
                'logits': logits,     # Raw model prediction scores
                'attention_scores': attention_scores,
                'origin_output': (hazards, S, Y_hat, attention_scores, logits)
            }
        
        # --- IMEX Regularization Forward Path ---
        # If 'imex' is requested, process both original and augmented features.
        if returnt == 'imex':
            out=h
            out_ssl = self.g(self.ssl_linear(out))
            return hazards, S, Y_hat, attention_scores, logits, F.normalize(out_ssl), F.normalize(self.classifier_projection(logits))

        # default returnt == 'out'
        return hazards, S, Y_hat, attention_scores, logits

    def forward_features(self, features, **kwargs):
        h = features
        task_label = kwargs['data_task_label']
        attention_scores = kwargs['attention_scores']
        ### Survival prediction layer
        # 1. Get logits through classifier
        if isinstance(self.classifier, MoEBase):
            full_logits = self.classifier(h, **kwargs)
        else:
            full_logits = self.classifier(h)
        # full_logits = self.classifier(h)
        
        # logits = self.classifiers[task_label](h)

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
            current_logits = full_logits  # If not task-il, use the full logits directly

        # 2. Predict the most probable class
        Y_hat = torch.topk(current_logits, 1, dim = 1)[1]
        # 3. Compute hazard probability at each time point
        hazards = torch.sigmoid(current_logits)
        # 4. Compute survival probability (cumulative probability)
        S = torch.cumprod(1 - hazards, dim=1)
        
        # For storing attention scores
        # attention_scores = {}

        # default returnt == 'out'
        return hazards, S, Y_hat, attention_scores, full_logits
        


# n_classes=16，4 classes for each task, depend on config
@register_backbone("mome_imex_reg")
def mome_imex_reg(n_bottlenecks=2, n_classes=16, omic_sizes=[100, 200, 300, 400, 500, 600], 
         model_size_wsi: str='small', model_size_omic: str='small', dropout=0.25) -> MoMETransformer_imex_reg:
    """Register MoME model as a backbone
    
    Args:
        omic_sizes (list): List of omic feature sizes
        n_classes (int): Number of output classes
        n_bottlenecks (int): Number of bottleneck tokens
        model_size_wsi (str): Size of WSI branch ('small' or 'big')
        model_size_omic (str): Size of omic branch ('small' or 'big') 
        dropout (float): Dropout rate
    
    Returns:
        MoMETransformer_imex_reg: Initialized MoME model
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
    return MoMETransformer_imex_reg(**model_dict)
