# # Configure layers to be converted to MoE
# moe_config = {
#     'moe_layers': ['classifier', 'multi_layer1'],
#     'num_experts': 8,
#     'top_k': 2,
#     'transformer_experts': 4,
#     'transformer_top_k': 1
# }

from argparse import Namespace
from models.ms_moe_utils.moe_base import MixtureOfExperts, FFNExpert, MoEBase, SequentialExpert
import torch.nn as nn
import inspect
import torch
import logging


class DynamicMoEFactory:
    """
    Dynamic MoE factory that supports per-layer MoE configuration with two integration modes: replace and append.
    """
    
    @classmethod
    def convert_backbone(cls, backbone, config_list, args: Namespace):
        """
        Dynamically replace layers based on configuration.
        
        Args:
            backbone: The original model.
            config_list: List of configurations, each element is a MoE config dict containing:
                - layer_name: Name of the layer to replace.
                - mode: 'replace' (no use_residual) or 'append' (use_residual).
                - input_dim: Input dimension (optional, auto-detected).
                - hidden_dim: Hidden layer dimension (optional, defaults to output_dim).
                - output_dim: Output dimension (optional, auto-detected).
                - num_experts: Number of experts.
                - k: Number of selected experts.
                - num_shared_experts: Number of shared experts.
                - num_tasks: Number of tasks.
                - expert_class: Expert class name or class object.
        
        Returns:
            The modified backbone.
        """
        # Iterate over each MoE configuration
        for moe_config in config_list:
            
            layer_name = moe_config['layer_name']
            
            # Set default values
            moe_config.setdefault('num_experts', args.num_experts)
            moe_config.setdefault('k', args.top_k_experts)
            moe_config.setdefault('num_shared_experts', args.num_shared_experts)
            moe_config.setdefault('num_tasks', args.num_tasks)

                
            original_layer = getattr(backbone, layer_name)
            
            moe_layer = cls._create_generic_moe(original_layer, moe_config)
            
            # Set the new layer
            setattr(backbone, layer_name, moe_layer)
        
        return backbone
    
    @classmethod
    def _create_generic_moe(cls, original_layer, config):
        """
        Generic MoE creation function.
        
        Args:
            original_layer: The original layer.
            config: MoE configuration dictionary.
        
        Returns:
            The MoE layer.
        """
        # Create MoE layer
        if config['mode'] == 'replace':
            return cls._create_replace_mode_moe(original_layer, config)
        else:  # 'append' mode
            return cls._create_append_mode_moe(original_layer, config)
    
    @classmethod
    def _detect_dimensions(cls, layer):
        """
        Auto-detect input/output dimensions of a layer.
        
        Args:
            layer: The layer to detect dimensions for.
            
        Returns:
            tuple: (input_dim, output_dim)
        """
        input_dim = None
        output_dim = None
        
        # Detect common layer types
        if isinstance(layer, nn.Linear):
            input_dim = layer.in_features
            output_dim = layer.out_features
        elif isinstance(layer, nn.Sequential):
            # Try to find the first and last Linear layers in Sequential
            first_linear = None
            last_linear = None
            for module in layer:
                if isinstance(module, nn.Linear):
                    if first_linear is None:
                        first_linear = module
                    last_linear = module
            
            if first_linear and last_linear:
                input_dim = first_linear.in_features
                output_dim = last_linear.out_features

        elif hasattr(layer, 'input_dim') and hasattr(layer, 'output_dim'):
            # If the layer has explicit input_dim and output_dim attributes
            input_dim = layer.input_dim
            output_dim = layer.output_dim
        
        if input_dim is None or output_dim is None:
            raise ValueError(f"Cannot auto-detect dimensions for layer {layer}")
        
        logging.info(f"Detected layer dimensions: input_dim={input_dim}, output_dim={output_dim}")
        return input_dim, output_dim
    
    @classmethod
    def _create_replace_mode_moe(cls, original_layer, config):
        """
        Create a replace-mode MoE layer.
        
        Args:
            original_layer: The original layer.
            config: MoE configuration dictionary.
            
        Returns:
            The MoE layer.
        """
        # Get configuration parameters
        if 'input_dim' not in config or 'output_dim' not in config:
            input_dim, output_dim = cls._detect_dimensions(original_layer)
            config['input_dim'] = input_dim
            config['output_dim'] = output_dim
        
        input_dim = config['input_dim']
        hidden_dim = config['hidden_dim'] if 'hidden_dim' in config else config['output_dim']
        output_dim = config['output_dim']
        num_experts = config['num_experts']
        k = config['k']
        num_shared_experts = config['num_shared_experts']
        num_tasks = config['num_tasks']
        
        # Get expert class.
        # In replace mode, the expert class is the original layer
        if original_layer is not None:
            expert_class = original_layer
        else:
            expert_class = None
        
        # Create MoE layer
        return MixtureOfExperts(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            num_experts=num_experts,
            k=k,
            expert_class=expert_class,
            num_shared_experts=num_shared_experts,
            num_tasks=num_tasks
        )
    
    @classmethod
    def _create_append_mode_moe(cls, original_layer, config):
        """
        Create an append-mode MoE layer.
        
        Args:
            original_layer: The original layer.
            config: MoE configuration dictionary.
            
        Returns:
            MoE layer wrapping the original layer.
        """

        return AppendMoEWrapper(original_layer, config)


# Create wrapper class
class AppendMoEWrapper(MoEBase):
    def __init__(self, original, moe_config):
        super().__init__()
        self.original_layer = original
        if 'output_dim' not in moe_config:
            input_dim, output_dim = DynamicMoEFactory._detect_dimensions(original)
            moe_config['output_dim'] = output_dim
        else:
            output_dim = moe_config['output_dim']
        
        moe_config['input_dim'] = output_dim
        moe_config['hidden_dim'] = output_dim
        # Use default expert class configuration
        self.moe = DynamicMoEFactory._create_replace_mode_moe(
            None, moe_config
        )
    
    def forward(self, *args, **kwargs):
        # First pass through the original layer with all arguments
        original_output = self.original_layer(*args, **kwargs)
        
        # Then pass through MoE layer
        moe_output = self.moe(original_output, **kwargs)
        
        # use_residual is True
        return original_output + moe_output
        
# Usage example:
"""
# Configuration example
config = {
    'moe_configs': [
        {
            'layer_name': 'classifier',
            'mode': 'replace',
            'num_experts': 8,
            'k': 2,
            'num_shared_experts': 1,
            'num_tasks': 4,
            'expert_class': 'SequentialExpert',
            'expert_params': {
                'dropout': 0.2,
                'num_layers': 2
            }
        },
        {
            'layer_name': 'wsi_net',
            'mode': 'append',
            'input_dim': 1024,
            'hidden_dim': 2048,
            'output_dim': 512,
            'num_experts': 6,
            'k': 3
        },
        {
            'layer_name': 'multi_layer1',
            'mode': 'replace',
            'num_experts': 4,
            'k': 1
        }
    ]
}

# Apply to model
model = DynamicMoMEFactory.convert_backbone(model, config)

# Forward pass
output = model(x, task_label=current_task)
"""