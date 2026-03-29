"""
This file contains the DynamicLoRAFactory, a utility class for dynamically
converting standard `nn.Linear` layers within a given model into LoRA-enhanced
layers, based on a provided configuration list. It supports both 'replace'
and 'append' modes for integrating LoRA.
"""
import torch.nn as nn
from argparse import Namespace
from .lora4cl_base import LoRALinear, FFN, LoraBase


class DynamicLoRAFactory:
    """
    A factory to dynamically replace or append LoRA modules to layers of a model.
    It reads a configuration list and applies LoRA modifications accordingly.
    """

    @classmethod
    def convert_backbone(cls, backbone, lora_config_list, args: Namespace):
        """
        Dynamically converts layers of the backbone model based on a configuration list.

        Args:
            backbone (nn.Module): The original model.
            args (Namespace): Global arguments, expected to contain `lora_rank`, `lora_alpha`, and `num_tasks`.
            lora_config_list (list): A list of dictionaries, each specifying a LoRA modification.
                - 'layer_name' (str): The name of the layer to modify.
                - 'mode' (str): 'replace' or 'append'.

        Returns:
            nn.Module: The modified backbone.
        """
        for config in lora_config_list:
            layer_name = config['layer_name']
            mode = config['mode']

            # Set default values
            config.setdefault('num_tasks', args.num_tasks)
            config.setdefault('lora_rank', args.lora_rank)
            config.setdefault('lora_alpha', args.lora_alpha)

            original_layer = getattr(backbone, layer_name)
            
            lora_layer = cls._create_generic_lora(original_layer, config)
            
            # Set the new layer
            setattr(backbone, layer_name, lora_layer)
            print(f"  - Successfully applied LoRA (mode: {mode}) to '{layer_name}'.")
        
        return backbone

    @classmethod
    def _create_generic_lora(cls, original_layer, config):
        """
        Generic LoRA creation function.
        
        Args:
            original_layer: The original layer to be adapted.
            config: LoRA configuration dictionary.
        
        Returns:
            The LoRA-adapted layer.
        """
        # Create LoRA layer
        if config['mode'] == 'replace':
            return cls._create_replace_mode_lora(original_layer, config)
        else:  # 'append' mode
            return cls._create_append_mode_lora(original_layer, config)

    @classmethod
    def _create_replace_mode_lora(cls, original_layer: nn.Module, config) -> LoRALinear:
        """
        Creates a LoRALinear module to completely replace the original layer.
        It uses the original layer's class as the base for the new LoRA layer.
        """
        if not isinstance(original_layer, nn.Linear):
            raise TypeError(f"Replace mode is only supported for nn.Linear, but got {type(original_layer).__name__}.")

        input_dim = original_layer.in_features
        output_dim = original_layer.out_features
        
        lora_layer = LoRALinear(
            rank=config['lora_rank'],
            alpha=config['lora_alpha'],
            num_tasks=config['num_tasks'],
            input_dim=input_dim,
            output_dim=output_dim,
            original_layer_class=type(original_layer) # Pass the class, e.g., nn.Linear
        )
        
        # Copy original weights to the new base layer within LoRALinear
        lora_layer.base_layer.weight.data.copy_(original_layer.weight.data)
        if original_layer.bias is not None:
            lora_layer.base_layer.bias.data.copy_(original_layer.bias.data)
            
        return lora_layer

    @classmethod
    def _create_append_mode_lora(cls, original_layer: nn.Module, config) :
        """
        Creates a wrapper that appends a LoRA module after the original layer.
        """
        return AppendLoRAWrapper(original_layer, config)

    

class AppendLoRAWrapper(LoraBase):
    """
    A wrapper to append a LoRALinear module after an original layer.
    The output of LoRA is added to the output of the original layer (residual connection).
    """
    def __init__(self, original_layer: nn.Module, config):
        super().__init__()
        self.original_layer = original_layer

        # The LoRALinear module in append mode will use its own FFN as a base,
        # but the input to that FFN is the output of the original layer.
        self.lora_module = LoRALinear(
            rank=config['lora_rank'],
            alpha=config['lora_alpha'],
            num_tasks=config['num_tasks'],
            input_dim=config['input_dim'],
            output_dim=config['output_dim'], # LoRA output must match original output for addition
            original_layer_class=None # Default to FFN
        )

    def forward(self, *args, **kwargs):
        """
        Forward pass: computes original_output + lora_output.
        """
        original_output = self.original_layer(*args, **kwargs)
        
        # Pass the output of the original layer to the LoRA module
        lora_output = self.lora_module(original_output, **kwargs)
        
        # The LoRALinear's forward handles the addition of its base layer (FFN) and adapter.
        # Here we add the original layer's output to the full LoRA module's output.
        # This creates: Y = Orig(X) + FFN(Orig(X)) + LoRA(Orig(X))
        # This is a powerful, doubly residual connection.
        if isinstance(original_output, tuple):
            # Create new tuple, replacing the first element
            result = (original_output[0] + lora_output,) + original_output[1:]
        else:
            result = original_output + lora_output
        return result

