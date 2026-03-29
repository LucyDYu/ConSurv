"""
This file contains the base building blocks for the LoRA for Continual Learning (LoRA4CL) model.
It includes the LoRA layer definition and the LoRALinear layer that integrates LoRA adapters
into a standard linear layer with task-based routing.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# Empty class as parent class for MoE classes
class LoraBase(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x, **kwargs):
        raise NotImplementedError("forward method not implemented")

    

class LoRALayer(LoraBase):
    """
    A single LoRA adapter layer.

    This module holds the two low-rank matrices A and B, which are trained
    for a specific task. The output of this layer is intended to be added
    to the output of a frozen pre-trained layer.
    """
    def __init__(self, in_features: int, out_features: int, rank: int, alpha: int):
        """
        Initializes the LoRALayer.

        Args:
            in_features (int): Number of input features to the original layer.
            out_features (int): Number of output features from the original layer.
            rank (int): The rank of the low-rank adaptation.
            alpha (int): The scaling factor for the LoRA output.
        """
        super().__init__()
        self.lora_A = nn.Parameter(torch.zeros(rank, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))
        self.scaling = alpha / rank
        
        # Initialize weights as in the original LoRA paper
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the LoRA layer.

        Computes `scaling * (B @ (A @ x))`.

        Args:
            x (torch.Tensor): The input tensor to the layer.

        Returns:
            torch.Tensor: The LoRA-adapted output.
        """
        # The multiplication order is B * (A * x)
        # x shape: (batch, ..., in_features)
        # lora_A: (rank, in_features) -> A @ x.T -> (rank, batch)
        # lora_B: (out_features, rank) -> B @ (A @ x.T) -> (out_features, batch)
        # Transpose back -> (batch, out_features)
        return (self.lora_B @ (self.lora_A @ x.transpose(-1, -2))).transpose(-1, -2) * self.scaling



class FFN(nn.Module):
    """
    Feed-forward network expert for LoRA.
    """
    def __init__(self, input_dim, output_dim):
        """
        Initialize the FFN expert.
        
        Args:
            input_dim (int): Dimension of input features
            output_dim (int): Dimension of output features
        """
        super(FFN, self).__init__()
        self.ffn = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.ReLU()
        )
    
    def forward(self, x):
        """
        Forward pass for the FFN expert.
        
        Args:
            x (torch.Tensor): Input tensor
            
        Returns:
            torch.Tensor: Output tensor
        """
        return self.ffn(x)



class LoRALinear(LoraBase):
    """
    A LoRA-enhanced linear layer with task-based routing.

    This module can either wrap an existing layer class or create its own
    `FFN` layer if no `original_layer_class` is provided. For the first task (task 0), 
    it trains the base layer. For subsequent tasks, the base layer is frozen, 
    and a new task-specific LoRA adapter is trained.
    """
    def __init__(self, rank: int, alpha: int, num_tasks=4, input_dim: int = None, output_dim: int = None, original_layer_class=None):
        """
        Initializes the LoRALinear layer.

        Args:
            rank (int): The rank for the LoRA adapters.
            alpha (int): The scaling factor for the LoRA adapters.
            original_layer (nn.Module, optional): An existing layer to wrap. Defaults to None.
            input_dim (int, optional): Input dimension, required if original_layer is None.
            output_dim (int, optional): Output dimension, required if original_layer is None.
        """
        super().__init__()
        self.rank = rank
        self.alpha = alpha

        if original_layer_class is not None:
            self.original_layer = original_layer_class(input_dim, output_dim)
            # Try to infer dimensions from the provided layer
            if hasattr(self.original_layer, 'in_features') and hasattr(self.original_layer, 'out_features'):
                self.in_features = self.original_layer.in_features
                self.out_features = self.original_layer.out_features
            else:
                self.in_features = input_dim
                self.out_features = output_dim
        else:
            self.original_layer = FFN(input_dim, output_dim)
            self.in_features = input_dim
            self.out_features = output_dim
        
        # A dictionary to hold LoRA adapters for each task (task_id > 0).
        self.lora_adapters = nn.ModuleDict()
        # Create LoRA modules for tasks 1 to num_tasks-1
        for i in range(1, num_tasks):
            self.lora_adapters[str(i)] = LoRALayer(input_dim, output_dim, rank, alpha)

    def forward(self, x, **kwargs) -> torch.Tensor:
        """
        Forward pass with task-based routing via kwargs.

        The `data_task_label` in kwargs determines which LoRA adapter to use.
        If the label is 0 or not present, only the base layer is used.
        """
        # if x is a tuple, use the first element
        if isinstance(x, tuple):
            x = x[0]

        squeeze_flag = False
        if x.dim() == 3 and x.shape[0] == 1:
            squeeze_flag = True
            x = x.squeeze(0) # default shape is [batch_size, feature_dim] 

        base_output = self.original_layer(x)
        result = base_output
        task_label_tensor = kwargs.get('data_task_label')
        if task_label_tensor is None:
            return base_output

        task_label = task_label_tensor.item()

        # For tasks > 0, add the output of the corresponding adapter.
        if task_label > 0:
            task_id_str = str(task_label)
            if task_id_str in self.lora_adapters:
                lora_output = self.lora_adapters[task_id_str](x)
                result = result + lora_output

        if squeeze_flag:
            result = result.unsqueeze(0)
        return result


