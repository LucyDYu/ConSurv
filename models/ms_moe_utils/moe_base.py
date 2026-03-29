from copy import deepcopy
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# Empty class as parent class for MoE classes
class MoEBase(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x, **kwargs):
        raise NotImplementedError("forward method not implemented")

        
class SparseDispatcher(nn.Module):
    """
    Sparse Dispatcher for Mixture of Experts.
    Routes the input to top-k experts and combines their outputs.
    """
    def __init__(self, num_experts, k, num_shared_experts=0):
        """
        Initialize the SparseDispatcher.
        
        Args:
            num_experts (int): Number of experts in the model
            k (int): Number of experts to be selected 
            num_shared_experts (int): Number of experts that are always selected
        """
        super(SparseDispatcher, self).__init__()
        self.num_experts = num_experts
        self.num_shared_experts = min(num_shared_experts, num_experts)
        # Adjust k to account for shared experts
        assert k >= num_shared_experts, "k must be greater than or equal to num_shared_experts"
        # fewer experts to select
        self.k_select = k-self.num_shared_experts
        self.k = k
        
    def forward(self, gates):
        """
        Forward pass for the dispatcher.
        
        Args:
            gates (torch.Tensor): Tensor of shape [batch_size, num_experts] containing gate values
            
        Returns:
            torch.Tensor: Combined output from selected experts
        """
        # Prepare to combine the outputs
        batch_size = gates.size(0)
        
        # Get the top-k experts for each sample
        # Keep gate values for non-shared experts; shared ones are at the end, not affecting indices
        end_idx = self.num_experts - self.num_shared_experts
        non_shared_gates = gates[..., :end_idx]
        
        if self.k_select > 0 and end_idx > 0:
            # If there are non-shared experts to select
            top_k_gates, top_k_indices = torch.topk(non_shared_gates, min(self.k_select, end_idx), dim=1)
            
            if self.num_shared_experts > 0:
                # If there are shared experts
                shared_gates = gates[...,  end_idx:]
                # Merge gate values of shared and selected non-shared experts
                top_k_gates_shared = torch.cat([top_k_gates, shared_gates], dim=1)
                # Merge indices of shared and selected non-shared experts
                shared_indices = torch.arange(end_idx, self.num_experts, device=gates.device).expand(batch_size, -1)
                top_k_indices_shared = torch.cat([top_k_indices, shared_indices], dim=1)
            else:
                # No shared experts
                top_k_gates_shared = top_k_gates
                top_k_indices_shared = top_k_indices
        else:
            # Only shared experts or k_select is 0
            top_k_gates_shared = gates[:, end_idx:]
            top_k_indices_shared = torch.arange(end_idx, self.num_experts, device=gates.device).expand(batch_size, -1)
        
        # Normalize the gate values for the top-k experts
        top_k_gates_shared = F.softmax(top_k_gates_shared, dim=1)
        
        return top_k_indices_shared, top_k_gates_shared

class MixtureOfExperts(MoEBase):
    """
    Mixture of Experts (MoE) module with sparse routing.
    """
    def __init__(self, input_dim, hidden_dim=None, output_dim=None, num_experts=None, k=2, 
                 expert_class=None, num_shared_experts=0, num_tasks=4):
        """
        Initialize the MoE module.
        
        Args:
            input_dim (int): Dimension of input features
            hidden_dim (int): Dimension of hidden layer in experts
            output_dim (int): Dimension of output features
            num_experts (int): Number of experts
            k (int): Number of experts to route each input to
            use_residual (bool): Whether to use residual connection
            expert_class (type or callable, optional): Expert module class or factory function
            num_shared_experts (int): Number of experts that are always selected
            num_tasks (int): Number of tasks for continual learning
        """
        super(MixtureOfExperts, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_experts = num_experts
        self.k = k
        self.num_shared_experts = num_shared_experts
        self.num_tasks = num_tasks
        
        # Create task-specific gate networks
        self.gates = nn.ModuleList([
                nn.Linear(input_dim, num_experts) for _ in range(num_tasks)
            ])
        
        # Create the experts
        self.experts = nn.ModuleList()
        for i in range(num_experts):
            if expert_class is not None:
                # Handle different types of expert_class input
                if isinstance(expert_class, type) and issubclass(expert_class, nn.Module):
                    # Case 1: An expert class is passed, needs instantiation
                    assert hidden_dim is not None and output_dim is not None, "hidden_dim and output_dim must be provided for expert class"
                    expert = expert_class(input_dim, hidden_dim, output_dim)
                elif isinstance(expert_class, nn.Module):
                    # Case 2: An expert instance is passed, needs deep copy
                    expert = deepcopy(expert_class)
                else:
                    raise ValueError(f"Expert class {expert_class} is not a valid nn.Module subclass")
                self.experts.append(expert)
            else:
                # Use the default expert module
                assert hidden_dim is not None and output_dim is not None, "hidden_dim and output_dim must be provided for expert class"
                self.experts.append(FFNExpert(input_dim, hidden_dim, output_dim))
        
        # Create the dispatcher
        self.dispatcher = SparseDispatcher(num_experts, k, num_shared_experts)
        
        # Initialize statistics data structures without specifying device
        # Will be automatically migrated when the model moves to a device
        self.register_buffer('expert_usage_counts', torch.zeros(num_tasks, num_experts, dtype=torch.int32)) # Usage count per expert per task
        self.register_buffer('expert_weight_sums', torch.zeros(num_tasks, num_experts, dtype=torch.float32)) # Weight sum per expert per task
    
    def forward(self, x, **kwargs):
        """
        Forward pass for the MoE module.
        
        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, input_dim]
            
        Returns:
            torch.Tensor: Output tensor of shape [batch_size, output_dim]
        """
        squeeze_flag = False
        if x.dim() == 3 and x.shape[0] == 1:
            squeeze_flag = True
            x = x.squeeze(0) # default shape is [batch_size, feature_dim] 
        batch_size = x.shape[0]
        # Calculate gate values
        task_label = kwargs['data_task_label'].item()
        gates = self.gates[task_label](x)

        # Dispatch inputs to experts
        top_k_indices, top_k_gates = self.dispatcher(gates)
        # Update statistics
        self.update_stats(task_label, top_k_indices, top_k_gates)

        # Get expert outputs
        expert_output_list = self.get_expert_output(x, top_k_indices)
        gate_values = self.convert_gate_values(top_k_indices, top_k_gates)
        # Shape: [batch_size, feature_dim]
        moe_features = self.get_moe_output(expert_output_list, gate_values)
        
        if squeeze_flag:
            moe_features = moe_features.unsqueeze(0)
            
        return moe_features


    def get_expert_output(self, x, top_k_indices):
        expert_output_list = [torch.zeros(x.shape[0], self.output_dim).to(x.device) for _ in range(self.num_experts)]
        # Calculate used expert outputs
        unique_expert_indices_all = torch.unique(top_k_indices.flatten())
        # For each expert index
        for expert_idx in unique_expert_indices_all:
            # Get samples assigned to the current expert
            sample_mask = (top_k_indices == expert_idx).any(dim=1)  # Shape: [batch_size]
            samples = x[sample_mask]  # Shape: [num_samples, feature_dim]

            # Compute expert output
            # Initialize expert_output_full with shape [batch_size, feature_dim]
            expert_output = self.experts[expert_idx](samples)  # Shape: [num_samples, feature_dim]

            # Fill expert output into the corresponding positions
            expert_output_list[expert_idx][sample_mask] = expert_output

        return expert_output_list

    def get_moe_output(self, expert_out_list, gate_values, multiply_by_gates=True):
        """
        Args:
            expert_out_list (list): List of expert outputs, each with shape [batch_size, feature_dim]
            gate_values (torch.Tensor): Tensor of shape [batch_size, num_experts] containing gate values
        """
        # Convert expert_out to tensor with shape [num_experts, batch_size, feature_dim]
        expert_out = torch.stack(expert_out_list, dim=0)  
        # If weighting is needed
        if multiply_by_gates:
            # Use einsum to compute weighted sum
            moe_features = torch.einsum('nbf,bn->bf', expert_out, gate_values)  # Shape: [batch_size, feature_dim]
        else:
            # Direct average
            moe_features = expert_out.sum(dim=0)  # Shape: [batch_size, feature_dim]

        # Return result
        return moe_features

    def convert_gate_values(self, top_k_indices, top_k_gates):
        """
        Convert top_k_gates_shared from [batch_size, selected_expert_num] to [batch_size, num_expert],
        filling 0 for unselected experts.

        Args:
            top_k_indices_shared: Shape [batch_size, selected_expert_num], indices of selected experts.
            top_k_gates_shared: Shape [batch_size, selected_expert_num], gate values of selected experts.

        Returns:
            Tensor of shape [batch_size, num_expert], with gate values at selected positions and 0 elsewhere.
        """
        # Get batch_size and num_expert
        batch_size, selected_expert_num = top_k_gates.shape

        # Create a zero tensor of shape [batch_size, num_expert]
        gate_values = torch.zeros(batch_size, self.num_experts, dtype=top_k_gates.dtype, device=top_k_gates.device)

        # Use scatter_ to fill top_k_gates_shared values into gate_values
        gate_values.scatter_(dim=1, index=top_k_indices, src=top_k_gates)

        return gate_values


    def update_stats(self, task_label, expert_indices, gate_values):
        """
        Update expert usage statistics during evaluation.
        
        Args:
            task_label (int): Current task label
            expert_indices (torch.Tensor): Indices of selected experts [batch_size, k]
            gate_values (torch.Tensor): Gate values for selected experts [batch_size, k]
        """
        if self.training:
            return
        
        # Ensure expert_indices and gate_values are 2D tensors
        assert expert_indices.dim() == 2, "expert_indices must be a 2D tensor"
        assert gate_values.dim() == 2, "gate_values must be a 2D tensor"
        assert expert_indices.shape == gate_values.shape, "expert_indices and gate_values shapes must match"

        # Count occurrences and sum weights per expert
        expert_usage_counts2 = torch.bincount(expert_indices.flatten(), minlength=self.num_experts)
        expert_weight_sums2 = torch.zeros(self.num_experts, dtype=torch.float32, device=gate_values.device)
        expert_weight_sums2.scatter_add_(0, expert_indices.flatten(), gate_values.flatten())
        # Update global statistics
        self.expert_usage_counts[task_label] += expert_usage_counts2  
        self.expert_weight_sums[task_label] += expert_weight_sums2



class FFNExpert(nn.Module):
    """
    Feed-forward network expert for MoE.
    """
    def __init__(self, input_dim, hidden_dim, output_dim):
        """
        Initialize the FFN expert.
        
        Args:
            input_dim (int): Dimension of input features
            hidden_dim (int): Dimension of hidden layer
            output_dim (int): Dimension of output features
        """
        super(FFNExpert, self).__init__()
        self.ffn = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
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

class SequentialExpert(nn.Module):
    """
    Sequential Expert for MoE with configurable layers.
    Uses a simple structure of Linear -> ReLU -> Dropout.
    """
    def __init__(self, input_dim, hidden_dim, output_dim, dropout=0.1, num_layers=1):
        """
        Initialize the Sequential Expert.
        
        Args:
            input_dim (int): Dimension of input features
            hidden_dim (int): Dimension of hidden layer (not used if num_layers=1)
            output_dim (int): Dimension of output features
            dropout (float): Dropout probability
            num_layers (int): Number of sequential layers
        """
        super(SequentialExpert, self).__init__()
        
        if num_layers == 1:
            # Single layer expert
            self.net = nn.Sequential(
                nn.Linear(input_dim, output_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            )
        else:
            # Multi-layer expert
            layers = []
            # First layer: input_dim -> hidden_dim
            layers.extend([
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            
            # Middle layers: hidden_dim -> hidden_dim
            for _ in range(num_layers - 2):
                layers.extend([
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout)
                ])
            
            # Last layer: hidden_dim -> output_dim
            if num_layers > 1:
                layers.extend([
                    nn.Linear(hidden_dim, output_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout)
                ])
            
            self.net = nn.Sequential(*layers)
    
    def forward(self, x):
        """
        Forward pass for the Sequential Expert.
        
        Args:
            x (torch.Tensor): Input tensor
            
        Returns:
            torch.Tensor: Output tensor
        """
        return self.net(x)
    
