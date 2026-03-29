import re
import torch
from torch import nn
from collections import OrderedDict

def detailed_parameter_count(model: nn.Module, dtype_bytes: int = 4):
    """
    Provides a detailed, hierarchical breakdown of parameter counts and memory cost for each module.
    It shows direct parameters (belonging only to the module itself), total parameters
    (including all submodules), and the memory cost in MB.

    Args:
        model (nn.Module): The PyTorch model.
        dtype_bytes (int): The number of bytes per parameter (e.g., 4 for float32, 2 for float16).
    """

    param_info = []
    max_name_len = 0

    def count_parameters(m: nn.Module):
        return sum(p.numel() for p in m.parameters() if p.requires_grad)

    # Recursively get information about parameters
    def get_info_recursive(module: nn.Module, prefix=''):
        nonlocal max_name_len
        direct_params = sum(p.numel() for p in module.parameters(recurse=False) if p.requires_grad)
        total_params = count_parameters(module)

        name = prefix if prefix else "model"
        if total_params > 0:
            param_info.append((name, direct_params, total_params))
            if len(name) > max_name_len:
                max_name_len = len(name)

        for child_name, child_module in module.named_children():
            get_info_recursive(child_module, prefix=f"{prefix}.{child_name}" if prefix else child_name)

    get_info_recursive(model)

    # Create a unique list of tuples based on name to avoid duplicates
    unique_param_info = list(OrderedDict.fromkeys(param_info))

    # Header
    header = f"{ 'Module Name'.ljust(max_name_len) } | {'Direct Params':>18} | {'Total (with submodules)':>25} | {'Memory (MB)':>15}"
    print(header)
    print('-' * len(header))

    # Print info for each module
    for name, direct_params, total_params in unique_param_info:
        memory_mb = (total_params * dtype_bytes) / (1024 * 1024)
        print(f"{ name.ljust(max_name_len) } | {direct_params:18,} | {total_params:25,} | {memory_mb:15.2f}")

    # Footer
    print('-' * len(header))
    total_model_params = count_parameters(model)
    total_memory_mb = (total_model_params * dtype_bytes) / (1024 * 1024)
    print(f"{ 'Total Model Trainable Params'.ljust(max_name_len) } | {'':>18} | {total_model_params:25,} | {total_memory_mb:15.2f}")


import datetime

def calculate_time_difference(time1_str, time2_str):
    """
    Calculates the difference (in hours) between two time points

    Args:
    time1_str (str): The first time point, format "YYYY-MM-DD HH:MM:SS"
    time2_str (str): The second time point, format "YYYY-MM-DD HH:MM:SS"

    Returns:
    float: The difference between the two time points (in hours)
    """
    # Define the time format
    time_format = "%Y-%m-%d %H:%M:%S"

    # Convert strings to datetime objects
    time1 = datetime.datetime.strptime(time1_str, time_format)
    time2 = datetime.datetime.strptime(time2_str, time_format)

    # Calculate the time difference
    time_diff = time2 - time1

    # Convert the time difference to hours (including decimals)
    hours_diff = time_diff.total_seconds() / 3600

    # Return absolute value, ensuring the result is positive
    return abs(hours_diff)

def convert_time_string_to_hours(time_str):
    # Define regex pattern to match hours, minutes, and seconds
    pattern = r'(\d+)\s*h.*?(\d+)\s*m.*?(\d+)\s*s'
    match = re.search(pattern, time_str, re.IGNORECASE)

    if match:
        hours = int(match.group(1))
        minutes = int(match.group(2))
        seconds = int(match.group(3))

        # Calculate total number of hours
        total_hours = hours + minutes / 60 + seconds / 3600
        return total_hours
    else:
        raise ValueError("Input format is invalid, expected a string like '2h 48m 2s'.")


def get_buffer_memory_usage(buffer):
    """
    Calculate the total memory (MB) used by all Tensors in a Mammoth Buffer object.
    """
    if buffer is None or buffer.num_seen_examples == 0:
        print("Buffer is empty or not initialized.")
        return 0

    total_bytes = 0

    # Attributes usually storing data in Mammoth buffer
    attributes_to_check = ['examples', 'labels', 'logits', 'task_labels', 'other_features']

    print("\n--- Analyzing model.buffer memory usage ---")

    for attr_name in attributes_to_check:
        if hasattr(buffer, attr_name):
            attribute = getattr(buffer, attr_name)
            # Ensure the attribute is a Tensor
            if isinstance(attribute, torch.Tensor):
                attr_bytes = attribute.element_size() * attribute.nelement()
                if attr_bytes > 0:
                    total_bytes += attr_bytes
                    print(f"Attribute `buffer.{attr_name}` (Tensor): {attr_bytes / (1024*1024):.4f} MB")

    total_mb = total_bytes / (1024 * 1024)
    print("---------------------------------------")
    print(f"Total buffer memory usage: {total_mb:.4f} MB")
    print("---------------------------------------")

    return total_mb


def calculate_module_params_and_memory_pure(module, module_name="module"):
    """
    A helper function specifically for calculating the total parameters and memory usage of a PyTorch module.
    """
    if not isinstance(module, torch.nn.Module):
        print(f"Error: '{module_name}' is not a valid PyTorch module.")
        return

    # Calculate the total number of trainable parameters
    total_params = sum(p.numel() for p in module.parameters() if p.requires_grad)

    # Assume float32, 4 bytes per parameter
    bytes_per_param = 4
    total_memory_mb = (total_params * bytes_per_param) / (1024 * 1024)

    print(f"\n--- Module analysis: {module_name} ---")
    print(f"Total trainable parameters: {total_params:,}")
    print(f"Total memory usage (MB): {total_memory_mb:.4f} MB")
    print("---------------------------------")




def calculate_module_params_and_memory(module, name="module"):
    """
    Calculates and prints the total trainable parameters and estimated memory usage for a given module.
    This provides a quick summary based on parameter count.

    Args:
        module (torch.nn.Module): The module to analyze.
        name (str): The name of the module for printing.
    """
    total_params = sum(p.numel() for p in module.parameters() if p.requires_grad)
    # Assumes float32, 4 bytes per parameter
    memory_mb = (total_params * 4) / (1024 * 1024)

    print(f"\n--- Module Analysis (Estimation): {name} ---")
    print(f"Total Trainable Parameters: {total_params:,}")
    print(f"Estimated Memory (MB): {memory_mb:.4f} MB")
    print("---------------------------------")


def report_allocated_memory(module: nn.Module, name: str):
    """
    Calculates and prints the total trainable parameters and their *actual allocated* memory usage.
    This method is consistent with memory reporters like MemReporter, as it checks the
    size of the underlying storage, which accounts for memory alignment by the allocator.

    Args:
        module (torch.nn.Module): The module to analyze.
        name (str): The name of the module for printing.
    """
    total_params = 0
    total_allocated_bytes = 0

    # Iterate through all parameters in the module
    for p in module.parameters():
        if p.requires_grad:
            total_params += p.numel()
            # Get the actual size of the allocated memory block for this tensor's storage
            total_allocated_bytes += p.untyped_storage().size()

    total_allocated_mb = total_allocated_bytes / (1024 * 1024)

    print(f"\n--- Allocated Memory Report: {name} ---")
    print(f"Total Trainable Parameters: {total_params:,}")
    print(f"Actual Allocated Memory (MB): {total_allocated_mb:.4f} MB")
    print("---------------------------------------")


def hierarchical_memory_report(model: nn.Module):
    """
    Provides a detailed, hierarchical breakdown of parameter counts and their true allocated memory cost,
    similar to torchinfo but with memory accuracy consistent with MemReporter.

    Args:
        model (nn.Module): The PyTorch model to analyze.
    """

    module_info = {}
    max_name_len = 0

    for name, module in model.named_modules():
        if not name:
            name = "model" # Root module

        direct_params = 0
        direct_memory_bytes = 0
        for p in module.parameters(recurse=False):
            if p.requires_grad:
                direct_params += p.numel()
                direct_memory_bytes += p.untyped_storage().size()

        module_info[name] = {
            'direct_params': direct_params,
            'direct_memory_bytes': direct_memory_bytes,
            'children': [child_name for child_name, _ in module.named_children()],
            'total_params': direct_params,
            'total_memory_bytes': direct_memory_bytes
        }
        max_name_len = max(max_name_len, len(name))

    # Calculate total params and memory by rolling up from children
    # We iterate in reverse to ensure children are calculated before parents
    for name in sorted(module_info.keys(), reverse=True):
        info = module_info[name]
        for child_name in info['children']:
            full_child_name = f"{name}.{child_name}" if name != "model" else child_name
            if full_child_name in module_info:
                info['total_params'] += module_info[full_child_name]['total_params']
                info['total_memory_bytes'] += module_info[full_child_name]['total_memory_bytes']

    # --- Formatting the output ---
    header = (
        f"{'Module Name':<{max_name_len}} | "
        f"{'Direct Params':>15} | {'Total Params':>15} | "
        f"{'Direct Mem (MB)':>18} | {'Total Mem (MB)':>18}"
    )
    separator = "-" * len(header)

    print("Hierarchical Memory Report (based on allocated memory)")
    print(header)
    print(separator)

    for name in sorted(module_info.keys()):
        info = module_info[name]
        # Only print modules that have parameters themselves or have children
        if info['total_params'] > 0:
            print(
                f"{name:<{max_name_len}} | "
                f"{info['direct_params']:>15,d} | {info['total_params']:>15,d} | "
                f"{info['direct_memory_bytes'] / (1024*1024):>18.4f} | "
                f"{info['total_memory_bytes'] / (1024*1024):>18.4f}"
            )

    print(separator)
    total_model_info = module_info['model']
    print(f"Total Trainable Parameters: {total_model_info['total_params']:,}")
    print(f"Total Allocated Memory: {total_model_info['total_memory_bytes'] / (1024*1024):.4f} MB")


def report_object_memory(obj: object, name: str):
    """
    Recursively finds all tensors within any Python object (lists, dicts, custom objects, etc.)
    and reports their total allocated memory. This is perfect for analyzing replay buffers
    or other data structures that are not part of the nn.Module parameters.

    Args:
        obj (object): The Python object to inspect.
        name (str): A descriptive name for the report.
    """

    seen_ids = set()
    tensors = []

    def find_tensors_recursive(item):
        # Avoid circular references and redundant checks
        if id(item) in seen_ids:
            return
        seen_ids.add(id(item))

        if isinstance(item, torch.Tensor):
            tensors.append(item)
            return # Tensors are leaves, no need to recurse further

        if isinstance(item, dict):
            for v in item.values():
                find_tensors_recursive(v)
        elif isinstance(item, (list, tuple)):
            for i in item:
                find_tensors_recursive(i)
        elif hasattr(item, '__dict__'): # Handle custom objects
            for attr_value in vars(item).values():
                find_tensors_recursive(attr_value)

    find_tensors_recursive(obj)

    total_tensors = len(tensors)
    total_allocated_bytes = sum(t.untyped_storage().size() for t in tensors)
    total_allocated_mb = total_allocated_bytes / (1024 * 1024)

    print(f"\n--- Memory Report for Python Object: '{name}' ---")
    if not total_tensors:
        print("No tensors found inside this object.")
    else:
        print(f"Found {total_tensors:,} tensors.")
        print(f"Total Allocated Memory: {total_allocated_mb:.4f} MB")
    print("-------------------------------------------------")
