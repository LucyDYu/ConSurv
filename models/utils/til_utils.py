

# Get class range for the current task
import torch


def get_task_range(dataset, task_idx):
    start_idx, end_idx = dataset.get_offsets(task_idx)
    return start_idx, end_idx

# mask out previous tasks - Task-IL forward
def mask_task_range(out, start_idx, end_idx):
    if start_idx > 0:
        out[:, :start_idx] = -torch.inf  # Mask out previous tasks' output
    out[:, end_idx:] = -torch.inf      # Mask out future tasks' output
    return out