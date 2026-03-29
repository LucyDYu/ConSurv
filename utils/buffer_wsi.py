# Copyright 2022-present, Lorenzo Bonicelli, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from argparse import Namespace
from copy import deepcopy
from typing import List, Tuple, TYPE_CHECKING

import numpy as np
import torch
import torch.nn as nn

from utils.augmentations import apply_transform
from utils.conf import create_seeded_dataloader, get_device
from utils.training_utils import full_data_to_device
from utils.utils import select_collate

if TYPE_CHECKING:
    from models.utils.continual_model import ContinualModel
    from datasets.utils.continual_dataset import ContinualDataset
    from backbone import MammothBackbone


def icarl_replay(self: 'ContinualModel', dataset: 'ContinualDataset', val_set_split=0):
    """
    Merge the replay buffer with the current task data.
    Optionally split the replay buffer into a validation set.

    Args:
        self: the model instance
        dataset: the dataset
        val_set_split: the fraction of the replay buffer to be used as validation set
    """

    if self.current_task > 0:
        buff_val_mask = torch.rand(len(self.buffer)) < val_set_split
        val_train_mask = torch.zeros(len(dataset.train_loader.dataset.data)).bool()
        val_train_mask[torch.randperm(len(dataset.train_loader.dataset.data))[:buff_val_mask.sum()]] = True

        if val_set_split > 0:
            self.val_dataset = deepcopy(dataset.train_loader.dataset)

        data_concatenate = torch.cat if isinstance(dataset.train_loader.dataset.data, torch.Tensor) else np.concatenate
        need_aug = hasattr(dataset.train_loader.dataset, 'not_aug_transform')
        if not need_aug:
            def refold_transform(x): return x.cpu()
        else:
            data_shape = len(dataset.train_loader.dataset.data[0].shape)
            if data_shape == 3:
                def refold_transform(x): return (x.cpu() * 255).permute([0, 2, 3, 1]).numpy().astype(np.uint8)
            elif data_shape == 2:
                def refold_transform(x): return (x.cpu() * 255).squeeze(1).type(torch.uint8)

        # REDUCE AND MERGE TRAINING SET
        dataset.train_loader.dataset.targets = np.concatenate([
            dataset.train_loader.dataset.targets[~val_train_mask],
            self.buffer.labels.cpu().numpy()[:len(self.buffer)][~buff_val_mask]
        ])
        dataset.train_loader.dataset.data = data_concatenate([
            dataset.train_loader.dataset.data[~val_train_mask],
            refold_transform((self.buffer.examples)[:len(self.buffer)][~buff_val_mask])
        ])

        if val_set_split > 0:
            # REDUCE AND MERGE VALIDATION SET
            self.val_dataset.targets = np.concatenate([
                self.val_dataset.targets[val_train_mask],
                self.buffer.labels.cpu().numpy()[:len(self.buffer)][buff_val_mask]
            ])
            self.val_dataset.data = data_concatenate([
                self.val_dataset.data[val_train_mask],
                refold_transform((self.buffer.examples)[:len(self.buffer)][buff_val_mask])
            ])

            self.val_loader = create_seeded_dataloader(self.args, self.val_dataset, batch_size=self.args.batch_size, shuffle=True)


class BaseSampleSelection:
    """
    Base class for sample selection strategies.
    """

    def __init__(self, buffer_size: int, device):
        """
        Initialize the sample selection strategy.

        Args:
            buffer_size: the maximum buffer size
            device: the device to store the buffer on
        """
        self.buffer_size = buffer_size
        self.device = device

    def __call__(self, num_seen_examples: int) -> int:
        """
        Selects the index of the sample to replace.

        Args:
            num_seen_examples: the number of seen examples

        Returns:
            the index of the sample to replace
        """

        raise NotImplementedError

    def update(self, *args, **kwargs):
        """
        (optional) Update the state of the sample selection strategy.
        """
        pass


class ReservoirSampling(BaseSampleSelection):
    def __call__(self, num_seen_examples: int) -> int:
        """
        Reservoir sampling algorithm.

        Args:
            num_seen_examples: the number of seen examples
            buffer_size: the maximum buffer size

        Returns:
            the target index if the current image is sampled, else -1
        """
        if num_seen_examples < self.buffer_size:  # If seen examples less than buffer size
            return num_seen_examples  # Return current sample count

        rand = np.random.randint(0, num_seen_examples + 1)  # Generate random index
        if rand < self.buffer_size:  # If random index less than buffer size
            return rand  # Return random index
        else:
            return -1  # Return -1, not sampled


class LARSSampling(BaseSampleSelection):
    def __init__(self, buffer_size: int, device):
        super().__init__(buffer_size, device)  # Call parent initialization
        # lossoir scores
        self.importance_scores = torch.ones(buffer_size, device=device) * -float('inf')  # Initialize importance scores to negative infinity

    def update(self, indexes: torch.Tensor, values: torch.Tensor):
        self.importance_scores[indexes] = values  # Update importance scores

    def normalize_scores(self, values: torch.Tensor):
        if values.shape[0] > 0:  # If values exist
            if values.max() - values.min() != 0:  # If max and min are not equal
                values = (values - values.min()) / ((values.max() - values.min()) + 1e-9)  # Normalize
            return values  # Return normalized values
        else:
            return None  # Return None

    def __call__(self, num_seen_examples: int) -> int:
        if num_seen_examples < self.buffer_size:  # If seen examples less than buffer size
            return num_seen_examples  # Return current sample count

        rn = np.random.randint(0, num_seen_examples)  # Generate random index
        if rn < self.buffer_size:  # If random index less than buffer size
            norm_importance = self.normalize_scores(self.importance_scores)  # Normalize importance scores
            norm_importance = norm_importance / (norm_importance.sum() + 1e-9)  # Normalize
            index = np.random.choice(range(self.buffer_size), p=norm_importance.cpu().numpy(), size=1)  # Select index based on importance scores
            return index  # Return selected index
        else:
            return -1  # Return -1, not sampled


class LossAwareBalancedSampling(BaseSampleSelection):
    """
    Combination of Loss-Aware Sampling (LARS) and Balanced Reservoir Sampling (BRS) from `Rethinking Experience Replay: a Bag of Tricks for Continual Learning`.
    """

    def __init__(self, buffer_size: int, device):
        super().__init__(buffer_size, device)
        # lossoir scores
        self.importance_scores = torch.ones(buffer_size, device=device) * -float('inf')
        # balancoir scores
        self.balance_scores = torch.ones(self.buffer_size, dtype=torch.float).to(self.device) * -float('inf') # Initialize balance scores to negative infinity
        # merged scores
        self.scores = torch.ones(self.buffer_size).to(self.device) * -float('inf') # Initialize merged scores to negative infinity

    def update(self, indexes: torch.Tensor, values: torch.Tensor):
        self.importance_scores[indexes] = values  # Update importance scores

    def merge_scores(self):
        scaling_factor = self.importance_scores.abs().mean() * self.balance_scores.abs().mean()  # Compute scaling factor
        norm_importance = self.importance_scores / scaling_factor  # Normalize importance scores
        presoftscores = 0.5 * norm_importance + 0.5 * self.balance_scores  # Merge scores

        if presoftscores.max() - presoftscores.min() != 0:  # If max and min are not equal
            presoftscores = (presoftscores - presoftscores.min()) / (presoftscores.max() - presoftscores.min() + 1e-9)  # Normalize
        self.scores = presoftscores / presoftscores.sum()  # Update merged scores

    def update_balancoir_scores(self, labels: torch.Tensor):
        unique_labels, orig_inputs_idxs, counts = labels.unique(return_counts=True, return_inverse=True)  # Get unique labels and counts
        # assert len(counts) > unique_labels.max(), "Some classes are missing from the buffer"
        self.balance_scores = torch.gather(counts, 0, orig_inputs_idxs).float()  # Update balance scores

    def __call__(self, num_seen_examples: int, labels: torch.Tensor) -> int:
        if num_seen_examples < self.buffer_size:  # If seen examples less than buffer size
            return num_seen_examples  # Return current sample count

        rn = np.random.randint(0, num_seen_examples)  # Generate random index
        if rn < self.buffer_size:  # If random index less than buffer size
            self.update_balancoir_scores(labels)  # Update balance scores
            self.merge_scores()  # Merge scores
            index = np.random.choice(range(self.buffer_size), p=self.scores.cpu().numpy(), size=1)  # Select index based on merged scores
            return index  # Return selected index
        else:
            return -1  # Return -1, not sampled


class ABSSampling(LARSSampling):
    def __init__(self, buffer_size: int, device: str, dataset: 'ContinualDataset'):
        super().__init__(buffer_size, device)  # Call parent initialization
        self.dataset = dataset  # Set dataset

    def scale_scores(self, past_indexes: torch.Tensor):
        # Normalize the two groups separately
        past_importance = self.normalize_scores(self.importance_scores[past_indexes])  # Normalize past samples importance scores
        current_importance = self.normalize_scores(self.importance_scores[~past_indexes])  # Normalize current samples importance scores
        current_scores, past_scores = None, None  # Initialize current and past scores
        if past_importance is not None:  # If past importance scores exist
            past_importance = 1 - past_importance  # Invert past importance scores
            past_scores = past_importance / past_importance.sum()  # Normalize past scores
        if current_importance is not None:  # If current importance scores exist
            if current_importance.sum() == 0:  # If current importance scores sum is 0
                current_importance += 1e-9  # Avoid numerical issues
            current_scores = current_importance / current_importance.sum()  # Normalize current scores

        return past_scores, current_scores  # Return past and current scores

    def __call__(self, num_seen_examples: int, labels: torch.Tensor) -> int:
        n_seen_classes, _ = self.dataset.get_offsets()  # Get number of seen classes

        if num_seen_examples < self.buffer_size:  # If seen examples less than buffer size
            return num_seen_examples  # Return current sample count

        rn = np.random.randint(0, num_seen_examples)  # Generate random index
        if rn < self.buffer_size:  # If random index less than buffer size
            past_indexes = labels < n_seen_classes  # Get past sample indices

            past_scores, current_scores = self.scale_scores(past_indexes)  # Compute scores
            past_percentage = np.float64(past_indexes.sum().cpu() / self.buffer_size)  # Compute past sample ratio
            pres_percetage = 1 - past_percentage  # Compute current sample ratio
            assert past_percentage + pres_percetage == 1, f"The sum of the percentages must be 1 but found {past_percentage+pres_percetage}: {past_percentage} + {pres_percetage}"  # Ensure ratios sum to 1
            rp = np.random.choice((0, 1), p=[past_percentage, pres_percetage])  # Randomly select past or current sample

            if not rp:  # If selecting past sample
                index = np.random.choice(np.arange(self.buffer_size)[past_indexes.cpu().numpy()], p=past_scores.cpu().numpy(), size=1)  # Select index based on past scores
            else:  # If selecting current sample
                index = np.random.choice(np.arange(self.buffer_size)[~past_indexes.cpu().numpy()], p=current_scores.cpu().numpy(), size=1)  # Select index based on current scores
            return index  # Return selected index
        else:
            return -1  # Return -1, not sampled


class Buffer_WSI:
    """
    The memory buffer of rehearsal method.
    """

    buffer_size: int  # the maximum size of the buffer
    device: str  # the device to store the buffer on
    num_seen_examples: int  # the total number of examples seen, used for reservoir
    attributes: List[str]  # the attributes stored in the buffer
    attention_maps: List[torch.Tensor]  # (optional) attention maps used by TwF
    sample_selection_strategy: str  # the sample selection strategy used to select samples to replace. By default, 'reservoir'

    examples: torch.Tensor  # (mandatory) buffer attribute: the tensor of images
    labels: torch.Tensor  # (optional) buffer attribute: the tensor of labels
    logits: torch.Tensor  # (optional) buffer attribute: the tensor of logits
    task_labels: torch.Tensor  # (optional) buffer attribute: the tensor of task labels
    true_labels: torch.Tensor  # (optional) buffer attribute: the tensor of true labels

    def __init__(self, buffer_size: int, dataset: 'ContinualDataset',
                 args: Namespace, device="cpu", sample_selection_strategy='reservoir', **kwargs):
        """
        Initialize a reservoir-based Buffer object.

        Supports storing images, labels, logits, task_labels, and attention maps. This can be extended by adding more attributes to the `attributes` list and updating the `init_tensors` method accordingly.

        To select samples to replace, the buffer supports:
        - `reservoir` sampling: randomly selects samples to replace (default). Ref: "Jeffrey S Vitter. Random sampling with a reservoir."
        - `lars`: prioritizes retaining samples with the *higher* loss. Ref: "Pietro Buzzega et al. Rethinking Experience Replay: a Bag of Tricks for Continual Learning."
        - `labrs` (Loss-Aware Balanced Reservoir Sampling): combination of LARS and BRS. Ref: "Pietro Buzzega et al. Rethinking Experience Replay: a Bag of Tricks for Continual Learning."
        - `abs` (Asymmetric Balanced Sampling): for samples from the current task, prioritizes retaining samples with the *lower* loss (i.e., inverse `lossoir`); for samples from previous tasks, prioritizes retaining samples with the *higher* loss (i.e., `lossoir`). Useful for settings with noisy labels. Ref: "Monica Millunzi et al. May the Forgetting Be with You: Alternate Replay for Learning with Noisy Labels".

        Args:
            buffer_size (int): The maximum size of the buffer.
            device (str, optional): The device to store the buffer on. Defaults to "cpu".
            sample_selection_strategy: The sample selection strategy. Defaults to 'reservoir'. Options: 'reservoir', 'lars', 'labrs', 'abs'.

        Note:
            If during the `get_data` the transform is PIL, data will be moved to cpu and then back to the device. This is why the device is set to cpu by default.
        """
        self._buffer_size = buffer_size  # Set buffer size
        self.dataset = dataset  # Set dataset
        self.args = args  # Set args
        self.device = device  # Set device
        self.num_seen_examples = 0  # Initialize seen examples count
        self.attributes = ['examples', 'labels', 'logits', 'task_labels', 'true_labels', 'final_features', 'wsi_features', 'omic_features']  # Define buffer attributes
        self.attention_maps = [None] * buffer_size  # Initialize attention maps
        self.sample_selection_strategy = sample_selection_strategy  # Set sample selection strategy

        assert sample_selection_strategy.lower() in ['reservoir', 'lars', 'labrs', 'abs', 'unlimited'], f"Invalid sample selection strategy: {sample_selection_strategy}"  # Ensure selection strategy is valid

        if sample_selection_strategy.lower() == 'abs':  # If selection strategy is abs
            # assert 'dataset' in kwargs, "The dataset is required for ABS sample selection"  # Ensure dataset is provided
            self.sample_selection_fn = ABSSampling(buffer_size, device, dataset)  # Initialize abs sampling
        elif sample_selection_strategy.lower() == 'lars':  # If selection strategy is lars
            self.sample_selection_fn = LARSSampling(buffer_size, device)  # Initialize lars sampling
        elif sample_selection_strategy.lower() == 'labrs':  # If selection strategy is labrs
            self.sample_selection_fn = LossAwareBalancedSampling(buffer_size, device)  # Initialize labrs sampling
        elif sample_selection_strategy.lower() == 'unlimited':  # If selection strategy is unlimited
            self.sample_selection_fn = lambda x: x  # Set unlimited sampling
            self._buffer_size = 10  # Initial buffer size, will expand when needed
        else:  # Default selection strategy is reservoir
            self.sample_selection_fn = ReservoirSampling(buffer_size, device)  # Initialize reservoir sampling

    def serialize(self, out_device='cpu'):
        """
        Serialize the buffer.

        Returns:
            A dictionary containing the buffer attributes.
        """
        serialized_attributes = {}
    
        for attr_str in self.attributes:
            if hasattr(self, attr_str):
                attr = getattr(self, attr_str)
                # If attribute is a list
                if isinstance(attr, list):
                    # Move each tensor in the list to target device
                    serialized_attributes[attr_str] = [tensor.to(out_device) if isinstance(tensor, torch.Tensor) else tensor for tensor in attr]
                else:
                    # If attribute is not a list, move directly to target device
                    serialized_attributes[attr_str] = attr.to(out_device)
        #  {attr_str: getattr(self, attr_str).to(out_device) for attr_str in self.attributes if hasattr(self, attr_str)}  # Return serialized attributes dictionary
        return serialized_attributes

    def to(self, device):
        """
        Move the buffer and its attributes to the specified device.

        Args:
            device: The device to move the buffer and its attributes to.

        Returns:
            The buffer instance with the updated device and attributes.
        """
        self.device = device  # Update device
        self.sample_selection_fn.device = device  # Update sample selection function device
        for attr_str in self.attributes:  # Iterate over all attributes
            if hasattr(self, attr_str):  # If attribute exists
                setattr(self, attr_str, getattr(self, attr_str).to(device))  # Move attribute to specified device
        return self  # Return buffer instance

    def __len__(self):
        """
        Returns the number items in the buffer.
        """
        if self.sample_selection_strategy == 'unlimited':  # If selection strategy is unlimited
            return self.num_seen_examples  # Return seen examples count
        return min(self.num_seen_examples, self.buffer_size)  # Return minimum of seen examples count and buffer size

    def init_tensors(self, examples: torch.Tensor, labels: torch.Tensor,
                     logits: torch.Tensor, task_labels: torch.Tensor,
                     true_labels: torch.Tensor, final_features: torch.Tensor, wsi_features: torch.Tensor, omic_features: torch.Tensor) -> None:
        """
        Initializes just the required tensors.

        Args:
            examples: tensor containing the images
            labels: tensor containing the labels
            logits: tensor containing the outputs of the network
            task_labels: tensor containing the task labels
            true_labels: tensor containing the true labels (used only for logging)
        """
        for attr_str in self.attributes:
            attr = eval(attr_str)
            if attr is not None and not hasattr(self, attr_str):
                setattr(self, attr_str, [None] * self.buffer_size)
            elif hasattr(self, attr_str):
                # Extend list length when buffer expands
                current_list = getattr(self, attr_str)
                if len(current_list) < self.buffer_size:
                    # Calculate number of None placeholders needed
                    pad_length = self.buffer_size - len(current_list)
                    # Extend list length with None
                    current_list += [None] * pad_length
                    setattr(self, attr_str, current_list)

    @property
    def buffer_size(self):
        """
        Returns the buffer size.
        """
        if self.sample_selection_strategy == 'unlimited':  # If selection strategy is unlimited
            return int(1e9)  # Return maximum integer
        return self._buffer_size  # Return buffer size

    @buffer_size.setter
    def buffer_size(self, value):
        """
        Sets the buffer size.
        """
        if self.sample_selection_strategy != 'unlimited':  # If selection strategy is not unlimited
            self._buffer_size = value  # Set buffer size

    @property
    def used_attributes(self):
        """
        Returns a list of attributes that are currently being used by the object.
        """
        return [attr_str for attr_str in self.attributes if hasattr(self, attr_str)]  # Return list of existing attributes

    def get_data_from_dataset(self, task_label: int, data_index: int) -> Tuple:
        train_loader = self.dataset.train_loaders[task_label]  # Get train loader
        data_index_int = int(data_index)
        inputs = [train_loader.dataset.__getitem__(data_index_int)]
                    
        collate = select_collate(self.args.mode)
        data = collate(inputs)
        return data


    def add_data(self, examples, labels=None, logits=None, task_labels=None, attention_maps=None, true_labels=None, sample_selection_scores=None, final_features=None, wsi_features=None, omic_features=None):
        """
        Adds the data to the memory buffer according to the reservoir strategy.

        Args:
            examples: list of data_index for the examples to store
            labels: tensor containing the labels
            logits: tensor containing the outputs of the network
            task_labels: tensor containing the task labels
            attention_maps: list of tensors containing the attention maps
            true_labels: if setting is noisy, the true labels associated with the examples. **Used only for logging.**
            sample_selection_scores: tensor containing the scores used for the sample selection strategy. NOTE: this is only used if the sample selection strategy defines the `update` method.

        Note:
            Only the examples are required. The other tensors are initialized only if they are provided.
        """
        if not hasattr(self, 'examples'):  # If examples attribute doesn't exist in buffer
            self.init_tensors(examples, labels, logits, task_labels, true_labels, final_features, wsi_features, omic_features)  # Initialize tensors
            
        for i in range(len(examples)):
        # for i in range(examples.shape[0]):  # Iterate over each example
            if self.sample_selection_strategy == 'abs' or self.sample_selection_strategy == 'labrs':  # If selection strategy is abs or labrs
                index = self.sample_selection_fn(self.num_seen_examples, labels=self.labels)  # Get replacement index
            else:
                index = self.sample_selection_fn(self.num_seen_examples)  # Get replacement index
            self.num_seen_examples += 1  # Increment seen examples count
            if index >= 0:  # If index is valid
                if self.sample_selection_strategy == 'unlimited' and self.num_seen_examples > self._buffer_size:  # If strategy is unlimited and seen examples exceed buffer size
                    self._buffer_size *= 2  # Expand buffer size
                    self.init_tensors(examples, labels, logits, task_labels, true_labels)  # Initialize tensors

                # self.examples[index] = examples[i].to(self.device)  # Add example to buffer
                self.examples[index] = full_data_to_device(examples[i], self.device, move_data_WSI=True)  # Add example to buffer
                if labels is not None:  # If labels are not None
                    self.labels[index] = labels[i].to(self.device)  # Add label to buffer
                if logits is not None:  # If logits are not None
                    self.logits[index] = logits[i].to(self.device)  # Add logits to buffer
                if task_labels is not None:  # If task labels are not None
                    self.task_labels[index] = task_labels[i].to(self.device)  # Add task label to buffer
                if attention_maps is not None:  # If attention maps are not None
                    self.attention_maps[index] = [at[i].byte().to(self.device) for at in attention_maps]  # Add attention maps to buffer
                if sample_selection_scores is not None:  # If sample selection scores are not None
                    self.sample_selection_fn.update(index, sample_selection_scores[i])  # Update sample selection scores
                if true_labels is not None:  # If true labels are not None
                    self.true_labels[index] = true_labels[i].to(self.device)  # Add true labels to buffer
                if final_features is not None:  # If final features are not None
                    self.final_features[index] = final_features[i].to(self.device)  # Add final features to buffer
                if wsi_features is not None:  # If WSI features are not None
                    self.wsi_features[index] = wsi_features[i].to(self.device)  # Add WSI features to buffer
                if omic_features is not None:  # If omic features are not None
                    self.omic_features[index] = omic_features[i].to(self.device)  # Add omic features to buffer

    def get_data(self, size: int, transform: nn.Module = None, return_index=False, device=None,
                 mask_task_out=None, cpt=None, return_not_aug=False, not_aug_transform=None) -> Tuple:
        """
        Random samples a batch of size items.

        Args:
            size: the number of requested items
            transform: the transformation to be applied (data augmentation)
            return_index: if True, returns the indexes of the sampled items
            mask_task: if not None, masks OUT the examples from the given task
            cpt: the number of classes per task (required if mask_task is not None and task_labels are not present)
            return_not_aug: if True, also returns the not augmented items
            not_aug_transform: the transformation to be applied to the not augmented items (if `return_not_aug` is True)

        Returns:
            a tuple containing the requested items. If return_index is True, the tuple contains the indexes as first element.
        """
        target_device = self.device if device is None else device  # Set target device
        samples_mask = None
        if mask_task_out is not None:  # If task masking is needed
            assert hasattr(self, 'task_labels') or cpt is not None  # Ensure task labels or class count exists
            assert hasattr(self, 'task_labels') or hasattr(self, 'labels')  # Ensure task labels or labels exists
            samples_mask = (self.task_labels != mask_task_out) if hasattr(self, 'task_labels') else self.labels // cpt != mask_task_out  # Create samples mask

        num_avail_samples = len(self.examples) if mask_task_out is None else samples_mask.sum().item()  # Get available samples count
        num_avail_samples = min(self.num_seen_examples, num_avail_samples)  # Take minimum of seen examples and available samples

        if size > min(num_avail_samples, len(self.examples)):  # If requested size exceeds available samples
            size = min(num_avail_samples, len(self.examples))  # Set requested size to available samples count

        choice = np.random.choice(num_avail_samples, size=size, replace=False)  # Randomly select sample indices
        # if transform is None:  # If no transform
        #     def transform(x): return x  # Define transform function

        assert hasattr(self, 'task_labels')

        selected_examples = self.selected_attr(self.examples, choice, mask_task_out, samples_mask)  # Get selected samples
        selected_examples_task_labels = self.selected_attr(self.task_labels, choice, mask_task_out, samples_mask)  # Get selected samples

        selected_samples_pairs=list(zip(selected_examples_task_labels, selected_examples))

        # Return tuple list of indices and labels
        # selected_samples=[]
        # for task_label, index in selected_samples_pairs:
        #     data=self.get_data_from_dataset(task_label, index)
        #     selected_samples.append(data)
        selected_samples = selected_examples

        if return_not_aug:  # If returning non-augmented samples
            if not_aug_transform is None:  # If no non-augmented transform
                def not_aug_transform(x): return x  # Define non-augmented transform function
            ret_tuple = (apply_transform(selected_samples, transform=not_aug_transform).to(target_device),)  # Apply non-augmented transform
        else:
            ret_tuple = tuple()  # Initialize return tuple

        # ret_tuple += (apply_transform(selected_samples, transform=transform).to(target_device),)  # Apply augmented transform
        trans_selected_samples = apply_transform(selected_samples, transform=transform)
        # Don't move data_WSI during get_data, subsequent training will move in mini batch
        selected_samples_to_device = [full_data_to_device(sample, target_device, move_data_WSI=False) for sample in trans_selected_samples]
        ret_tuple += (selected_samples_to_device,)  # Apply augmented transform
        for attr_str in self.attributes[1:]:  # Iterate over all attributes
            if hasattr(self, attr_str):  # If attribute exists
                attr = getattr(self, attr_str)  # Get attribute
                selected_attr = self.selected_attr(attr, choice, mask_task_out, samples_mask)  # Get selected attribute
                selected_attr_to_device = [tensor.to(target_device) for tensor in selected_attr]
                ret_tuple += (selected_attr_to_device,)  # Add selected attribute to return tuple

        if not return_index:  # If not returning index
            return ret_tuple  # Return tuple
        else:
            return (torch.tensor(choice).to(target_device), ) + ret_tuple  # Return index and tuple

    def get_data_by_index(self, indexes, transform: nn.Module = None, device=None) -> Tuple:
        """
        Returns the data by the given index.

        Args:
            index: the index of the item
            transform: the transformation to be applied (data augmentation)

        Returns:
            a tuple containing the requested items. The returned items depend on the attributes stored in the buffer from previous calls to `add_data`.
        """
        target_device = self.device if device is None else device  # Set target device

        if transform is None:  # If no transform
            def transform(x): return x  # Define transform function
        ret_tuple = (apply_transform(self.examples[indexes], transform=transform).to(target_device),)  # Apply transform
        for attr_str in self.attributes[1:]:  # Iterate over all attributes
            if hasattr(self, attr_str):  # If attribute exists
                attr = getattr(self, attr_str).to(target_device)  # Get attribute and move to target device
                ret_tuple += (attr[indexes],)  # Add selected attribute to return tuple
        return ret_tuple  # Return tuple

    def selected_attr(self, attr, choice, mask_task_out=None, samples_mask=None):
        # selected_attr = attr[choice] if mask_task_out is None else attr[samples_mask][choice] # Get selected samples
        if mask_task_out is None:
            selected_attr = [attr[i] for i in choice]
        else:
            masked_examples = [attr[i] for i, mask in enumerate(samples_mask) if mask]
            selected_attr = [masked_examples[i] for i in choice]

        return selected_attr
    
    def is_empty(self) -> bool:
        """
        Returns true if the buffer is empty, false otherwise.
        """
        if self.num_seen_examples == 0:  # If seen examples count is 0
            return True  # Return True, buffer is empty
        else:
            return False  # Return False, buffer is not empty

    def get_all_data(self, transform: nn.Module = None, device=None) -> Tuple:
        """
        Return all the items in the memory buffer.

        Args:
            transform: the transformation to be applied (data augmentation)

        Returns:
            a tuple with all the items in the memory buffer
        """
        target_device = self.device if device is None else device  # Set target device
        if transform is None:  # If no transform
            ret_tuple = (self.examples[:len(self)].to(target_device),)  # Get all examples
        else:
            ret_tuple = (apply_transform(self.examples[:len(self)], transform=transform).to(target_device),)  # Apply transform
        for attr_str in self.attributes[1:]:  # Iterate over all attributes
            if hasattr(self, attr_str):  # If attribute exists
                attr = getattr(self, attr_str)[:len(self)].to(target_device)  # Get attribute and move to target device
                ret_tuple += (attr,)  # Add attribute to return tuple
        return ret_tuple  # Return tuple

    def empty(self) -> None:
        """
        Set all the tensors to None.
        """
        for attr_str in self.attributes:  # Iterate over all attributes
            if hasattr(self, attr_str):  # If attribute exists
                delattr(self, attr_str)  # Delete attribute
        self.num_seen_examples = 0  # Reset seen examples count


@torch.no_grad()
def fill_buffer(buffer: Buffer_WSI, dataset: 'ContinualDataset', t_idx: int, net: 'MammothBackbone' = None, use_herding=False,
                required_attributes: List[str] = None, normalize_features=False, extend_equalize_buffer=False) -> None:
    """
    Adds examples from the current task to the memory buffer.
    Supports images, labels, task_labels, and logits.

    Args:
        buffer: the memory buffer
        dataset: the dataset from which take the examples
        t_idx: the task index
        net: (optional) the model instance. Used if logits are in buffer. If provided, adds logits.
        use_herding: (optional) if True, uses herding strategy. Otherwise, random sampling.
        required_attributes: (optional) the attributes to be added to the buffer. If None and buffer is empty, adds only examples and labels.
        normalize_features: (optional) if True, normalizes the features before adding them to the buffer
        extend_equalize_buffer: (optional) if True, extends the buffer to equalize the number of samples per class for all classes, even if that means exceeding the buffer size defined at initialization
    """
    if net is not None:  # If model is provided
        mode = net.training  # Get model training mode
        net.eval()  # Set model to evaluation mode
    else:
        assert not use_herding, "Herding strategy requires a model instance"  # If using herding strategy, model instance is required

    device = net.device if net is not None else get_device()  # Set device

    n_seen_classes = dataset.N_CLASSES_PER_TASK * (t_idx + 1) if isinstance(dataset.N_CLASSES_PER_TASK, int) else \
        sum(dataset.N_CLASSES_PER_TASK[:t_idx + 1])  # Get number of seen classes
    n_past_classes = dataset.N_CLASSES_PER_TASK * t_idx if isinstance(dataset.N_CLASSES_PER_TASK, int) else \
        sum(dataset.N_CLASSES_PER_TASK[:t_idx])  # Get number of past classes

    mask = dataset.train_loader.dataset.targets >= n_past_classes  # Create mask
    dataset.train_loader.dataset.targets = dataset.train_loader.dataset.targets[mask]  # Update target labels
    dataset.train_loader.dataset.data = dataset.train_loader.dataset.data[mask]  # Update data

    buffer.buffer_size = dataset.args.buffer_size  # Reset initial buffer size

    if extend_equalize_buffer:  # If buffer extension is needed
        samples_per_class = np.ceil(buffer.buffer_size / n_seen_classes).astype(int)  # Calculate samples per class
        new_bufsize = int(n_seen_classes * samples_per_class)  # Calculate new buffer size
        if new_bufsize != buffer.buffer_size:  # If new buffer size differs from current
            print('Buffer size has been changed to:', new_bufsize)  # Print new buffer size
        buffer.buffer_size = new_bufsize  # Update buffer size
    else:
        samples_per_class = buffer.buffer_size // n_seen_classes  # Calculate samples per class

    # Check required attributes
    required_attributes = required_attributes or ['examples', 'labels']  # Set required attributes
    assert all([attr in buffer.used_attributes for attr in required_attributes]) or len(buffer) == 0, \
        "Required attributes not in buffer: {}".format([attr for attr in required_attributes if attr not in buffer.used_attributes])  # Ensure required attributes are in buffer

    if t_idx > 0:  # If task index is greater than 0
        # 1) First, subsample prior classes
        buf_data = buffer.get_all_data()  # Get all data from buffer
        buf_y = buf_data[1]  # Get labels

        buffer.empty()  # Clear buffer
        for _y in buf_y.unique():  # Iterate over unique labels
            idx = (buf_y == _y)  # Get indices for current label
            _buf_data_idx = {attr_name: _d[idx][:samples_per_class] for attr_name, _d in zip(required_attributes, buf_data)}  # Get data for current label
            buffer.add_data(**_buf_data_idx)  # Add data to buffer

    # 2) Then, fill with current tasks
    loader = dataset.train_loader  # Get data loader
    norm_trans = dataset.get_normalization_transform()  # Get normalization transform
    if norm_trans is None:  # If no normalization transform
        def norm_trans(x): return x  # Define normalization transform function

    if 'logits' in buffer.used_attributes:  # If logits are used in buffer
        assert net is not None, "Logits in buffer require a model instance"  # Ensure model instance is provided

    # 2.1 Extract all features
    a_x, a_y, a_f, a_l = [], [], [], []  # Initialize feature lists
    for data in loader:  # Iterate over data loader
        x, y, not_norm_x = data[0], data[1], data[2]  # Get data
        if not x.size(0):  # If data is empty
            continue  # Skip
        a_x.append(not_norm_x.cpu())  # Add non-normalized data
        a_y.append(y.cpu())  # Add labels

        if net is not None:  # If model is provided
            feats = net(norm_trans(not_norm_x.to(device)), returnt='features')  # Get features
            outs = net.classifier(feats)  # Get outputs
            if normalize_features:  # If feature normalization is needed
                feats = feats / feats.norm(dim=1, keepdim=True)  # Normalize features

            a_f.append(feats.cpu())  # Add features
            a_l.append(torch.sigmoid(outs).cpu())  # Add outputs
    a_x, a_y = torch.cat(a_x), torch.cat(a_y)  # Concatenate data and labels
    if net is not None:  # If model is provided
        a_f, a_l = torch.cat(a_f), torch.cat(a_l)  # Concatenate features and outputs

    # 2.2 Compute class means
    for _y in a_y.unique():  # Iterate over unique labels
        idx = (a_y == _y)  # Get indices for current label
        _x, _y = a_x[idx], a_y[idx]  # Get data for current label

        if use_herding:  # If using herding strategy
            _l = a_l[idx]  # Get outputs for current label
            feats = a_f[idx]  # Get features for current label
            mean_feat = feats.mean(0, keepdim=True)  # Compute mean feature

            running_sum = torch.zeros_like(mean_feat)  # Initialize running sum
            i = 0  # Initialize counter
            while i < samples_per_class and i < feats.shape[0]:  # Loop until reaching sample count or feature count
                cost = (mean_feat - (feats + running_sum) / (i + 1)).norm(2, 1)  # Compute cost

                idx_min = cost.argmin().item()  # Get index of minimum cost

                buffer.add_data(  # Add data to buffer
                    examples=_x[idx_min:idx_min + 1].to(device),
                    labels=_y[idx_min:idx_min + 1].to(device),
                    logits=_l[idx_min:idx_min + 1].to(device) if 'logits' in required_attributes else None,
                    task_labels=torch.ones(len(_x[idx_min:idx_min + 1])).to(device) * t_idx if 'task_labels' in required_attributes else None
                )

                running_sum += feats[idx_min:idx_min + 1]  # Update running sum
                feats[idx_min] = feats[idx_min] + 1e6  # Update feature
                i += 1  # Increment counter
        else:  # If not using herding strategy
            idx = torch.randperm(len(_x))[:samples_per_class]  # Randomly select sample indices

            buffer.add_data(  # Add data to buffer
                examples=_x[idx].to(device),
                labels=_y[idx].to(device),
                logits=_l[idx].to(device) if 'logits' in required_attributes else None,
                task_labels=torch.ones(len(_x[idx])).to(device) * t_idx if 'task_labels' in required_attributes else None
            )

    assert len(buffer.examples) <= buffer.buffer_size, f"buffer overflowed its maximum size: {len(buffer)} > {buffer.buffer_size}"  # Ensure buffer has not overflowed
    assert buffer.num_seen_examples <= buffer.buffer_size, f"buffer has been overfilled, there is probably an error: {buffer.num_seen_examples} > {buffer.buffer_size}"  # Ensure seen examples count has not overflowed

    if net is not None:  # If model is provided
        net.train(mode)  # Restore model training mode
