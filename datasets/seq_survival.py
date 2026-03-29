from __future__ import print_function, division
from argparse import Namespace
import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, ConcatDataset
import bisect
import numpy as np

from datasets import register_dataset
from datasets.utils import set_default_from_args
from datasets.utils.continual_dataset import ContinualDataset, MammothDatasetWrapper
from datasets.dataset_survival import Generic_MIL_Survival_Dataset

from torch.utils.data import Dataset
import h5py

from utils.evaluate_wsi import evaluate_wsi
from utils.utils import get_split_loader, NLLSurvLoss, CrossEntropySurvLoss, CoxSurvLoss
from utils.wsi_metrics import Surv, _extract_survival_metadata


class ConcatDataset(Dataset):
    """
    Dataset to concatenate multiple datasets.
    Purpose: useful to assemble different existing datasets, possibly
    large-scale datasets as the concatenation operation is done in an
    on-the-fly manner.

    Arguments:
        datasets (sequence): List of datasets to be concatenated
    """

    @staticmethod
    def cumsum(sequence):
        # Compute cumulative sum
        r, s = [], 0
        for e in sequence:
            l = len(e)
            r.append(l + s)
            s += l
        return r

    def __init__(self, datasets):
        super(ConcatDataset, self).__init__()
        assert len(datasets) > 0, 'datasets should not be an empty iterable'
        self.datasets = list(datasets)
        # Compute cumulative size of each dataset
        self.cumulative_sizes = self.cumsum(self.datasets)

    def __len__(self):
        # Return total length of concatenated datasets
        return self.cumulative_sizes[-1]

    def __getitem__(self, idx):
        # Find the corresponding dataset index
        dataset_idx = self.get_dataset_idx(idx)
        # Compute the sample index within the corresponding dataset
        if dataset_idx == 0:
            sample_idx = idx
        else:
            sample_idx = idx - self.cumulative_sizes[dataset_idx - 1]
        # Return the corresponding sample
        return self.datasets[dataset_idx][sample_idx]


    def getlabel(self, idx):
        # Find the corresponding dataset index
        dataset_idx = self.get_dataset_idx(idx)
        # Compute the sample index within the corresponding dataset
        if dataset_idx == 0:
            sample_idx = idx
        else:
            sample_idx = idx - self.cumulative_sizes[dataset_idx - 1]
        # Return the corresponding sample
        return self.datasets[dataset_idx].getlabel(sample_idx)    


    def get_dataset_idx(self, idx):
        # Handle negative indexing
        if idx < 0:
            if -idx > len(self):
                raise ValueError("absolute value of index should not exceed dataset length")
            idx = len(self) + idx
        # Find the corresponding dataset index
        dataset_idx = bisect.bisect_right(self.cumulative_sizes, idx)
        return dataset_idx
    


@register_dataset("seq-survival")
class Sequential_Generic_Survival_Dataset(ContinualDataset):
    NAME = 'seq-survival'
    SETTING = 'task-il'
    # SETTING = 'domain-il'
    N_CLASSES_PER_TASK = 4  # Each cancer type has 4 classes (survival month bins)
    N_TASKS = 4  # 4 cancer types
    
    TRANSFORM = None
    SIZE = (32,32) # not needed

    def __init__(self, args: Namespace):
        """
        Args:
            args: Arguments containing default parameters
            dataset_configs (list): List of dictionaries containing dataset configurations.
                                  If None, will use default configurations.
        """
        super().__init__(args)

        self.train_loaders = []
        self.train_loaders_dict = {}

        self.test_loaders_dict = {}

        self.eval_fn = evaluate_wsi

        
        
        

        # dataset = Generic_MIL_Survival_Dataset(csv_path=csv_path,
        #                                             mode=args.mode,
        #                                             apply_sig=args.apply_sig,
        #                                             data_dir=args.data_root_dir,
        #                                             shuffle=False,
        #                                             seed=args.seed,
        #                                             print_info=True,
        #                                             patient_strat=False,
        #                                             n_bins=4,
        #                                             label_col='survival_months',
        #                                             ignore=[])
        # Default dataset parameters

        

        # dataset = Generic_MIL_Survival_Dataset(csv_path=csv_path,
        #                                             mode=args.mode,
        #                                             apply_sig=args.apply_sig,
        #                                             data_dir=args.data_root_dir,
        #                                             shuffle=False,
        #                                             seed=args.seed,
        #                                             print_info=True,
        #                                             patient_strat=False,
        #                                             n_bins=4,
        #                                             label_col='survival_months',
        #                                             ignore=[])
        # Default dataset parameters
        self.dataset_params = {
            'mode': self.args.mode,
            'apply_sig': self.args.apply_sig,
            'data_dir': self.args.data_root_dir,
            'shuffle': False,
            'seed': self.args.seed,
            'print_info': True,
            'patient_strat': False,
            'n_bins': 4,
            'label_col': 'survival_months',
            'ignore': []
        }

        # Get task order from args
        self.task_name_order = self.args.task_name_order 
        
        # Generate configs based on task order
        self.configs = {}
        for task_name in self.task_name_order:
            config = {
                'cancer_type': task_name,
                'csv_path': os.path.join(self.args.csv_path_folder, f"{task_name}{self.args.csv_path_suffix}"),
                'split_dir': os.path.join('./splits',  self.args.which_splits, task_name)
            }
            self.configs[task_name] = config

        
        # Initialize datasets based on configurations
        self.datasets = {}
        self.split_dirs = {}
    
        for task_name, config in self.configs.items():
            dataset = Generic_MIL_Survival_Dataset(
                csv_path=config['csv_path'],
                mode=self.dataset_params['mode'],
                apply_sig=self.dataset_params['apply_sig'],
                data_dir=self.dataset_params['data_dir'],
                shuffle=self.dataset_params['shuffle'],
                seed=self.dataset_params['seed'],
                print_info=self.dataset_params['print_info'],
                patient_strat=self.dataset_params['patient_strat'],
                n_bins=self.dataset_params['n_bins'],
                label_col=self.dataset_params['label_col'],
                ignore=self.dataset_params['ignore'],
                task_id=self.task_name_order.index(task_name)
            )
            self.datasets[task_name] = dataset
            self.split_dirs[task_name] = config['split_dir']
            # self.metrics_helper[task_name] = {}
        

        # Setup genomic dimensions
        train_split, val_split = self.get_train_val_splits(self.task_name_order[0], 0)
        self.setup_genomic_dimensions(train_split)


    def get_train_val_splits(self,task_name, fold):
        """
        Returns train and validation splits for a given task and fold
        """
        dataset = self.datasets[task_name]
        split_dir = os.path.join('./splits',  self.args.which_splits, task_name)
        print("split_dir", split_dir)
        # Get train and validation splits for the current task
        # Split according to the fold
        train_split, val_split = dataset.return_splits(
            from_id=False,
            csv_path='{}/splits_{}.csv'.format(split_dir, fold)
        )
        return train_split, val_split

    def get_data_loaders(self, task_name, fold):
        """
        Returns data loaders for current task with MammothDatasetWrapper
        """
        train_split, val_split = self.get_train_val_splits(task_name, fold)

        print('training: {}, validation: {}'.format(
            len(train_split), len(val_split)))

        print('\nInit Loaders...', end=' ')
        
        train_split = MammothDatasetWrapper(train_split)
        train_split.extra_return_fields += ('indexes',)  # Add indexes to extra return fields
        val_split = MammothDatasetWrapper(val_split)
        val_split.extra_return_fields += ('indexes',)  # Add indexes to extra return fields
        
        task_label = self.task_name_order.index(task_name)
        train_split.data_task_label = np.full(len(train_split.indexes), task_label)  # Add data_task_label list
        train_split.extra_return_fields += ('data_task_label',)  # Add data_task_label to extra return fields
        val_split.data_task_label = np.full(len(val_split.indexes), task_label)  # Add data_task_label list
        val_split.extra_return_fields += ('data_task_label',)  # Add data_task_label to extra return fields

        # Create base loaders
        train_loader = get_split_loader(train_split, training=True, testing=False,
                                        weighted=self.args.weighted_sample, mode=self.args.mode, batch_size=self.args.batch_size, args=self.args)
        val_loader = get_split_loader(val_split, testing=False, mode=self.args.mode, 
                                      batch_size=self.args.batch_size, args=self.args)
        print('Done!')

        # Wrap the dataset with MammothDatasetWrapper
        # train_loader.dataset = MammothDatasetWrapper(train_loader.dataset)
        # val_loader.dataset = MammothDatasetWrapper(val_loader.dataset)


        self.setup_genomic_dimensions(train_loader.dataset)

        # self.i += self.N_CLASSES_PER_TASK
        self.train_loader = train_loader
        self.train_loaders.append(train_loader)
        self.train_loaders_dict[task_name] = train_loader
        # val_loader is used as test_loader in x fold cross validation
        self.test_loaders.append(val_loader)
        self.test_loaders_dict[task_name] = val_loader


        return train_loader, val_loader

    def setup_genomic_dimensions(self, dataset):
        """
        Sets up genomic feature dimensions based on the data source.
        
        Args:
            data_source: Either a DataLoader or Dataset containing genomic data
        """
        
        # Set genomic dimensions based on mode
        if 'omic' in self.args.mode or self.args.mode in ['cluster', 'graph', 'pyramid']:
            self.omic_input_dim = dataset.genomic_features.shape[1]
            print("Genomic Dimension", self.omic_input_dim)
        elif 'coattn' in self.args.mode:
            self.omic_sizes = dataset.omic_sizes
            print('Genomic Dimensions', self.omic_sizes)
        else:
            self.omic_input_dim = 0

    def get_joint_loaders(self, fold):
        """
        Returns data loaders for joint training, containing data from all tasks.
        
        Args:
            fold (int): Current cross-validation fold
        Returns:
            train_loader, val_loader: Data loaders for training and validation sets
        """
        print('\nInit Joint Loaders...\n', end=' ')

        train_split_list, val_split_list = [], []
        
        # Iterate over all tasks
        for task_name in self.task_name_order:
            # Get training and validation splits for current task
            train_split, val_split = self.get_train_val_splits(task_name, fold)
            
            train_split_list.append(train_split)
            val_split_list.append(val_split)

        # Merge all datasets
        train_split_joint = ConcatDataset(train_split_list)
        val_split_joint = ConcatDataset(val_split_list)

        train_split_joint = MammothDatasetWrapper(train_split_joint)
        train_split_joint.extra_return_fields += ('indexes',)  # Add indexes to extra return fields
        val_split_joint = MammothDatasetWrapper(val_split_joint)
        val_split_joint.extra_return_fields += ('indexes',)  # Add indexes to extra return fields
        
        data_task_label_train, data_task_label_val = [], []
        for i, task_name in enumerate(self.task_name_order):
            data_task_label_train.extend(np.full(len(train_split_list[i]), i))
            data_task_label_val.extend(np.full(len(val_split_list[i]), i))

        train_split_joint.data_task_label = data_task_label_train  # Add data_task_label list
        train_split_joint.extra_return_fields += ('data_task_label',)  # Add data_task_label to extra return fields
        val_split_joint.data_task_label = data_task_label_val  # Add data_task_label list
        val_split_joint.extra_return_fields += ('data_task_label',)  # Add data_task_label to extra return fields

        # Create base loaders
        train_loader = get_split_loader(train_split_joint, training=True, testing=False,
                                        weighted=self.args.weighted_sample, mode=self.args.mode, batch_size=self.args.batch_size, joint_flag=True)
        val_loader = get_split_loader(val_split_joint, testing=False, mode=self.args.mode, 
                                      batch_size=self.args.batch_size, joint_flag=True)

        
        # Save data loaders for later use
        self.train_loader = train_loader
        self.train_loaders.append(train_loader)
        self.train_loaders_dict[self.args.joint_task_name] = train_loader
        # val_loader is used as test_loader in x fold cross validation
        self.test_loaders.append(val_loader)
        self.test_loaders_dict[self.args.joint_task_name] = val_loader
        
        train_size_list = [len(dataset) for dataset in train_split_list]
        val_size_list = [len(dataset) for dataset in val_split_list]
        print(f'Individual dataset sizes - Training: {train_size_list} (sum: {np.sum(train_size_list)}), Validation: {val_size_list} (sum: {np.sum(val_size_list)})')
        print(f'Joint training dataset sizes - Training: {len(train_split_joint)}, Validation: {len(val_split_joint)}')
        print('Done!')
        
        return train_loader, val_loader

    @staticmethod
    def get_backbone():
        return  "mome"

    @staticmethod
    def get_transform():
        return None

    @staticmethod
    def get_loss(**kwargs):
        if 'args' in kwargs:
            args = kwargs['args']
            if args.task_type == 'survival':
                if args.bag_loss == 'ce_surv':
                    loss_fn = CrossEntropySurvLoss(alpha=args.alpha_surv)
                elif args.bag_loss == 'nll_surv':
                    loss_fn = NLLSurvLoss(alpha=args.alpha_surv)
                elif args.bag_loss == 'cox_surv':
                    loss_fn = CoxSurvLoss()
                else:
                    raise NotImplementedError
            else:
                raise NotImplementedError
        else:
            # default in this setting
            alpha_surv = 0 #args.alpha_surv
            loss_fn = NLLSurvLoss(alpha=alpha_surv)

        return loss_fn
        

    
    @staticmethod
    def get_normalization_transform():
        return None

    @staticmethod
    def get_denormalization_transform():
        return None

    @set_default_from_args('n_epochs')
    def get_epochs(self):
        return 20