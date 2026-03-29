# Copyright 2022-present, Lorenzo Bonicelli, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import random
import warnings
import numpy as np
import copy
import math
import os
import sys
from argparse import Namespace
from time import time
from typing import Iterable, Tuple
import logging
import pandas as pd
import torch
from tqdm import tqdm
from sklearn.utils import check_array, check_consistent_length
from sksurv.metrics import concordance_index_censored, concordance_index_ipcw
from datasets.utils.continual_dataset import ContinualDataset

from torch.utils.data import DataLoader, Dataset
try:
    import wandb
except ImportError:
    wandb = None



def _extract_survival_metadata(train_loader, val_loader):
    r"""
    Extract censorship and survival times from the train and val loader and combine to get numbers for the fold
    We need to do this for train and val combined because when evaulating survival metrics, the function needs to know the 
    distirbution of censorhsip and survival times for the trainig data
    
    Args:
        - train_loader : Pytorch Dataloader
        - val_loader : Pytorch Dataloader
    
    Returns:
        - all_survival : np.array
    
    """
    if isinstance(train_loader, DataLoader) and isinstance(val_loader, DataLoader):
        train_dataset = train_loader.dataset.dataset
        val_dataset = val_loader.dataset.dataset
    elif isinstance(train_loader, Dataset) and isinstance(val_loader, Dataset):
        # Pass in a dataset (default type Generic_Split)
        train_dataset = train_loader
        val_dataset = val_loader
    else:
        raise ValueError("Invalid input types for _extract_survival_metadata: train_loader: {}, val_loader: {}".format(type(train_loader), type(val_loader)))
    
    all_censorships = np.concatenate(
        [train_dataset.slide_data['censorship'].to_numpy(),
        val_dataset.slide_data['censorship'].to_numpy()],
        axis=0)
    all_event_times = np.concatenate(
        [train_dataset.slide_data['survival_months'].to_numpy(),
        val_dataset.slide_data['survival_months'].to_numpy()],
        axis=0)
    all_survival = Surv.from_arrays(event=(1-all_censorships).astype(bool), time=all_event_times)
    return all_survival


def _calculate_metrics(dataset: ContinualDataset, task_name: str, survival_train, all_risk_scores, all_censorships, all_event_times):
    r"""
    Calculate survival metrics: C-index and C-index IPCW
    
    Args:
        - dataset : ContinualDataset
        - task_name : str
        - survival_train : np.array
        - all_risk_scores : np.array
        - all_censorships : np.array
        - all_event_times : np.array
        
    Returns:
        - c_index : Float
        - c_index_ipcw : Float
    """
    #---> delete the nans and corresponding elements from other arrays
    original_risk_scores = all_risk_scores
    all_risk_scores = np.delete(all_risk_scores, np.argwhere(np.isnan(original_risk_scores)))
    all_censorships = np.delete(all_censorships, np.argwhere(np.isnan(original_risk_scores)))
    all_event_times = np.delete(all_event_times, np.argwhere(np.isnan(original_risk_scores)))
    #<---

    c_index = concordance_index_censored((1-all_censorships).astype(bool), all_event_times, all_risk_scores, tied_tol=1e-08)[0]
    c_index_ipcw = 0.

    try:
        survival_test = Surv.from_arrays(event=(1-all_censorships).astype(bool), time=all_event_times)
    except:
        print("Problem converting survival test datatype, so all metrics 0.")
        return c_index, c_index_ipcw

    try:
        c_index_ipcw = concordance_index_ipcw(survival_train, survival_test, estimate=all_risk_scores)[0]
    except:
        print('An error occured while computing c-index ipcw')
        c_index_ipcw = 0.

    return c_index, c_index_ipcw


class Surv:
    """
    Helper class to construct structured array of event indicator and observed time.
    """

    @staticmethod
    def from_arrays(event, time, name_event=None, name_time=None):
        """Create structured array.

        Parameters
        ----------
        event : array-like
            Event indicator. A boolean array or array with values 0/1.
        time : array-like
            Observed time.
        name_event : str|None
            Name of event, optional, default: 'event'
        name_time : str|None
            Name of observed time, optional, default: 'time'

        Returns
        -------
        y : np.array
            Structured array with two fields.
        """
        name_event = name_event or "event"
        name_time = name_time or "time"
        if name_time == name_event:
            raise ValueError("name_time must be different from name_event")

        time = np.asanyarray(time, dtype=float)
        y = np.empty(time.shape[0], dtype=[(name_event, bool), (name_time, float)])
        y[name_time] = time

        event = np.asanyarray(event)
        check_consistent_length(time, event)

        if np.issubdtype(event.dtype, np.bool_):
            y[name_event] = event
        else:
            events = np.unique(event)
            events.sort()
            if len(events) != 2:
                raise ValueError("event indicator must be binary")

            if np.all(events == np.array([0, 1], dtype=events.dtype)):
                y[name_event] = event.astype(bool)
            else:
                raise ValueError("non-boolean event indicator must contain 0 and 1 only")

        return y

    @staticmethod
    def from_dataframe(event, time, data):
        """Create structured array from data frame.

        Parameters
        ----------
        event : object
            Identifier of column containing event indicator.
        time : object
            Identifier of column containing time.
        data : pandas.DataFrame
            Dataset.

        Returns
        -------
        y : np.array
            Structured array with two fields.
        """
        if not isinstance(data, pd.DataFrame):
            raise TypeError(f"expected pandas.DataFrame, but got {type(data)!r}")

        return Surv.from_arrays(
            data.loc[:, event].values, data.loc[:, time].values, name_event=str(event), name_time=str(time)
        )





def update_metric_dict_tuple(task_name, metric_output_tuple, metric_dict_tuple):
    for metric_value, metric_dict in zip(metric_output_tuple, metric_dict_tuple):
        metric_dict[f'val_{task_name}'] = metric_value
    return metric_dict_tuple


def update_metric_matrix_tuple(task_name, metric_dict_tuple, metric_matrix_tuple):
    for metric_dict, metric_matrix in zip(metric_dict_tuple, metric_matrix_tuple):
        metric_matrix[f'train_{task_name}'] = metric_dict
    return metric_matrix_tuple

def get_last_value_from_metric_dict_tuple(metric_dict_tuple):
    result = tuple()
    for metric_dict in metric_dict_tuple:
        result = result + (list(metric_dict.items())[-1][1],)
    return result

def df_add_column(df, column_name, column_values):
    # if a single value, add it to the last row
    if not isinstance(column_values, list):
        new_column = [0] * (len(df) - 1) + [column_values]  # Construct new column
        df[column_name] = new_column
    # if a list of column_values matching the length of the dataframe, add the column to the dataframe
    elif len(column_values) == len(df):
        df[column_name] = column_values
    else:
        raise ValueError(f"The length of column_values ({len(column_values)}) does not match the length of the dataframe ({len(df)})")
    return df

def df_add_columns(df, forgetting, bwt, fwt, random_res_class): 
  
    value_list=[forgetting, bwt, fwt]
    if any([v is None or np.isnan(v) for v in value_list]):
        logging.warning("Some values are None or NaN, so no columns are added. forgetting: {}, bwt: {}, fwt: {}, random_res_class: {}".format(forgetting, bwt, fwt, random_res_class))
        return df
        
    df['Avg(on trained)'] = [row[:i + 1].mean() for i, row in enumerate(df.values)]
    df = df_add_column(df, 'Forgetting', forgetting)
    df = df_add_column(df, 'BWT', bwt)
    df = df_add_column(df, 'FWT', fwt)
    df = df_add_column(df, 'Random', np.concatenate(random_res_class).tolist())
    return df


def matrix_remove_keys(matrix):
    return [list(inner_dict.values()) for inner_dict in matrix.values()]


def replace_none_with_zero(results):
    """Recursively replace None with 0 in nested lists."""
    if not isinstance(results, list):
        return results if results is not None else 0
    return [replace_none_with_zero(item) for item in results]
