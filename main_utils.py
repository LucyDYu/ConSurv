"""
This script is the main entry point for the Mammoth project. It contains the main function `main()` that orchestrates the training process.

The script performs the following tasks:
- Imports necessary modules and libraries.
- Sets up the necessary paths and configurations.
- Parses command-line arguments.
- Initializes the dataset, model, and other components.
- Trains the model using the `train()` function.

To run the script, execute it directly or import it as a module and call the `main()` function.
"""
# Copyright 2022-present, Lorenzo Bonicelli, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# needed (don't change it)
import numpy  # noqa
import logging
import os
import sys
import time
import importlib
import socket
import datetime
import uuid
import argparse
import torch

from utils.utils import get_custom_exp_code
# Set the number of threads Torch will use for parallel operations
torch.set_num_threads(2)

# Determine the path to the Mammoth project root directory based on the location of this script
# if file is launched inside the `utils` folder
if os.path.dirname(__file__) == 'utils':
    # Get the absolute path of the parent directory of this script's location
    mammoth_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
else:
    # Get the absolute path of this script's location
    mammoth_path = os.path.dirname(os.path.abspath(__file__))

# Add the Mammoth project root directory to the beginning of the Python module search path
sys.path.insert(0, mammoth_path)

# Import and set up logging configuration
from utils import setup_logging
setup_logging()


# needed (don't change it)
def lecun_fix():
    """
    This function is needed to access Yann LeCun's website via urllib.
    It sets the User-Agent header to 'Mozilla/5.0' to avoid issues caused by Cloudflare's anti-bot protection.
    """
    # Yann moved his website to CloudFlare. You need this now
    from six.moves import urllib  # pyright: ignore
    opener = urllib.request.build_opener()
    opener.addheaders = [('User-agent', 'Mozilla/5.0')]
    urllib.request.install_opener(opener)


# Check arguments 
def check_args(args, dataset=None):
    """
    Validates the arguments to ensure they are compatible with each other and suitable for the chosen dataset.
    
    This function performs a series of assertions to check various argument combinations and values.
    It covers checks for argument conflicts, value ranges, and dataset-specific requirements.
    
    Parameters:
    args (namespace): A namespace containing command-line arguments or parameters.
    dataset (Dataset, optional): The dataset object to be used, which may influence the argument validity checks.
    
    Returns:
    None
    """
    # Ensure that the arguments related to label percentage by class and overall are not both set, to avoid conflict
    assert args.label_perc_by_class == 1 or args.label_perc == 1, "Cannot use both `label_perc_by_task` and `label_perc_by_class`"

    # For joint training, ensure that certain arguments are not set, as they are incompatible
    if args.joint:
        assert args.start_from is None and args.stop_after is None, "Joint training does not support start_from and stop_after"
        assert not args.enable_other_metrics, "Joint training does not support other metrics"
        assert not args.eval_future, "Joint training does not support future evaluation (what is the future?)"

    # Ensure the label percentage is within the valid range
    assert 0 < args.label_perc <= 1, "label_perc must be in (0, 1]"

    # Ensure checkpoints are not saved during inference-only mode
    if args.savecheck:
        assert not args.inference_only, "Should not save checkpoint in inference only mode"

    # Ensure the noise rate is within the valid range
    assert (args.noise_rate >= 0.) and (args.noise_rate <= 1.), "Noise rate must be in [0, 1]"

    # If a dataset is provided, perform additional checks based on the dataset type
    if dataset is not None:
        from datasets.utils.gcl_dataset import GCLDataset, ContinualDataset

        # Special checks for GCLDataset type
        if isinstance(dataset, GCLDataset):
            assert args.n_epochs == 1, "GCLDataset is not compatible with multiple epochs"
            assert args.enable_other_metrics == 0, "GCLDataset is not compatible with other metrics (i.e., forward/backward transfer and forgetting)"
            assert args.eval_future == 0, "GCLDataset is not compatible with future evaluation"
            assert args.noise_rate == 0, "GCLDataset is not compatible with automatic noise injection"

        # Ensure the dataset is an instance of ContinualDataset or GCLDataset
        assert issubclass(dataset.__class__, ContinualDataset) or issubclass(dataset.__class__, GCLDataset), "Dataset must be an instance of `ContinualDataset` or `GCLDataset`"

        # Specific checks for datasets with 'biased-class-il' setting
        if dataset.SETTING == 'biased-class-il':
            assert not args.eval_future, 'Evaluation of future tasks is not supported for biased-class-il.'
            assert not args.enable_other_metrics, 'Other metrics are not supported for biased-class-il.'

        # For datasets using cross-entropy loss, issue a warning if trying to use label noise (except for noise rate of 1)
        if 'cross_entropy' in str(dataset.get_loss()) or 'CrossEntropy' in str(dataset.get_loss()):
            if args.noise_rate != 1:
                logging.warning('Label noise is not available with multi-label datasets. If this is not multi-label, ignore this warning.')



def load_configs(parser: argparse.ArgumentParser) -> dict:
    from models import get_model_class
    from models.utils import load_model_config

    from datasets import get_dataset_class
    from datasets.utils import get_default_args_for_dataset, load_dataset_config
    from utils.args import fix_model_parser_backwards_compatibility, get_single_arg_value

    args = parser.parse_known_args()[0]

    # load the model configuration
    # - get the model parser and fix the get_parser function for backwards compatibility
    model_group_parser = parser.add_argument_group('Model-specific arguments')
    model_parser = get_model_class(args).get_parser(model_group_parser)
    parser = fix_model_parser_backwards_compatibility(model_group_parser, model_parser)
    is_rehearsal = any([p for p in parser._actions if p.dest == 'buffer_size'])
    buffer_size = None
    if is_rehearsal:  # get buffer size
        buffer_size = get_single_arg_value(parser, 'buffer_size')
        assert buffer_size is not None, "Buffer size not found in the arguments. Please specify it with --buffer_size."
        try:
            buffer_size = int(buffer_size)  # try convert to int, check if it is a valid number
        except ValueError:
            raise ValueError(f'--buffer_size must be an integer but found {buffer_size}')

    # - get the defaults that were set with `set_defaults` in the parser
    base_config = parser._defaults.copy()

    # - get the configuration file for the model
    model_config = load_model_config(args, buffer_size=buffer_size)

    # update the dataset class with the configuration
    dataset_class = get_dataset_class(args)

    # load the dataset configuration. If the model specified a dataset config, use it. Otherwise, use the dataset configuration
    base_dataset_config = get_default_args_for_dataset(args.dataset)
    if 'dataset_config' in model_config:  # if the dataset specified a dataset config, use it
        cnf_file_dataset_config = load_dataset_config(model_config['dataset_config'], args.dataset)
    else:
        cnf_file_dataset_config = load_dataset_config(args.dataset_config, args.dataset)

    dataset_config = {**base_dataset_config, **cnf_file_dataset_config}
    dataset_config = dataset_class.set_default_from_config(dataset_config, parser)  # the updated configuration file is cleaned from the dataset-specific arguments

    # - merge the dataset and model configurations, with the model configuration taking precedence
    config = {**dataset_config, **base_config, **model_config}

    return config


def add_help(parser):
    """
    Add the help argument to the parser
    """
    parser.add_argument('-h', '--help', action='help', default=argparse.SUPPRESS, help='Show this help message and exit.')

def add_wsi_args(parser):
    
    ### Checkpoint + Misc. Pathing Parameters
    parser.add_argument('--data_root_dir',   type=str, default='path/to/data_root_dir',
                        help='Data directory to WSI features (extracted via CLAM')
    parser.add_argument('--k', 			     type=int, default=5,
                        help='Number of folds (default: 5)')
    parser.add_argument('--k_start',		 type=int, default=-1,
                        help='Start fold (Default: -1, last fold)')
    parser.add_argument('--k_end',			 type=int, default=-1,
                        help='End fold (Default: -1, first fold)')
    parser.add_argument('--results_dir',     type=str, default='./results',
                        help='Results directory (Default: ./results)')
    parser.add_argument('--which_splits',    type=str, default='5foldcv',
                        help='Which splits folder to use in ./splits/ (Default: ./splits/5foldcv')
    parser.add_argument('--split_dir',       type=str, default='tcga_blca',
                        help='Which cancer type within ./splits/<which_splits> to use for training. Used synonymously for "task" (Default: tcga_blca)')
    parser.add_argument('--log_data',        action='store_true', 
                        help='Log data using tensorboard')
    parser.add_argument('--overwrite',     	 action='store_true', default=False,
                        help='Whether or not to overwrite experiments (if already ran)')
    parser.add_argument('--load_model',        action='store_true',
                        default=False, help='whether to load model')
    parser.add_argument('--path_load_model', type=str,
                        default='/path/to/load', help='path of ckpt for loading')
    parser.add_argument('--start_epoch',              type=int,
                        default=0, help='start_epoch.')

    ### Model Parameters.
    parser.add_argument('--model_type',      type=str, choices=['snn', 'amil', 'mcat', 'motcat', 'mome'], 
                        default='motcat', help='Type of model (Default: motcat)')
    parser.add_argument('--mode',            type=str, choices=['omic', 'path', 'pathomic', 'cluster', 'coattn'],
                        default='coattn', help='Specifies which modalities to use / collate function in dataloader.')
    parser.add_argument('--fusion',          type=str, choices=[
                        'None', 'concat'], default='concat', help='Type of fusion. (Default: concat).')
    parser.add_argument('--apply_sig',		 action='store_true', default=False,
                        help='Use genomic features as signature embeddings.')
    parser.add_argument('--apply_sigfeats',  action='store_true',
                        default=False, help='Use genomic features as tabular features.')
    parser.add_argument('--drop_out',        action='store_true',
                        default=True, help='Enable dropout (p=0.25)')
    parser.add_argument('--model_size_wsi',  type=str,
                        default='small', help='Network size of AMIL model')
    parser.add_argument('--model_size_omic', type=str,
                        default='small', help='Network size of SNN model')

    ### Optimizer Parameters + Survival Loss Function
    parser.add_argument('--opt',             type=str,
                        choices=['adam', 'sgd'], default='adam')
    # parser.add_argument('--batch_size',      type=int, default=1,
    #                     help='Batch Size (Default: 1, due to varying bag sizes)')
    parser.add_argument('--gc',              type=int,
                        default=32, help='Gradient Accumulation Step.')
    parser.add_argument('--max_epochs',      type=int, default=20,
                        help='Maximum number of epochs to train (default: 20)')
    # parser.add_argument('--lr',				 type=float, default=2e-4,
    #                     help='Learning rate (default: 0.0002)')
    parser.add_argument('--bag_loss',        type=str, choices=['svm', 'ce', 'ce_surv', 'nll_surv',
                        'cox_surv'], default='nll_surv', help='slide-level classification loss function (default: nll_surv)')
    parser.add_argument('--label_frac',      type=float, default=1.0,
                        help='fraction of training labels (default: 1.0)')
    parser.add_argument('--bag_weight',      type=float, default=0.7,
                        help='clam: weight coefficient for bag-level loss (default: 0.7)')
    parser.add_argument('--reg', 			 type=float, default=1e-5,
                        help='L2-regularization weight decay (default: 1e-5)')
    parser.add_argument('--alpha_surv',      type=float, default=0.0,
                        help='How much to weigh uncensored patients')
    parser.add_argument('--reg_type',        type=str, choices=['None', 'omic', 'pathomic'],
                        default='None', help='Which network submodules to apply L1-Regularization (default: None)')
    parser.add_argument('--lambda_reg',      type=float, default=1e-4,
                        help='L1-Regularization Strength (Default 1e-4)')
    parser.add_argument('--weighted_sample', action='store_true',
                        default=True, help='Enable weighted sampling')
    parser.add_argument('--early_stopping',  action='store_true',
                        default=False, help='Enable early stopping')

    ### MoME Parameters
    parser.add_argument('--bs_micro',      type=int, default=4096,
                        help='The Size of Micro-batch (Default: 4096)')
    parser.add_argument('--n_bottlenecks', 			 type=int, default=2,
                        help='number of bottleneck features (Default: 2)')

    ### WSI CL Parameters
    parser.add_argument('--fold',      type=int, default=0,
                        help='Run tasks on this fold')
    parser.add_argument('--scenario',      type=str, default='task-il',
                        help='Run tasks on this fold')
    parser.add_argument('--data_tuple_size',      type=int, default=10,
                        help='Default data tuple size, e.g. (WSI, omic1, omic2, omic3, omic4, omic5, omic6, label, event_time, c)')
    parser.add_argument('--bs_micro_fix_shuffle',  action='store_true', default=False,
                        help='Whether micro batches are fixed or random')
    parser.add_argument('--n_bins',  type=int, default=4,
                        help='Number of bins for survival analysis')
    parser.add_argument('--joint_training', action='store_true', default=False,
                        help='Whether to use joint training. Still need to specify the task_name_order to know which datasets to join.')
    parser.add_argument('--loadcheck_base_name', type=str, default=None, help='Path of the checkpoint to load (without _task_label.pt)')
    parser.add_argument('--warm_up_epochs', type=int, default=0, help='warm up epochs before adding data to buffer')

def parse_args():
    """
    Parse command line arguments for the mammoth program and sets up the `args` object.

    Returns:
        args (argparse.Namespace): Parsed command line arguments.
    """
    # Import required utility functions
    from utils import create_if_not_exists
    from utils.conf import warn_once
    from utils.args import add_initial_args, add_management_args, add_experiment_args, add_configuration_args, clean_dynamic_args, \
        check_multiple_defined_arg_during_string_parse, add_dynamic_parsable_args, update_cli_defaults, get_single_arg_value

    from models import get_all_models

    # Check for duplicate argument definitions
    check_multiple_defined_arg_during_string_parse()

    # Create argument parser without help message initially
    parser = argparse.ArgumentParser(description='Mammoth - A benchmark Continual Learning framework for Pytorch', allow_abbrev=False, add_help=False)

    # 1) add arguments that include model, dataset, and backbone. These define the rest of the arguments.
    #   the backbone is optional as may be set by the dataset or the model. The dataset and model are required.
    add_initial_args(parser)
    args = parser.parse_known_args()[0]
    
    # Add initial arguments for seq-survival
    if args.dataset == 'seq-survival':
        add_wsi_args(parser)
        args = parser.parse_known_args()[0]

    # Warn if no backbone is specified
    if args.backbone is None:
        logging.warning('No backbone specified. Using default backbone (set by the dataset).')

    # 2) load the configuration arguments for the dataset and model
    add_configuration_args(parser, args)
    config = load_configs(parser)
    add_help(parser)

    # 3) add the remaining arguments

    # - get the chosen backbone. The CLI argument takes precedence over the configuration file.
    backbone = args.backbone
    if backbone is None:
        if 'backbone' in config:
            backbone = config['backbone']
        else:
            backbone = get_single_arg_value(parser, 'backbone')
    assert backbone is not None, "Backbone not found in the arguments. Please specify it with --backbone or in the model or dataset configuration file."

    # - add the dynamic arguments defined by the chosen dataset and model
    add_dynamic_parsable_args(parser, args.dataset, backbone)

    # - add the main Mammoth arguments
    add_management_args(parser)
    add_experiment_args(parser)

    # 4) Once all arguments are in the parser, we can set the defaults using the loaded configuration
    update_cli_defaults(parser, config)

    # force call type on all default values to fix values (https://docs.python.org/3/library/argparse.html#type)
    for action in parser._actions:
        if action.default is not None and action.type is not None:
            if action.nargs is None or action.nargs == 0:
                action.default = action.type(action.default)
            else:
                if not isinstance(action.default, (list, tuple)) or (action.type is not list and action.type is not tuple):
                    action.default = [action.type(v) for v in action.default]

    # 5) parse the arguments
    if args.load_best_args:
        # Load best known arguments if requested
        from utils.best_args import best_args

        warn_once("The `load_best_args` option is untested and not up to date.")

        # Check if model uses a buffer
        is_rehearsal = any([p for p in parser._actions if p.dest == 'buffer_size'])

        args = parser.parse_args()
        if args.model == 'joint':
            best = best_args[args.dataset]['sgd']
        else:
            best = best_args[args.dataset][args.model]
        if is_rehearsal:
            best = best[args.buffer_size]
        else:
            best = best[-1]

        # Reconstruct command line with best args
        to_parse = sys.argv[1:] + ['--' + k + '=' + str(v) for k, v in best.items()]
        to_parse.remove('--load_best_args')
        args = parser.parse_args(to_parse)
        if args.model == 'joint' and args.dataset == 'mnist-360':
            args.model = 'joint_gcl'
    else:
        args = parser.parse_args()

    # 6) clean dynamically loaded args
    args = clean_dynamic_args(args)

    # 7) final checks and updates to the arguments
    models_dict = get_all_models()
    args.model = models_dict[args.model]

    # Handle learning rate scheduler
    if args.lr_scheduler is not None:
        logging.info('`lr_scheduler` set to {}, overrides default from dataset.'.format(args.lr_scheduler))

    # Set random seed if specified
    if args.seed is not None:
        from utils.conf import set_random_seed
        set_random_seed(args.seed)

    # Add uuid, timestamp and hostname for logging
    args.conf_jobnum = str(uuid.uuid4())
    args.conf_timestamp = str(datetime.datetime.now())
    args.conf_host = socket.gethostname()

    # Add the current git commit hash to the arguments if available
    try:
        import git
        repo = git.Repo(path=mammoth_path)
        args.conf_git_hash = repo.head.object.hexsha
    except Exception:
        logging.error("Could not retrieve git hash.")
        args.conf_git_hash = None

    # Setup checkpoint saving
    if args.savecheck:
        if not os.path.isdir('checkpoints'):
            create_if_not_exists("checkpoints")

        now = time.strftime("%Y%m%d-%H%M%S")
        uid = args.conf_jobnum.split('-')[0]
        extra_ckpt_name = "" if args.ckpt_name is None else f"{args.ckpt_name}_"
        args.ckpt_name = f"{extra_ckpt_name}{args.model}_{args.dataset}_{args.dataset_config}_{args.buffer_size if hasattr(args, 'buffer_size') else 0}_{args.n_epochs}_{str(now)}_{uid}"
        print("Saving checkpoint into", args.ckpt_name, file=sys.stderr)

    if args.dataset == 'seq-survival':
        args = extend_args_wsi(args)

    # Validate arguments
    check_args(args)

    # Log validation settings if enabled
    if args.validation is not None:
        logging.info(f"Using {args.validation}% of the training set as validation set.")
        logging.info(f"Validation will be computed with mode `{args.validation_mode}`.")

    return args

def extend_args(args, dataset):
    """
    Extend the command-line arguments with the default values from the dataset and model.
    
    This function takes the parsed arguments and dataset object, and sets default values
    for various parameters if they were not explicitly specified. It handles:
    - Number of classes
    - Training epochs/iterations
    - Batch sizes
    - Validation settings
    - Debug mode configuration
    - Weights & Biases logging setup
    
    Args:
        args: The parsed command line arguments
        dataset: The dataset object containing default configurations
    """
    from datasets import ContinualDataset
    dataset: ContinualDataset = dataset  # noqa, used for type hinting

    # Set number of classes if not specified
    if hasattr(args, 'num_classes') and args.num_classes is None:
        args.num_classes = dataset.N_CLASSES

    # Set training duration based on fitting mode (epochs or iterations)
    if args.fitting_mode == 'epochs' and args.n_epochs is None and isinstance(dataset, ContinualDataset):
        args.n_epochs = dataset.get_epochs()
    elif args.fitting_mode == 'iters' and args.n_iters is None and isinstance(dataset, ContinualDataset):
        args.n_iters = dataset.get_iters()

    # Configure batch sizes
    if args.batch_size is None:
        args.batch_size = dataset.get_batch_size()
        # Set minibatch size for models with buffer if not specified
        if hasattr(importlib.import_module('models.' + args.model), 'Buffer') and (not hasattr(args, 'minibatch_size') or args.minibatch_size is None):
            args.minibatch_size = dataset.get_minibatch_size()
    else:
        args.minibatch_size = args.batch_size

    # Validate settings for validation mode
    if args.validation:
        if args.validation_mode == 'current':
            assert dataset.SETTING in ['class-il', 'task-il'], "`current` validation modes is only supported for class-il and task-il settings (requires a task division)."

    # Configure debug mode settings
    if args.debug_mode:
        print('Debug mode enabled: running only a few forward steps per epoch with W&B disabled.')
        # set logging level to debug
        args.nowand = 1

    # Setup Weights & Biases logging configuration
    if args.wandb_entity is None:
        args.wandb_entity = os.getenv('WANDB_ENTITY', None)
    if args.wandb_project is None:
        args.wandb_project = os.getenv('WANDB_PROJECT', None)

    # Disable W&B if credentials are missing
    if args.wandb_entity is None or args.wandb_project is None:
        logging.info('`wandb_entity` and `wandb_project` not set. Disabling wandb.')
        args.nowand = 1
    else:
        print('Logging to wandb: {}/{}'.format(args.wandb_entity, args.wandb_project))
        args.nowand = 0

def extend_args_wsi(args):
    if args.dataset == 'seq-survival':
        # Add exp_code for seq-survival
        args = get_custom_exp_code(args)
        exp_code = "CL_" + str(args.exp_code) + '_microb{}'.format(args.bs_micro) + '_s{}'.format(args.seed)
        args.task = '_'.join(args.split_dir.split('_')[:2]) + '_survival'
        print("===="*30)
        print("Experiment Name:", exp_code)
        print("===="*30)
        args.results_dir = os.path.join(args.results_dir, args.which_splits, args.param_code, exp_code)
        if not os.path.isdir(args.results_dir):
            os.makedirs(args.results_dir)
        print("logs saved at ", args.results_dir)
        if ('summary_latest.csv' in os.listdir(args.results_dir)) and (not args.overwrite):
            print("Exp Code <%s> already exists! Exiting script." % args.exp_code)
            sys.exit()

        # Sets the absolute path of split_dir
        args.split_dir = os.path.join('./splits', args.which_splits, args.split_dir)
        settings = {'data_root_dir': args.data_root_dir,
            'num_splits': args.k,
            'k_start': args.k_start,
            'k_end': args.k_end,
            'task': args.task,
            'max_epochs': args.max_epochs,
            'results_dir': args.results_dir,
            'lr': args.lr,
            'experiment': args.exp_code,
            'reg': args.reg,
            'label_frac': args.label_frac,
            'bag_loss': args.bag_loss,
            'bag_weight': args.bag_weight,
            'seed': args.seed,
            'model_type': args.model_type,
            'model_size_wsi': args.model_size_wsi,
            'model_size_omic': args.model_size_omic,
            "use_drop_out": args.drop_out,
            'weighted_sample': args.weighted_sample,
            'gc': args.gc,
            'opt': args.opt,
            'bs_micro': args.bs_micro}
        
        print("split_dir", args.split_dir)
        assert os.path.isdir(args.split_dir)
        settings.update({'split_dir': args.split_dir})

        with open(args.results_dir + '/experiment_{}.txt'.format(args.exp_code), 'w') as f:
            print(settings, file=f)
        f.close()

        print("################# Settings ###################")
        for key, val in settings.items():
            print("{}:  {}".format(key, val))
            
    return args
