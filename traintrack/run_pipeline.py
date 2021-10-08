"""The main logic of the library to run a traintrack pipeline.

This module contains the main steps of running a pipeline, and is called from the `command_line_pipe` module, which is really just a helper for the command line script `traintrack`. This module is also called in batch mode to directly handle the hyperparameters of each stage, passed in as an arbitrary number of command line args.

Example:
    Typically, one wouldn't call this module directly, but given a valid set of hyperparams, and a model name, one could call the `batch_stage` function from the command line. Indeed, this may end up being the way for external libraries to interact with the traintrack pipeline (e.g. a Weights & Biases hyperparameter optimization agent).
        
Todo:
    * Remove unnecessary run args
    * Handle verbose properly as passed from either batch or inside command_line_pipe
"""

import sys, os
import argparse
import yaml
import logging
import torch

from traintrack.utils.config_utils import load_config, combo_config, submit_batch
from traintrack.utils.data_utils import autocast
from traintrack.utils.model_utils import (
    get_resume_id,
    get_logger,
    build_model,
    build_trainer,
)

def parse_batch_pipeline():

    """Parse command line arguments."""

    run_parser, model_parser = (
        argparse.ArgumentParser("run_pipeline.py"),
        argparse.ArgumentParser("run_pipeline.py"),
    )
    add_run_arg, add_model_arg = run_parser.add_argument, model_parser.add_argument
    add_run_arg("--verbose", action="store_true")
    add_run_arg("pipeline_config", nargs="?", default="configs/pipeline_test.yaml")

    run_parsed, model_to_parse = run_parser.parse_known_args()
    [
        add_model_arg(arg, nargs="+")
        for arg in model_to_parse
        if arg.startswith(("-", "--"))
    ]

    run_parsed, _ = run_parser.parse_known_args()
    model_parsed, _ = model_parser.parse_known_args()

    return run_parsed, model_parsed


def batch_stage():
    print("Running batch from top with args:", sys.argv)
    run_args, model_args = parse_batch_pipeline()
    model_config = vars(model_args)
    if "inference" not in model_config: model_config["inference"] = False

    logging_level = logging.INFO if run_args.verbose else logging.WARNING
    logging.basicConfig(level=logging_level)
    #     logging.basicConfig(level=logging.INFO)
    logging.info("Parsed run args: {}".format(run_args))
    logging.info("Parsed model args: {}".format(model_args))

    run_stage(**model_config)


@autocast
def run_stage(**model_config):

    print("Running stage, with args, and model library:", model_config["model_library"])
    sys.path.append(model_config["model_library"])

    # Load the model and configuration file for this stage
    model_class = build_model(model_config)
    model = model_class(model_config)
    logging.info("Model found")

    # Test if the model is TRAINABLE (i.e. a learning stage) or NONTRAINABLE (i.e. a processing stage)
    if callable(getattr(model, "training_step", None)):
        train_stage(model, model_config)
    else:
        data_stage(model, model_config)


def train_stage(model, model_config):

    # Define a logger (default: Weights & Biases)
    logger = get_logger(model_config)

    # Load the trainer, handling any resume conditions
    trainer = build_trainer(model_config, logger)

    # Run training, unless in inference mode
    if not model_config["inference"]:
        trainer.fit(model)
    else:
        # Run testing and, if requested, inference callbacks to continue the pipeline
        if model_config["checkpoint_path"]:
            print("Loading state dict")
            model.load_state_dict(torch.load(model_config["checkpoint_path"])["state_dict"])
        else:
            logging.error("Cannot run inference without a resume ID")
            
    trainer.test(model)

def data_stage(model, model_config):
    logging.info("Preparing data")
    model.prepare_data()


def start(args):

    print(args)

    with open(args.pipeline_config) as f:
        pipeline_config = yaml.load(f, Loader=yaml.FullLoader)

    with open("configs/project_config.yaml") as f:
        project_config = yaml.load(os.path.expandvars(f.read()), Loader=yaml.FullLoader)

    # Make models available to the pipeline
    sys.path.append(
        project_config["libraries"]["model_library"]
    )  #  !!  TEST WITHOUT THIS LINE IN MAIN()

    # This is the current slurm ID to handle serial dependency
    running_id = None
    for stage in pipeline_config["stage_list"]:

        # Set resume_id if it is given, else it is None and new model is built
        resume_id = get_resume_id(stage)

        # Get config file, from given location OR from ckpnt
        model_config = load_config(stage, resume_id, project_config, args)
        logging.info("Single config: {}".format(model_config))
        
        model_config_combos = (
            combo_config(model_config) if resume_id is None else [model_config]
        )
        logging.info("Combo configs: {}".format(model_config_combos))

        for config in model_config_combos:
            if args.slurm:
                running_id = submit_batch(
                    config, project_config, running_id
                )
            else:
                run_stage(**config)
