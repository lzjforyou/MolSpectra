import argparse
import os
import numpy as np
import yaml

from runner import train_and_eval

def load_config(custom_fp, device_id, checkpoint_name):
    """
    Load and merge configuration from YAML config file
    This function also handles additional parameters like device ID and checkpoint name
    
    Parameters:
        custom_fp: Custom configuration file path, used to modify or extend template configuration
        device_id: Device ID, may be used for device-related settings
        checkpoint_name: Checkpoint name
    
    Returns:
        run_name: Run name
        data_d: Data configuration dictionary
        model_d: Model configuration dictionary
        run_d: Run configuration dictionary
        config_file_path: Configuration file path for automatic pretrained weight finding
    """
    assert custom_fp, "custom_fp cannot be empty"
    assert os.path.isfile(custom_fp), custom_fp

    # Open custom config file and load YAML content using yaml.load function
    # Using yaml.FullLoader ensures safe file loading
    with open(custom_fp, "r") as custom_file:
        config_d = yaml.load(custom_file, Loader=yaml.FullLoader)

    # Get run name, if None, set it to custom config filename (without extension)
    run_name = config_d.get("run_name")
    if run_name is None:
        run_name = os.path.splitext(os.path.basename(custom_fp))[0]
        config_d["run_name"] = run_name

    data_d = config_d["data"]
    model_d = config_d["model"]
    run_d = config_d["run"]

    # Overwrite device if necessary
    if device_id is not None:
        if device_id < 0:
            run_d["device"] = "cpu"
        else:
            run_d["device"] = f"cuda:{device_id}"

    # Overwrite checkpoint if necessary
    if checkpoint_name:
        model_d["checkpoint_name"] = checkpoint_name

    # Add custom_fp filename (without .yml extension) to run_d with key name custom_name
    custom_name = os.path.splitext(os.path.basename(custom_fp))[0]
    run_d["custom_name"] = custom_name

    # Return config file path for automatic pretrained weight finding
    config_file_path = custom_fp if custom_fp else ""
    return run_name, data_d, model_d, run_d, config_file_path

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--device_id", type=int, required=False, help="device id (-1 for cpu)")
    parser.add_argument("-c", "--custom_fp", type=str, required=False, help="path to custom config file")
    parser.add_argument("-k", "--checkpoint_name", type=str, required=False, help="name of checkpoint to load (from checkpoint_dp)")
    parser.add_argument("-n", "--num_splits", type=int, default=1, help="number of different split seeds to run")
    flags = parser.parse_args()

    run_name, data_d, model_d, run_d, config_file_path = load_config(flags.custom_fp, flags.device_id, flags.checkpoint_name)

    base_split_seed = run_d["split_seed"]
    
    if flags.num_splits > 1:
        # Define multiple split seeds for cross-validation or multiple runs
        split_seeds = [520, 521, 522, 523, 524][:flags.num_splits]
    else:
        split_seeds = [base_split_seed]
    
    print(f"Will run {len(split_seeds)} training sessions with the following split_seeds: {split_seeds}")
    
    # Run training for each split seed
    for i, split_seed in enumerate(split_seeds):
        print(f"\n\n===== Running training {i+1}/{len(split_seeds)}, split_seed: {split_seed} =====\n")
    
        run_d["split_seed"] = split_seed
    
        train_and_eval(data_d, model_d, run_d, config_file_path)
        
        print(f"\n===== Completed training {i+1}/{len(split_seeds)} =====\n")
    
    # Example usage: python src/train.py -c config/nist23_P.yml
