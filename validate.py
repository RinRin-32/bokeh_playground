import argparse
import h5py
import os
import sys
import json


def print_h5_structure(h5_file, project_type):
    """Print the structure of the HDF5 file."""
    try:
        with h5py.File(h5_file, "r") as f:
            def print_group(name, obj):
                """Recursive function to print the structure of the HDF5 file."""
                if isinstance(obj, h5py.Group):
                    print(f"Group: {name}")
                    for key, val in obj.items():
                        print_group(f"{name}/{key}", val)
                elif isinstance(obj, h5py.Dataset):
                    print(f"Dataset: {name} | Shape: {obj.shape} | Dtype: {obj.dtype}")

            print(f"Structure of the HDF5 file '{h5_file}':")
            for name, obj in f.items():
                if project_type == "evolving_server" and "scores" in name and isinstance(obj, h5py.Group):
                    # Print only one of the steps and the total count of 'step_{n}'
                    print(f"Group: {name}")
                    steps = [key for key in obj.keys() if key.startswith("step_")]
                    if steps:
                        print(f"  Found {len(steps)} steps (e.g., 'step_0', 'step_1', ...) under {name}")
                        step_example = steps[0]
                        print(f"  Example step: {name}/{step_example} | Datasets: {', '.join(obj[step_example].keys())}")
                    else:
                        print(f"  No 'step_' groups found under {name}")
                else:
                    print_group(name, obj)

            # Print the content of 'config' group if it exists
            if "config" in f:
                print(f"\nContent of the 'config' group:")
                config_data = f["config"].get("config_data")
                if config_data:
                    config_json = config_data[()].decode('utf-8')
                    config = json.loads(config_json)
                    print(json.dumps(config, indent=4))
                else:
                    print("  'config_data' not found in 'config' group.")

    except Exception as e:
        print(f"Error: Failed to read HDF5 file '{h5_file}' to print its structure. {str(e)}")


def validate_evolving_server(h5_file):
    try:
        with h5py.File(h5_file, "r") as f:
            # Check for essential groups and datasets for evolving_server.py
            required_groups = ['config', 'coord', 'scores']
            for group in required_groups:
                if group not in f:
                    print(f"Error: Missing required group '{group}' in the HDF5 file for evolving_server.py.")
                    return False

            # Validate required datasets in 'coord' group
            if 'X_train' not in f['coord'] or 'y_train' not in f['coord']:
                print("Error: Missing required datasets 'X_train' or 'y_train' in 'coord' group for evolving_server.py.")
                return False

            # Check 'scores' group for essential datasets
            total_steps = f['config']['config_data'][()].decode('utf-8')
            config = json.loads(total_steps)
            for step in range(config.get("total_step", 0)):
                step_group = f.get(f"scores/step_{step}")
                if not step_group:
                    print(f"Error: Missing 'step_{step}' group in 'scores' for evolving_server.py.")
                    return False
                required_datasets = ['bpe', 'bls', 'softmax_deviations', 'decision_boundary']
                for dataset in required_datasets:
                    if dataset not in step_group:
                        print(f"Error: Missing '{dataset}' in 'step_{step}' group for evolving_server.py.")
                        return False
    except Exception as e:
        print(f"Error: Failed to read HDF5 file '{h5_file}' for evolving_server.py. {str(e)}")
        return False

    return True


def validate_mpe_server(h5_file):
    try:
        with h5py.File(h5_file, "r") as f:
            # Check for essential groups and datasets for mpe_server.py
            scores_group = f.get("scores")
            if not scores_group:
                print("Error: Missing 'scores' group in the HDF5 file for mpe_server.py.")
                return False

            required_datasets = ['X_train', 'y_train', 'sensitivities', 'softmax_deviations', 'bpe', 'bls']
            for dataset in required_datasets:
                if dataset not in scores_group:
                    print(f"Error: Missing '{dataset}' dataset in 'scores' group for mpe_server.py.")
                    return False

            # Check the config group
            if "config" not in f or "config_data" not in f["config"]:
                print("Error: Missing 'config' or 'config_data' in the HDF5 file for mpe_server.py.")
                return False

            # Load config to validate further
            read = f["config"]["config_data"][()]
            config_json = read.decode("utf-8")
            config = json.loads(config_json)

    except Exception as e:
        print(f"Error: Failed to read HDF5 file '{h5_file}' for mpe_server.py. {str(e)}")
        return False

    return True


def validate_h5_file(h5_file, project_type):
    # Validate the file based on the selected project type
    if project_type == "evolving_server":
        return validate_evolving_server(h5_file)
    elif project_type == "mpe_server":
        return validate_mpe_server(h5_file)
    else:
        print(f"Error: Unknown project type '{project_type}'. Please specify either 'evolving_server' or 'mpe_server'.")
        return False


def main():
    parser = argparse.ArgumentParser(description="Validate HDF5 file for a specified project.")
    parser.add_argument("--file", type=str, required=True, help="Path to the HDF5 file")
    parser.add_argument("--project", type=str, required=True, choices=["evolving_server", "mpe_server"], help="Specify the project type")
    args = parser.parse_args()

    h5_file = args.file
    project_type = args.project

    # Validate the HDF5 file for the specified project
    if validate_h5_file(h5_file, project_type):
        print(f"The HDF5 file '{h5_file}' is valid for the '{project_type}' project.")
        # Print the structure of the HDF5 file
        print_h5_structure(h5_file, project_type)
    else:
        sys.exit(1)


if __name__ == "__main__":
    main()