# Interactive MPE Tool

Dependencies
make sure to git clone the following repository into the same root that this repository stays in
```
git clone https://github.com/RinRin-32/memory-perturbation/tree/memorymaps_exp
```

## Installation

- To create a conda environment `mpe` with all necessary dependencies run: `conda env create --file environment.yml`
- We use torch 2.2.1 with cuda 12.1.1
- The current environment.yml file limits python to be lower than version 3.13

## Before you launch a Bokeh server
Validate your h5 file with
```
python ./validate.py --file <path to your h5 file> --project <mpe_server/evolving_server>
```

### Expected response for correct h5 format for mpe_server
```
The HDF5 file './mpe_data.h5' is valid for the 'mpe_server' project.
Structure of the HDF5 file './mpe_data.h5':
Group: config
Dataset: config/config_data | Shape: () | Dtype: object
Group: scores
Dataset: scores/X_train | Shape: (800, 2) | Dtype: float32
Dataset: scores/bls | Shape: (800,) | Dtype: float32
Dataset: scores/bpe | Shape: (800,) | Dtype: float32
Dataset: scores/indices_retrain | Shape: (800,) | Dtype: int64
Dataset: scores/sensitivities | Shape: (800,) | Dtype: float64
Dataset: scores/softmax_deviations | Shape: (800,) | Dtype: float64
Dataset: scores/y_train | Shape: (800,) | Dtype: int64
```

### Expected response for correct h5 format for evolving_server
```
The HDF5 file './evolving_data.h5' is valid for the 'evolving_server' project.
Structure of the HDF5 file './evolving_data.h5':
Group: config
Dataset: config/config_data | Shape: () | Dtype: object
Group: coord
Dataset: coord/X_train | Shape: (800, 2) | Dtype: float32
Dataset: coord/y_train | Shape: (800,) | Dtype: int64
Group: scores
  Found 124 steps (e.g., 'step_0', 'step_1', ...) under scores
  Example step: scores/step_0 | Datasets: bls, bpe, decision_boundary, sensitivities, softmax_deviations
```

## Serving your Bokeh server
```
bokeh serve --show <server you'd like to run> --args --file <path/to/your/experiment>.h5
```