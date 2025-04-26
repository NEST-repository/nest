# Nest

## Installation

To install the dependencies for Nest, run:

```bash
./setup.sh
```

Install Astra sim and Chakra into third_party_for_nest
Refer for Installation:
[Astra Sim](https://astra-sim.github.io/astra-sim-docs/getting-started/build.html)
[Chakra](https://github.com/astra-sim/chakra/tree/214f2c559c10f897bcc395f8e1502d80d14f1541)

- Nest uses Gurobi 10.0.1 to solve the ILP formulations. To run the ILP solver, obtain a Gurobi license from the [The Gurobi Website](https://www.gurobi.com/).

Add the following path variables in `~/.bashrc`:
```bash
export THIRD_PARTY_PATH=$(pwd)/nest/third_party_for_nest
export WHAM_PATH=$THIRD_PARTY_PATH/wham/
export SUNSTONE_PATH=$THIRD_PARTY_PATH/sunstone/
export PYTHONPATH=$THIRD_PARTY_PATH:$WHAM_PATH:$SUNSTONE_PATH:$PYTHONPATH
export CHAKRA_PATH=$THIRD_PARTY_PATH/chakra/
export PYTHONPATH=$THIRD_PARTY_PATH:$CHAKRA_PATH:$PYTHONPATH
export LD_LIBRARY_PATH=$THIRD_PARTY_PATH/astra-sim/extern/network_backend/ns-3/build/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$CONDA_ENVS_PATH/nest/lib/:$LD_LIBRARY_PATH
export PYTHONPATH=$(pwd)/nest/Estimator/utils/:$PYTHONPATH
```

## Quick Start

We provide scripts to run the experiments described in the paper.

```bash
cd scripts
./<model.sh> "<microbatch_sizes>"
```

## Nest Execution and Code Structure

Nest can be executed with the following command:
```bash
python3 nest.py --nest_model <model_name> --nest_exec_type <execution_mode> 
 --nest_micro_batch_size <microbatch_sizes> --nest_max_tmp_width <tmp> \
--nest_sequence_length <seq_len>  --nest_hbm_size <hbm> --nest_exp_parallel_degree <ep_degree>
```

### Inputs
- `model_name` = Bert, GPT, OPT, llama2, llama3, mixtral variants
- `execution_mode` = ["run_solver", "prepopulate_estimates", "extract_graph"]
- `seq_len`= Sequence length of the model
- `micro_batch_size` = List of microbatch sizes to explore
- `max_tmp_width` = Maximum Tensor Model Parallel width for megatron models
- `exp_parallel_degree` = Expert Parallel Degree for Expert Parallel supported models
- `hbm_size` = High Bandwidth Memory size (GB)

### Code Structure
```bash
/                           : NEST_ROOT
|-- GraphExtractor          : Extract model operator graphs
|-- Estimator               : Generate architectures and estimate latencies
|-- Solver                  : ILP and DP solver
|   |-- device_placement
|   |   |-- device_placement.cpp
|-- third_party_for_nest
|   |-- Wham                : For operator mapping and estimating area
|   |-- Sunstone            : For estimating operator latency
|   |-- Megatron            : For Megatron Models
|-- nest.py                : Python source for Nest
```