from collections import namedtuple
import json
import math
import subprocess
import os
import re

from .all_reduce_generator import generate_allreduce_files
from .all_gather_generator import generate_allgather_files
from .reduce_scatter_generator import generate_reducescatter_files
from .all_to_all_generator import generate_alltoall_files


def workload_config_generation(comm_grp_ip, operation):
    if(operation == 0):
        generate_allreduce_files(comm_grp_ip.sub_network.num_nodes, comm_grp_ip.datasize, comm_grp_ip.node_ids)
        return "allreduce"
    elif(operation == 1):
        print(comm_grp_ip.sub_network.num_nodes)
        print(comm_grp_ip.datasize)
        print(comm_grp_ip.node_ids)
        generate_allgather_files(comm_grp_ip.sub_network.num_nodes, comm_grp_ip.datasize, comm_grp_ip.node_ids)
        return "allgather"
    elif(operation == 2):
        generate_reducescatter_files(comm_grp_ip.sub_network.num_nodes, comm_grp_ip.datasize, comm_grp_ip.node_ids)
        return "reducescatter"
    elif(operation == 3):
        generate_alltoall_files(comm_grp_ip.sub_network.num_nodes, comm_grp_ip.datasize, comm_grp_ip.node_ids)
        return "alltoall"
    

def physical_network_generation(sub_network):
    #Given the subnetwork generate the physical network txt file
    script_dir = os.path.dirname(os.path.realpath(__file__))
    target_dir = os.path.join(script_dir, "tmp/network/inputs")
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    file_name = os.path.join(target_dir, f'physical_topology.txt')
    with open(file_name, "w") as file:
        num_nodes = sub_network.num_nodes
        sub_network.dimension.append(1)
        num_switches = sum(sub_network.dimension[i+1] * sub_network.dimension[i+2] for i in range(len(sub_network.dimension) - 2)) + 1
        num_links = 0
        for i in range(len(sub_network.dimension)-1):
            result = 1
            for dim in sub_network.dimension[i:]:
                result *= dim
            num_links += result
        print(f'{num_nodes+num_switches} {num_switches} {num_links}', file=file)
        start_idxs = [-1] * (len(sub_network.dimension))
        start_idxs[0] = 0
        tot_nodes = 0
        for i in range(len(sub_network.dimension)):
            result = 1
            for j in sub_network.dimension[i:]:
                result *= j
            if(i != len(sub_network.dimension)-1):
                tot_nodes += result
                start_idxs[i+1] = tot_nodes
        switch_ids = list(range(start_idxs[1], (num_nodes + num_switches)))
        print(" ".join(map(str, switch_ids)), file=file)


        for i in range(len(start_idxs)-1):
            if(i < len(start_idxs)-2):
                sub_switches = list(range(start_idxs[i+1], start_idxs[i+2]))
            else:
                sub_switches = [start_idxs[i+1]]
            num_children = sub_network.dimension[i]
            move_nodes = 0
            for switch in sub_switches:
                for child in range(num_children):
                    print(f'{switch} {start_idxs[i] + child + move_nodes} {sub_network.bandwidth[i]}Gbps {sub_network.latency[i]}ms 0', file=file)
                move_nodes += num_children

    return "physical_topology.txt"

def insert_interpolated(mapping, new_key):
    if new_key in mapping:
        return  # Already exists, do nothing

    keys = sorted(mapping.keys())
    if new_key < keys[0]:
        mapping[new_key] = mapping[keys[0]]
    elif new_key > keys[-1]:
        mapping[new_key] = mapping[keys[-1]]
    else:
        # Find the two surrounding keys
        for i in range(1, len(keys)):
            if keys[i-1] < new_key < keys[i]:
                val1 = mapping[keys[i-1]]
                val2 = mapping[keys[i]]
                mapping[new_key] = (val1 + val2) / 2
                break

def format_map(name, mapping):
    items = [f"{key} {value}" for key, value in sorted(mapping.items())]
    return f"{name} {len(mapping)} " + " ".join(items)

def update_all_maps(kmax_map, kmin_map, pmax_map, new_key):
    insert_interpolated(kmax_map, new_key)
    insert_interpolated(kmin_map, new_key)
    insert_interpolated(pmax_map, new_key)

def network_config_generation(sub_network):
    #Generate the physical network txt file 
    physical_network_file = physical_network_generation(sub_network)

    kmax_map = {
    12500000000: 400,
    25000000000: 400,
    40000000000: 800,
    100000000000: 1600,
    200000000000: 2400,
    400000000000: 3200,
    900000000000: 3200,
    2400000000000: 3200
    }
    kmin_map = {
        12500000000: 100,
        25000000000: 100,
        40000000000: 200,
        100000000000: 400,
        200000000000: 600,
        400000000000: 800,
        900000000000: 800,
        2400000000000: 800
    }
    pmax_map = {
        12500000000: 0.2,
        25000000000: 0.2,
        40000000000: 0.2,
        100000000000: 0.2,
        200000000000: 0.2,
        400000000000: 0.2,
        900000000000: 0.2,
        2400000000000: 0.2
    }

    for i in range(len(sub_network.bandwidth)):
        update_all_maps(kmax_map, kmin_map, pmax_map, sub_network.bandwidth[i]*1000000000)

    script_dir = os.path.dirname(os.path.realpath(__file__))
    target_dir = os.path.join(script_dir, "tmp/network/inputs")

    #Create the network config file including the name of the physical network file 
    #obtained from physical_network_generation
    config_content = f"""ENABLE_QCN 1
USE_DYNAMIC_PFC_THRESHOLD 1

PACKET_PAYLOAD_SIZE 1000

TOPOLOGY_FILE {script_dir}/tmp/network/inputs/{physical_network_file}
FLOW_FILE ../../scratch/output/flow.txt
TRACE_FILE ../../scratch/output/trace.txt
TRACE_OUTPUT_FILE ../../scratch/output/mix.tr
FCT_OUTPUT_FILE ../../scratch/output/fct.txt
PFC_OUTPUT_FILE ../../scratch/output/pfc.txt
QLEN_MON_FILE ../../scratch/output/qlen.txt
QLEN_MON_START 0
QLEN_MON_END 20000

SIMULATOR_STOP_TIME 40000000000000.00

CC_MODE 12
ALPHA_RESUME_INTERVAL 1
RATE_DECREASE_INTERVAL 4
CLAMP_TARGET_RATE 0
RP_TIMER 900 
EWMA_GAIN 0.00390625
FAST_RECOVERY_TIMES 1
RATE_AI 50Mb/s
RATE_HAI 100Mb/s
MIN_RATE 100Mb/s
DCTCP_RATE_AI 1000Mb/s

ERROR_RATE_PER_LINK 0.0000
L2_CHUNK_SIZE 4000
L2_ACK_INTERVAL 1
L2_BACK_TO_ZERO 0

HAS_WIN 1
GLOBAL_T 0
VAR_WIN 1
FAST_REACT 1
U_TARGET 0.95
MI_THRESH 0
INT_MULTI 1
MULTI_RATE 0
SAMPLE_FEEDBACK 0
PINT_LOG_BASE 1.05
PINT_PROB 1.0
NIC_TOTAL_PAUSE_TIME 0

RATE_BOUND 1

ACK_HIGH_PRIO 0

LINK_DOWN 0 0 0

ENABLE_TRACE 1

{format_map("KMAX_MAP", kmax_map)}
{format_map("KMIN_MAP", kmin_map)}
{format_map("PMAX_MAP", pmax_map)}

BUFFER_SIZE 32"""

    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    file_name = os.path.join(target_dir, f'config_clos.txt')
    with open(file_name, "w") as file:
        file.write(config_content)

    #create logical topology file based on the number of nodes in the subnetwork
    logical_topo_file = logical_network_generation(sub_network)

    #return the name of the network config file generated
    return "config_clos.txt"

def system_config_generation():
    data = {
        "scheduling-policy": "LIFO",
        "endpoint-delay": 10,
        "active-chunks-per-dimension": 1,
        "preferred-dataset-splits": 4,
        "all-reduce-implementation": ["ring"],
        "all-gather-implementation": ["ring"],
        "reduce-scatter-implementation": ["ring"],
        "all-to-all-implementation": ["ring"],
        "collective-optimization": "localBWAware",
        "local-mem-bw": 50,
        "boost-mode": 0
    }

    script_dir = os.path.dirname(os.path.realpath(__file__))
    target_dir = os.path.join(script_dir, "tmp/network/inputs")
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    file_name = os.path.join(target_dir, f'Ring_sys.json')
    with open(file_name, "w") as file:
        json.dump(data, file, indent=4)

def remote_memory_config_generation():
    data = {
        "memory-type": "NO_MEMORY_EXPANSION"
    }

    
    script_dir = os.path.dirname(os.path.realpath(__file__))
    target_dir = os.path.join(script_dir, "tmp/network/inputs")

    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    file_name = os.path.join(target_dir, f'RemoteMemory.json')

    with open(file_name, "w") as file:
        json.dump(data, file, indent=4)
    
def run_file_generation(workload_config_file, network_config_file, comm_grp_file):
    script_dir = os.path.dirname(os.path.realpath(__file__))
    target_dir = os.path.join(script_dir, "tmp/network")

    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
    run_sh_file = os.path.join(target_dir, f"run_{workload_config_file}.sh")
    
    # Define the ASTRA SIM path
    astra_sim_path = os.path.join(script_dir, "../../third_party_for_nest/astra-sim/extern/network_backend/ns-3/build/scratch/")

    script_content = f"""#!/bin/bash
set -e

# Path
SCRIPT_DIR=$(dirname "$(realpath $0)")
ASTRA_SIM_BUILD_DIR={astra_sim_path}
ASTRA_SIM=./ns3.42-AstraSimNetwork-default

# Run ASTRA-sim
(           
cd ${{ASTRA_SIM_BUILD_DIR}}
touch ../../scratch/output/flow.txt
${{ASTRA_SIM}} \\
    --workload-configuration=${{SCRIPT_DIR}}/{workload_config_file}/{workload_config_file} \\
    --system-configuration=${{SCRIPT_DIR}}/inputs/Ring_sys.json \\
    --remote-memory-configuration=${{SCRIPT_DIR}}/inputs/RemoteMemory.json \\
    --logical-topology-configuration=${{SCRIPT_DIR}}/inputs/logical_topology.json \\
    --network-configuration=${{SCRIPT_DIR}}/inputs/config_clos.txt \\
    --comm-group-configuration=${{SCRIPT_DIR}}/inputs/comm_grp_config.json
)"""

    # Write the script content to run.sh
    with open(run_sh_file, "w") as file:
        file.write(script_content)
    
    # Make the script executable
    os.chmod(run_sh_file, 0o755)  # This sets the script to be executable

def logical_network_generation(sub_network):
    #Given the subnetwork generate the logical network json file
    # Define the content for the JSON file
    num_nodes = sub_network.num_nodes
    logical_data = {
    "logical-dims": [str(num_nodes)]
    }

    script_dir = os.path.dirname(os.path.realpath(__file__))
    target_dir = os.path.join(script_dir, "tmp/network/inputs")
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    file_name = os.path.join(target_dir, f'logical_topology.json')
    with open(file_name, "w") as file:
        json.dump(logical_data, file, separators=(",", ": "))

    #return the name of the logical network json file generated
    return "logical_topology.json"

def comm_groups_config_generaton(node_ids):

    # Define the content for the JSON file
    comm_group_data = {
        "0": node_ids
    }

    # Create and write the JSON content to the file 'comm_grp_config.json'
    script_dir = os.path.dirname(os.path.realpath(__file__))
    target_dir = os.path.join(script_dir, "tmp/network/inputs")

    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    file_name = os.path.join(target_dir, f'comm_grp_config.json')

    with open(file_name, "w") as file:
        json.dump(comm_group_data, file, separators=(",", ": "))

    # print("comm_grp_config.json file has been created.")
    return "comm_grp_config.json"

def run_shell_script(script_path, output_file, timeout_duration):
    try:
        # Ensure the output directory exists
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, 'w') as f:
            result = subprocess.run(
                ['bash', script_path],
                stdout=f,
                stderr=subprocess.STDOUT,
                check=True,
                timeout=timeout_duration  # Set timeout in seconds
            )
    except FileNotFoundError as e:
        print(f"Error: {e}")  # Handle missing script file
    except subprocess.TimeoutExpired:
        print(f"Script timed out after {timeout_duration} seconds and was terminated.")
        pass
    except subprocess.CalledProcessError as e:
        print(f"Error occurred: {e}")

def extract_numbers_from_file(file_path):
    # Initialize an empty list to store the extracted numbers
    numbers = []
    
    # Open the file and read line by line
    with open(file_path, 'r') as file:
        for line in file:
            match = re.search(r'finished,\s*(\d+)\s*cycles', line)
            if match:
                # Append the matched number to the list
                numbers.append(int(match.group(1)))
    
    return numbers

def equal_numbers(lst):
    if len(set(lst)) == 1:
        return lst[0]  # Return the common number
    return None  # Return None if numbers are not the same

def communication_estimator(comm_grp_ip, operation):
    #send additional input to this function
    workload_config_file = workload_config_generation(comm_grp_ip, operation)
    network_config_file = network_config_generation(comm_grp_ip.sub_network)
    print("completed network file")
    system_config_generation()
    remote_memory_config_generation()
    comm_grp_file = comm_groups_config_generaton(comm_grp_ip.node_ids)

    #send additional input to this function
    run_file_generation(workload_config_file, network_config_file, comm_grp_file)

    script_dir = os.path.dirname(os.path.realpath(__file__))
    target_dir = os.path.join(script_dir, "tmp/network")
    run_shell_script(os.path.join(target_dir, f"run_{workload_config_file}.sh"), os.path.join(target_dir, "output.txt"), 7200)

    num_cycles = extract_numbers_from_file(os.path.join(target_dir,"output.txt"))
    if(len(num_cycles) == 0):
        print("No cycles found in the output file.")
        return 0
    else:
        return max(num_cycles)

network_config = namedtuple(
    "network_config", ["topology", "num_nodes", "dimension", "bandwidth", "latency", "utilization", "switches"])
comm_group_config = namedtuple(
    "comm_group_config", ["node_ids", "sub_network", "datasize"])
