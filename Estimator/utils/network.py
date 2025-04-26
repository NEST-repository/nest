from collections import namedtuple
import math
from functools import reduce
import operator
from .comm_estimator import communication_estimator

total_devices = 1024
bandwidths_per_level = [900, 100, 400] #GBps
latency_per_level = [0.0001,0.0002,0.0003] #s
devices_per_level = [8, 4, 32]
bws = bandwidths_per_level.copy()
bws = [x*1024*1024*1024 for x in bandwidths_per_level]

# Full network configuration tuple
# Topology: string - specifies the base topology of the network
# Num_nodes: int - total number of nodes in the network
# Dimension: [] - number of nodes in each level 
# Note: the product of all dimensions should equal to Num_nodes
#  [2,2,1] for Hierarchical: 
#  level 2:              switch
#                   |            |
# Level 1:       switch        switch
#                |     |      |     |
#               gpu   gpu    gpu   gpu 
# level 0:       nvswitch    nvswitch
#
# Bandwidth: [] - bw of the links per each dimension
#   index 0 = nvswitch (dgx box) bw
#   index 1 = bw of gpu to first IB switch
# Latency:[] - lantecy of the links (similar structure as BW)

network_config = namedtuple(
    "network_config", ["topology", "num_nodes", "dimension", "bandwidth", "latency", "utilization", "switches"])

comm_group_config = namedtuple(
    "comm_group_config", ["node_ids", "sub_network", "datasize"])

supported_topology = ["Hierarchical"]

# Initialization
full_network = -1


def create_full_network(topology, num_nodes, dimension, bandwidth, latency, utilization, switches):
    if(topology in supported_topology):
        full_network = network_config(topology, num_nodes, dimension, bandwidth, latency, utilization, switches)
        topo_support = 1
    else:
        print(f"{topology} is not supported")
        full_network = -1
        topo_support = 0
    return full_network, topo_support

def create_comm_group(node_ids, datasize, operation):
    # node_ids = [] of nodes involded in the all reduce 
    sub_network, mod_node_ids = create_partial_network(node_ids)
    #reassigning node ids
    comm_group = comm_group_config(mod_node_ids, sub_network, datasize)
    cycles = get_comm_estimate(comm_group, operation)
    return cycles

def create_partial_network(node_ids):
    # IDEA: create the smallest subset of the network that includes lowest nodes involved in the communication
    all_level_ids = []
    full_network.dimension.append(1)
    for i in range(len(node_ids)):
        level_ids = [-1] * (len(full_network.dimension)-1)
        for j in range(len(level_ids)):
            if(j == 0):
                level_ids[j] = int(node_ids[i]/full_network.dimension[j])
            else:
                level_ids[j] = int(level_ids[j-1]/full_network.dimension[j])
        all_level_ids.append(level_ids)

    source_idx, source_level = next(((all_level_ids[0][i], i) for i in range(len(all_level_ids[0]))
                               if all(lst[i] == all_level_ids[0][i] for lst in all_level_ids)), (None, None))
    source_level += 1
    full_network.dimension.pop()
    partial_num_nodes = reduce(operator.mul, full_network.dimension[:source_level], 1)
    partial_dimension = full_network.dimension[:source_level]
    partial_network = network_config("Hierarchical", partial_num_nodes, partial_dimension, full_network.bandwidth[:source_level], full_network.latency[:source_level], full_network.utilization[:source_level], full_network.switches[:source_level])
    #reassigning node ids
    result = 1
    for i in range(len(partial_dimension)):
        result *= partial_dimension[i]
    min_node_idx = source_idx * result
    mod_node_ids = [-1]*len(node_ids)
    for i in range(len(node_ids)):
        mod_node_ids[i] = node_ids[i] - min_node_idx
    return partial_network, mod_node_ids

def get_comm_estimate(comm_group_config, operation):
    # call network simulator to get estimation 
    cycles = communication_estimator(comm_group_config, operation)
    return cycles

def get_allreduce_latency(node_ids, datasize):
    if(topo_support):
        cycles = create_comm_group(node_ids, datasize, 0)
        return cycles

def get_alltoall_latency(node_ids, datasize):
    if(topo_support):
        cycles = create_comm_group(node_ids, datasize, 3)
        return cycles

def get_allgather_latency(node_ids, datasize):
    if(topo_support):
        cycles = create_comm_group(node_ids, datasize, 1)
        return cycles

def get_reduce_scatter_latency(node_ids, datasize):
    if(topo_support):
        cycles = create_comm_group(node_ids, datasize, 2)
        return cycles

full_network, topo_support = create_full_network("Hierarchical", total_devices, devices_per_level, bandwidths_per_level, latency_per_level, [1,1,1], [1,1,1])