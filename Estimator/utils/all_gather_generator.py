import os
import argparse

from chakra.schema.protobuf.et_def_pb2 import (
    ALL_GATHER,
    ALL_REDUCE,
    ALL_TO_ALL,
    BARRIER,
    BROADCAST,
    REDUCE,
    COMM_COLL_NODE,
    COMM_RECV_NODE,
    COMM_SEND_NODE,
    COMP_NODE,
    MEM_LOAD_NODE,
    MEM_STORE_NODE,
    METADATA_NODE,
    REDUCE_SCATTER,
    BoolList,
    BytesList,
    DoubleList,
    Fixed32List,
    Fixed64List,
    FloatList,
    GlobalMetadata,
    Int32List,
    Int64List,
    Sfixed32List,
    Sfixed64List,
    Sint32List,
    Sint64List,
    StringList,
    Uint32List,
    Uint64List,
)
from chakra.schema.protobuf.et_def_pb2 import (
    AttributeProto as ChakraAttr,
)
from chakra.schema.protobuf.et_def_pb2 import (
    Node as ChakraNode,
)
from chakra.schema.protobuf.et_def_pb2 import (
    NodeType as ChakraNodeType,
)
from chakra.src.third_party.utils.protolib import encodeMessage as encode_message

NODE_ID = 0

def get_node(node_name: str, node_type: ChakraNodeType) -> ChakraNode:
    """Generate a new ChakraNode with a unique ID."""
    global NODE_ID
    node = ChakraNode()
    node.id = NODE_ID
    node.name = node_name
    node.type = node_type
    NODE_ID += 1
    return node

def get_comm_type_attr(comm_type: int) -> ChakraAttr:
    """Create a communication type attribute."""
    return ChakraAttr(name="comm_type", int64_val=comm_type)

def generate_allgather_files(npus_count: int, coll_size: int, gather_ranks: list) -> None:
    """
    Function to generate Chakra ET files for all-gather communication.

    :param npus_count: Number of NPUs
    :param coll_size: Size of the communication (in bytes)
    """
    # create directories if they don't exist
    script_dir = os.path.dirname(os.path.realpath(__file__))
    target_dir = os.path.join(script_dir, "tmp/network/allgather")
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
    
    for npu_id in range(npus_count):
        output_filename = os.path.join(target_dir, f"allgather.{npu_id}.et")
        with open(output_filename, "wb") as et:
            # Chakra Metadata
            encode_message(et, GlobalMetadata(version="0.0.4"))

            if npu_id in gather_ranks:  # Use the passed list of ranks
                node = get_node("ALL_GATHER", COMM_COLL_NODE)
                node.attr.append(ChakraAttr(name="is_cpu_op", bool_val=False))
                node.attr.extend([get_comm_type_attr(ALL_GATHER), ChakraAttr(name="comm_size", int64_val=coll_size)])
                encode_message(et, node)