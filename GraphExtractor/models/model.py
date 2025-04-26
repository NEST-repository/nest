# internal code import
from ..utils import NestGraph
from ..utils import language_models
from ..utils import store_obj_to_file, load_obj_from_file

supported_models = language_models


class BaseModelIR:
    def __init__(self, model_name, tmp_width=1) -> None:
        self.model = None
        self.summary = None
        self.tmp_width = tmp_width
        self.model_name = model_name

        self.graphmodule = None
        self.nestgraph = NestGraph(self.tmp_width, self.model_name)

        # source of model graph and trace - either extracted model or obtained only the trace from file
        self.trace_only_model = None

        # These values are set when model estimator is executed
        # model tensor properties
        self.model_inputs = None
        self.number_of_parameters = 0
        self.model_outputs = None

    def get_nest_graph(self):
        return self.nestgraph

    def get_op_graph(self, layer_id):
        return self.nestgraph.get_op_layer_graph(layer_id)

    def get_op_graphs(self):
        return self.nestgraph.get_op_layer_graphs()

    def get_unique_op_graphs(self):
        return self.nestgraph.get_unique_op_graphs()

    def get_layer_ids(self):
        return self.nestgraph.get_layer_ids()

    def get_layer_graph(self):
        return self.nestgraph.get_layer_graph()

    def is_layer_info_extended(self):
        try:
            return self.nestgraph.layer_info_extended
        except:
            return False

    def create_graph_from_symbolic_trace(self):
        curr_layer_id = -1
        graphmodule = self.graphmodule.graph

        layer_ids = set()

        # create nodes in the graph
        g = self.get_nest_graph()

        for n in graphmodule.nodes:
            (if_layer_identifier, layer_id) = self.get_layer_id(n, curr_layer_id)
            if if_layer_identifier:
                curr_layer_id = layer_id
            g.setup_node_and_edges(graphmodule, n, curr_layer_id)

            # add layer information
            layer_ids.add(curr_layer_id)

        g.set_layer_ids(layer_ids)

    def get_repeat_layer_ids(self):
        g = self.get_nest_graph()
        return g.get_repeat_layer_ids()

    def generate_layer_info(self):
        g = self.nestgraph

        if g is None:
            raise ValueError(
                "Model for model name" + self.model_name,
                "and tensor model parallel width" + self.tmp_width + "does not exist",
            )

        g.set_repeat_layer_ids(language_models)
        g.generate_layer_info()
        g.set_op_layer_graphs()
        g.contract_layer_graph()

    # This function generates the memory footprint for layer/layer slice, latter if tmp_width > 1
    # 1. weights (times 2, plus optimizer state)
    # 2. peak memory usage, DEFINED AS sum of output activations across all nodes in the layer/layer slice (ergo, equal for fw and bw pass)
    # 3. input size to each layer/layer slice

    def get_layerwise_memory_footprint(self):
        g = self.nestgraph
        return g.get_layerwise_memory_footprint()

    # This function accumulates memory footprint for the entire model
    def get_memory_footprint(self):
        g = self.nestgraph

        print(g.get_number_of_parameters())

        return g.get_memory_footprint()

    def set_model(self):
        raise NotImplementedError("Must override set_model")

    def obtain_symbolic_trace_model(self, micro_batch_size, sequence_length):
        raise NotImplementedError("Must override obtain_symbolic_trace_model")

    def get_model_type(self):
        raise NotImplementedError("Must override get_out_dir")

    def get_out_dir(self):
        raise NotImplementedError("Must override get_out_dir")

    def get_layer_id(self, node, curr_layer_id=-1):
        raise NotImplementedError("Must override get_layer_id")

    def load_language_model(self, out_dir, micro_batch_size=1, sequence_length=64, force_reextract_model=False,):

        if self.tmp_width is None:
            raise ValueError("Tensor Model Width is set as None")

        nestgraph = None
        if not force_reextract_model:
            nestgraph = load_obj_from_file(
                self.model_name, micro_batch_size, self.tmp_width, sequence_length, module_type="graph",)

        if isinstance(nestgraph, NestGraph):
            print("Loaded nest graph successfully from file for model",
                  self.model_name,)
            self.nestgraph = nestgraph
            # this implies self.model and self.graphmodule is None but nestgraph is populated
            self.trace_only_model = True
        else:
            self.set_model()
            self.obtain_symbolic_trace_model(micro_batch_size, sequence_length)
            self.create_graph_from_symbolic_trace()
            store_obj_to_file(self.model_name, self.nestgraph, micro_batch_size,
                              self.tmp_width, sequence_length, module_type="graph",)

        self.generate_layer_info()

        g = self.get_nest_graph()

        # if g.get_num_nodes() < 1000:
        #g.print_graph(out_dir, micro_batch_size, sequence_length)
        #g.print_graph(out_dir, micro_batch_size, sequence_length)

        # print the memory of the model
        # model_memory = g.get_memory_footprint()
        # print("Model parameter size", model_memory.parameter_size)

        g.print_layer_graph(out_dir, micro_batch_size, sequence_length)

        # storing entire models can take a lot of space
        # store_obj_to_file(self.model_name, self, micro_batch_size,
        #                  self.tmp_width, sequence_length,)
