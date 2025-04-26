# internal nest code
from .model import BaseModelIR
from ..utils import ShapeProp, NestGraph
from ..utils import store_obj_to_file, load_obj_from_file

# torch module loads
import torch
from transformers import LlamaTokenizer, LlamaForCausalLM, AutoConfig, AutoModelForCausalLM, AutoTokenizer
from transformers.utils import is_torch_fx_available

import os
from pathlib import Path

hf_token = os.getenv("HF_TOKEN")


if is_torch_fx_available():
    from transformers.utils.fx import (
        symbolic_trace as symbolic_trace_transformers,
    )


class LlamaIR(BaseModelIR):
    def __init__(self, model_name="llama2", tmp_width=1):
        super().__init__(model_name, tmp_width)

        self.out_dir = None
        self.graphmodule = None

        self.out_dir = self.create_out_dir()

    def set_model(self):
        self.trace_only_model = True

        if self.model_name == "llama2":
            # Initializing a LLaMA llama-7b style configuration LlamaForCausalLM
            config = AutoConfig.from_pretrained(
                    "meta-llama/Llama-2-7b-hf",
                    token=hf_token,
                    num_hidden_layers=1, 
                    attn_implementation="eager",
                    )

            self.model = AutoModelForCausalLM.from_config(
               config=config)
            self.tokenizer = AutoTokenizer.from_pretrained(
                "meta-llama/Llama-2-7b-hf", token=hf_token, legacy=True)
        elif self.model_name == "llama2-70":
            # Initializing a LLaMA llama-7b style configuration LlamaForCausalLM
            config = AutoConfig.from_pretrained(
                    "meta-llama/Llama-2-70b-hf",
                    token=hf_token,
                    num_hidden_layers=1, 
                    attn_implementation="eager",
                    )

            self.model = AutoModelForCausalLM.from_config(
               config=config)
            self.tokenizer = AutoTokenizer.from_pretrained(
                "meta-llama/Llama-2-70b-hf", token=hf_token, legacy=True)
        elif self.model_name == "llama3":
            # Initializing a LLaMA llama-7b style configuration LlamaForCausalLM
            config = AutoConfig.from_pretrained(
                    "meta-llama/Meta-Llama-3.1-70B",
                    token=hf_token,
                    num_hidden_layers=1, 
                    attn_implementation="eager",
                    )
            self.model = AutoModelForCausalLM.from_config(
               config=config)
            self.tokenizer = AutoTokenizer.from_pretrained(
                "meta-llama/Meta-Llama-3.1-70B", token=hf_token, legacy=True)
            
        else:
            raise TypeError("Model type not found in llama", self.model_name)
        print("done set model")

    def get_model_type(self):
        return "llama"

    def create_out_dir(self):
        curr_dir = Path(__file__).parent.absolute()
        curr_dir = os.path.join(curr_dir, "../out/Llama/")
        isExist = os.path.exists(curr_dir)
        if not isExist:
            # Create a new directory because it does not exist
            os.makedirs(curr_dir)
            print("The new directory is created!")

        return curr_dir

    def get_out_dir(self):
        if not self.out_dir:
            raise ValueError("Out directory not setup for", self.model_name)

        return self.out_dir

    def print_graphmodule(self):
        self.graphmodule.print_readable()

    def obtain_symbolic_trace_model(self, micro_batch_size=1, sequence_length=1):
        input_ids = torch.ones(
            micro_batch_size, sequence_length, dtype=torch.int32)

        self.graphmodule: torch.fx.GraphModule = symbolic_trace_transformers(
            self.model)

        model_shapeprop = ShapeProp(self.graphmodule)
        model_shapeprop.propagate(input_ids)

    def get_layer_id(self, n, curr_layer_id):
        layer_annotations = ["layer", "layers"]

        node_name = n.name
        layer_details = node_name.split("_")
        for l in range(0, len(layer_details)):
            if layer_details[l] in layer_annotations:
                if layer_details[l + 1] and layer_details[l + 1].isdigit():
                    return (True, int(layer_details[l + 1]))
        return (False, 0)

    def create_graph_from_symbolic_trace(self):
        super().create_graph_from_symbolic_trace()
    
    # Note: this code does not add more layers to the graph, just populates the layer info
    def add_more_layer_info(self, ex_num_layers, repeat_layer_id):
        print("\033[96m" + "Extending number of layers (only) in the graph to",
              ex_num_layers, "for model", self.model_name, "\033[0m")
        self.nestgraph.extend_layer_info_sans_graph(
            ex_num_layers, repeat_layer_id)

    def extract_model_graph(self, micro_batch_size=1, sequence_length=64, force_reextract_model=False,):
        self.load_language_model(
            self.out_dir, micro_batch_size, sequence_length, force_reextract_model)
        
        num_layers = 0
        if self.model_name == "llama2":
            num_layers = 32
        elif self.model_name == "llama2-70":
            num_layers = 80
        elif self.model_name == "llama3":
            num_layers = 80

        if num_layers:
            self.add_more_layer_info(num_layers, repeat_layer_id=0)
