# internal nest code
from .model import BaseModelIR
from ..utils import ShapeProp, NestGraph
from ..utils import store_obj_to_file, load_obj_from_file, custom_tracer_for_megatron

# torch module loads
import torch
from transformers.utils import is_torch_fx_available

import os, sys
from pathlib import Path


if is_torch_fx_available():
    from transformers.utils.fx import (
        symbolic_trace as symbolic_trace_transformers,
    )

try:
    import pretrain_gpt
    from megatron.training.initialize import initialize_megatron
    from megatron.core.enums import ModelType
    from megatron.training.global_vars import get_args
except:
    Warning("Megatron related models import unsuccessful. Please ensure Megatron support for this model")
    pass

class MixtralIR(BaseModelIR):
    def __init__(self, model_name="mixtral", tmp_width=1, ep_degree=1):
        super().__init__(model_name, tmp_width, ep_degree)

        self.out_dir = None
        self.graphmodule = None
        self.ep_degree = ep_degree

        self.out_dir = self.create_out_dir()

    def set_model(self):
        self.trace_only_model = True

        if self.model_name == "mixtral":
            print(
                    "mixtral_megatron is initialized with each exp degree: ",
                    self.ep_degree,
                )
            self.set_args_megatron()
            self.model = self.megatron_model_provider()
        else:
            raise TypeError("Model type not found in mixtral", self.model_name)

    def get_model_type(self):
        return "mixtral"

    def create_out_dir(self):
        curr_dir = Path(__file__).parent.absolute()
        curr_dir = os.path.join(curr_dir, "../out/Mixtral/")
        isExist = os.path.exists(curr_dir)
        if not isExist:
            # Create a new directory because it does not exist
            os.makedirs(curr_dir)
            print("The new directory is created!")

        return curr_dir

    def megatron_model_provider(
        self,
        pre_process=True,
        post_process=True,
    ):
        """Build the megatron model."""

        if not torch.cuda.is_available():
            ValueError(
                "Cuda is not available on the machine, cannot extract megatron graph")

        device = torch.device("cuda")

        args = get_args()
        args.model_type = ModelType.encoder_or_decoder
        model = pretrain_gpt.model_provider(
            pre_process=pre_process,
            post_process=post_process,
        ).to(device)
        model.half()

        return model
    

    def set_args_megatron(self):
        WORLD_SIZE = self.tmp_width * self.ep_degree
        RANK = 0

        os.environ["CUDA_DEVICE_MAX_CONNECTIONS"] = str(1)
        os.environ["WORLD_SIZE"] = str(WORLD_SIZE)
        os.environ["RANK"] = str(RANK)
        os.environ["MASTER_PORT"] = "6000"
        os.environ["MASTER_ADDR"] = "localhost"

        TENSOR_MP_SIZE = self.tmp_width
        PIPELINE_MP_SIZE = 1
        EXPERT_MP_SIZE = self.ep_degree

        DISTRIBUTED_ARGS = {
            "--nproc_per_node": WORLD_SIZE,
            "--nnodes": 1,
            "--node_rank": RANK,
            "--master_addr": "localhost",
            "--master_port": 6000,
        }

        CHECKPOINT_PATH = self.out_dir

        MODEL_ARGS = {
            "--use-mcore-models": True,
            "--disable-bias-linear": True,
            "--seq-length": 4096,
            "--max-position-embeddings": 32768,
            "--num-layers": 1,
            "--hidden-size": 4096,
            "--ffn-hidden-size": 14336,
            "--num-attention-heads": 32,
            "--init-method-std": 0.01,
            "--attention-dropout": 0.0,
            "--hidden-dropout": 0.0,
            "--normalization": "LayerNorm",
            "--position-embedding-type": "rope",
            "--no-rope-fusion": True,
            "--swiglu": True,
            "--untie-embeddings-and-output-weights": True,
            "--group-query-attention": True,
            "--num-query-groups": 8,
            "--no-masked-softmax-fusion": True,
            "--no-position-embedding": True,
            "--rotary-base": 1000000,
            "--no-async-tensor-model-parallel-allreduce": True,
        }

        MOE_ARGS = {
            "--num-experts": 8,
            "--moe-router-topk": 2,
            "--moe-router-load-balancing-type": "sinkhorn",
            # "--moe-aux-loss-coeff": 1e-2,
            # "--moe-grouped-gemm": True,
            "--moe-token-dispatcher-type": "alltoall",
            # "--overlap-param-gather": True,
            # "--overlap-grad-reduce": True
        }


        TRAINING_ARGS = {
            "--micro-batch-size": 1,
            "--global-batch-size": self.ep_degree,
            "--lr": 1e-4,
            "--train-iters": 500000,
            "--lr-decay-iters": 320000,
            "--lr-decay-style": "cosine",
            "--min-lr": 1.0e-5,
            "--weight-decay": 0.1,
            "--lr-warmup-iters": 500,
            "--clip-grad": 1.0,
            "--bf16": True
        }

        MODEL_PARALLEL_ARGS = {
            "--tensor-model-parallel-size": TENSOR_MP_SIZE,
            "--pipeline-model-parallel-size": PIPELINE_MP_SIZE,
            "--expert-model-parallel-size": EXPERT_MP_SIZE,
            "--DDP-impl": "torch",
            "--no-masked-softmax-fusion": True,
        }
        OUTPUT_ARGS = {
            "--log-interval": 10,
            "--save-interval": 500,
            "--eval-interval": 100,
            "--eval-iters": 10,
            "--activations-checkpoint-method": "uniform",
        }
        DATA_ARGS={
            "--tokenizer-type": "Llama2Tokenizer",
            "--tokenizer-model": "./vocabfiles/tokenizer.model",
            "--data-path": "my-gpt_text_sentence",
            "--split": "99990,8,2",
        }

        ALL_ARGS = {**DISTRIBUTED_ARGS, **MODEL_ARGS, **MOE_ARGS,
                    **OUTPUT_ARGS, **MODEL_PARALLEL_ARGS, **TRAINING_ARGS, **DATA_ARGS}

        def add_args(argdict):
            for key, val in argdict.items():
                sys.argv.append(key)
                sys.argv.append(str(val))

        add_args(ALL_ARGS)

        initialize_megatron(ignore_unknown_args=True)
    

    def get_out_dir(self):
        if not self.out_dir:
            raise ValueError("Out directory not setup for", self.model_name)

        return self.out_dir

    def print_graphmodule(self):
        self.graphmodule.print_readable()

    def obtain_symbolic_trace_model(self, micro_batch_size=1, sequence_length=1):

        device = torch.device("cuda")

        input_ids = torch.ones(
            micro_batch_size, sequence_length, dtype=torch.long,).to(device)
        position_ids = torch.ones(
            micro_batch_size, sequence_length, dtype=torch.long,).to(device)
        attention_mask = torch.ones(
            micro_batch_size, sequence_length, dtype=torch.float,).to(device)
        decoder_input = None
        labels = torch.ones(
            micro_batch_size, sequence_length, dtype=torch.long,).to(device)
        inference_params = None
        packed_seq_params = None
        extra_block_kwargs = None
        runtime_gather_output = None

        graphmodule: torch.fx.GraphModule = custom_tracer_for_megatron(
            self.model)

        model_shapeprop = ShapeProp(graphmodule)
        self.graphmodule = model_shapeprop.propagate(
            input_ids, position_ids, attention_mask, decoder_input, labels, inference_params, packed_seq_params, extra_block_kwargs, runtime_gather_output)

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
            self.out_dir, micro_batch_size, sequence_length, force_reextract_model, ep_degree=self.ep_degree)
        
        num_layers = 32
        if num_layers:
            self.add_more_layer_info(num_layers, repeat_layer_id=0)
