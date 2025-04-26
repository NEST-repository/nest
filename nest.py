# internal nest imports
from arguments import process_nest_arguments
from exec_modes import extract_and_prepopulate, extract_and_solve, extract_only
from GraphExtractor import supported_models

import time


def main(args):

    nest_mbs = args.nest_micro_batch_size
    nest_model_names = args.nest_model_names
    nest_exec_type = args.nest_exec_type
    nest_seq_len = args.nest_sequence_length
    nest_max_tmpc = args.nest_max_tmp_width
    force_reextract_model = args.force_reextract_model
    hbm_size_list = args.nest_hbm_size
    ep_degree = args.nest_exp_parallel_degree

    assert set(nest_model_names).issubset(
        set(supported_models)
    ), "Model not supported by Nest. Please check the list of supported models in GraphExtractor.py"

    print("Running for hbm: ", hbm_size_list, " mbs: ",
          nest_mbs, " max tmp: ", nest_max_tmpc)
    if nest_exec_type == "extract_graph":
        # Extract graph using Torch.fx
        # Fill details about the tensor sizes - weights, activations, and
        # intermediate results
        for micro_batch_size in nest_mbs:
            extract_only(nest_model_names, nest_max_tmpc, micro_batch_size,
                         nest_seq_len, force_reextract_model,ep_degree)

    elif nest_exec_type == "prepopulate_estimates":
        # Every node has a corresponding estimates in a 3D matrix <TMP
        # strategy, core dimensions, and number of cores>
        for micro_batch_size in nest_mbs:
            extract_and_prepopulate(nest_model_names, nest_max_tmpc,
                                    micro_batch_size, nest_seq_len, force_reextract_model,ep_degree)

    elif nest_exec_type == "run_solver":

        # initialize variables for final "best" config
        final_config = None
        final_total_time = 0
        final_ilp_time = 0
        final_dp_time = 0
        final_estimation_time = 0
        final_micro_batch_size = 0
        final_hbm_size = 0
        final_throughput = 0
        final_activation_recomputation = False

        # search both activation recomp true and false
        activation_recomputations = [False, True]

        for micro_batch_size in nest_mbs:
            for hbm_size in hbm_size_list:
                for activation_recomputation in activation_recomputations:

                    print("mbs: ", micro_batch_size, " HBM size: " + str(hbm_size) +
                          " activation_recomputation: " + str(activation_recomputation))

                    start = time.time()

                    final_nest_config, estimation_time, ilp_time, dp_time = extract_and_solve(
                        nest_model_names, nest_max_tmpc, micro_batch_size, nest_seq_len, force_reextract_model, activation_recomputation, hbm_size*1024*1024*1024, ep_degree)

                    end = time.time()

                    print("Best nest config for mbs: ", micro_batch_size, " HBM: ", hbm_size, " Activation Recomp: ", activation_recomputation, "\n",
                          "Config ", final_nest_config, "\n",
                          "Models", nest_model_names, "\n",
                          "total solving time, ilptime and dptime", end - start, ilp_time, dp_time, "\n",
                          "estimation time", estimation_time)

                    final_total_time += end - start
                    final_ilp_time += ilp_time
                    final_dp_time += dp_time

                    cc, strategy = final_nest_config

                    if strategy != None:
                        if strategy[nest_model_names[0]].throughput > final_throughput:
                            final_config = final_nest_config
                            final_micro_batch_size = micro_batch_size
                            final_hbm_size = hbm_size
                            final_throughput = strategy[nest_model_names[0]].throughput
                            final_activation_recomputation = activation_recomputation
            final_estimation_time += estimation_time

        if final_nest_config != None:
            print("Best nest config for single model comparison: mbs: ", final_micro_batch_size, " HBM: ", final_hbm_size, " Activation Recomp: ", final_activation_recomputation, "\n",
                  "Config ", final_config, "\n",
                  "Model", strategy[nest_model_names[0]], "\n",
                  "total solving time, ilptime and dptime", final_total_time, final_ilp_time, final_dp_time, "\n",
                  "estimation time", final_estimation_time)
        else:
            print("No valid configuration found")


if __name__ == "__main__":
    args = process_nest_arguments()
    main(args)
