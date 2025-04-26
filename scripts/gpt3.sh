mbs=$1
mode = run_solver # prepopulate_estimates, extract_graph
# if the model is to be extracted from scratch, then the following flag should be added
#   --force_reextract_model True
#   This mode requires space to extract and execute the forward pass of the model
# else the model from GraphExtractor/out folder will be used.

cd ..

if [ ! -d "GraphExtractor/out/GPT" ]; then
  mkdir GraphExtractor/out/GPT
fi

# GPT3
echo "mbs: $mbs";
python3 nest.py \
        --nest_model megatrongpt3 \
        --nest_exec_type run_solver \
        --nest_micro_batch_size $mbs \
        --nest_sequence_length 2048 \
        --nest_max_tmp_width 8 \
        --nest_hbm_size 32 64 80
