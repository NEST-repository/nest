mbs=$1
#mode = run_solver # prepopulate_estimates, extract_graph

# if the model is to be extracted from scratch, then the following flag should be added
#   --force_reextract_model True
#   This mode requires space to extract and execute the forward pass of the model
# else the model from GraphExtractor/out folder will be used.

cd ..

if [ ! -d "GraphExtractor/out/GPT" ]; then
  mkdir GraphExtractor/out/GPT
fi

# GPT2
echo "mbs: $mbs";
python3 nest.py \
        --nest_model gpt2 \
        --nest_exec_type run_solver \
        --nest_micro_batch_size $mbs \
        --nest_sequence_length 1024 \
        --nest_hbm_size 32 64 80
