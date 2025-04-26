mbs=$1
mode = run_solver # prepopulate_estimates, extract_graph

# if the model is to be extracted from scratch, then the following flag should be added
#   --force_reextract_model True
#   This mode requires space to extract and execute the forward pass of the model
# else the model from GraphExtractor/out folder will be used.

cd ..

if [ ! -d "GraphExtractor/out/Llama" ]; then
  mkdir GraphExtractor/out/Llama
fi

echo "mbs: $mbs";
python3 nest.py \
        --nest_model llama2 \
        --nest_exec_type run_solver \
        --nest_micro_batch_size $mbs \
        --nest_sequence_length 4096 \
        --nest_hbm_size 32 64 80 
