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

# Model hyperparameter details: https://arxiv.org/pdf/1909.08053.pdf
echo "mbs: $mbs";

Megatron GPT2-54
python3 nest.py \
        --nest_model megatrongpt2-54 \
        --nest_exec_type extract_graph \
        --nest_micro_batch_size 16 \
        --nest_sequence_length 1024 \
        --nest_max_tmp_width 4 \
        --nest_hbm_size 32 64 80  

# # Megatron GPT2-72
python3 nest.py \
        --nest_model megatrongpt2-72 \
        --nest_exec_type run_solver \
        --nest_micro_batch_size $mbs \
        --nest_sequence_length 1024 \
        --nest_max_tmp_width 8 \
        --nest_hbm_size 32 64 80  
