conda env create -f environment.yml
conda activate nest_env

pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 --no-cache-dir
pip3 install transformers==4.45.2 --no-cache-dir
pip3 install graphviz pygraphviz --no-cache-dir
pip3 install transformer-engine[pytorch]
pip install accelerate

# Megatron specific installations
######################################################
pip3 install git+https://github.com/huggingface/Megatron-LM.git --no-cache-dir
pip3 install six --no-cache-dir
pip3 install pybind11 --no-cache-dir
pip3 install ninja --no-cache-dir

pip3 install scipy --no-cache-dir
pip3 install networkx --no-cache-dir
pip3 install gurobipy --no-cache-dir
pip3 install wandb --no-cache-dir

cd ..
git clone https://github.com/NVIDIA/apex
cd apex
pip3 install -v --disable-pip-version-check --no-cache-dir  --no-build-isolation  --global-option="--cpp_ext" --global-option="--cuda_ext" ./
cp -r nest-megatron-lm/megatron/*  ${$CONDA_ENVS_PATH}/lib/python3.10/site-packages/megatron/
cd ..

######################################################

git clone https://github.com/Accelergy-Project/accelergy.git
cd accelergy
git checkout 0278a565187dc019ca40043ed486bf94b645327e
pip3 install .
accelergy
cd ..

git clone https://github.com/Accelergy-Project/accelergy-cacti-plug-in.git
cd accelergy-cacti-plug-in
git checkout ba5468303c27b4a1a317742a4eaf147065b907e5
pip3 install .

git clone https://github.com/HewlettPackard/cacti.git 
cd cacti
make
export PATH=$(pwd):${PATH}
cd ../..

git clone https://github.com/Accelergy-Project/accelergy-table-based-plug-ins.git
cd accelergy-table-based-plug-ins/
git checkout 223039ffbf0e034f3b09c2b80074ad398fbaf03e 
pip3 install .
cd ..

git clone https://github.com/Accelergy-Project/accelergy-library-plug-in.git
cd accelergy-library-plug-in/
git checkout 0cab62c3631dbbe9a7925ff795285619a1bd6538
pip3 install .
cd ..

cp nest/Estimator/arch_configs/area_files/*.csv $CONDA_PREFIX/share/accelergy/estimation_plug_ins/accelergy-table-based-plug-ins/set_of_table_templates/data/.
cp -r nest/Estimator/arch_configs/area_files/tablePluginData $CONDA_PREFIX/share/accelergy/estimation_plug_ins/accelergy-library-plugin/library
cp -r cacti $CONDA_PREFIX/share/accelergy/estimation_plug_ins/accelergy-cacti-plug-in

cd nest/Solver/device_placement/
g++ device_placement.cpp -o device_placement `python3-config --cflags` `python3-config --ldflags` -lpython3.10
cd ../..