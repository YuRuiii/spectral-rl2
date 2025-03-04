conda create --name rl2 python=3.10
conda activate rl2
conda install --file image_pomdp_deps.txt
pip install -r pypi.txt
cd ..
git clone git@github.com:Farama-Foundation/Metaworld
cd Metaworld && git checkout 04be337a
pip install -e .