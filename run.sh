export PYTHONPATH=$PYTHONPATH:/home/wangyc/yr/spectral-rl2/ && export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia && export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/wangyc/.mujoco/mujoco210/bin && conda activate yr
export PYTHONPATH=$PYTHONPATH:~/spectral-rl2 && export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia && export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/amax/.mujoco/mujoco210/bin

python examples/main_visual.py task=metaworld_assembly device=cuda:0 # 没学会
python examples/main_visual.py task=metaworld_box-close device=cuda:1 # -> 已 1.0
python examples/main_visual.py task=metaworld_button-press device=cuda:2 # -> 已 1.0
python examples/main_visual.py task=metaworld_coffee-push device=cuda:3 # -> 已 1.0
python examples/main_visual.py task=metaworld_door-close device=cuda:0 # -> 已 1.0
python examples/main_visual.py task=metaworld_drawer-open device=cuda:1 # 还没学会
python examples/main_visual.py task=metaworld_hammer device=cuda:2 # 还没学会
python examples/main_visual.py task=metaworld_reach device=cuda:3 # -> 已 1.0
python examples/main_visual.py task=metaworld_soccer device=cuda:0 # 最高0.8
python examples/main_visual.py task=metaworld_window-close device=cuda:2 # -> 已 1.0

python examples/main_visual.py task=metaworld_assembly algo=diffsr_drqv2 device=cuda:0 # 已1.0
python examples/main_visual.py task=metaworld_drawer-open algo=diffsr_drqv2 device=cuda:1 # 好像没跑起来
python examples/main_visual.py task=metaworld_hammer algo=diffsr_drqv2 device=cuda:2 # 已1.0

python examples/main_visual.py task=metaworld_coffee-push device=cuda:0 && python examples/main_visual.py task=metaworld_door-close device=cuda:0 && python examples/main_visual.py task=metaworld_window-close device=cuda:0
python examples/main_visual.py task=metaworld_assembly algo=diffsr_drqv2 device=cuda:1
python examples/main_visual.py task=metaworld_drawer-open algo=diffsr_drqv2 device=cuda:2
python examples/main_visual.py task=metaworld_hammer algo=diffsr_drqv2 device=cuda:3


python examples/main_visual.py mode=collect task=metaworld_box-close debug=True eval_episode=500 # 0
python examples/main_visual.py mode=collect task=metaworld_button-press debug=True eval_episode=500 # 好了
python examples/main_visual.py mode=collect task=metaworld_coffee-push debug=True eval_episode=500 device=cuda:1 # 0
python examples/main_visual.py mode=collect task=metaworld_door-close debug=True eval_episode=500 device=cuda:2 model_dir=/home/amax/yr/spectral-rl2/log/drqv2/debug/metaworld_door-close/seed0_debug-03-04-17-47-2358609 # 0
python examples/main_visual.py mode=collect task=metaworld_reach debug=True eval_episode=500 device=cuda:3 # 好了
python examples/main_visual.py mode=collect task=metaworld_soccer debug=True eval_episode=500 device=cuda:0 # 好了
python examples/main_visual.py mode=collect task=metaworld_window-close debug=True eval_episode=500 device=cuda:0 # 0

python examples/main_visual.py mode=collect task=metaworld_assembly debug=True algo=diffsr_drqv2 eval_episode=500 device=cuda:0 # 全0
python examples/main_visual.py mode=collect task=metaworld_drawer-open debug=True algo=diffsr_drqv2 eval_episode=500 device=cuda:0 # 全0
python examples/main_visual.py mode=collect task=metaworld_hammer debug=True algo=diffsr_drqv2 eval_episode=500 device=cuda:1 #全0

