export PYTHONPATH=$PYTHONPATH:/home/wangyc/yr/spectral-rl2/
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/wangyc/.mujoco/mujoco210/bin


python examples/main_visual.py --task metaworld_assembly --device cuda:0
python examples/main_visual.py --task metaworld_box-close --device cuda:1
python examples/main_visual.py --task metaworld_button-press --device cuda:2
python examples/main_visual.py --task metaworld_coffee-push --device cuda:3
python examples/main_visual.py --task metaworld_door-close --device cuda:4
python examples/main_visual.py --task metaworld_drawer-open --device cuda:5
python examples/main_visual.py --task metaworld_hammer --device cuda:6
python examples/main_visual.py --task metaworld_reach --device cuda:7
python examples/main_visual.py --task metaworld_soccer --device cuda:8
python examples/main_visual.py --task metaworld_window-close
python examples/main_visual.py --task metaworld_open-drawer
python examples/main_visual.py --task metaworld_open-slide
python examples/main_visual.py --task metaworld_push-green