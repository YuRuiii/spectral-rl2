import os
import numpy as np
from tqdm import trange


folder_list = os.listdir('data')


def split_medium(folder):
    print(f'split {folder}...')
    new_folder = folder.replace('medium', 'medium_split')
    os.makedirs(f'data/{new_folder}', exist_ok=True)
    
    # 只读第一个文件
    file = os.listdir(f'data/{folder}')[0]
    
    data = np.load(f'data/{folder}/{file}')
    last_index = np.where(data['is_last'] == True)[0]
    last_count = last_index.shape[0]
    
    for i in trange(last_count):
        observation = data['observation'][last_index[i]+1:last_index[i+1]+1]
        action = data['action'][last_index[i]+1:last_index[i+1]+1]
        reward = data['reward'][last_index[i]+1:last_index[i+1]+1]
        is_terminal = data['is_terminal'][last_index[i]+1:last_index[i+1]+1]
        is_success = data['is_success'][last_index[i]+1:last_index[i+1]+1]
        is_first = data['is_first'][last_index[i]+1:last_index[i+1]+1]
        is_first[0] = True
        is_last = data['is_last'][last_index[i]+1:last_index[i+1]+1]
        
        assert is_first[0] == True
        assert is_last[-1] == True
        
        np.savez(
            f'data/{new_folder}/{i}_success{int(sum(is_success))}.npz',
            observation=observation,
            action=action,
            reward=reward,
            is_terminal=is_terminal,
            is_success=is_success,
            is_first=is_first,
            is_last=is_last
        )
        

for folder in folder_list:
    if folder.endswith('medium'):
        split_medium(folder)