import os
import json
import jsonlines
import os.path as osp
import numpy as np


def read_jsonl(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        lines = (line.strip() for line in file)
        try:
            data = [json.loads(line) for line in lines if line]
        except json.JSONDecodeError as e:
            print(f"error: {e}")
            return []
    return data


def cal_statistic(data, fps):
    task_dict = dict()
    for x in data:
        tt = x['tasks'][0]
        if tt not in task_dict:
            task_dict[tt] = {
                'episodes': 0,
                'frames': 0, 
                'hour': 0
            }
        task_dict[tt]['episodes'] += 1
        task_dict[tt]['frames'] += x['length']
    for key in task_dict:
        frames = task_dict[key]['frames']
        task_dict[key]['hour'] = frames / (fps*3600)
    return task_dict


if __name__ == "__main__":
    data_folders = "/pfstem/lyc/dataset/lerobot/20250312_MultiTask4"
    
    fps = 30
    data = read_jsonl(osp.join(data_folders, 'meta/episodes.jsonl'))
    
    stats = cal_statistic(data, fps)
    all_episodes, all_frames, all_hour = 0, 0, 0
    for key in stats:
        info = stats[key]
        print('{}: episodes={}, frames={}, hour={:0.2f}'.format(key, info['episodes'], info['frames'], info['hour']))
        all_episodes += info['episodes']
        all_frames += info['frames']
        all_hour += info['hour']
    print('all episodes={}, frame={}, hour={:0.2f}'.format(all_episodes, all_frames, all_hour))
