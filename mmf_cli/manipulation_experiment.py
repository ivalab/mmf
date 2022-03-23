import argparse
import os
import cv2

from omegaconf import DictConfig

from mmf.utils.flags import flags
from mmf.utils.configuration import Configuration
from mmf.utils.env import set_seed, setup_imports
from mmf.utils.build import build_config
from mmf.utils.inference_resnet_bert_tdatt_triple import Inference_R_B_TT
from mmf.utils.inference_resnet_bert_concat import Inference_R_B_C

def main():
    parser = flags.get_parser()
    args_config = parser.parse_args()

    configuration = Configuration(args_config)
    configuration.args = args_config

    setup_imports()
    configuration.import_user_dir()
    config = build_config(configuration)

    if config.model == 'resnet_bert_concat':
        inferencer = Inference_R_B_C(config.inference.checkpoint_path, config)
    elif config.model == 'resnet_bert_tdatt_triple':
        inferencer = Inference_R_B_TT(config.inference.checkpoint_path, config)

    root = config.inference.root
    save_root = config.inference.save_root
    task_type = config.inference.task_type
    difficulty_level = config.inference.difficulty_level

    if task_type is not None:
        if difficulty_level is not None:
            evaluation_loop(root, save_root, task_type, difficulty_level, inferencer)
        else:
            for difficulty_level in os.listdir(os.path.join(root, task_type)):
                evaluation_loop(root, save_root, task_type, difficulty_level, inferencer)
    else:
        for task_type in os.listdir(root):
            if difficulty_level is not None:
                evaluation_loop(root, save_root, task_type, difficulty_level, inferencer)
            else:
                for difficulty_level in os.listdir(os.path.join(root, task_type)):
                    evaluation_loop(root, save_root, task_type, difficulty_level, inferencer)

def evaluation_loop(root_path, save_root_path, task_path, diff_level_path, inferencer):
    for folder in sorted(os.listdir(os.path.join(root_path, task_path, diff_level_path))):
        path = os.path.join(root_path, task_path, diff_level_path, folder)
        save_path = os.path.join(save_root_path, task_path, diff_level_path, folder)
        img_path = os.path.join(path, 'rgb_image.png')
        with open(os.path.join(path, 'natural_language.txt'), 'r') as f:
            text = f.readline().split('\n')[0]

        action_pr, subject_pr, object_pr = inferencer.forward(img_path, text)
        with open(os.path.join(save_path, 'pred_pddl_goal_state.txt'), 'w') as f:
            f.write(action_pr + ' ' + subject_pr + ' ' + object_pr)

        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        cv2.namedWindow('RGB Image', cv2.WINDOW_AUTOSIZE)
        cv2.imshow('RGB Image', img)
        print("Paired natural language: {}".format(text))
        print("Predicted PDDL goal state: {} {} {}".format(action_pr, subject_pr, object_pr))
        cv2.waitKey(30)

if __name__ == "__main__":
    main()
