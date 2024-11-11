# Created by xunannancy at 2024/11/04
import argparse
import itertools
import torch.cuda

from language_llm import model_lan_dict
import os
from tqdm import tqdm
import json
from model_utilities import (
    _get_response_from_llama2_chat,
    _get_response_from_vicuna,
    _get_response_from_wizardlm,
    _get_response_from_guanaco,
    _get_response_from_mpt,
    _get_response_from_chatgpt,
    _get_response_from_llama2_chat_2turns,
    _get_response_from_vicuna_2turns,
    _get_response_from_wizardlm_2turns,
    _get_response_from_guanaco_2turns,
    _get_response_from_mpt_2turns,
    _get_response_from_chatgpt_2turns
)
from utils import (
    perform_translation,
    compute_success_rate
)

"""
- LLMs:
Llama 2 (7B-chat and 13B-chat) (Touvron et al., 2023b), 
Vicuna (7B and 13B) (Chiang et al., 2023), 
WizardLM (7B and 13B) (Xu et al., 2023), 
Guanaco (7B and 13B) (Dettmers et al., 2023) 
MPT (7b-instruct and 7b-chat) (Team, 2023)
ChatGPT (gpt-3.5- turbo-0301)
- datasets
AdvBench
MasterKey
"""

def perform_monolingual_attack(attack, dataset, model_name, max_batch_size, max_tokens, load_8bit, cpu_offloading,
                               repetition_penalty, overwrite, dataset_dir, res_dir, translation_model, num_gpus=1, debug=False):
    if attack == 'en':
        lan_list = ['en']
        evaluate_jb = True
    elif attack == 'monolingual':
        lan_list = model_lan_dict[model_name][1:]
        evaluate_jb = False
    if dataset == 'MasterKey':
        file_path_list, saved_file_list = list(), list()
        for lan in lan_list:
            file_path_list.append(list())
            saved_file_list.append(list())
            if lan == 'en':
                dir_path = f'{dataset_dir}/MasterKey'
            else:
                dir_path = f'{dataset_dir}/MasterKey/{lan}'
            for control_file in os.listdir(dir_path):
                if not control_file.endswith('_controls.json'):
                    continue
                control_type = control_file.split('_')[0]
                file_path_list[-1].append(f'{dir_path}/{control_file}')
                if attack == 'en':
                    saved_file_list[-1].append(f'{res_dir}/{lan}/MasterKey/MasterKey_{control_type}_{model_name}_response.json')
                else:
                    saved_file_list[-1].append(f'{res_dir}/monolingual/MasterKey/{lan}/MasterKey_{control_type}_{model_name}_response.json')
    elif dataset == 'AdvBench':
        file_path_list, saved_file_list = list(), list()
        for lan in lan_list:
            if lan == 'en':
                file_path_list.append([f'{dataset_dir}/AdvBench/attacks.json'])
                saved_file_list.append([f'{res_dir}/{lan}/AdvBench/llm_attack_{model_name}_response.json'])
            else:
                file_path_list.append([f'{dataset_dir}/AdvBench/{lan}/attacks.json'])
                saved_file_list.append([f'{res_dir}/monolingual/AdvBench/{lan}/llm_attack_{model_name}_response.json'])

    for lan_file_path, lan_save_file in tqdm(zip(file_path_list, saved_file_list), total=len(file_path_list)):
        for file_path, save_file in zip(lan_file_path, lan_save_file):
            if os.path.exists(save_file) and not overwrite:
                print(f'find response at {save_file}...')
                continue
            os.makedirs(os.path.dirname(save_file), exist_ok=True)
            saved_goals = list()
            for idx, line in enumerate(open(file_path)):
                content = json.loads(line)
                saved_goals.append(content['goal'])
                if idx>=7 and debug:
                    break
            if model_name == 'chatgpt':
                # NOTE: The model `gpt-3.5-turbo-0301` used in paper has been deprecated, we used "gpt-3.5-turbo" instead for implementation
                # used_model_name = "gpt-3.5-turbo-0301"
                used_model_name = "gpt-3.5-turbo"
                _get_response_from_chatgpt(saved_goals, used_model_name, save_file, max_tokens=max_tokens, evaluate_jb=evaluate_jb)
            elif model_name in ['llama2-7b-chat', 'llama2-13b-chat']:
                if model_name == 'llama2-7b-chat':
                    used_model_name = 'meta-llama/Llama-2-7b-chat-hf'
                elif model_name == 'llama2-13b-chat':
                    used_model_name = 'meta-llama/Llama-2-13b-chat-hf'
                _get_response_from_llama2_chat(saved_goals, used_model_name, save_file, max_tokens=max_tokens,
                                               max_batch_size=max_batch_size, evaluate_jb=evaluate_jb)
            elif model_name in ['vicuna-7b', 'vicuna-13b']:
                _get_response_from_vicuna(saved_goals, f'lmsys/{model_name}-v1.3', save_file, max_tokens=max_tokens,
                                          max_batch_size=max_batch_size,
                                          load_8bit=load_8bit, cpu_offloading=cpu_offloading,
                                          repetition_penalty=repetition_penalty, num_gpus=num_gpus, evaluate_jb=evaluate_jb)
            elif model_name in ['WizardLM-7B-V1.0', 'WizardLM-13B-V1.2']:
                if model_name == 'WizardLM-7B-V1.0':
                    model_path = f'./models/{model_name}_recovered'
                else:
                    model_path = f'WizardLM/{model_name}'
                _get_response_from_wizardlm(saved_goals, model_path, save_file,
                                               max_tokens=max_tokens,
                                               max_batch_size=max_batch_size, load_8bit=load_8bit, evaluate_jb=evaluate_jb)
            elif model_name in ['guanaco-7b', 'guanaco-13b']:
                _get_response_from_guanaco(saved_goals, model_name, save_file,
                                               max_tokens=max_tokens,
                                               max_batch_size=max_batch_size, evaluate_jb=evaluate_jb)
            elif model_name in ['mpt-7b-chat', 'mpt-7b-instruct']:
                _get_response_from_mpt(saved_goals, model_name, save_file,
                                       max_tokens=max_tokens,
                                       max_batch_size=max_batch_size, evaluate_jb=evaluate_jb)
    saved_files = list(itertools.chain(*saved_file_list))
    if attack == 'monolingual':
        perform_translation(saved_files, lan_list, translation_model, max_batch_size, evaluate_jb=True, response_name='response')
        compute_success_rate([i[:-len(".json")]+'_translated.json' for i in saved_files])
    else:
        compute_success_rate(saved_files)
    return

def perform_multilingual_attack(dataset, model_name, max_batch_size, max_tokens, load_8bit, cpu_offloading, repetition_penalty, overwrite, dataset_dir, res_dir,
                                translation_model, en_first, num_gpus, debug):
    context_source = 'wiki'
    if dataset == 'MasterKey':
        if en_first:
            beginning_path_list = [[f'./datasets/MasterKey/dpr/vlt5-base-keywords/{i[:-len("_controls.json")]}_controls.json' for i in
                              sorted(os.listdir('./datasets/MasterKey/')) if i.endswith('_controls.json')] for lan in model_lan_dict[model_name][1:]]
            ending_path_list = [[f'./datasets/MasterKey/{lan}/{i[:-len("_controls.json")]}_controls.json' for i in
                              sorted(os.listdir('./datasets/MasterKey/')) if i.endswith('_controls.json')] for lan in model_lan_dict[model_name][1:]]
            save_path_list = [[f'./results/multilingual/MasterKey/en_first/{context_source}/{lan}/MasterKey_{i[:-len("_controls.json")]}_controls_{model_name}.json' for i in
                                sorted(os.listdir('./datasets/MasterKey/')) if i.endswith('_controls.json')] for lan in model_lan_dict[model_name][1:]]
            lan_list = [[lan for i in sorted(os.listdir('./datasets/MasterKey/')) if i.endswith('_controls.json')] for lan in model_lan_dict[model_name][1:]]
        else:
            beginning_path_list = [[f'./datasets/MasterKey/{lan}/context_{context_source}_{i[:-len("_controls.json")]}.json' for i in
                              sorted(os.listdir('./datasets/MasterKey/')) if i.endswith('_controls.json')] for lan in model_lan_dict[model_name][1:]]
            ending_path_list = [[f'./datasets/MasterKey/{i[:-len("_controls.json")]}_controls.json' for i in
                                  sorted(os.listdir('./datasets/MasterKey/')) if i.endswith('_controls.json')] for lan in model_lan_dict[model_name][1:]]
            save_path_list = [[f'./results/multilingual/MasterKey/en_last/{context_source}/{lan}/MasterKey_{i[:-len("_controls.json")]}_controls_{model_name}.json' for i in
                                sorted(os.listdir('./datasets/MasterKey/')) if i.endswith('_controls.json')] for lan in model_lan_dict[model_name][1:]]
            lan_list = [[lan for i in sorted(os.listdir('./datasets/MasterKey/')) if i.endswith('_controls.json')] for lan in model_lan_dict[model_name][1:]]

        # flatten
        beginning_path_list = list(itertools.chain(*beginning_path_list))
        ending_path_list = list(itertools.chain(*ending_path_list))
        save_path_list = list(itertools.chain(*save_path_list))
        lan_list = list(itertools.chain(*lan_list))

    elif dataset == 'AdvBench':
        if en_first:
            beginning_path_list = ['./datasets/AdvBench/dpr/vlt5-base-keywords/attacks.json'] * len(
                model_lan_dict[model_name][1:])  # random model's output file should work
            ending_path_list = [f'./datasets/AdvBench/{lan}/attacks.json' for lan in model_lan_dict[model_name][1:]]
            save_path_list = [
                f'./results/multilingual/AdvBench/en_first/{context_source}/{lan}/llm_attack_{model_name}_response.json' for lan in
                model_lan_dict[model_name][1:]]
            lan_list = [lan for lan in model_lan_dict[model_name][1:]]
        else:
            # NOTE: context translated in extract_context_lan.py
            beginning_path_list = [f'./datasets/AdvBench/{lan}/context_{context_source}.json' for lan in
                                   model_lan_dict[model_name][1:]]
            ending_path_list = ['./datasets/AdvBench/attacks.json'] * len(model_lan_dict[model_name][1:])
            save_path_list = [f'./results/multilingual/AdvBench/en_last/{context_source}/{lan}/llm_attack_{model_name}_response.json'
                              for lan in model_lan_dict[model_name][1:]]
            lan_list = [lan for lan in model_lan_dict[model_name][1:]]

    if debug:
        save_path_list = [[i.replace('results', 'results_debug') for i in save_path_list][0]]
    for beginning_path, ending_path, save_file in tqdm(zip(beginning_path_list, ending_path_list, save_path_list),
                                                       total=len(beginning_path_list)):
        if os.path.exists(save_file) and not overwrite:
            print(f'find response at {save_file}...')
            continue
        os.makedirs(os.path.dirname(save_file), exist_ok=True)
        beginning_res = json.load(open(beginning_path))
        beginning_text = list()
        if en_first:
            for idx, item in enumerate(beginning_res):
                beginning_text.append(item['context'])
                if idx>=7 and debug:
                    break
        else:
            for idx, item in enumerate(beginning_res):
                beginning_text.append(item['response_0'])
                if idx>=7 and debug:
                    break
        ending_text = list()
        for line in open(ending_path):
            ending_text.append(json.loads(line)['goal'])

        if model_name == 'chatgpt':
            # NOTE: The model `gpt-3.5-turbo-0301` used in paper has been deprecated, we used "gpt-3.5-turbo" instead for implementation
            # used_model_name = "gpt-3.5-turbo-0301"
            used_model_name = "gpt-3.5-turbo"
            _get_response_from_chatgpt_2turns(beginning_text, ending_text, used_model_name, save_file, max_tokens=max_tokens,
                                       evaluate_jb=False)
        elif model_name in ['llama2-7b-chat', 'llama2-13b-chat']:
            if model_name == 'llama2-7b-chat':
                used_model_name = 'meta-llama/Llama-2-7b-chat-hf'
            elif model_name == 'llama2-13b-chat':
                used_model_name = 'meta-llama/Llama-2-13b-chat-hf'
            _get_response_from_llama2_chat_2turns(beginning_text, ending_text, used_model_name, save_file, max_tokens=max_tokens,
                                           max_batch_size=max_batch_size, evaluate_jb=False)
        elif model_name in ['vicuna-7b', 'vicuna-13b']:
            _get_response_from_vicuna_2turns(beginning_text, ending_text, f'lmsys/{model_name}-v1.3', save_file, max_tokens=max_tokens, max_batch_size=max_batch_size,
                                      load_8bit=load_8bit, cpu_offloading=cpu_offloading, repetition_penalty=repetition_penalty, num_gpus=num_gpus, evaluate_jb=False)
        elif model_name in ['WizardLM-7B-V1.0', 'WizardLM-13B-V1.2']:
            if model_name == 'WizardLM-7B-V1.0':
                model_path = f'./models/{model_name}_recovered'
            else:
                model_path = f'WizardLM/{model_name}'
            _get_response_from_wizardlm_2turns(beginning_text, ending_text, model_path, save_file,
                                           max_tokens=max_tokens,
                                           max_batch_size=max_batch_size, load_8bit=load_8bit, evaluate_jb=False)
        elif model_name in ['guanaco-7b', 'guanaco-13b']:
            _get_response_from_guanaco_2turns(beginning_text, ending_text, model_name, save_file, max_tokens=max_tokens,
                                           max_batch_size=max_batch_size, evaluate_jb=False)
        elif model_name in ['mpt-7b-chat', 'mpt-7b-instruct']:
            _get_response_from_mpt_2turns(beginning_text, ending_text, model_name, save_file, max_tokens=max_tokens,
                                           max_batch_size=max_batch_size, evaluate_jb=False)
        if debug:
            break
    perform_translation(save_path_list, lan_list, translation_model, max_batch_size, evaluate_jb=True, response_name='response_1')
    compute_success_rate([i[:-len(".json")]+'_translated.json' for i in save_path_list])

    return

if __name__ == '__main__':
    parser = argparse.ArgumentParser('perform multilingual attacks')
    parser.add_argument('--dataset',  choices=['AdvBench', 'MasterKey'], default='AdvBench')
    parser.add_argument('--model-name', type=str, default='llama2-7b-chat', choices=[
        'llama2-7b-chat', 'llama2-13b-chat',
        'vicuna-7b', 'vicuna-13b',
        'WizardLM-13B-V1.2', 'WizardLM-7B-V1.0',
        'guanaco-7b', 'guanaco-13b',
        'mpt-7b-chat', 'mpt-7b-instruct',
        'chatgpt',
    ], help='model name')
    parser.add_argument('--attack', type=str, choices=['en', 'monolingual', 'multilingual'], default='en')
    parser.add_argument('--max-batch-size', type=int, default=8, help='max batch size')
    parser.add_argument('--max-tokens', type=int, default=128, help='max batch size')
    parser.add_argument('--load-8bit', action='store_true', help='for vicuna')
    parser.add_argument('--cpu-offloading', action='store_true', help='for vicuna')
    parser.add_argument('--repetition-penalty', type=float, default=1.0, help='repetition penalty')
    parser.add_argument('--overwrite', action='store_true')
    parser.add_argument('--dataset-dir', type=str, default='/datadrive/nan/CognitiveOverload/datasets')
    parser.add_argument('--res-dir', type=str, default='/datadrive/nan/CognitiveOverload/results')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--num-gpus', type=int, default=1)
    parser.add_argument('--translation-model', type=str, default='nllb-200-distilled-1.3B', choices=['nllb-200-distilled-1.3B'])
    parser.add_argument('--en-first', action='store_true', help='whether to use en first or other lan first')

    args = parser.parse_args()
    if args.debug and not args.res_dir.endswith('_debug'):
        args.res_dir += '_debug'

    if args.attack in ['monolingual', 'en']:
        perform_monolingual_attack(args.attack, args.dataset, args.model_name, args.max_batch_size, args.max_tokens, args.load_8bit, args.cpu_offloading,
                                    args.repetition_penalty, args.overwrite, args.dataset_dir, args.res_dir, args.translation_model, args.num_gpus, args.debug)
    elif args.attack == 'multilingual':
        perform_multilingual_attack(args.dataset, args.model_name, args.max_batch_size, args.max_tokens, args.load_8bit, args.cpu_offloading,
                                    args.repetition_penalty, args.overwrite, args.dataset_dir, args.res_dir, args.translation_model,
                                    args.en_first, args.num_gpus, args.debug)

    """
    CUDA_VISIBLE_DEVICES=1 python perform_multilingual_attack.py \
        --dataset AdvBench \
        --model-name vicuna-7b  \
        --max-tokens 128 \
        --attack en \
        --max-batch-size 16 \
        --debug \
        --en-first
        
                --en-first \

    """