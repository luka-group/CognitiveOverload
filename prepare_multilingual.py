# Created by xunannancy at 2024/11/09
import argparse
import os
import json
from tqdm import tqdm
import torch
from utils import chunk
from transformers import T5Tokenizer, T5ForConditionalGeneration, AutoTokenizer, AutoModelForSeq2SeqLM
from pyserini.search.lucene import LuceneSearcher
from pyserini.search.faiss import FaissSearcher
from language_llm import language_llm_dict, llm_coded_flores_code_dict
import itertools

def vlt5_keyword_extraction(model_name, goal_list, max_batch_size, save_file_path):
    if torch.cuda.is_available():
        device = 'cuda:0'
    else:
        device = 'cpu'
    model = T5ForConditionalGeneration.from_pretrained(f"Voicelab/{model_name}").to(device)
    tokenizer = T5Tokenizer.from_pretrained(f"Voicelab/{model_name}")
    task_prefix = "Keywords: "

    batched_goals = list(chunk(goal_list, max_batch_size))
    res = list()
    for one_goal in tqdm(batched_goals, total=len(batched_goals)):
        inputs = [task_prefix+i for i in one_goal]
        input_ids = tokenizer(
            inputs, return_tensors="pt", truncation=True, padding=True
        ).input_ids
        input_ids = input_ids.to(device)
        output = model.generate(input_ids, no_repeat_ngram_size=3, num_beams=4)
        predicted = tokenizer.batch_decode(output, skip_special_tokens=True)
        for single_goal, single_prediction in zip(one_goal, predicted):
            result = {
                'target': 'random',
                'goal': single_goal,
                'control': '',
                'keywords': single_prediction,
            }
            res.append(result)
    with open(save_file_path, 'w') as f:
        json.dump(res, f, indent=4)

    return
def extract_keywords(dataset, keyword_model, max_batch_size, overwrite):
    if dataset == 'MasterKey':
        file_path_list = [f'./datasets/MasterKey/{i}' for i in sorted(os.listdir('./datasets/MasterKey')) if i.endswith('_controls.json')]
        saved_file_list = [f'./datasets/MasterKey/keywords/{keyword_model}/{i}' for i in sorted(os.listdir('./datasets/MasterKey')) if i.endswith('_controls.json')]
    elif dataset == 'AdvBench':
        file_path_list = ['./datasets/AdvBench/attacks.json']
        saved_file_list = [f'./datasets/AdvBench/keywords/{keyword_model}/attacks.json']
    for file_path, save_file in zip(file_path_list, saved_file_list):
        if os.path.exists(save_file) and not overwrite:
            print(f'find keywords at {save_file}...')
            continue
        os.makedirs(os.path.dirname(save_file), exist_ok=True)
        saved_goals = list()
        for line in open(file_path):
            content = json.loads(line)
            saved_goals.append(content['goal'])
        if keyword_model == 'vlt5-base-keywords':
            vlt5_keyword_extraction(keyword_model, saved_goals, max_batch_size, save_file)

    return

def perform_retrieval(dataset, keyword_model, retrieval_method, max_batch_size, overwrite):
    if dataset == 'MasterKey':
        file_path_list = [f'./datasets/MasterKey/keywords/{keyword_model}/{i}' for i in sorted(os.listdir('./datasets/MasterKey')) if i.endswith('_controls.json')]
        saved_file_list = [f'./datasets/MasterKey/{retrieval_method}/{keyword_model}/{i}' for i in sorted(os.listdir('./datasets/MasterKey')) if i.endswith('_controls.json')]
    elif dataset == 'AdvBench':
        file_path_list = [f'./datasets/AdvBench/keywords/{keyword_model}/attacks.json']
        saved_file_list = [f'./datasets/AdvBench/{retrieval_method}/{keyword_model}/attacks.json']

    if retrieval_method == 'dpr':
        searcher = FaissSearcher.from_prebuilt_index(
            'wikipedia-dpr-100w.dpr-multi', #'wikipedia-dpr-multi-bf',
            'facebook/dpr-question_encoder-multiset-base'  # encoder
        )
        prompt_format = 'What is {}?'
        if torch.cuda.is_available():
            searcher.query_encoder.device = 'cuda:0'
            searcher.query_encoder.model.to('cuda:0')
    else:
        searcher = LuceneSearcher.from_prebuilt_index('wikipedia-dpr')
        prompt_format = '{}'
    for file_path, save_file in zip(file_path_list, saved_file_list):
        if os.path.exists(save_file) and not overwrite:
            print(f'find keywords at {save_file}...')
            continue
        os.makedirs(os.path.dirname(save_file), exist_ok=True)
        content_list = json.load(open(file_path))
        res = list()
        query_list, goal_list = list(), list()
        for content in content_list:
            keyword_1st = content['keywords'].split(', ')[0]
            query = prompt_format.format(keyword_1st) #content['goal'] + f' What is {keyword_1st}?'
            query_list.append(query)
            goal_list.append(content['goal'])

        batched_queries = list(chunk(query_list, max_batch_size))
        batched_goals = list(chunk(goal_list, max_batch_size))
        for queries, goals in tqdm(zip(batched_queries, batched_goals), total=len(batched_queries)):
            hits = searcher.batch_search(
                queries=queries,
                q_ids=[str(i) for i in range(len(queries))],
                k=1)
            for hit, cur_goal, cur_query in zip(hits.values(), goals, queries):
                response = json.loads(searcher.doc(hit[0].docid).raw())['contents']
                # print(f'response: {response}')
                result = {
                    'target': 'random',
                    'goal': cur_goal,
                    'control': '',
                    'context': response,
                    'query': cur_query,
                }
                res.append(result)
        with open(save_file, 'w') as f:
            json.dump(res, f, indent=4)

    return

def translate_context(dataset, keyword_model, retrieval_method, translation_model, max_batch_size, overwrite):
    context_source = 'wiki'
    if dataset == 'MasterKey':
        beginning_path_list = [[f'./datasets/MasterKey/{retrieval_method}/{keyword_model}/{i}' for i in sorted(os.listdir('./datasets/MasterKey')) if i.endswith('_controls.json')] for lan in language_llm_dict]
        save_path_list = [[f'./datasets/MasterKey/{lan}/context_{context_source}_{i[:-len("_controls.json")]}.json' for i in sorted(os.listdir('./datasets/MasterKey/')) if i.endswith('_controls.json')] for lan in language_llm_dict]
        beginning_path_list = list(itertools.chain(*beginning_path_list))
        save_path_list = list(itertools.chain(*save_path_list))
    elif dataset == 'AdvBench':
        beginning_path_list = [f'./datasets/AdvBench/{retrieval_method}/{keyword_model}/attacks.json' for lan in language_llm_dict]
        save_path_list = [f'./datasets/AdvBench/{lan}/context_{context_source}.json' for lan in language_llm_dict]

    if torch.cuda.is_available():
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu')
    mt_tokenizer = AutoTokenizer.from_pretrained(f"facebook/{translation_model}", use_auth_token=True)
    mt_model = AutoModelForSeq2SeqLM.from_pretrained(f"facebook/{translation_model}", use_auth_token=True).to(device)

    for beginning_path, save_file in tqdm(zip(beginning_path_list, save_path_list), total=len(language_llm_dict)):
        if os.path.exists(save_file) and not overwrite:
            print(f'find response at {save_file}...')
            continue
        os.makedirs(os.path.dirname(save_file), exist_ok=True)
        beginning_res = json.load(open(beginning_path))
        beginning_text = list()
        for item in beginning_res:
            beginning_text.append(item['context'])
        translated_context = list()
        llm_code = save_file.split('/')[-2]
        flores_code = llm_coded_flores_code_dict[llm_code]
        batched_beginning_text = list(chunk(beginning_text, max_batch_size))
        for batched_text in tqdm(batched_beginning_text, total=len(batched_beginning_text)):
            inputs = mt_tokenizer.prepare_seq2seq_batch(batched_text, return_tensors="pt").to(device)
            translated_tokens = mt_model.generate(
                **inputs,
                forced_bos_token_id=mt_tokenizer.lang_code_to_id[flores_code],
            )
            gen_str = mt_tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)
            for one_gen_str in gen_str:
                translated_context.append({
                    'response_0': one_gen_str
                })
        with open(save_file, 'w') as f:
            json.dump(translated_context, f, ensure_ascii=False, indent=4)
    return

if __name__ == '__main__':
    parser = argparse.ArgumentParser('prepare multilingual')
    parser.add_argument('--dataset',  choices=['AdvBench', 'MasterKey'], default='AdvBench')
    parser.add_argument('--keyword-model', type=str, choices=['vlt5-base-keywords'], default='vlt5-base-keywords')
    parser.add_argument('--stage', type=str, choices=['extract_keywords', 'wiki_definition', 'translate_context'], default='translate_context')
    parser.add_argument('--max-batch-size', type=int, default=8)
    parser.add_argument('--overwrite', action='store_true')
    parser.add_argument('--retriever', type=str, choices=['dpr', 'bm25'], default='dpr')
    parser.add_argument('--translation-model', type=str, default='nllb-200-distilled-1.3B', choices=['nllb-200-distilled-1.3B'])

    args = parser.parse_args()

    if args.stage == 'extract_keywords':
        extract_keywords(args.dataset, args.keyword_model, args.max_batch_size, args.overwrite)
    elif args.stage == 'wiki_definition':
        perform_retrieval(args.dataset, args.keyword_model, args.retriever, args.max_batch_size, args.overwrite)
    elif args.stage == 'translate_context':
        translate_context(args.dataset, args.keyword_model, args.retriever, args.translation_model, args.max_batch_size, args.overwrite)


    """
    CUDA_VISIBLE_DEVICES=1 python prepare_multilingual.py \
        --stage translate_context \
        --dataset AdvBench \
        --max-batch-size 16
    """
