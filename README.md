# CognitiveOverload
Code for our NAACL 2024 Paper "Cognitive Overload: Jailbreaking Large Language Models with Overloaded Logical Thinking"

## Datasets
We adopt the following two datasets with malicious prompts: 
1. **_AdvBench_** from [Universal and transferable adversarial attacks on aligned language models](https://arxiv.org/abs/2307.15043) 
2. **_MasterKey_** from [Jailbreaker: Automated jailbreak across multiple large language model chatbots.](https://arxiv.org/abs/2307.08715)

We translate the original English prompts to 52 other languages with Google Cloud API. You can find the original and translated prompts in folder` ./datasets`
### Jailbreaking with Multilingual Cognitive Overload
Run the following script to get baseline performance, i.e., LLMs prompted with English prompts
```shell
export dataset="AdvBench"
export model_name="vicuna-7b"
CUDA_VISIBLE_DEVICES=0 python perform_multilingual_attack.py \
    --dataset $dataset \
    --model-name $model_name  \
    --max-tokens 128 \
    --attack en \
    --max-batch-size 16 
```
#### Harmful Prompting in Various Languages
Run the following script to get results when prompting LLMs with all other languages that the LLM supports:
```shell
export dataset="AdvBench"
export model_name="vicuna-7b"
CUDA_VISIBLE_DEVICES=0 python perform_multilingual_attack.py \
    --dataset $dataset \
    --model-name $model_name  \
    --max-tokens 128 \
    --attack monolingual \
    --max-batch-size 16 
```
#### Language Switching: from English to Lan X vs. from Lan X to English
1. Extract keywords from prompts
```shell
export dataset="AdvBench"
CUDA_VISIBLE_DEVICES=0 python prepare_multilingual.py \
    --stage extract_keywords \
    --dataset $dataset
```
2. retrieve definition of keywords from wikipedia
```shell
CUDA_VISIBLE_DEVICES=0 python prepare_multilingual.py \
    --stage wiki_definition \
    --dataset $dataset 
```
3. translate definitions into 52 other languages
```shell
CUDA_VISIBLE_DEVICES=0 python prepare_multilingual.py \
    --stage translate_context \
    --dataset $dataset
```
4. attack LLMs in 2-turn, first english then other language or in reverse order
```shell
export model_name="vicuna-7b"
echo "English first then other language"
CUDA_VISIBLE_DEVICES=0 python perform_multilingual_attack.py \
    --dataset $dataset \
    --model-name $model_name  \
    --max-tokens 128 \
    --attack multilingual \
    --max-batch-size 16 \
    --en-first
echo "other language first then English"
CUDA_VISIBLE_DEVICES=0 python perform_multilingual_attack.py \
    --dataset $dataset \
    --model-name $model_name  \
    --max-tokens 128 \
    --attack multilingual \
    --max-batch-size 16 
```
### Jailbreaking with Veiled Expressions
1. extract sensitive words
```shell
export dataset="AdvBench"
CUDA_VISIBLE_DEVICES=0 python perform_veiled_attack.py \
    --dataset $dataset \
    --stage extract_sensitive \
    --max-batch-size 16 \
    --model-name Mistral-7B-Instruct-v0.1
```
2. clean extracted sensitive words
```shell
CUDA_VISIBLE_DEVICES=0 python perform_veiled_attack.py \
    --dataset $dataset \
    --stage clean_word \
    --model-name Mistral-7B-Instruct-v0.1
```
3. replace sensitive words with veiled expressions
```shell
CUDA_VISIBLE_DEVICES=0 python perform_veiled_attack.py \
    --dataset $dataset \
    --stage replace_sensitive \
    --max-batch-size 16 \
    --model-name Mistral-7B-Instruct-v0.1
```
4. attack LLMs with veiled expressions
```shell
export model_name="vicuna-7b"
CUDA_VISIBLE_DEVICES=0 python perform_veiled_attack.py \
    --dataset $dataset \
    --stage sensitive_attack \
    --max-batch-size 16 \
    --model-name $model_name \
    --num-demo 8
```
### Jailbreaking with Effect-to-Cause Cognitive Overload
1. extract events with in-context learning
```shell
export dataset="AdvBench"
for model_name in 'Mistral-7B-v0.1' 'llama2-13b'
do
    CUDA_VISIBLE_DEVICES=0 python perform_effect_to_cause.py \
        --dataset $dataset \
        --model-name $model_name \
        --stage extract_event \
        --max-batch-size 16
done
```
2. filter extracted events
```shell
python perform_effect_to_cause.py \
    --dataset $dataset \
    --stage filter_event
```
3. perform effect-to-cause attacks
```shell
export model_name="vicuna-7b"
CUDA_VISIBLE_DEVICES=0 python perform_effect_to_cause.py \
    --dataset $dataset \
    --model-name $model_name \
    --stage attack \
    --max-batch-size 16 
```
### Citation
We appreciate data contribution from these two papers:
```
@article{zou2023universal,
  title={Universal and transferable adversarial attacks on aligned language models},
  author={Zou, Andy and Wang, Zifan and Carlini, Nicholas and Nasr, Milad and Kolter, J Zico and Fredrikson, Matt},
  journal={arXiv preprint arXiv:2307.15043},
  year={2023}
}
@article{deng2023jailbreaker,
  title={Jailbreaker: Automated jailbreak across multiple large language model chatbots},
  author={Deng, Gelei and Liu, Yi and Li, Yuekang and Wang, Kailong and Zhang, Ying and Li, Zefeng and Wang, Haoyu and Zhang, Tianwei and Liu, Yang},
  journal={arXiv preprint arXiv:2307.08715},
  year={2023}
}
```
If you find our work helpful, please consider cite our work:
```
@inproceedings{xu2024cognitive,
  title={Cognitive overload: Jailbreaking large language models with overloaded logical thinking},
  author={Xu, Nan and Wang, Fei and Zhou, Ben and Li, Bang Zheng and Xiao, Chaowei and Chen, Muhao},
  booktitle={NAACL - Findings},
  year={2024}
}
```
