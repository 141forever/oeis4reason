import asyncio
from tqdm import tqdm
from utils import Agent, load_jsonl, save_jsonl

import json
import sys
import os
import pdb
import copy
import re
import ast
import random
import time
# import anthropic
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# print(sys.path)
from openai import OpenAI
from utils_ import *
from prompt import *
from concurrent.futures import ThreadPoolExecutor


from dotenv import load_dotenv

def read_jsonl(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():  # 跳过空行
                try:
                    item = json.loads(line)
                    data.append(item)
                except json.JSONDecodeError as e:
                    print(f"解析出错: {e}，内容为: {line}")
    return data
def is_valid_dict(string,example_value):
    # 检查字符串是否以 '{' 开头并以 '}' 结尾
    if not (string.startswith('{') and string.endswith('}')):
        return False
    try:
        # 尝试将字符串解析为字典
        parsed_dict = json.loads(string)
        if not judge_all_number(parsed_dict,example_value):
            print("judge_all_number Error!!!!!!!!!",parsed_dict['example'],example_value)
        return isinstance(parsed_dict, dict) and judge_all_number(parsed_dict,example_value)
    except (SyntaxError, ValueError):
        return False
def get_valid_string(result, example_value, split=None):
    def replace_newlines_in_quotes(text):
            def replacer(match):
                return match.group().replace('\n', '\\n')

            pattern = r'"[^"]*"'
            return re.sub(pattern, replacer, text)
    if not result.count(split):
        # print(result)
        return ""
    result = result.split(split)[-1]
    if split.count("```") and result.count("```"):
        result = result[:result.rfind('```')]
    return result.strip()



agent = Agent(apikeys="EMPTY",base_url=")
def roll_answer(ll):
    # print(ll)
    ans = asyncio.run(
                agent(
                    ll,#[[{"role":"user", "content": "Hello"}],[{"role":"user", "content": "No"}]],
                    {
                        "model": ",
                        "timeout": 1000,
                        "temperature": 1.0,
                        "max_tokens": 10*1024,
                    },
                    max_concurrency=32,
                )
            )
    # pdb.set_trace()
    return ans
def read_oeis(path):
    data_list = []
    file = Utils_.read_files(path)
    for d in file:
        data_list.append(d)
    return file
def generate_python(dict_now):
        
    # prompt = copy.deepcopy(prompt_dict["answer_generation_prompt5"])
    # prompt = prompt.format(problem=dict_now['description'],
    #                            input_format=dict_now['input requirement'],
    #                            output_format=dict_now['output requirement'],
    #                            input1=dict_now['example'][0]['input'],
    #                            output1=dict_now['example'][0]['output'],
    #                            explanation1=dict_now['example'][0]['explanation'],
    #                            input2=dict_now['example'][1]['input'],
    #                            output2=dict_now['example'][1]['output'],
    #                            explanation2=dict_now['example'][1]['explanation'], ).strip()
    prompt = dict_now['prompt'][0]['content']
    l = []
    for i in range(32):
        l.append([{"role":"user", "content": prompt}])

    code_now_list = roll_answer(l)
    roll_results = []
    for can in code_now_list:
        final_string = get_valid_string(result=can,example_value=[],split="```python")
        roll_results.append(final_string)
    assert len(roll_results)==32
    dict_now["roll_results"] = roll_results
    dict_now['problem_id'] = dict_now ['reward_model']['ground_truth']['id']
    tmp = read_jsonl('')
    for ddd in tmp:
        if ddd['id'] == dict_now['problem_id']:
            dict_now['test'] = ddd['test']
            break
    with open(".", "a", encoding='utf-8') as f:
        f.write(json.dumps(dict_now, ensure_ascii=False) + "\n")



if __name__ == "__main__":
    
    path1 = ''

    data_list = read_oeis(path1)
    target_ids = []
    for i in range(len(data_list)):
        target_ids.append(data_list[i]['reward_model']['ground_truth']['id'])
    print(len(target_ids))

    # path2 = ''
    # data_list2 = read_oeis(path2)
    # data_list3 = []
    # for dd in data_list2:
    #     if dd['problem_id'] in target_ids:
    #         data_list3.append(dd)
    # print(len(data_list3))
    data_list3 = read_jsonl("")


    save_path = ''
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    vis = set()
    if os.path.exists(save_path):
        with open(save_path,"r",encoding='utf-8',errors='ignore') as f:
            for l in f:
                try:
                    sample = json.loads(l)
                    vis.add(sample['problem_id'])
                except:
                    pass
    print("vis:",len(vis))
    
    actual_list = []
    for d in data_list3:
        if d['reward_model']['ground_truth']['id'] not in vis:
            actual_list.append(d)
    data_list = actual_list
    
    for dict_now in data_list:
        generate_python(dict_now)
        

