# coding=utf-8
import json
import sys
import os
import pdb
import copy
import re
import ast
import random
import time
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from openai import OpenAI
from prompt import *
from utils import *
from prompt import *
from concurrent.futures import ThreadPoolExecutor


def judge_all_number(pre_dict,shulie):
    if shulie == []:
        return True
    if len(pre_dict['example'])==2 and int(pre_dict['example'][0]['output'])==shulie[0] and int(pre_dict['example'][1]['output'])==shulie[1]:
        return True
    return False

def contains_axxxxx(string):
    # 正则表达式匹配 'A' 后跟六个数字
    pattern = r'A\d{6}'
    return bool(re.search(pattern, string))


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

def is_valid_list(string):
    # 检查字符串是否以 '[' 开头并以 ']' 结尾
    if not (string.startswith('[') and string.endswith(']')):
        return False
    try:
        # 尝试将字符串解析为字典
        parsed_list = ast.literal_eval(string)
        return isinstance(parsed_list, list)
    except (SyntaxError, ValueError):
        return False
def get_valid_string(prompt, example_value, mode='dict', split=None, model_name="o1-mini"): # mode: dict,list
    try_cnt = 0
    while True:
        result,usage = generate_one(prompt,model_name=model_name)
        try_cnt += 1
        if try_cnt > 3:
            return "", usage
        if result=="":
            time.sleep(2)
            continue
            # print(f"try_cnt: {try_cnt}")
        print("try_cnt:",try_cnt)
        def replace_newlines_in_quotes(text):
            def replacer(match):
                return match.group().replace('\n', '\\n')

            pattern = r'"[^"]*"'
            return re.sub(pattern, replacer, text)
        if mode == 'dict':
            result = result[result.find('{'):result.rfind('}')+1]
            # 修复引号中\n问题
            result = replace_newlines_in_quotes(result)
            if is_valid_dict(result,example_value):
                return result,usage
            else:
                # print(result)
                return "","dict error"
        elif mode == 'split-dict':
            if not result.count(split):
                print(result)
                return "", f"not count {split} error"
                
            result = result.split(split)[-1]
            if split.count("```") and result.count("```"):
                result = result[:result.rfind('```')]
            result = result[result.find('{'):result.rfind('}')+1]
            # 修复引号中\n问题
            result = replace_newlines_in_quotes(result)
            if is_valid_dict(result[:],example_value):
                return result,usage
            else:
                print(result)
                return "","split-dict error"
        elif mode == 'list':
            st = result.find('{') if mode == 'dict' else result.find('[')
            ed = result.rfind('}') if mode == 'dict' else result.rfind(']')
            result = result[st:ed + 1]
            if is_valid_list(result):
                r2 = result
                if mode == 'dict':
                    return r2[result.find('{'):result.rfind('}') + 1], usage
                else:
                    return r2[result.find('['):result.rfind(']') + 1], usage
            else:
                return "", f"{mode} Error"
        elif mode == 'python':
            st = result.find('{')
            ed = result.rfind('}')
            result = result[st:ed + 1]
            if is_valid_dict(result,example_value):
                r2 = result
                return r2[result.find('{'):result.rfind('}') + 1], usage
            else:
                return "", f"{mode} Error"

        else:
            return "",f"{mode} Error"

def generate_one(prompt,temp = 1, model_name = "o1-preview"):
    client = OpenAI(
    api_key='', base_url="https://api.deepseek.com"
    )
    response = (client.chat.completions.create
                (temperature = temp,
                model = model_name,
                 messages=[{"role": "user", "content":prompt}],))

    usage = response.usage
    text = response.choices[0].message.content
    return text,usage


def generate_problem(save_path, vis, data_list):
    global prompt
    sum_input_token = 0
    sum_output_token = 0
    total_cnt_usage = 0
    have_num = 0
    for dict in data_list:
        have_num+=1
        print(have_num)
        dict2 = {}
        for key,value in dict.items():
            if 'A' in key:
                dict2 = value
                break
        assert 'id' in dict2.keys()
        if dict2['id'] in vis:
            print(f"vis {dict2['id']}")
            continue
        shulie = dict2['seq_data']
        shulie_list = list(map(int,shulie.split(',')))
        shulie_list = [(i+1,shulie_list[i]) for i in range(len(shulie_list))]
        # print(shulie_list[0])
        if len(shulie_list)>4 and shulie_list[0][-1]<=5 and shulie_list[1][-1]<=20:
            shulie_list = shulie_list[1:]
        # 增加随机性
        if random.randint(0,2)==0 and len(shulie_list)>6 and shulie_list[2][-1]<=10:
            shulie_list = shulie_list[random.randint(1,2):]
        # 样例选择前2个
        example_list = shulie_list[:2]
        # test选择剩下的前8个
        test_list = shulie_list[2:][:8]
        
        # 样例情况
        sample_test = prompt.format(content=dict2['content'],e1in=example_list[0][0],e1out=example_list[0][1],e2in=example_list[1][0],e2out=example_list[1][1])
        # print(sample_test)
        result_dict = ""
        try_now = 0
        while True:
            try:
                try_now += 1
                result_dict, usage = get_valid_string(sample_test,[example_list[0][1],example_list[1][1]],mode='split-dict',split="```json",model_name='claude-3-5-sonnet-20241022')
                break
            except:
                if try_now > 0:
                    break
                time.sleep(3)
                pass
        # print(result_dict)
        if result_dict == "":
            print(f"Error {usage}")
            continue
        if usage:
            sum_input_token += int(usage.prompt_tokens)
            sum_output_token += int(usage.completion_tokens)
            total_cnt_usage += 1
        result_dict = json.loads(result_dict)
        result_dict['usage'] = {"prompt_tokens": usage.prompt_tokens if usage else -1,"completion_tokens": usage.completion_tokens if usage else -1,}
        
        # 这里转int
        try:
            result_dict['example'][0]['input'] = int(result_dict['example'][0]['input'])
            result_dict['example'][0]['output'] = int(result_dict['example'][0]['output'])
            result_dict['example'][1]['input'] = int(result_dict['example'][1]['input'])
            result_dict['example'][1]['output'] = int(result_dict['example'][1]['output'])
        except:
            continue
        # offset一致
        if int(result_dict['example'][0]['input'])-example_list[0][0]==int(result_dict['example'][1]['input'])-example_list[1][0]:
            
            result_dict['test'] = []
            for i in range(len(test_list)):
                test_list[i] = list(test_list[i])
                test_list[i][0] += (int(result_dict['example'][0]['input'])-example_list[0][0])
                result_dict['test'].append({"input":test_list[i][0],"output":test_list[i][1]})
        else:
            print(f"Offset error {int(result_dict['example'][0]['input'])} , {example_list[0][0]} , {int(result_dict['example'][1]['input'])}, {example_list[1][0]}")
        print(json.dumps(result_dict,indent=2,ensure_ascii=False)+"\n")

        result_dict['problem_id'] = dict2['id']
        with open(save_path,"a",encoding='utf-8') as f:
            f.write(json.dumps(result_dict,ensure_ascii=False)+"\n")
        # if total_cnt_usage%10 == 0:
        #     print(f"总计输入token个数:{sum_input_token} 平均输入token数:{sum_input_token/total_cnt_usage}",)
        #     print(f"总计输出token个数:{sum_output_token} 平均输入token数:{sum_output_token/total_cnt_usage}")

def check_problem_direct(save_path, vis, data_list):
    sum_input_token = 0
    sum_output_token = 0
    have_num = 0
    for item in data_list:
        have_num += 1
        print(have_num)
        dict_now = item
        if dict_now['problem_id'] in vis:
            print(f"vis {dict_now['problem_id']}")
            continue

        prompt = copy.deepcopy(prompt_dict["answer_generation_prompt2"])
        prompt = prompt.format(problem=dict_now['description'],
                               input_format=dict_now['input requirement'],
                               output_format=dict_now['output requirement'],
                               input1=dict_now['example'][0]['input'],
                               input2=dict_now['example'][1]['input'],
                               json_type='''[{'thought1':xxxx,'answer1':xxxx},{'thought2':xxxx,'answer2':xxxx}]''').strip()
        result_list,usage = get_valid_string(prompt,example_value=[],mode='list',model_name='claude-3-5-sonnet-20241022')
        if result_list == "":
            result_list, usage = get_valid_string(prompt, example_value=[], mode='list',
                                                  model_name='claude-3-5-sonnet-20241022')
            if result_list == "":
            # dict_now['if_pass'] = False
            # with open(save_path, "a", encoding='utf-8') as f:
            #     f.write(json.dumps(dict_now, ensure_ascii=False) + "\n")
                    continue
        # sum_input_token += int(usage.prompt_tokens)
        # sum_output_token += int(usage.completion_tokens)
        try:
            result_list = ast.literal_eval(result_list)
            assert len(result_list) == 2
            ans1 = str(result_list[0]['answer1'])
            thought1 = str(result_list[0]['thought1'])
            ans2 = str(result_list[1]['answer2'])
            thought2 = str(result_list[1]['thought2'])
            example_in_dict = dict_now['example']
            gt1 = example_in_dict[0]['output']
            gt2 = example_in_dict[1]['output']

            dict_now['example'][0]['get_ans1'] = ans1
            dict_now['example'][0]['thought1'] = thought1
            dict_now['example'][1]['get_ans2'] = ans2
            dict_now['example'][1]['thought2'] = thought2
            assert str(ans1)==str(gt1) and str(ans2)==str(gt2)
            dict_now['if_pass'] = True
        except:
            result_list, usage = get_valid_string(prompt, example_value=[], mode='list',
                                                  model_name='claude-3-5-sonnet-20241022')
            try:
                result_list = ast.literal_eval(result_list)
                assert len(result_list) == 2
                ans1 = str(result_list[0]['answer1'])
                thought1 = str(result_list[0]['thought1'])
                ans2 = str(result_list[1]['answer2'])
                thought2 = str(result_list[1]['thought2'])
                example_in_dict = dict_now['example']
                gt1 = example_in_dict[0]['output']
                gt2 = example_in_dict[1]['output']

                dict_now['example'][0]['get_ans1'] = ans1
                dict_now['example'][0]['thought1'] = thought1
                dict_now['example'][1]['get_ans2'] = ans2
                dict_now['example'][1]['thought2'] = thought2
                if str(ans1) == str(gt1) and str(ans2) == str(gt2):
                    dict_now['if_pass'] = True
                else:
                    dict_now['if_pass'] = False
            except:
                continue
        with open(save_path, "a", encoding='utf-8') as f:
            f.write(json.dumps(dict_now, ensure_ascii=False) + "\n")
    # print("总计输入token个数:", sum_input_token)
    # print("总计输出token个数:", sum_output_token)

def generate_python(save_path, vis, data_list):
    have_num = 0
    for item in data_list:
        have_num += 1
        print(have_num)
        dict_now = item
        if dict_now['if_pass'] == False:
            continue
        if dict_now['problem_id'] in vis:
            print(f"vis {dict_now['problem_id']}")
            continue
        prompt = copy.deepcopy(prompt_dict["python_generation_prompt"])
        prompt = prompt.format(problem=dict_now['description'],
                               input_format=dict_now['input requirement'],
                               output_format=dict_now['output requirement'],
                               input1=dict_now['example'][0]['input'],
                               input2=dict_now['example'][1]['input'],
                               output1=dict_now['example'][0]['output'],
                               output2=dict_now['example'][1]['output'],
                               json_type='''{"process":xxx,"code":xxx}''').strip()
        result_list,usage = get_valid_string(prompt,example_value=[],mode='python',model_name='deepseek-chat')
        if result_list == "":
                    continue
        try:
            result_list = ast.literal_eval(result_list)
            assert 'process' in result_list.keys() and 'code' in result_list.keys()
            dict_now['python_process'] = result_list['process']
            dict_now['python_code'] = result_list['code']

        except:
                continue
        with open(save_path, "a", encoding='utf-8') as f:
            f.write(json.dumps(dict_now, ensure_ascii=False) + "\n")
def read_oeis():
    # result_list = []
    # for i in range(0,129):
    #     name = r"" +str(i) +"_content.jsonl"
    #     with open(name,"r",encoding='utf-8') as f:
    #         for l in f:
    #             result_list.append(json.loads(l))
    #
    # content_list = []
    # for i in range(0, 129):
    #     name = r"" + str(i) + "_content.jsonl"
    #     with open(name, "r", encoding='utf-8') as f:
    #         for l in f:
    #             content_list.append(json.loads(l))
    #
    # for d in result_list:
    #     dd = {}
    #     dd[d['id']] = d
    #     for tmp_d in content_list:
    #         if tmp_d['id'] == d['id']:
    #             dd[d['id']]['content'] = tmp_d['content']
    #             # pdb.set_trace()
    #             with open('./oeis_all_information.json', "a", encoding='utf-8') as f:
    #                 f.write(json.dumps(dd, ensure_ascii=False) + "\n")
    #

    # pdb.set_trace()
    return Utils.read_files('')


if __name__ == "__main__":




