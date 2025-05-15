prompt_dict = {
    # TODO：1.混合所有信息 2.测试与示例不放在一起 3.5和6不一定需要 4.第四个点改一下
    "problem_generation_prompt":
        """
        现在我会给你一些信息，请你根据这些信息生成一道ACM算法题。
        
        我首先给你如下数列: {shulie}。
        关于这个数列的一个系统性的描述: {descrip}。
        关于这个数列的一些相关信息:{comment}。
        关于这个数列的一些参考信息:{reference}。
        关于这个数列的一些链接信息:{link}。
        可能相关这个数列的的一个例子:{examp}。
        关于该数列的类似于伪代码一样的描述:{maple}。
        关于数列的一个公式以方便你理解: {formula}。
        关于该数列的数学描述以方便你理解: {math}。
        关于该数列通项的代码表述: {program}。
        
        现在请你根据上面信息，帮我生成一道ACM形式的算法题目，要求：
        1. 包括题目描述description，输入格式input，输出格式output。
        2. 然后你需要给出2组示例输入示例输出和示例解释,放入一个example对应的数组，其中每个元素是一个字典，现在有2组示例，所以有2个字典。这些example的output数据必须来自原数列，且input需要尽可能简单，使用个位数(0-9)当成input。
        3. 接下来再请你给出5-8个测试输入测试输出，但是这次不用示例解释，放入test数组重，用字典表示，注意input和output都要是数字，并且output要来自于原数列，不要超出数列范围。
        4. 你生成的算法题是根据给定的数列、该数列的公式和数学描述，使用一个故事包装描述数列的内在逻辑，而不能仅仅输出上面的系统性描述，也不要直接说出有关输入数列的基本信息和基本逻辑。
        5. 你需要满足仅仅根据你生成的description，能够让做题者反推数列的数学内核，从而完成这道算法题。
        6. 你需要保证输出内容都是源于给定的数列。
        7. 题目请用英文进行展示。
        8. 请你按照如下json字典形式进行输出: {json_type}。
        
        现在请你给出你输出的json字典,请不要输出任何别的内容。
        """,
    "problem_generation_prompt2":
        """
        现在我会给你一些信息，请你根据这些信息生成一道ACM算法题。

        这些信息是: {shulie}。{descrip}。{comment}。{reference}。{link}。{examp}。{maple}。{formula}。{math}。{program}。

        现在请你根据上面信息，帮我生成一道ACM形式的算法题目，要求：
        1. 包括题目描述description，输入格式input，输出格式output。
        2. 然后你需要给出2组示例输入示例输出和示例解释,放入一个example对应的数组，其中每个元素是一个字典，现在有2组示例，所以有2个字典。这些example的output数据必须来自原数列，且input需要尽可能简单，使用个位数(0-9)当成input。
        3. 接下来再请你给出5-8个测试输入测试输出，但是这次不用示例解释，放入test数组重，用字典表示，注意input和output都要是数字，并且output要来自于原数列，不要超出数列范围。
        4. 你生成的算法题是根据给定的数列、该数列的公式和数学描述，使用一个故事包装描述数列的内在逻辑，而不能仅仅输出上面的系统性描述，也不要直接说出有关输入数列的基本信息和基本逻辑。
        5. 你需要满足仅仅根据你生成的description，能够让做题者反推数列的数学内核，从而完成这道算法题。
        6. 你需要保证输出内容都是源于给定的数列。
        7. 题目请用英文进行展示。
        8. 请你按照如下json字典形式进行输出: {json_type}。

        现在请你给出你输出的json字典,请不要输出任何别的内容。
        """,
    "problem_generation_prompt3":
        """
        现在我会给你一些信息，请你根据这些信息生成一道ACM算法题。

        这些信息是: {content}。

        现在请你根据上面信息，帮我生成一道ACM形式的算法题目，要求：
        1. 包括题目描述description，输入格式input，输出格式output。
        2. 然后你需要给出2组示例输入示例输出和示例解释,放入一个example对应的数组，其中每个元素是一个字典，现在有2组示例，所以有2个字典。这些example的output数据必须来自原数列，且input需要尽可能简单，使用个位数(0-9)当成input。
        3. 接下来再请你给出5-8个测试输入测试输出，但是这次不用示例解释，放入test数组重，用字典表示，注意input和output都要是数字，并且output要来自于原数列，不要超出数列范围。
        4. 示例输出要尽量来自数列的{example_can}这几项。测试输出则限制在数列的{test_can}这几项中挑选5-8条。
        5. 你生成的算法题是根据给定的数列、该数列的公式和数学描述，使用一个故事包装描述数列的内在逻辑，而不能仅仅输出上面的系统性描述，也不要直接说出有关输入数列的基本信息和基本逻辑。
        6. 你需要满足仅仅根据你生成的description，能够让做题者反推数列的数学内核，从而完成这道算法题。
        7. 你需要保证输出内容都是源于给定的数列。
        8. 题目请用英文进行展示。
        9. 请你按照如下json字典形式进行输出: {json_type}。

        现在请你给出你输出的json字典,请不要输出任何别的内容。
        """,

    "answer_generation_prompt":
    '''
    现在请你扮演一名ACM算法竞赛选手，我给你一个题目和两个输入input，请你告诉给出我最后的思路和答案。
    
    我会给出你这道ACM题目:{problem}。
    这道ACM题目的输入格式是:{input_format}。
    这道ACM题目的输出格式是:{output_format}。
    然后我会给你两个输入{input1}和{input2}。
    
    请你分别输出两个题目的思路thought和答案answer。
    请你think step by step，一步一步计算，仔细计算，仔细读题。
    请你按照如下json字典形式进行输出:{json_type},即输出一个python的list，每个元素是一个字典，每个字典包含一个题目的thought和answer。
    ''',

    "answer_generation_prompt2":
    '''
    我有你一个acm题目和题目对应的两个样例输入input，请你给出每个样例的完整详细的求解过程和最后的答案。
    题目：{problem}。
    输入格式：{input_format}。
    输出格式：{output_format}。
    输入1：{input1}
    输入2：{input2}。

    请你分别给出不同input的详细作答过程process和最终的答案answer。
    在作答过程process中请你一步一步求解，列出每一步求解过程。
    请你按照如下json字典形式进行输出:{json_type},即输出一个python的list，每个元素是一个字典，每个字典包含一个题目的详细作答过程process和最后的answer。
    ''',
    "answer_generation_prompt3":
    '''
    I have an ACM problem and two sample inputs corresponding to the problem. Please provide the detailed solution process and the final answer for each sample input.
    Problem: {problem}.
    Input format: {input_format}.
    Output format: {output_format}.
    Input 1: {input1}.
    Input 2: {input2}.

    For each input, please provide the step-by-step solution process and the final answer. In the solution process, please solve step by step and list each intermediate step.
    The output should be in the following JSON dictionary format: {json_type}, where each element must be a python dictionary containing the detailed solution process and the final answer for the problem.
    ''',
    "python_generation_prompt":
        '''
        现在请你扮演一名ACM算法竞赛选手，我给你一个题目和两个样例，请你告诉给出这道题的python的解题代码code。
    
        我会给出你这道ACM题目:{problem}。
        这道ACM题目的输入格式是:{input_format}。
        这道ACM题目的输出格式是:{output_format}。
        然后我会给你两个输入{input1}和{input2}，这两个输入分别对应的输出应该是{output1}和{output2}。
    
        请你给出这道题目的思路过程process和python code。
        请你注意：你给出的代码应该是这道题的通解，不能局限于两个样例。
        请你按照如下json字典形式进行输出:{json_type},即输出一个python的字典,字典中包含process和code两个键，分别代表思路过程process和python code。
        注意，你的python code字段请不要输出除了代码以外的任何东西，包括“以下是python代码”，“python code:”等等提示字样，都不需要！
        ''',


        "prompt_problem_generate_rl" : '''
I want to generate an ACM problem based on the following OEIS sequence:
{content}

You need to embed a sequence into an ACM-style problem by leveraging the properties of the sequence **without directly referencing sequence-related information**. The problem should have the sequence's logic as its core mechanism but must **not explicitly ask for solving the sequence**. Craft a vivid story for the problem description while ensuring clarity, meaning the problem description alone should fully explain the requirements.
Additionally, construct detailed explanations for the 2 examples in the JSON (examples derived from the sequence). **Do not modify the example outputs**, but you may adjust the input with an appropriate offset based on the problem context. Provide thorough explanations for the examples, preferably including specific cases. **Carefully verify** that the problem description aligns with the examples in the JSON.
First analyze the sequence and examples to ensure correctness. Ensure reasonable input constraints and avoid excessively large output numbers (consider applying a modulo operation if needed). The problem description should not mention coding requirements—simply describe the problem clearly. Ensure that for the sample input, the sample output can be obtained solely based on the problem description.
All your output should be in English.
Please directly return an English ACM-style problem in the following Python JSON format, your output must have ```json...``` out of a dictionary:
```json
{{
  "description": "
  Problem description......
  ", 
  "example": [
    {{
      "input": "{e1in}",
      "output": "{e1out}",
      "explanation": "
      Example 1 explanation......
      "
    }},
    {{
      "input": "{e2in}",
      "output": "{e2out}",
      "explanation": "
      Example 2 explanation......
      """
    }},
  ],
  "input requirement": "Input requirement",
  "output requirement": "Output requirement"
}}
```''',
  "answer_generation_prompt4":
    '''
    I have an ACM problem. I provide the input and output requirements for this problem, along with two examples and their explanations. 
    Problem: {problem}.

    Input requirements: {input_format}.
    Output requirements: {output_format}.

    Input 1: {input1}.
    Output 1: {output1}.
    Explanation 1: {explanation1}.

    Input 2: {input2}.
    Output 2: {output2}.
    Explanation 2: {explanation2}.

    Now please provide the solution code for this problem along with your thought process.
    Pay attention: you must use `input()` rather than `sys.stdin` as the input, `print()` rather than `sys.stdout` as the output,
    Please directly return an English ACM-style problem in the following Python JSON format, your output must have ```json...``` out of a dictionary:
```json
{{
  "code": "
  python code......
  ", 
  "thought process": "
  your thought process......
  ",
}}
```''',

  "answer_generation_prompt5":
    '''
    I have an ACM problem. I provide the input and output requirements for this problem, along with two examples and their explanations. 
    Problem: {problem}.

    Input requirements: {input_format}.
    Output requirements: {output_format}.

    Input 1: {input1}.
    Output 1: {output1}.
    Explanation 1: {explanation1}.

    Input 2: {input2}.
    Output 2: {output2}.
    Explanation 2: {explanation2}.

    Now please provide the solution code for this problem along with your thought process.
    Pay attention: you must use `input()` rather than `sys.stdin` as the input, `print()` rather than `sys.stdout` as the output,
    Please first generate an explanation in English, and then output the Python code:
    Explanation: (your explanation of your code). 
    ```python
    (your code here)
    ```
    ''',
}