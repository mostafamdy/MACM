import os
import re
import json
import random
from prompt.prompts import *
from collections import Counter
from macm.executor import Execute_steps
from macm.judge import Judge_statement, Judge_answer, Judge_condition
from macm.thinker import Analysis_conditions, Think_thoughts, Think_Steps
from utils.gpt_robots import set_model
import argparse
import gc
import torch
from transformers import BitsAndBytesConfig, AutoModelForCausalLM, GemmaTokenizer

def check_condition(question,condition, n):
    """
    Use several Judges to check the statement
    Input:
    conditions, unchecked_conditions, the number of the inspectors (List, Str, int)
    Output:
    True/False (bool)
    """
    for _ in range(n):
        if Judge_condition(question,condition).strip() == "False":
            return False
    return True

def check_statement(conditions,statement, n):
    """
    Use several Judges to check the statement
    Input:
    conditions, unchecked_conditions, the number of the inspectors (List, Str, int)
    Output:
    True/False (bool)
    """
    for _ in range(n):
        answer = Judge_statement(conditions,statement)
        if  "False" in answer or "false" in answer:
            return False
    return True


def check_answer(conditions,statement):
    """
    Use several Judges to check the answer
    Input:
    unchecked_conditions, the number of the inspectors (Str, int)
    Output:
    True/False (bool)
    """
    if_got_answer = Judge_answer(conditions,statement)
    if "False" in if_got_answer or "false" in if_got_answer:
        return False
    return True


def check_if_got_answer(conditions,statement,n):
    for _ in range(n):
        if check_answer(conditions,statement) == False:
            return False
    return True    

def main(question, times, n, min_voters, max_voters,model,tokenizer):
    """
    Input question and get the final answer from muti-Agent got
    Input:
    quesion, the number of times new conditions are identified, the number of the inspectors  (Str, int, int)
    Output:
    final answer (Str)
    """
    set_model(model,tokenizer)
    possible_answers = []
    try:
        voter_count = 0
        tie = True
        
        # Vote
        while tie or voter_count < min_voters:
            voter_count += 1
            print(f"\n# {voter_count} Thinker is analyzing the question...")
            print(f"Question:\n {question}")
            conditions,objectives = Analysis_conditions(question)
            Initial_condition_numbers = len(conditions) # This line will be used for the $while$ mode
            print(f"conditions:\n {conditions}")
            print(f"objectives:\n {objectives}")
            
            # Think thoughts
            # while len(conditions) - Initial_condition_numbers <= times: 
            for time in range(times): # Try to reduce the LLM queries.
                print(f"\n# {voter_count} Thinker is thinking new thoughts...")
                unchecked_conditions = Think_thoughts(conditions,objectives)
                print(f"unchecked_conditions:\n {unchecked_conditions}")
                checked_conditions = []
                for unchecked_condition in unchecked_conditions:
                    print(f"\n# {voter_count} Judge is checking conditions...")
                    if check_statement(conditions,unchecked_condition,n):
                        start = unchecked_condition.find("we can get: ")
                        if start != -1:
                            unchecked_condition = unchecked_condition[start + len("we can get: "):]
                            unchecked_condition = unchecked_condition.split("Reason:")[0]
                        checked_conditions.append(unchecked_condition)
                conditions = conditions + checked_conditions
                print(f"conditions:\n {conditions}")
                if_got_answer = check_if_got_answer(conditions,objectives,1)
                if if_got_answer:
                    break
            print(f"\n# {voter_count} thinker is thinking steps...")
            steps = Think_Steps(conditions,objectives)
            print(f"steps:\n {steps}")
            
            print(f"\n# {voter_count} Executor is trying to calculate the answer...")
            final_answer = Execute_steps(conditions,objectives,steps)
            print(f"final_answer:\n {final_answer}")
            # Achieve one potiential answer
            Answer = re.search(r'\\boxed\{(.*)(?=\})', final_answer)  
            if Answer:
                Answer_boxed = Answer.group(1)
            else:
                Answer_boxed = "No match found"
            possible_answers.append(Answer_boxed)
            if voter_count >= min_voters:
                counter = Counter(possible_answers)
                most_votes = counter.most_common(1)[0][1]  
                tie_count = len(list(filter(lambda x: x[1] == most_votes, counter.items())))
                
                tie = tie_count > 1
                print("\nThere is a tie vote. We need to add another voter.")
                if voter_count >= max_voters:
                    print("\nReached maximum voter limit.")
                    break
        most_possible_answer, count = counter.most_common(1)[0]
        print(f"\nThe final answer is {most_possible_answer}")
        
        return most_possible_answer
    except Exception as e:
        print(f"Error processing file: {e}")


def evaluate_dataset(folder_path, times, n, limit=5):
    all_files = []
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith('.json'):
                file_path = os.path.join(root, file)
                all_files.append(file_path)

    random.shuffle(all_files)  # Shuffle the order of files randomly

    for count, file_path in enumerate(all_files[:limit]):
        with open(file_path, 'r') as json_file:
            try:
                data = json.load(json_file)
                problem = data.get("problem")
                if problem:
                    print(f"#{count} Problem:\n", problem)
                    solution = data.get("solution")
                    print(f"#{count} Solution\n", solution)
                    main(problem, times, n)
            except json.JSONDecodeError:
                print(f"Error reading file {file_path}")
            except Exception as e:
                print(f"Error processing file {file_path}: {e}")
                            
                                          
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MACM parser we execpt n times min_voters max_voters question")
    # Define required and optional arguments
    parser.add_argument("-q","--question", type=str, help="Question")
    parser.add_argument("-minv","--min_voters", type=int, help="min voters count")
    parser.add_argument("-maxv","--max_voters", type=int, help="max voters count")
    parser.add_argument("-t","--times", type=int, help="The upper limit of the mining times")
    parser.add_argument("-n","--verification_times", type=int, help="verification times")
    parser.add_argument("-m","--model_path", type=int, help="model path")
    # Parse arguments from the command line
    args = parser.parse_args()
    n=args.verification_times
    times=args.times
    min_voters=args.min_voters
    max_voters=args.max_voters
    question=args.question
    model_path=args.model_path
    
    tokenizer = GemmaTokenizer.from_pretrained(model_path)
    
    _model = AutoModelForCausalLM.from_pretrained(
        model_path,#"/kaggle/input/codegemma/transformers/7b-it/1",
        device_map="auto",
        torch_dtype="auto",
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True,  # Loading weights in 4-bit format
            bnb_4bit_quant_type="nf4",  # Using non-linear quantization with 4 bits
            bnb_4bit_compute_dtype=torch.bfloat16,  # Using bfloat16 for computation
            bnb_4bit_use_double_quant=True  # Using double quantization
        )
    )
    
    main(question, times, n, min_voters, max_voters,_model,tokenizer)  # Assuming these are defined elsewhere
