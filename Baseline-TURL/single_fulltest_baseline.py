import os
from openai import OpenAI
import csv
import json
from pathlib import Path
from random import shuffle, seed
import numpy as np
from tqdm import tqdm
from utils import *
import sys
from SPARQLWrapper import SPARQLWrapper, JSON
import os
import openai
import pandas as pd

seed(0)

openai.api_key = ""

def main():

    data_dir = '/mmfs1/gscratch/balazinska/linxiwei/TURL'
    type_vocab = load_type_vocab(data_dir)
    OUTPUT = "GPT3.5_output_single_fulltest_baseline_1.csv"
    results = []
    res_format = "{'type': []}"
    with open(os.path.join(data_dir, 'test.table_col_type.json'), 'r') as f:
        examples = json.load(f)
    num_examples = len(examples)

    for id, example in enumerate(tqdm(examples)):
        
        table_raw, table_with_en, label, headers, meta_data, num_col, max_num_row, colset = parse_example(example)
        x = min(len(table_raw),10)
        y = len(table_raw[0])
        table = []
        for col in range(y):
            column = []
            for row in range(x):
                if table_raw[row][col] == None:
                    column.append("")
                else: column.append(table_raw[row][col])
            table.append(column)
        num_col = len(table)
        table = list(zip(*table))
        headers =  tuple('' for _ in range(num_col)) 
        df = pd.DataFrame(table[0:], columns=headers, index=None)
        table = map(lambda x: ", ".join(x), table)
        CSV_like = ",\n".join(table)
        
        all_preds = []
        messages=[
            {
                "role": "system",
                "content": f"Be a helpful, accurate assistant for data discovery and exploration desiged to output valid JSON in the format {res_format}",
            },
            {
                "role": "user",
                "content": f"""Consider this table given in Comma-separated Values format:
                            ```
                            {CSV_like}
                            ```
                Your task is to assign only one semantic class to the first column that best represents all cells of this column. Solve this task by following these steps: 
                1. Look at the cells in the first column of the above table. 
                2. Choose only one valid type from the given list of types: {type_vocab.keys()}. Check that the type MUST be in the list. Give the answer in valid JSON format.
                """,
            }
        ]

        for i, col in enumerate(df.columns):
            chatgpt_msg = response(messages, 40)
            prediction = chatgpt_msg.content
            all_preds.append(prediction)
            messages.append(dict(chatgpt_msg))
            if i+1 >= len(colset): break
            messages.append(
                { "role": "user",
                "content": f"""Your task is to assign only one semantic class to the {i+2} column that best represents all cells of this column. Solve this task by following these steps: 
                1. Look at the cells in the {i+2} column of the above table.
                2. Choose only one valid type from the given list of types. Check that the type MUST be in the list. Give the answer in valid JSON format. 
                """
            })
        
        for i, p in enumerate(all_preds):
            results.append([id, i, p])

        with open(OUTPUT, "a", newline="", encoding="UTF-8") as output:
            writer = csv.writer(output)
            for i, p in enumerate(all_preds):
                writer.writerow([id, i, p])


if __name__ == "__main__":
    main()