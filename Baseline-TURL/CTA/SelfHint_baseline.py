import ast
import os
import csv
import json
from random import seed
import re
from tqdm import tqdm
from utils import parse_example, response
from data.TURL_CTA_label_reduction import reduced_label_set
import os
import openai

seed(0)
openai.api_key = os.getenv("OPENAI_API_KEY")

def main():

    data_dir = 'data/'
    model = "gpt-4o-mini"
    OUTPUT = f"{model}/SelfHint_baseline.csv"
    res_format = "{'type': []}"

    with open(os.path.join(data_dir, 'test.table_col_type.json'), 'r') as f:
        examples = json.load(f)

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
        table = list(zip(*table))
        table = map(lambda x: ", ".join(x), table)
        CSV_like = ",\n".join(table)
        
        # step 1: input table -> output : every column topic/type/.. without providing a label set
        all_preds1 = []
        messages1=[
            {
                "role": "system",
                "content": f"You are a helpful and precise assistant for data discovery and exploration. "
                        f"Your responses must always be valid JSON, strictly following this format: {res_format}. "
                        f"Do not include any explanations or additional text."
            },
            {
                "role": "user",
                "content": f"""Analyze the following table in Comma-Separated Values (CSV) format:
                ```
                {CSV_like}
                ```
                Your task is to classify the **first column** by assigning **one semantic class** that best represents all its cells. 

                Follow these steps:
                1. Examine the table to understand its overall theme.
                2. Analyze the values in the first column and determine their semantic meaning.
                3. Briefly summarize the common characteristic of the values in the first column.
                4. Assign a **single best-fitting type** to the column.
                
                Return the result **only** as a valid JSON object, following this format:  
                ```json
                {res_format}
                ```  
                Do not provide explanations or extra text."""
            }
        ]

        coarse_types = []
        for i in range(num_col):
            chatgpt_msg = response(messages1, model)
            prediction = chatgpt_msg.content
            prediction = re.sub(r"```json|```", "", prediction).strip()
            try:
                data = ast.literal_eval(prediction)
            except (SyntaxError, ValueError) as e:
                print(f"Failed to parse prediction: {prediction}")
                print(f"Error: {e}")
                data = {'type': []}  
            if 'type' not in data or len(data['type'])<1: info = None
            else: info = data['type'][0]
            if info: coarse_types.append(info)
            else: coarse_types.append("")
            all_preds1.append(prediction)
            messages1.append(dict(chatgpt_msg))
            messages1.append(
                { "role": "user",
                "content": f"""Your task is to classify the **{i+1} column** by assigning **one semantic class** that best represents all its cells. 

                    Follow these steps:
                    1. Examine the table to understand its overall theme.
                    2. Analyze the values in the {i+2} column and determine their semantic meaning.
                    3. Briefly summarize the common characteristic of the values in the {i+2} column.
                    4. Assign a **single best-fitting type** to the column.
                    
                    Return the result **only** as a valid JSON object, following this format:  
                    ```json
                    {res_format}
                    ```  
                    Do not provide explanations or extra text."""
                })
        
        # step 2: 1st LLM output along with the label set, output the prediction
        all_preds2 = []

        messages2=[
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
                There are a list of 127 valid types for each column: {reduced_label_set}. Your task is to choose only one type from the list to annotate the first column. Solve this task by following these steps: 
                1. Look at the cells in the first column of the above table. 
                2. Take the coarse type for this column given by another LLM as a reference: {coarse_types[0]}.
                3. Choose only one valid type from the given list of types. Check that the type MUST MUST MUST be in the given list. Give the answer in valid JSON format.
                """,
            }
        ]

        for i in range(num_col):
            chatgpt_msg = response(messages2, model)
            prediction = chatgpt_msg.content
            all_preds2.append(prediction)
            messages2.append(dict(chatgpt_msg))
            messages2.append(
                { "role": "user",
                "content": f"""Your task is to assign only one semantic class to the {i+2} column that best represents all cells of this column. Solve this task by following these steps: 
                1. Look at the cells in the {i+2} column of the above table.
                2. Take the coarse type for this column given by another LLM as a reference: {coarse_types[i+1]}.
                3. Choose only one valid type from the given list of types. Check that the type MUST MUST MUST be in the given list. Give the answer in valid JSON format.
                """
            })

        with open(OUTPUT, "a", newline="", encoding="UTF-8") as output:
            writer = csv.writer(output)
            for i, p in enumerate(all_preds2):
                writer.writerow([id, i, p, coarse_types[i]])


if __name__ == "__main__":
    main()