import os
import sys
import csv
import json
from random import seed
from tqdm import tqdm
from utils import parse_example, response
from pathlib import Path
data_path = Path(__file__).parent.parent / 'data'
sys.path.append(str(data_path))
from TURL_CTA_label_reduction import reduced_label_set
import os
import openai

seed(0)
openai.api_key = os.getenv("OPENAI_API_KEY")

def main():

    data_dir = os.path.join(os.getenv('DATA_DIR', '.'), 'data')
    model = "gpt-4o-mini"
    OUTPUT = f"{model}/NoHint_baseline.csv"
    os.makedirs(os.path.dirname(OUTPUT), exist_ok=True)
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
                There are a list of 127 valid types for each column: {reduced_label_set}. Your task is to choose only one type from the list to annotate the first column. Solve this task by following these steps: 
                1. Look at the cells in the first column of the above table. 
                2. Choose only one valid type from the given list of types. Check that the type MUST MUST MUST be in the given list. Give the answer in valid JSON format.
                """,
            }
        ]

        for i in range(num_col):
            chatgpt_msg = response(messages, model)
            prediction = chatgpt_msg.content
            all_preds.append(prediction)
            messages.append(dict(chatgpt_msg))
            messages.append(
                { "role": "user",
                "content": f"""Your task is to choose only one type from the list to annotate the {i+2} column. Solve this task by following these steps: 
                1. Look at the cells in the {i+2} column of the above table. 
                2. Choose only one valid type from the given list of types. Check that the type MUST MUST MUST be in the given list. Give the answer in valid JSON format.
                """
            })

        with open(OUTPUT, "a", newline="", encoding="UTF-8") as output:
            writer = csv.writer(output)
            for i, p in enumerate(all_preds):
                writer.writerow([id, i, p])


if __name__ == "__main__":
    main()