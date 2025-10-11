import os
import csv
import json
from pathlib import Path
from tqdm import tqdm
import os
import openai
import sys
from utils import *
data_path = Path(__file__).parent.parent / 'data'
sys.path.append(str(data_path))
from TURL_RE_label_reduction import RE_label_set
openai.api_key = os.getenv("OPENAI_API_KEY")


def main():
    data_dir = os.path.join(os.getenv('DATA_DIR', '.'), 'data')
    model = "gpt-4o-mini"
    OUTPUT = f"{model}/SelfHint_baseline.csv"
    os.makedirs(os.path.dirname(OUTPUT), exist_ok=True)

    with open(os.path.join(data_dir, 'test.table_rel_extraction.json'), 'r') as f:
        examples = json.load(f)
    res_format = "{'relation': []}"

    for id, example in enumerate(tqdm(examples)):
        hints = []
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
        table = map(lambda x: ", ".join(x), table)
        CSV_like = "\n".join(table)


        # step 1: input table -> output : every column pair output a relation without providing a label set
        all_preds1 = []

        messages1=[
                {
                    "role": "system",
                    "content": f"Be a helpful, accurate assistant for data discovery and exploration desiged to output JSON in the format {res_format}",
                },
                {
                    "role": "user",
                    "content": f"""
                    Relation extraction is mapping subject-object column pairs in a table to a given set of relations. The subject column is the first column of the table, and the object columns are the rest.
                    Consider this table given in Comma-separated Values format:
                    {CSV_like}

                    Your task is to assign only one relation that correctly describes how Column 1 (subject) relates to Column 2 (object), in the direction from subject to object.

                    Solve this task by following these steps: 
                    1. Carefully examine the values in Column 1 and Column 2.
                    2. Identify the semantics of each column.
                    3. For each row, imagine a relation in the form:  
                    <Column 1 entity>, RELATION, <Column 2 entity> 
                    4. Assign the single best-fitting relation that applies to all rows.
                    5. Output only the chosen relation in valid JSON format. Do not provide explanations or extra text.
                    """,
                }
            ]
        
        coarse_rels = []
        for i in range(1, num_col):
            chatgpt_msg = response(messages1, model)
            prediction = chatgpt_msg.content
            prediction = re.sub(r"```json|```", "", prediction).strip()
            print(prediction)
            try:
                data = ast.literal_eval(prediction)
                if len(data['relation']) < 1:
                    info = None
                else:
                    info = data['relation'][0]
            except (ValueError, SyntaxError, KeyError, TypeError):
                info = None
            if info: coarse_rels.append(info)
            else: coarse_rels.append("")
            all_preds1.append(prediction)
            messages1.append(dict(chatgpt_msg))

            messages1.append(
            {
                "role": "user",
                "content": f"""Your task is to choose the most suitable relation to annotate Column 1 (the subject column) and Column {i+2} (the object column) in the table so that the relation holds between all entity pairs in the columns. 
                Solve this task by following these steps: 
                1. Carefully examine the values in Column 1 and Column {i+2}.
                2. Identify the semantics of each column.
                3. For each row, imagine a relation in the form:  
                <Column 1 entity>, RELATION, <Column {i+2} entity>
                4. Choose the single best-fitting relation that applies to all rows.
                5. Output only the chosen relation in valid JSON format. Do not provide explanations or extra text.
                """,
            })
        
        # step 2: 1st LLM output along with the label set, output the prediction
        all_preds2 = []
        messages2=[
                {
                    "role": "system",
                    "content": f"Be a helpful, accurate assistant for data discovery and exploration desiged to output JSON in the format {res_format}",
                },
                {
                    "role": "user",
                    "content": f"""
                    Relation extraction is mapping subject-object column pairs in a table to a given set of relations. The subject column is the first column of the table, and the object columns are the rest.
                    Consider this table given in Comma-separated Values format:
                    {CSV_like}

                    This is the label set of 66 relations: {RE_label_set}

                    Your task is to choose only one relation from the list that correctly describes how Column 1 (subject) relates to Column 2 (object), in the direction from subject to object.

                    Solve this task by following these steps: 
                    1. Carefully examine the values in Column 1 and Column 2.
                    2. Identify the semantics of each column.
                    3. For each row, imagine a relation in the form:  
                    <Column 1 entity>, RELATION, <Column 2 entity> 
                    4. Take the coarse relation for this column pair given by another LLM as a reference: {coarse_rels[0]}.
                    5. Choose the single best-fitting relation from the label set that applies to all rows.
                    6. The chosen relation must be exactly one string from the list. Output only that relation in valid JSON format.
                    """,
                }
            ]

        for i in range(1, num_col):
            chatgpt_msg = response(messages2, model)
            prediction = chatgpt_msg.content
            all_preds2.append(prediction)
            messages2.append(dict(chatgpt_msg))
            if i+2 >= len(colset): break
            messages2.append(
            {
                "role": "user",
                "content": f"""Your task is to choose the most suitable relation from the list to annotate Column 1 (the subject column) and Column {i+2} (the object column) in the table so that the relation holds between all entity pairs in the columns. 
            Solve this task by following these steps: 
                1. Carefully examine the values in Column 1 and Column {i+2}.
                2. Identify the semantics of each column.
                3. For each row, imagine a relation in the form:  
                <Column 1 entity>, RELATION, <Column {i+2} entity>
                4. Take the coarse relation for this column pair given by another LLM as a reference: {coarse_rels[i]}.
                5. Choose the single best-fitting relation from the label set that applies to all rows.
                6. The chosen relation must be exactly one string from the list. Output only that relation in valid JSON format.
                """,
            })

        with open(OUTPUT, "a", newline="", encoding="UTF-8") as output:
            writer = csv.writer(output)
            for i, p in enumerate(all_preds2):
                writer.writerow([id, i, p, coarse_rels[i]])


if __name__ == "__main__":
    main()