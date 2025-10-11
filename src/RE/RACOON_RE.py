import os
from openai import OpenAI
import csv
import json
from pathlib import Path
from random import shuffle, seed
import numpy as np
from tqdm import tqdm
import os
import openai
import sys
from utils import *
import pandas as pd
data_path = Path(__file__).parent.parent / 'data'
sys.path.append(str(data_path))
from TURL_RE_label_reduction import RE_label_set
from openai import OpenAI
import argparse
openai.api_key = os.getenv("OPENAI_API_KEY")

from refined.inference.processor import Refined
import sys
sys.path.append('..')
from KG_Linker import get_column_wise_spans
from pruning import *

refined = Refined.from_pretrained(model_name='wikipedia_model',
                                  entity_set="wikidata")
preprocessor = refined.preprocessor

def main():
    parser = argparse.ArgumentParser(description='Relation Extraction with RACOON+')
    parser.add_argument('--model', type=str, default='gpt-4o-mini',
                        help='OpenAI model to use (default: gpt-4o-mini)')
    parser.add_argument('--context', type=str, default='hybrid',
                        choices=['wikiAPI', 'cell', 'table', 'col', 'hybrid'],
                        help='Context type (default: hybrid)')
    parser.add_argument('--info', type=str, default='type',
                        choices=['entity', 'des', 'type', 'relation'],
                        help='Information type (default: type)')

    args = parser.parse_args()

    data_dir = os.path.join(os.getenv('DATA_DIR', '.'), 'data')
    model = args.model
    context = args.context
    info = args.info

    OUTPUT = f"{model}/RACOON_{context}_{info}.csv"
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
        all_preds = []
        hints = []
        for col in range(y):
            column = []
            for row in range(x):
                if table_raw[row][col] == None:
                    column.append("")
                else: column.append(table_raw[row][col])
            table.append(column)
        table_cols = table.copy()
        table = list(zip(*table))
        table_row = table.copy()
        table = map(lambda x: ", ".join(x), table)
        CSV_like = ",\n".join(table)

        # --------------------- KG-Linker --------------------- #
        if context == "wikiAPI":
            for col in colset:
                col_qids = get_qid_wk(col)
                EL_result_table.append(col_qids)
        else:
            column_wise_table_spans = get_column_wise_spans(context, table_row, table_cols)
            EL_result_table = []
            for column_wise_table_span in column_wise_table_spans:
                EL_result_col = process_EL_res(column_wise_table_span)
                EL_result_table.append(EL_result_col)

        # --------------------- KG-Explorer --------------------- #
        hint = ""
        if info == "entity":
            table_entities = get_entity_label(EL_result_table)
            # hint = f"Each cell may be linked to an entity in the Wikidata knowledge graph. When available, the entity labels corresponding to the cells in the first column are provided as a list: {col_info}."
            orig_table_hint = []
            for c in range(num_col):
                t = "Column " + str(c) + ": " + str(table_entities[c])
                orig_table_hint.append(t)
            pruned_table_hint = pruning_orig(CSV_like, orig_table_hint)
            pattern = r"-?\s*Column \d+:(.*?)(?:\n|$)"
            hint_strings = re.findall(pattern, pruned_table_hint)
            hint = f"The entity labels in the Wikidata knowledge graph corresponding to the cells in this column are provided as a list: {hint_strings[0]}."
        
        elif info == "type":
            types,_ = get_types(EL_result_table)
            pruned_hint = pruning_orig(CSV_like,types)
            pattern = r"-?\s*Column \d+:\s*(.+?)(?=\n-?\s*Column \d+:|\Z)"
            dict_strings = re.findall(pattern, pruned_hint, re.DOTALL)
            dict_strings = [s.strip().strip('[]').strip() for s in dict_strings]
            hint_list = [safe_parse_dict(d) for d in dict_strings]
            hint = serialize_dict(hint_list[0])

        elif info == "des":
            table_entities = get_entity_label_des(EL_result_table)
            orig_table_hint = []
            for c in range(num_col):
                t = "Column " + str(c) + ": " + str(table_entities[c])
                orig_table_hint.append(t)
            pruned_table_hint = pruning_orig(CSV_like, orig_table_hint)
            pattern = r"-?\s*Column \d+:(.*?)(?:\n|$)"
            hint_strings = re.findall(pattern, pruned_table_hint)
            hint = f"The entity labels in the Wikidata knowledge graph corresponding to the cells in this column are provided as a list: {hint_strings[0]}."
        
        elif info == "relation":
            table_relations = get_entity_relation(EL_result_table)
            pruned_table_rel = pruning_rel(CSV_like, table_relations)
            hint = "This table's columns are related in Wikidata as follows: " + pruned_table_rel


        hints.append(hint)

        
        
        messages=[
            {
                "role": "system",
                "content": f"""You are a data analysis agent designed to annotate columns pairs with semantic relations. Your task is to:
                1. **Analyze** the given column pair in the context of the entire table.
                2. **Consider** a Wikidata hint if given. 
                3. **Reason** carefully to select the single best-fitting relation from the given list.
                4. **Output** ONLY in valid JSON format: {res_format}.
                
                Follow this format precisely:
                THOUGHT PROCESS: [Brief reasoning, mention if using/ignoring hint]
                FINAL ANSWER: [only the JSON output here]
                """
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
                4. For your reference, the entity pairs in Column 1 and Column 2 are connected by predicates: {hint} in the Wikidata knowledge graph.
                Use these predicates to help you understand the relation between the two column, but note that these relations might not be in the provided label set, so you still need to remap them to a relation in the given list.
                5. Choose the single best-fitting relation from the label set that applies to all rows. The chosen relation must be exactly one string from the list. 
                6.  **Output**:
                - First, explain your reasoning (non-JSON).
                - Finally, output **ONLY** the JSON answer in this format: {res_format}.
                Example:
                THOUGHT: The data in Column 1 are countries. The data in Column 2 are cities. The hint lists 'capital (6 entity pairs)', which confirms that the relation between Column 1 and Column 2 is has capital.
                FINAL ANSWER: {{"relation": ["has capital"]}}
                """,
            }
        ]

        chatgpt_msg = response(messages, model)
        prediction = chatgpt_msg.content
        predicted_type = parse_pred_RE(prediction)
        messages.append(dict(chatgpt_msg))

        if predicted_type not in RE_label_set:
            messages.append({"role": "user",
                    "content": f"""Your prediction {predicted_type} is not in the given label set. Please try again to choose the closest label in the given set to annotate the relation between Column 1 and Column 2. Choose only one valid relation from the given list of relations. Check that the relation MUST MUST MUST be in the list. Give the answer in valid JSON format. DO NOT include any explanation or additional text."""})
            chatgpt_msg = response(messages, 40)
            prediction = chatgpt_msg.content
            messages.append(dict(chatgpt_msg))
            predicted_type = parse_json_pred_RE(prediction)

        all_preds.append((predicted_type, prediction))

        for i in range(1, num_col):
            hint = ""
            hints.append(hint)
            messages.append(
            {
                "role": "user",
                "content": f"""Your task is to choose the most suitable relation from the list to annotate Column 1 (the subject column) and Column {i+2} (the object column) in the table so that the relation holds between all entity pairs in the columns. 
            Solve this task by following these steps: 
            1. Carefully examine the values in Column 1 and Column {i+2}.
            2. Identify the semantics of each column.
            3. For each row, imagine a relation in the form:  
            <Column 1 entity>, RELATION, <Column {i+2} entity>
            4. For your reference, the entity pairs in Column 1 and Column {i+2} are connected by predicates: {hint} in the Wikidata knowledge graph.
            Use these predicates to help you understand the relation between the two column, but note that these relations might not be in the provided label set, so you still need to remap them to a relation in the given list.
            5. Choose the single best-fitting relation from the label set that applies to all rows. The chosen relation must be exactly one string from the list. 
            6.  **Output**:
            - First, explain your reasoning (non-JSON).
            - Finally, output **ONLY** the JSON answer in this format: {res_format}.
            Example:
            THOUGHT: The data in Column 1 are countries. The data in Column {i+2} are cities. The hint lists 'capital (6 entity pairs)', which confirms that the relation between Column 1 and Column {i+2} is has capital.
            FINAL ANSWER: {{"relation": ["has capital"]}}
                """,
            })

            chatgpt_msg = response(messages, model)
            prediction = chatgpt_msg.content
            predicted_type = parse_pred_RE(prediction)
            messages.append(dict(chatgpt_msg))

            if predicted_type not in RE_label_set:
                messages.append({"role": "user",
                        "content": f"""Your prediction {predicted_type} is not in the given label set. Please try again to choose the closest label in the given set to annotate the relation between Column 1 and Column {i+2}. Choose only one valid relation from the given list of relations. Check that the relation MUST MUST MUST be in the list. Give the answer in valid JSON format. DO NOT include any explanation or additional text."""})
                chatgpt_msg = response(messages, model)
                prediction = chatgpt_msg.content
                messages.append(dict(chatgpt_msg))
                predicted_type = parse_json_pred_RE(prediction)
            all_preds.append((predicted_type, prediction))

        with open(OUTPUT, "a", newline="", encoding="UTF-8") as output:
            writer = csv.writer(output)
            for i, p in enumerate(all_preds):
                writer.writerow([id, i, p])
            

if __name__ == "__main__":
    main()