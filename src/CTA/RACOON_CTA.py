import os
import csv
import json
from random import seed
from tqdm import tqdm
from pathlib import Path
data_path = Path(__file__).parent.parent / 'data'
sys.path.append(str(data_path))
from TURL_CTA_label_reduction import reduced_label_set
import os
import openai
import re
import argparse

seed(0)
openai.api_key = os.getenv("OPENAI_API_KEY")

from refined.inference.processor import Refined
import sys
sys.path.append('..')
from KG_Linker import get_column_wise_spans
from pruning import *
from utils import *

refined = Refined.from_pretrained(model_name='wikipedia_model',
                                  entity_set="wikidata")
preprocessor = refined.preprocessor

def get_info_re(column):
    en_labels = []
    text = ""
    for (pid, em) in column:
        spans = refined.process_text(em)
        if (len(spans) < 1): continue
        item = spans[0]
        item_string = item.__repr__()
        res = item_string.strip("[]").split(", ")
        label_match = re.search(r'wikipedia_entity_title=([^,]+)', res[2][:-1])
        if label_match:
            entity = label_match.group(1)
            en_labels.append(entity)

    return en_labels


def get_col_entity(col_spans):
    en_labels = []
    for span in col_spans:
        if len(span) < 1: continue
        item = span[0] # for each cell, only take the first entity linked
        item_string = item.__repr__()
        res = item_string.strip("[]").split(", ")
        label_match = re.search(r'wikipedia_entity_title=([^,]+)', res[2][:-1])
        if label_match:
            entity = label_match.group(1)
            en_labels.append(entity)
    return en_labels

def main():
 
    
    parser = argparse.ArgumentParser(description='Column Type Annotation with RACOON+')
    parser.add_argument('--model', type=str, default='gpt-4o-mini',
                        help='OpenAI model to use (default: gpt-4o-mini)')
    parser.add_argument('--context', type=str, default='col',
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

    res_format = "{'type': []}"

    with open(os.path.join(data_dir, 'test.table_col_type.json'), 'r') as f:
        examples = json.load(f)

    for id, example in enumerate(tqdm(examples)):
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
            orig_types = []
            for c in range(num_col):
                t = "Column " + str(c) + ": " + str(types[c])
                orig_types.append(t)
            pruned_hint = pruning_orig(CSV_like,orig_types)
            pattern = r"-?\s*Column \d+:(.*?)(?:\n|$)"
            dict_strings = re.findall(pattern, pruned_hint)
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
            hint = f"The entity labels with descriptions in the Wikidata knowledge graph corresponding to the cells in this column are provided as a list: {hint_strings[0]}."
        
        elif info == "relation":
            table_relations = get_entity_relation(EL_result_table)
            pruned_table_rel = pruning_rel(CSV_like, table_relations)
            hint = "This table's columns are related in Wikidata as follows: " + pruned_table_rel

        hints.append(hint)

        # --------------------- Generation --------------------- #
        
        messages = [{
                "role": "system",
                "content": f"""You are a data type analysis agent designed to annotate columns with semantic types. Your task is to:
            1. **Analyze** the given column data in the context of the entire table.
            2. **Consider** a Wikidata hint if given. 
            3. **Reason** carefully to select the best single type from the given list.
            4. **Output** ONLY in valid JSON format: {res_format}.

            Follow this format precisely:
            THOUGHT PROCESS: [Brief reasoning, mention if using/ignoring hint]
            FINAL ANSWER: [only the JSON output here]"""
            },
            {
                "role": "user",
                "content": f"""Consider this table with {num_col} columns given in Comma-separated Values format:
                        {CSV_like}
                        There are a list of 126 valid types for each column: {reduced_label_set}. Your task is to choose only one type from the list to annotate Column {1}. 
            Solve this task by following these steps:
            1. **Analyze the data in Column {1} in the context of the entire table**
            2. **Evaluate the Wikidata hint if not empty: {hint}**:
            - Use the hint to enhance your understanding of the column data.
            3. **Select the best type** from the given label set. 
            - Ensure the type best summarizes every cell in the column.
            - Ensure it aligns with the context of the whole table.
            - Verify your chosen type MUST be in the provided list.
            4. **Output**:
            - First, explain your reasoning (non-JSON).
            - Finally, output **ONLY** the JSON answer in this format: {res_format}.
            Example:
            THOUGHT: The data in this column consists of names of sports teams. The hint lists 'football_club (7 cells)', which confirms that the column is about sports teams. I select 'sports_team'.
            FINAL ANSWER: {{"type": ["sports_team"]}}"""}
        ]

        chatgpt_msg = response(messages, model)
        prediction = chatgpt_msg.content
        predicted_type = parse_pred(prediction)
        messages.append(dict(chatgpt_msg))

        if predicted_type not in reduced_label_set:
            messages.append({"role": "user",
                "content": f"""Your prediction {predicted_type} is not in the given label set. Please try again to choose the closest label in the given set to annotate the first column. Choose only one valid type from the given list of types. Check that the type MUST MUST MUST be in the list. Give the answer in valid JSON format. DO NOT include any explanation or additional text."""})
            chatgpt_msg = response(messages, model)
            prediction = chatgpt_msg.content
            messages.append(dict(chatgpt_msg))
            predicted_type = parse_json_pred(prediction)
                
        all_preds.append((predicted_type, prediction))

        for i in range(1, num_col):

            if info == "type":
                if i >= len(hint_list): hint = "Entities in this column are instances of the following wikidata entities: "
                else: 
                    hint = serialize_dict(hint_list[i])
            elif info == "entity":
                hint = f"The entity labels in the Wikidata knowledge graph corresponding to the cells in this column are provided as a list: {hint_strings[i]}."
            elif info == "des":
                hint = f"The entity labels with descriptions in the Wikidata knowledge graph corresponding to the cells in this column are provided as a list: {hint_strings[i]}."
            hints.append(hint)
            
            messages.append(
            {
                "role": "user",
                "content": f"""Your task is to choose only one type from the list to annotate Column {i+1}. 
            Solve this task by following these steps:
            1. **Analyze the data in Column {i+1} in the context of the entire table**
            2. **Evaluate the Wikidata hint if not empty: {hint}**:
            - Use the hint to enhance your understanding of the column data.
            3. **Select the best type** from the given label set. 
            - Ensure the type best summarizes every cell in the column.
            - Ensure it aligns with the context of the whole table.
            - Verify your chosen type MUST be in the provided list.
            4. **Output**:
            - First, explain your reasoning (non-JSON).
            - Finally, output **ONLY** the JSON answer in this format: {res_format}."""
            })
            
            chatgpt_msg = response(messages, model)
            prediction = chatgpt_msg.content
            predicted_type = parse_pred(prediction)
            messages.append(dict(chatgpt_msg))

            if predicted_type not in reduced_label_set:
                messages.append({"role": "user",
                    "content": f"""Your prediction {predicted_type} is not in the given label set. Please try again to choose the closest label in the given set to annotate the first column. Choose only one valid type from the given list of types. Check that the type MUST MUST MUST be in the list. Give the answer in valid JSON format. DO NOT include any explanation or additional text."""})
                chatgpt_msg = response(messages, model)
                pred = chatgpt_msg.content
                messages.append(dict(chatgpt_msg))
                predicted_type = parse_json_pred(pred)
                    
            all_preds.append((predicted_type, prediction))

        with open(OUTPUT, "a", newline="", encoding="UTF-8") as output:
            writer = csv.writer(output)
            for i, p in enumerate(all_preds):
                writer.writerow([id, i, p[0], hints[i], p[1]])

if __name__ == "__main__":
    main()