import csv
import ast
import argparse
import sys
import os
from pathlib import Path
data_path = Path(__file__).parent.parent / 'data'
sys.path.append(str(data_path))
from TURL_CTA_label_reduction import reduced_label_set
from utils import f1_score_multilabel

types = reduced_label_set


data_dir = os.path.join(os.getenv('DATA_DIR', '.'), 'data')
labels = []
with open(os.path.join(data_dir, 'CTA_remapped_test_labels.txt'), 'r') as file:
    labels = file.readlines()
labels = [line.strip() for line in labels]

def serialize(answers):
    cnt = 0
    result = {}
    for type in types:
        result[type] = 0
    for item in answers:
        if item in result:
            result[item] = 1
        else: 
            cnt += 1
    return result, cnt


def eval(data_dir, labels):
    wrong_cols = {}
    preds = []
    ground_truth = []
    empty = serialize([])[0].values()
    empty = [*empty]
    num_oov = 0
    json_error = 0
    num_acc = 0
    cur_id = 0
    all_preds = {}
    with open(data_dir, mode ='r') as file:
        csvFile = csv.reader(file)
        for id, lines in enumerate(csvFile):
            cur_id += 1
            label = labels[id]
            table_id = lines[0]
            col_id = lines[1]
            type_value = [lines[2]]
            all_preds[(table_id, col_id)] = type_value
            pred, cnt = serialize(type_value)
            pred = pred.values()
            num_oov += cnt
            pred = [*pred]
            preds.append(pred)
            if len(type_value) > 0 and type_value[0] in label:
                num_acc += 1
                gt = pred
            else:
                label = ast.literal_eval(label)
                label = label[:1]
                gt = serialize(label)[0].values()
                gt = [*gt]
                wrong_cols[(table_id, col_id)] = (type_value, label)
            ground_truth.append(gt)
    micro_f1, macro_f1, class_f1, conf_mat, precision, recall = f1_score_multilabel(ground_truth, preds)
    return micro_f1, macro_f1, class_f1, conf_mat, precision, recall, num_oov, json_error, wrong_cols, all_preds
            
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Evaluate Column Type Annotation Results')

    parser.add_argument('file_dir', type=str,
                        help='Path to the evaluation CSV file')
    args = parser.parse_args()

    micro_f1, macro_f1, class_f1, conf_mat, precision, recall, num_oov, json_error, wrong_cols,all_preds_4O = eval(args.file_dir, labels)
    print(micro_f1)