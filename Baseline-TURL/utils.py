import numpy as np
import os
from openai import OpenAI
import csv
import json
from pathlib import Path
from random import shuffle, seed
from tqdm import tqdm
from sklearn.metrics import multilabel_confusion_matrix
import sys
sys.path.append("/mmfs1/gscratch/balazinska/linxiwei/baseline/freebase-wikidata-convert")
from converter import EntityConverter
entity_converter = EntityConverter("https://query.wikidata.org/sparql")
from SPARQLWrapper import SPARQLWrapper, JSON
import openai
#from openai.error import InvalidRequestError
#from ratelimiter import RateLimiter

import pandas as pd
data_dir = '/mmfs1/gscratch/balazinska/linxiwei/TURL'

pid_to_qid = {}
with open(os.path.join(data_dir,'Baseline-TURL/map_pid_to_qid.json'), 'r') as file:
    pid_to_qid = json.load(file)

pid_to_mid = dict()
pid_to_label = dict()

def parse_file_to_dict(file_path):
    result_dict = {}
    pid_to_label = {}
    with open(file_path, 'r') as file:
        for line in file:
            parts = line.strip().split('\t')
            if len(parts) == 5:
                vocab_id, wikipedia_id, wikipedia_title, freebase_mid, entity_count = parts
                result_dict[int(wikipedia_id)] = freebase_mid
                pid_to_label[int(wikipedia_id)] = wikipedia_title
    return result_dict, pid_to_label

file_path = os.path.join(data_dir, "entity_vocab.txt")
pid_to_mid, pid_to_label = parse_file_to_dict(file_path)

def response(messages, n_tokens):
    completion = openai.chat.completions.create(
        #model="gpt-4o-mini",
        model="gpt-3.5-turbo",
        messages=messages,
        max_tokens=n_tokens,
    )
    return completion.choices[0].message

def parse_example(example):
    label = example[7]
    meta_data = {}
    meta_data["page title"] = example[1]
    meta_data["section_title"] = example[3]
    meta_data["caption"] = example[4]
    headers = example[5]
    num_col = len(example[6])
    colset = []
    max_num_row = -1

    # build up a 2-D array for the table
    for i in range(num_col):
        col = example[6][i]

        num_row = len(col)
        table_col = []
        for j in range(num_row):
            cell = col[j]
            idx = cell[0]
            text = cell[1][1]
            given_link = cell[1][0]
            if idx[0] > max_num_row:
                max_num_row = idx[0]
            table_col.append((given_link, text))
        colset.append(table_col)

    max_num_row += 1
    table = np.empty([max_num_row, num_col], dtype = object) 
    table_with_en = np.empty([max_num_row, num_col], dtype = object) 

    for i in range(num_col):
        col = example[6][i]

        num_row = len(col)
        for j in range(num_row):
            cell = col[j]
            idx = cell[0]
            text = cell[1][1]
            page_id = cell[1][0]
            table[idx[0]][idx[1]] = text
            table_with_en[idx[0]][idx[1]] = (page_id, text)
            
    return table, table_with_en, label, headers, meta_data, num_col, max_num_row, colset

def f1_score_multilabel(true_list, pred_list):
    conf_mat = multilabel_confusion_matrix(np.array(true_list),
                                           np.array(pred_list))
    agg_conf_mat = conf_mat.sum(axis=0)
    # Note: Pos F1
    # [[TN FP], [FN, TP]] if we consider 1 as the positive class
    p = agg_conf_mat[1, 1] / agg_conf_mat[1, :].sum()
    r = agg_conf_mat[1, 1] / agg_conf_mat[:, 1].sum()
    
    micro_f1 = 2 * p * r / (p  + r) if (p + r) > 0 else 0.
    class_p = conf_mat[:, 1, 1] /  conf_mat[:, 1, :].sum(axis=1)
    class_r = conf_mat[:, 1, 1] /  conf_mat[:, :, 1].sum(axis=1)
    class_f1 = np.divide(2 * (class_p * class_r), class_p + class_r,
                         out=np.zeros_like(class_p), where=(class_p + class_r) != 0)
    class_f1 = np.nan_to_num(class_f1)
    macro_f1 = class_f1.mean()
    return (micro_f1, macro_f1, class_f1, conf_mat)

def get_info(column):
    en_labels = []
    for (pid, em) in column:
        if pid in pid_to_label:
            entity = pid_to_label[pid]
            en_labels.append(entity)
    return en_labels

def get_gt_qid(column):
   qid_set = set()
   for (pid, em) in column:
      if pid not in pid_to_mid: 
        continue
      
      mid = pid_to_mid[pid]
      mid = "/" + mid.replace(".", "/")
      qid = entity_converter.get_wikidata_id(mid)
      if qid is not None:
         qid_set.add(qid)
   return qid_set


def get_instance_of(qid):
   sparql = SPARQLWrapper("http://query.wikidata.org/sparql")
   sparql.setQuery(f"""SELECT ?type ?typeLabel ?pq_obj ?pq_objLabel 
      WHERE {{
      wd:{qid} p:P31 ?statementNode.
      ?statementNode ps:P31 ?type.
      OPTIONAL {{
         ?statementNode pq:P642 ?pq_obj.
      }}
      SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en". }}
      }}
      """)
   sparql.setReturnFormat(JSON)
   qids = []
   entity_set = set()

   try :
      ret = sparql.query()
      results = ret.convert()
      data = results["results"]["bindings"]
   except :
      return qids, entity_set


   for binding in data:
      if 'pq_obj' in binding:
        qid = binding['pq_obj']['value'].split('/')[-1]
        label = binding['pq_objLabel']['value']
      else:
         qid = binding['type']['value'].split('/')[-1]
         label = binding['typeLabel']['value']
      qids.append(qid)
      entity_set.add(label)
   return qids, entity_set

def get_relation(qid1, qid2):
    sparql = SPARQLWrapper("http://query.wikidata.org/sparql")
    sparql.setQuery(f"""SELECT ?property ?propertyLabel WHERE {{
        VALUES (?entity1 ?entity2) {{(wd:{qid1} wd:{qid2})}}
        
        {{
            ?entity1 ?property ?entity2.
        }}
        UNION
        {{
            ?entity2 ?property ?entity1.
        }}
        
        ?propertyEntity wikibase:directClaim ?property.
        ?propertyEntity rdfs:label ?propertyLabel.
        FILTER(LANG(?propertyLabel) = "en")
        }}
        """)
    sparql.setReturnFormat(JSON)
    rel_set = set()
    try :
        ret = sparql.query()
        results = ret.convert()
        data = results["results"]["bindings"]
    except :
        return rel_set

    for binding in data:
        pid = binding['property']['value'].split('/')[-1]
        label = binding['propertyLabel']['value']
        rel_set.add(label)
    return rel_set

def get_col_relations(column1, column2):
    rel_set = set()
    length = min(len(column1), len(column2))
    for i in range(length):
        pid1, em1 = column1[i]
        pid2, em2 = column2[i]
        # if pid1 not in pid_to_qid or pid2 not in pid_to_qid: 
        #     print("NO QID")
        #     continue
        if pid1 not in pid_to_mid: continue
        mid1 = pid_to_mid[pid1]
        mid1 = "/" + mid1.replace(".", "/")
        qid1 = entity_converter.get_wikidata_id(mid1)
        if qid1 is None: continue
        if pid2 not in pid_to_mid: continue
        mid2 = pid_to_mid[pid2]
        mid2 = "/" + mid2.replace(".", "/")
        qid2 = entity_converter.get_wikidata_id(mid2)
        if qid2 is None: continue
        cur_set = get_relation(qid1, qid2)
        rel_set = rel_set.union(cur_set)
    return rel_set

def get_triplets(column):
   info_set = {}
   for (pid, em) in column:
      if pid not in pid_to_mid or pid not in pid_to_label: continue
      mid = pid_to_mid[pid]
      mid = "/" + mid.replace(".", "/")
      qid = entity_converter.get_wikidata_id(mid)
      if qid is None: continue
      entity_label = pid_to_label[pid]
      _, entity_set = get_instance_of(qid)
      for item in entity_set:
         if item not in info_set:
               info_set[item] = []
         info_set[item].append(entity_label)
      
   return info_set



def serialize_dict(data):
    serialized_str = "Entities in this column are instances of the following wikidata entities: "
    cnt = 0
    for key, value in data.items():
        cnt += 1
        count = len(value)
        serialized_str += f"{key} ({count} cells), "
        if cnt == 5: break

    serialized_str = serialized_str.rstrip(", ")
    return serialized_str

def load_type_vocab(data_dir):
    type_vocab = {}
    with open(os.path.join(data_dir, "type_vocab.txt"), "r") as f:
        for line in f:
            index, t = line.strip().split('\t')
            type_vocab[t] = int(index)
    return type_vocab
    
type_vocab = load_type_vocab(data_dir)

