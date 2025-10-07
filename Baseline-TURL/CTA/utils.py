import numpy as np
import os
import csv
import json
import time
import subprocess
from sklearn.metrics import multilabel_confusion_matrix
import sys
from SPARQLWrapper import SPARQLWrapper, JSON
import openai
import ast
import re
from collections import defaultdict, Counter
data_dir = os.getenv('DATA_DIR', './data')

from refined.inference.processor import Refined

refined = Refined.from_pretrained(model_name='wikipedia_model',
                                  entity_set="wikidata")
preprocessor = refined.preprocessor

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


def response(messages, model):
    completion = openai.chat.completions.create(
        model=model,
        messages=messages
    )
    return completion.choices[0].message

def get_instance_of(qid):
    """
    Retrieve entity types (P31: instance of) for a given Wikidata QID.
    
    Queries Wikidata to find what types/classes an entity belongs to, with support
    for qualified statements (P642: of). For example, if an entity is "instance of: 
    position" with qualifier "of: basketball", returns the qualifier object instead.
    
    Args:
        qid: Wikidata identifier
    
    Returns:
        qids: List of type QIDs found for this entity
        entity_set: Set of human-readable labels for the types
    """

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

    try:
        ret = sparql.query()
        results = ret.convert()
        data = results["results"]["bindings"]
    except:
        return qids, entity_set

    for binding in data:
        # Prefer qualifier object (P642) if present, otherwise use the main type
        if 'pq_obj' in binding:
            qid = binding['pq_obj']['value'].split('/')[-1]
            label = binding['pq_objLabel']['value']
        else:
            qid = binding['type']['value'].split('/')[-1]
            label = binding['typeLabel']['value']
        qids.append(qid)
        entity_set.add(label)
    
    return qids, entity_set

def get_entity_info(qid):
    """
    Retrieve the label and description for a Wikidata entity.
    
    Queries the Wikidata SPARQL endpoint to fetch human-readable information
    about an entity given its QID (Wikidata identifier).
    
    Args:
        qid: Wikidata identifier 
    
    Returns:
        dict: A dictionary containing:
            - 'label': The entity's English label/name
            - 'description': The entity's English description (empty string if unavailable)
        None: If the query fails or the entity is not found
    
    """

    query = f"""
    SELECT ?itemLabel ?itemDescription
    WHERE {{
      BIND(wd:{qid} AS ?item)
      SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en". }}
    }}
    """
    
    endpoint_url = "https://query.wikidata.org/sparql"
    sparql = SPARQLWrapper(endpoint_url)
    sparql.setQuery(query)
    sparql.setReturnFormat(JSON)
    
    try:
        results = sparql.query().convert()
        bindings = results["results"]["bindings"][0]
        
        return {
            "label": bindings["itemLabel"]["value"],
            "description": bindings.get("itemDescription", {}).get("value", "")
        }
        
    except Exception as e:
        print(f"Error fetching data for {qid}: {str(e)}")
        return None

def get_relation_single_direction(qid1, qid2):
    sparql = SPARQLWrapper("http://query.wikidata.org/sparql")
    sparql.setQuery(f"""SELECT ?property ?propertyLabel WHERE {{
        VALUES (?entity1 ?entity2) {{(wd:{qid1} wd:{qid2})}}
        
        {{
            ?entity1 ?property ?entity2.
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
        label = binding['propertyLabel']['value']
        rel_set.add(label)

    return rel_set
    
def curl_request(url):
    
    # Define the command to execute using curl
    command = ['curl', '-s', '-o', '-', url]
    result = subprocess.run(command, capture_output=True, text=True)
    return result.stdout

def wikidata_lookup(em):
    em = em.replace(" ", "_")
    url = f'https://www.wikidata.org/w/api.php?action=wbsearchentities&search={em}&format=json&language=en&uselang=en&type=item&limit=1'
    response = curl_request(url)
    qid = None
    entity = None
    if response:  # Check if the response is not empty
        try:
            data = json.loads(response)
            content = data.get('search', [])
            if content:
                item = content[0]
                qid = item.get('id', {})
                entity = item.get('label', {})

        except json.JSONDecodeError as e:
            return qid, entity
    return qid, entity

def get_qid_wk(column):
    col_qids = []
    for (pid, em) in column:
      qid, entity_label = wikidata_lookup(em)
      if qid is None: qid = ''
      print(qid)
      col_qids.append(qid)
    return col_qids

def get_types(EL_res):
    """
    Extract and aggregate entity types for each column in a table.
    
    This function processes entity linking results at the table level, where each column's
    entities are analyzed to determine their types. For cells with multiple possible types,
    the most frequent type across the column is selected.
    
    Args:
        EL_res: A 2D array where each element represents a column's entity linking results.
                Each column contains a list of QIDs (Wikidata identifiers) for its cells.
    
    Returns:
        types: A list of dictionaries (one per column), where each dictionary maps
               type QIDs to lists of cell indices that belong to that type.
        col_type_size: A dictionary mapping column indices to the number of distinct
                       types found in that column.
    """
    col_type_size = {}
    types = []  # List of type counters for each column

    for i in range(len(EL_res)):  # Iterate through each column
        counter = Counter()
        col_qids = EL_res[i]
        cell_to_types_set = {}

        # First pass: collect all types for each cell and count type frequencies
        for j, qid in enumerate(col_qids):  # Iterate through each cell
            type_qids, entity_set = get_instance_of(qid)
            cell_to_types_set[j] = entity_set
            if len(entity_set) < 1: 
                continue
            counter.update(entity_set)

        # Second pass: assign the most appropriate type to each cell
        final_counter = {}
        for cell, types_set in cell_to_types_set.items():
            types_set = list(types_set)
            if len(types_set) < 1: 
                continue
            elif len(types_set) == 1:
                # Single type: assign directly
                t = types_set[0]
                if t not in final_counter: 
                    final_counter[t] = []
                final_counter[t].append(cell)
            else:
                # Multiple types: select the most frequent type in the column
                max_count = 0
                max_type = None
                for t in types_set:
                    count = counter[t]
                    if count > max_count:
                        max_count = count
                        max_type = t
                if max_type not in final_counter: 
                    final_counter[max_type] = []
                final_counter[max_type].append(cell)

        types.append(final_counter)
        size = len(final_counter)
        col_type_size[i] = size
    
    return types, col_type_size

def get_entity_label(EL_res):
    """
    Extract human-readable entity labels for all cells in a table.
    
    This is a table-wise function that converts entity linking results (QIDs)
    into their corresponding Wikidata labels. For each column in the table,
    it retrieves the entity label for every cell that has been linked to a
    Wikidata entity.
    
    Args:
        EL_res: A 2D array where each element represents a column's entity linking
                results. Each column contains a list of QIDs corresponding to the cells in that column.
    
    Returns:
        table_res: A 2D array with the same structure as EL_res, but with QIDs
                   replaced by their human-readable labels.
    
    """
    table_res = []
    for i in range(len(EL_res)):  # For each column
        entity_labels = []
        col_qids = EL_res[i]

        # Loop through all cells in the column to get entity labels
        for j, qid in enumerate(col_qids):  # For each cell
            res_dict = get_entity_info(qid)
            entity_label = res_dict.get("label") if res_dict else None
            entity_labels.append(entity_label)

        table_res.append(entity_labels)
    
    return table_res

def get_entity_label_des(EL_res):
    """
    Extract entity labels and descriptions for all cells in a table.
    
    This is a table-wise function that converts entity linking results (QIDs)
    into their corresponding Wikidata labels and descriptions. For each column
    in the table, it retrieves both the entity label and description for every
    cell that has been linked to a Wikidata entity, combining them in the format
    "label: description".
    
    Args:
        EL_res: A 2D array where each element represents a column's entity linking
                results. Each column contains a list of QIDs (Wikidata identifiers)
                corresponding to the cells in that column.
    
    Returns:
        table_res: A 2D array with the same structure as EL_res, but with QIDs
                   replaced by formatted strings combining labels and descriptions.
    """
    table_res = []
    for i in range(len(EL_res)):  # For each column
        entity_info = []
        col_qids = EL_res[i]

        # Loop through all cells in the column to get entity labels and descriptions
        for j, qid in enumerate(col_qids):  # For each cell
            res_dict = get_entity_info(qid)
            if res_dict is None:
                entity_info.append("")
            else: 
                entity_label = res_dict.get("label")
                entity_des = res_dict.get("description")
                entity_info.append(entity_label + ": " + entity_des)

        table_res.append(entity_info)
    
    return table_res

def RE_get_col_relations(column1, column2):
    rel_set = Counter()
    length = min(len(column1), len(column2))
    for i in range(length):
        qid1 = column1[i]
        qid2 = column2[i]
        if qid1 == "" or qid2 == "": continue
        cur_set = get_relation_single_direction(qid1, qid2)
        rel_set.update(cur_set)
    return rel_set

def serialize_counter(counter):
    serialized_str = ""
    for item, count in counter.items():
        serialized_str += f"{item} ({count} cells), "

    return serialized_str.rstrip(", ")

def get_entity_relation(EL_res):
    """
    Extract relationships between the first column and all other columns in a table.
    
    This is a table-wise function that identifies Wikidata relationships (properties)
    between entities in the first column and entities in each subsequent column.
    
    Args:
        EL_res: A 2D array where each element represents a column's entity linking
                results. Each column contains a list of QIDs (Wikidata identifiers)
                corresponding to the cells in that column.
    
    Returns:
        table_res: A list of strings, one for each column (excluding the first column).
                   Each string describes the relationships found between the first column
                   and that column, formatted as:
                   "- Column 1 & Column {i+1}: relationship1 (count), relationship2 (count), ..."
    """
    table_res = []
    first_col = EL_res[0]
    for i in range(1, len(EL_res)):  # For each column (starting from the second)
        col_qids = EL_res[i]
        rel_set = RE_get_col_relations(first_col, col_qids)
        rel_str = f"- Column 1 & Column {i+1}: " + serialize_counter(rel_set)
        table_res.append(rel_str)
    
    return table_res


def serialize_dict(data):
    serialized_str = "Entities in this column are instances of the following wikidata entities: "
    for key, value in data.items():
        serialized_str += f"{key} ({value} cells), "

    serialized_str = serialized_str.rstrip(", ")
    return serialized_str

def compute_cell_boundaries(flattened_table: str):
    """
    Compute token-level boundaries for each cell in a flattened table string.
    
    This function tokenizes a table where cells are separated by '||' delimiters
    and returns the start and end token indices for each cell. These boundaries
    are used for entity linking to identify which entities belong to which cells.
    
    Args:
        flattened_table: A string representation of a table with cells separated by '||' delimiters
    
    Returns:
        boundaries: A list of tuples (start_idx, end_idx) where each tuple represents
                   the token index range for one cell. 
    """
    tokens = preprocessor.tokenize(flattened_table)
    boundaries = []
    cell_start = None  # Start index of the current cell

    for i in range(len(tokens)):
        token = tokens[i]
        if token.text == "||" or token.text == "Ġ||":  # Cell delimiter
            if cell_start is not None:
                # Close the current cell
                boundaries.append((cell_start, i))
                cell_start = None
        else:
            if cell_start is None:  # Start a new cell
                cell_start = i

    return boundaries

def process_EL_res(col_spans):
    EL_col_res = []
    for span in col_spans:
        if len(span) < 1:
            EL_col_res.append("")
            continue
        item = span[0] # for each cell, only take the first entity linked
        item_string = item.__repr__()
        res = item_string.strip("[]").split(", ")
        qid_match = re.search(r'wikidata_entity_id=(\w+)', res[1])
        if qid_match:
            qid = qid_match.group(1)
            EL_col_res.append(qid)
        else:
            EL_col_res.append("")
            continue
    return EL_col_res


def safe_parse_dict(d):
    d = d.strip()
    if not d:
        return {}
    try:
        val = ast.literal_eval(d)
        return val if isinstance(val, dict) else {}
    except (SyntaxError, ValueError):
        return {}
    

def parse_pred(prediction): # parse the prediction returned by the row-linker-agent
    try:
        final_answer_start = prediction.find("FINAL ANSWER:") + len("FINAL ANSWER:")
        final_answer_str = prediction[final_answer_start:].strip()

        # Convert the string to a valid JSON (replace single quotes with double quotes)
        final_answer_json = json.loads(final_answer_str.replace("'", '"'))

        # Extract the type
        predicted_type = final_answer_json['type'][0]  # Get the first element in the list
    except (json.JSONDecodeError, KeyError, IndexError, TypeError):
        predicted_type = "None"
        
    return predicted_type

def parse_json_pred(prediction):  # parse a JSON response
    try:
        # Extract JSON substring if there is a prefix like "FINAL ANSWER: "
        match = re.search(r'\{.*\}', prediction)
        if not match:
            return "NA"
        
        json_str = match.group(0)

        # Convert the string to a valid JSON (replace single quotes with double quotes)
        final_answer_json = json.loads(json_str.replace("'", '"'))

        # Extract the type
        predicted_type = final_answer_json['type'][0]  # Get the first element in the list
    except (json.JSONDecodeError, KeyError, IndexError, TypeError):
        predicted_type = "None"

    return predicted_type