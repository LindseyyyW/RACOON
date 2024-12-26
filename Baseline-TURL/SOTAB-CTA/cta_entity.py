import pandas as pd
import os
import tqdm
from collections import defaultdict
import sys
sys.path.append("/mmfs1/gscratch/balazinska/linxiwei/TURL/Baseline-TURL")
from utils import *
openai.api_key = ""
df = pd.read_excel('test_label_name.xlsx', header=None)
table_names = df[0].tolist()
import tiktoken
enc = tiktoken.encoding_for_model("gpt-3.5-turbo-0125")

# prepare the type vocab
type_vocab = [
    "DateTime", "Duration", "Mass", "Date", "currency", "Person/name", "Number",
    "telephone", "Place/name", "price", "Organization", "addressLocality", "IdentifierAT",
    "Country", "addressRegion", "URL", "MusicAlbum/name", "streetAddress", "MusicArtistAT",
    "ItemAvailability", "CoordinateAT", "postalCode", "MusicRecording/name", "priceRange",
    "Event/name", "weight", "Book/name", "QuantitativeValue", "email", "Product/name",
    "Movie/name", "Recipe/name", "category", "Distance", "unitCode", "LocalBusiness/name",
    "Hotel/name", "ItemList", "faxNumber", "Brand", "PostalAddress", "Time", "Energy",
    "Review", "EducationalOccupationalCredential", "Restaurant/name", "OfferItemCondition",
    "CategoryCode", "EventStatusType", "BookFormatType", "SportsEvent/name",
    "CreativeWork/name", "Language", "openingHours", "ProductModel",
    "EventAttendanceModeEnumeration", "SportsTeam", "unitText",
    "OccupationalExperienceRequirements", "workHours", "DayOfWeek", "Photograph",
    "paymentAccepted", "TVEpisode/name", "MonetaryAmount", "Boolean", "DeliveryMethod",
    "Rating", "CreativeWorkSeries", "GenderType", "RestrictedDiet", "Product/description",
    "Recipe/description", "Event/description", "Book/description", "JobPosting/description",
    "Hotel/description", "Movie/description", "LocationFeatureSpecification",
    "Museum/name", "JobPosting/name", "CreativeWork"
]

# prepare ground truth labels
df = pd.read_csv('sotab_v2_cta_test_set.csv')  
table_labels = defaultdict(lambda: [])

for _, row in df.iterrows():
    table_name = row['table_name']
    col_index = row['column_index']
    label = row['label']
    
    if len(table_labels[table_name]) <= col_index:
        table_labels[table_name].extend([None] * (col_index + 1 - len(table_labels[table_name])))
    table_labels[table_name][col_index] = label

table_labels = dict(table_labels)

res_format = "{'type': []}"
OUTPUT = "GPT3.5_output_sotabv2_re_entity_||.csv"
gt_labels = []

from refined.data_types.base_types import Span, Entity
from refined.inference.processor import Refined
import re

refined = Refined.from_pretrained(model_name='wikipedia_model',
                                  entity_set="wikidata",
                                  use_precomputed_descriptions=False)
def get_info(column):
    en_labels = []
    text = ""
    for em in column:
        em = str(em)
        encoded_str = enc.encode(em)
        if len(encoded_str) >= 50: continue
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

def get_info_triplet(column):
    info_set = {}
    for em in column:
        em = str(em)
        encoded_str = enc.encode(em)
        if len(encoded_str) >= 50: continue
        spans = refined.process_text(em)
        if (len(spans) < 1): continue
        item = spans[0]
        item_string = item.__repr__()
        res = item_string.strip("[]").split(", ")
        qid_match = re.search(r'wikidata_entity_id=(\w+)', res[1])
        if qid_match:
            qid = qid_match.group(1)
        else:
            continue
        _, entity_set = get_instance_of(qid)

        label_match = re.search(r'wikipedia_entity_title=([^,]+)', res[2])
        if label_match:
            entity_label = label_match.group(1)
        else:
            continue
        
        cnt = 0
        for item in entity_set:
            cnt += 1
            if item not in info_set:
                info_set[item] = []
            info_set[item].append(entity_label)
    sorted_info_set = dict(sorted(info_set.items(), key=lambda x: len(x[1]), reverse=True))
    return sorted_info_set


col_pairs = set()
with open("col_pairs.txt", "r") as f:
    for line in f:
        # Convert the string back to a tuple
        pair = eval(line.strip())
        col_pairs.add(pair)


cell_length_total = 0
num_cells = 0
long_cells = 0
for name in tqdm(table_names):
    valid_cols = []
    df = pd.read_json(os.path.join('Test',str(name)), compression='gzip', lines=True)
    #CSV_like = df.head(5).to_csv(header=False, index=False)
    CSV_like = '\n'.join(['||'.join(map(str, row)) for row in df.head(5).values])

    df.columns = [f"Column_{i}" for i in range(df.shape[1])]
    
    for id,item in enumerate(table_labels[name]):
        if item is not None:
            valid_cols.append(id)
            gt_labels.append(item)

    all_preds = []
    hints = []
    flag = 0
    
    for i, col in enumerate(df.columns):
        if i not in valid_cols: continue
        hint = get_info(df[f"Column_{i}"].head(10))
        hints.append(hint)
        if flag == 0: 
            if len(hint) > 0:
                messages=[
                {
                    "role": "system",
                    "content": f"Be a helpful, accurate assistant for data discovery and exploration desiged to output valid JSON in the format {res_format}",
                },
                {
                    "role": "user",
                    "content": f"""Consider this table, where each cell in each row is separated by "||":
                                ```
                                {CSV_like}
                                ```
                    There are a list of 82 valid types for each column: {type_vocab}. Your task is to choose only one type from the list to annotate the {i+1} column. Solve this task by following these steps: 
                    1. Look at the cells in the {i+1} column of the above table (columns are indexed starting from 1). Some column cells can be linked to entities in the Wikidata knowledge graph. The entity labels correspond to the cells in the {i+1} column are presented as a list delimited by triple quotes.
                    ```{hint}```
                    2. Understand the entities in the first column
                    3. Choose only one valid type from the given list of types. Check that the type MUST be in the list. Give the answer in valid JSON format.
                    """,
                }
                ]
            else:
                messages=[
                {
                    "role": "system",
                    "content": f"Be a helpful, accurate assistant for data discovery and exploration desiged to output valid JSON in the format {res_format}",
                },
                {
                    "role": "user",
                    "content": f"""Consider this table, where each cell in each row is separated by "||":
                                ```
                                {CSV_like}
                                ```
                    There are a list of 82 valid types for each column: {type_vocab}. Your task is to choose only one type from the list to annotate the {i+1} column. Solve this task by following these steps: 
                    1. Look at the cells in the {i+1} column of the above table (columns are indexed starting from 1). 
                    2. Choose only one valid type from the given list of types. Check that the type MUST be in the list. Give the answer in valid JSON format.
                    """,
                }
                ]
            flag = 1
        else:
            if len(hint) > 0:
                messages.append(
                {
                    "role": "user",
                    "content": f"""Your task is to choose only one type from the list to annotate the {i+1} column. Solve this task by following these steps: 
                    1. Look at the cells in the {i+1} column of the above table (columns are indexed starting from 1). Some column cells can be linked to entities in the Wikidata knowledge graph. The entity labels correspond to the cells in the {i+1} column are presented as a list delimited by triple quotes.
                    ```{hint}```
                    2. Understand the entities in the first column
                    3. Choose only one valid type from the given list of types. Check that the type MUST be in the list. Give the answer in valid JSON format.
                    """,
                })
            else:
                messages.append(
                {
                    "role": "user",
                    "content": f"""Your task is to choose only one type from the list to annotate the {i+1} column. Solve this task by following these steps: 
                    1. Look at the cells in the {i+1} column of the above table (columns are indexed starting from 1). 
                    2. Choose only one valid type from the given list of types. Check that the type MUST be in the list. Give the answer in valid JSON format.
                    """,
                })

        chatgpt_msg = response(messages, 40)
        prediction = chatgpt_msg.content
        
        all_preds.append(prediction)
        messages.append(dict(chatgpt_msg))
        with open(OUTPUT, "a", newline="", encoding="UTF-8") as output:
            writer = csv.writer(output)
            writer.writerow([name, i, prediction,hint])





  




### entity labels
# for i, col in enumerate(df.columns):
#     if i not in valid_cols: continue
#     hint = get_info(df[f"Column_{i}"].head(10))
#     hints.append(hint)
#     if flag == 0: 
#         if len(hint) > 0:
#             messages=[
#             {
#                 "role": "system",
#                 "content": f"Be a helpful, accurate assistant for data discovery and exploration desiged to output valid JSON in the format {res_format}",
#             },
#             {
#                 "role": "user",
#                 "content": f"""Consider this table given in Comma-separated Values format:
#                             ```
#                             {CSV_like}
#                             ```
#                 There are a list of 82 valid types for each column: {type_vocab}. Your task is to choose only one type from the list to annotate the {i+1} column. Solve this task by following these steps: 
#                 1. Look at the cells in the {i+1} column of the above table (columns are indexed starting from 1). Some column cells can be linked to entities in the Wikidata knowledge graph. The entity labels correspond to the cells in the {i+1} column are presented as a list delimited by triple quotes.
#                 ```{hint}```
#                 2. Understand the entities in the first column
#                 3. Choose only one valid type from the given list of types. Check that the type MUST be in the list. Give the answer in valid JSON format.
#                 """,
#             }
#             ]
#         else:
#             messages=[
#             {
#                 "role": "system",
#                 "content": f"Be a helpful, accurate assistant for data discovery and exploration desiged to output valid JSON in the format {res_format}",
#             },
#             {
#                 "role": "user",
#                 "content": f"""Consider this table given in Comma-separated Values format:
#                             ```
#                             {CSV_like}
#                             ```
#                 There are a list of 82 valid types for each column: {type_vocab}. Your task is to choose only one type from the list to annotate the {i+1} column. Solve this task by following these steps: 
#                 1. Look at the cells in the {i+1} column of the above table (columns are indexed starting from 1). 
#                 2. Choose only one valid type from the given list of types. Check that the type MUST be in the list. Give the answer in valid JSON format.
#                 """,
#             }
#             ]
#         flag = 1
#     else:
#         if len(hint) > 0:
#             messages.append(
#             {
#                 "role": "user",
#                 "content": f"""Your task is to choose only one type from the list to annotate the {i+1} column. Solve this task by following these steps: 
#                 1. Look at the cells in the {i+1} column of the above table (columns are indexed starting from 1). Some column cells can be linked to entities in the Wikidata knowledge graph. The entity labels correspond to the cells in the {i+1} column are presented as a list delimited by triple quotes.
#                 ```{hint}```
#                 2. Understand the entities in the first column
#                 3. Choose only one valid type from the given list of types. Check that the type MUST be in the list. Give the answer in valid JSON format.
#                 """,
#             })
#         else:
#             messages.append(
#             {
#                 "role": "user",
#                 "content": f"""Your task is to choose only one type from the list to annotate the {i+1} column. Solve this task by following these steps: 
#                 1. Look at the cells in the {i+1} column of the above table (columns are indexed starting from 1). 
#                 2. Choose only one valid type from the given list of types. Check that the type MUST be in the list. Give the answer in valid JSON format.
#                 """,
#             })



### entity triplets
# for i, col in enumerate(df.columns):
#     if i not in valid_cols: continue
#     info_set = get_info_triplet(df[f"Column_{i}"].head(10))
#     hint = serialize_dict(info_set)
#     hints.append(hint)
#     if flag == 0: 
#         if len(hint) > 0:
#             messages=[
#             {
#                 "role": "system",
#                 "content": f"Be a helpful, accurate assistant for data discovery and exploration desiged to output valid JSON in the format {res_format}",
#             },
#             {
#                 "role": "user",
#                 "content": f"""Consider this table given in Comma-separated Values format:
#                             ```
#                             {CSV_like}
#                             ```
#                 There are a list of 82 valid types for each column: {type_vocab}. Your task is to choose only one type from the list to annotate the {i+1} column. Solve this task by following these steps: 
#                 1. Look at the cells in the {i+1} column of the above table (columns are indexed starting from 1).
#                 2. Consider this information carefully: {hint}
#                 3. Choose only one valid type from the given list of types. Check that the type MUST be in the list. Give the answer in valid JSON format.
#                 """,
#             }
#             ]
#         else:
#             messages=[
#             {
#                 "role": "system",
#                 "content": f"Be a helpful, accurate assistant for data discovery and exploration desiged to output valid JSON in the format {res_format}",
#             },
#             {
#                 "role": "user",
#                 "content": f"""Consider this table given in Comma-separated Values format:
#                             ```
#                             {CSV_like}
#                             ```
#                 There are a list of 82 valid types for each column: {type_vocab}. Your task is to choose only one type from the list to annotate the {i+1} column. Solve this task by following these steps: 
#                 1. Look at the cells in the {i+1} column of the above table (columns are indexed starting from 1). 
#                 2. Choose only one valid type from the given list of types. Check that the type MUST be in the list. Give the answer in valid JSON format.
#                 """,
#             }
#             ]
#         flag = 1
#     else:
#         if len(hint) > 0:
#             messages.append(
#             {
#                 "role": "user",
#                 "content": f"""Your task is to choose only one type from the list to annotate the {i+1} column. Solve this task by following these steps: 
#                 1. Look at the cells in the {i+1} column of the above table (columns are indexed starting from 1).
#                 2. Consider this information carefully: {hint}
#                 3. Choose only one valid type from the given list of types. Check that the type MUST be in the list. Give the answer in valid JSON format.
#                 """,
#             })
#         else:
#             messages.append(
#             {
#                 "role": "user",
#                 "content": f"""Your task is to choose only one type from the list to annotate the {i+1} column. Solve this task by following these steps: 
#                 1. Look at the cells in the {i+1} column of the above table (columns are indexed starting from 1). 
#                 2. Choose only one valid type from the given list of types. Check that the type MUST be in the list. Give the answer in valid JSON format.
#                 """,
#             })
