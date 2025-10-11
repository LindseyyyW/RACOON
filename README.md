# RACOON<sup>+</sup>

This repository hosts the source code of the RACOON<sup>+</sup> prototype system, which is introduced in the paper: *"RACOON<sup>+</sup>: A System for LLM-based Table Understanding with a Knowledge Graph"*.

## Repository Structure

```
RACOON/
├── README.md                           
├── requirements.txt                   # Python dependencies
└── src/                               # Source code directory
    ├── CTA/                           # Column Type Annotation module
    ├── RE/                            # Relation Extraction module
    ├── data/                          # Datasets used and label mappings
    ├── KG_Linker.py/                  # The KG-Linker component in RACOON+
    ├── pruning.py/                    # The pruning module in RACOON+
    └── utils.py                       # Shared utility functions for both the CTA and RE workloads
    
```

## Installation

You can install the required packages with `pip install -r requirements.txt`. The experiments were run on Python 3.12.

## Datasets

The datasets used in the paper experiments are available on Google Drive:

[Download Datasets](https://drive.google.com/drive/folders/1XX5B2Z0QUOR1zp0Ja5N_4k5RN4HFwQB_?usp=drive_link)

Please download the datasets and place them in the `src/data/` directory before running the experiments.

## Usage

The RACOON+ system supports column type annotation and relation extraction using different context types for entity linking and different types of retrieved information to augment LLM prompts. Run the CTA or RE module with the following commands:

```bash
# Column Type Annotation
cd src/CTA
python RACOON_CTA.py [OPTIONS]

# Relation Extraction
cd src/RE
python RACOON_RE.py [OPTIONS]
```

#### Command Line Arguments

- `--model`: OpenAI model to use (default: `gpt-4o-mini`)
  - Options: Any valid OpenAI model name
  - Example: `--model gpt-4o`

- `--context`: Context type for knowledge graph linking (default: `col`)
  - Options: `wikiAPI`, `cell`, `table`, `col`, `hybrid`
  - `wikiAPI`: Uses Wikipedia API for entity linking
  - `cell`: Cell context for entity linking
  - `table`: Table context for entity linking
  - `col`: Column context for entity linking
  - `hybrid`: Hybrid context for entity linking

- `--info`: Information type to extract from knowledge graph (default: `type`)
  - Options: `entity`, `des`, `type`, `relation`
  - `entity`: Extract entity labels from Wikidata
  - `des`: Extract entity labels and descriptions from Wikidata
  - `type`: Extract entity types from Wikidata
  - `relation`: Extract entity relations between cell pairs from Wikidata

#### Example Usage

```bash

# Use GPT-4o with hybrid context and type information
python RACOON_CTA.py --model gpt-4o --context hybrid --info type

# Use cell-level context with entity information
python RACOON_CTA.py --context cell --info entity
```


#### Output

Both programs generate CSV files with the format: `{model}/RACOON_{context}_{info}.csv`

**CTA Output** - Each row contains:
- Table ID
- Column index
- Predicted type
- Wikidata hint used

**RE Output** - Each row contains:
- Table ID
- Column pair index
- Predicted relation
- Wikidata hint used

#### Environment Variables

Make sure to set the following environment variables:
- `OPENAI_API_KEY`: Your OpenAI API key
- `DATA_DIR`: Path to the data directory (defaults to `../data` relative to the script)

## Evaluation

To evaluate the results of CTA or RE experiments, use the evaluation scripts:

**Column Type Annotation (CTA):**
```bash
cd src/CTA
python CTA_eval.py <path_to_results_csv>
```

**Relation Extraction (RE):**
```bash
cd src/RE
python RE_eval.py <path_to_results_csv>
```

**Output:**
The evaluation scripts compute and print the micro-averaged F1 score, which is the main metric used in the paper.
