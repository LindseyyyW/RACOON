# RACOON<sup>+</sup>

This repository hosts the source code of the RACOON<sup>+</sup> prototype system, which is introduced in the paper: *"RACOON<sup>+</sup>: A System for LLM-based Table Understanding with a Knowledge Graph"*.

## Repository Structure

```
RACOON/
├── README.md                           
├── requirements.txt                    # Python dependencies
└── src/                               # Source code directory
    ├── CTA/                           # Column Type Annotation module
    ├── RE/                            # Relation Extraction module
     ├── data/                          # Datasets used and label mappings
     └── utils.py                       # Shared utility functions for both the CTA and RE workloads
```

## Installation

You can install the required packages with `pip install -r requirements.txt`. The experiments were run on Python 3.12.