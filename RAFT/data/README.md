# Data Processing for Synthetic Data Generation

This directory contains the dataset processing pipeline for CLAPNQ-based synthetic data generation.
The CLAPNQ dataset used here is an example, for any kind of long or medium textual corpus, we can use 
RAFT method to 
- break down text into chunks 
- generate synthetic quetsion-answer pair based on the chunks

## Make sure
Clean the texts, otherwise LLMs may generate useless data over meaningless texts!

## Directory Structure

### `/raw_data`
Contains the original CLAPNQ dataset (a subset of Natural Questions), sourced from [CLAPNQ repository](https://github.com/primeqa/clapnq/tree/main/original_documents/dev). These are unprocessed Wikipedia articles that include:
- Full article content
- External links
- References
- Other metadata
Link to raw data used here: https://drive.google.com/file/d/192Voqfs72rpG5iyT1me2ZAcT3BPLzwtS/view?usp=drive_link

### `/processed_data`
Contains cleaned Wikipedia articles that have been preprocessed to remove noise and improve quality for synthetic data generation. These articles:
- Contain only meaningful paragraphs
- Have noise (links, references, etc.) removed
- Are ready for chunking and further processing

### `CLAPQN_document_processing.ipynb`
Jupyter notebook containing the preprocessing pipeline that:
- Explores the raw CLAPNQ dataset
- Implements cleaning functions
- Processes the articles
- Saves the cleaned data to the processed_data directory
