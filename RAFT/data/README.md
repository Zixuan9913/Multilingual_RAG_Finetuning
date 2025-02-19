# Data Processing for Synthetic Data Generation

This directory contains the dataset processing pipeline for CLAPNQ-based synthetic data generation.

## Directory Structure

### `/raw_data`
Contains the original CLAPNQ dataset (a subset of Natural Questions), sourced from [CLAPNQ repository](https://github.com/primeqa/clapnq/tree/main/original_documents/dev). These are unprocessed Wikipedia articles that include:
- Full article content
- External links
- References
- Other metadata

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
