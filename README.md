# Multilingual_RAG_Finetuning

The aim of the project is to see if RAG+Fine-tuning can give a better performance than RAG only for QA task in general domain and specific domains (Medical and Technical), especially when the original LLM is trained on English while the input question is in another language (eg. Spanish and Vietnamese) so that the final answer should also be in that language. 

Therefore we have the following steps for the project: 
## Project structure
1. Training data generation with RAFT 
2. LLM fine-tuning 
3. Evaluation 

### Training data generation 
We take articles from QA dataset and break them into smaller chunks, then we ask a LLM to generate synthetic QA pairs, the pairs can be in English or in other languages (monolingual data or multilingual data)

### LLM Fine-tuning 
Then we use the data generated from the first step to fine-tune the LLAMA3-8b model, which is an English model but it has some multilingual potentials 

There are several Fine-tuning strategies:
1. English monolingual training (with English Only data)
2. Multilingual training (with context in English but QA pair in Spanish or Vietnamese)
3. Mixed training (mixture of 1 and 2)

### Model Evaluation
Then we use the Fine-tuned LLM as the generator in the RAG pipeline to generate responses. Then we compare the generated answer to gold answer in the test dataset 

Metircs:
1. Accuracy: This is for binary QA task 
2. Semantic metrics: Sentence Bert and BLEU

## Conclusions:
- We found that especially for Vietnamese, the FTed LLM has a better understanding of the task requirements and gives higher-quality results than original model 
- The task understanding ability is boosted after fine-tuning in terms of format requirements 
- Semantic wise, the improvment for English and Spanish is not significant because the base model already perform well in these two languages 
- Training the model on English only data makes the model loses Vietnamese ability (catastrophic forgetting)
