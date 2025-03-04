## Evaluation 
In this folder there are two files for the evaluation:
- Eva_en.py 
- Similarity_Calculation.ipynb 

The first file **Eva-en.py** takes the fine-tuned LLM to generate responses to the input question.
The steps are:
1. Load a fine-tuned language model with LoRA or use the base Llama-3-8b model.
2. Process medical dataset by building a vector index from question-context pairs.
3. For each question, retrieve the most relevant contexts using vector similarity search.
4. Generate structured answers with Chain of Thought reasoning and a final "yes", "no", or "maybe" response.\\
The final answer should depend on the task, it can also be long sentences.
5. Evaluate accuracy by comparing generated answers to gold standard answers. (Only for binary QA task, long answer generation metrics are calculated by Similarity_Calculation.ipynb)
6. Save comprehensive results including questions, answers, reasoning chains, and accuracy metrics to CSV.
**The prompt and final answer can be in any language because the LLM can be fine-tuned on Non-English data**

The second file **Similarity_Calculation.ipynb** takes the LLM output from the first step to calculate the similarity between generated responses and gold answers from the test data 
The steps are: 
1. In case generated answer is in a different language than the gold answer, we need to transalte the response into that language using Helsinki MT model 
2. Then we compare the responses with the gold answer using various metrics: SBert and BLEU.
3. Finally we save the results. 




