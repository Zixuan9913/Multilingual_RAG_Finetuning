import argparse
import json
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModelForSeq2SeqLM
from transformers import MarianMTModel, MarianTokenizer
from unsloth import FastLanguageModel, is_bfloat16_supported
from peft import PeftModel, PeftConfig
from datasets import load_dataset
import torch
from llama_index.core import Document, VectorStoreIndex, ServiceContext, Settings
from llama_index.core.node_parser import SimpleNodeParser
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import Settings, StorageContext, load_index_from_storage
import os
import re
from tqdm import tqdm
import csv
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

#function to calculate BLEU, if necessary 
smoothie = SmoothingFunction().method1
def calculate_bleu(reference, candidate):
    """Calculate BLEU score between reference and candidate sentences."""
    return sentence_bleu([reference.split()], candidate.split(), smoothing_function=smoothie)
#function for raw accuracy    
def calculate_accuracy(gold_answers, predicted_answers):
    """
    Calculate accuracy for yes/no/maybe predictions.
    
    Args:
        gold_answers (list): List of ground truth answers
        predicted_answers (list): List of predicted answers
        
    Returns:
        float: Accuracy score between 0 and 1
    """
    if not gold_answers or not predicted_answers:
        return 0.0
    
    if len(gold_answers) != len(predicted_answers):
        raise ValueError("Gold and predicted answer lists must be the same length")
    
    correct = sum(1 for g, p in zip(gold_answers, predicted_answers) 
                 if g.lower().strip() == p.lower().strip())
    
    return correct / len(gold_answers)

#calculate accuracy of valid examples (which contain both CoT and final answer)
def calculate_valid_accuracy(results):
    """
    Calculate accuracy for examples with both valid CoT and Answer.
    
    Args:
        results (list of dicts): List of results containing 'correct' and 'valid' keys.
        
    Returns:
        float: Accuracy score for valid examples (correct numbers/total valid examples).
    """
    valid_results = [r for r in results if r['valid']]
    if not valid_results:
        return 0.0  # Avoid division by zero if there are no valid examples
    
    correct_valid = sum(1 for r in valid_results if r['correct'])
    return correct_valid / len(valid_results)

# Load your fine-tuned model with LoRA
def load_finetuned_model(base_model_dir, adapter_model_dir):
    tokenizer = AutoTokenizer.from_pretrained(base_model_dir)
    model = AutoModelForCausalLM.from_pretrained(base_model_dir, torch_dtype=torch.float16)
    peft_config = PeftConfig.from_pretrained(adapter_model_dir)
    model = PeftModel.from_pretrained(model, adapter_model_dir)
    model.to('cuda')
    return model, tokenizer

# Generate response from the model
def generate_response(model, tokenizer, prompt, max_tokens=512, temperature=0.0):
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True).to('cuda')
    with torch.no_grad():
        output_ids = model.generate(**inputs, 
        max_new_tokens=max_tokens, 
        temperature=temperature,  # Add temperature parameter
        do_sample=temperature > 0)
    response_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return response_text

def build_index(dataset, num_samples):

    items = list(dataset.items())
    if num_samples and num_samples < len(items):
        sampled_items = items[:num_samples]  # Take first num_samples items
    else:
        sampled_items = items
    
    documents = []
    sampled_data = {}  # Keep track of sampled data
    
    print(f"Building index with first {len(sampled_items)} examples...")

    for key, item in sampled_items:
        # Combine all contexts into one document
        combined_context = "\n".join(item['CONTEXTS'])
        doc = Document(text=combined_context)
        documents.append(doc)
        # Store sampled data
        sampled_data[key] = item

    parser = SimpleNodeParser()
    nodes = parser.get_nodes_from_documents(documents)
    
    embed_model = HuggingFaceEmbedding(model_name="intfloat/multilingual-e5-small")
    Settings.embed_model = embed_model
    
    index = VectorStoreIndex(nodes)
    return index, sampled_data

def retrieve_and_answer(index, model, tokenizer, query, top_k,temperature=0.0):
    retriever = index.as_retriever(similarity_top_k=top_k)
    retrieved_nodes = retriever.retrieve(query)
    
    retrived_context = [node.node.text for node in retrieved_nodes]

    prompt = f"""
<|begin_of_text|>
<|start_header_id|>system<|end_header_id|>
You are a professional answer-answering expert in Medicine domain in English. Find the most relevant sentences from the context and answer the user's question using only the relevant part, discarding distracting texts.
Instructions:
1. Provide a step-by-step Chain of Thought (CoT) in ENGLISH in "<CoT>:" following this EXACTLY numbered format:
1) Analysis of the question: [State the intent of the question in one sentence]
2) Relevant cite: Analyze the texts in the context and indicate which one is relevant to the question. Cite the context in: ##begin_quote##[ONLY ONE quote - the most relevant part of the context identified as relevant]##end_quote##
3) Analysis of the quote: [Analyze the evidence in the quote, focusing on significant results]
4) Summarization: [Summarize your conclusion based on the evidence]
2. End with <ANSWER>: which MUST be ONLY one of these words:
   - "yes" - Use "yes" when there is significant evidence supporting a positive result or when an important factor influences the decisions or actions described in the context. If a reason or factor is mentioned by most participants or has a significant impact on behavior, select "yes."
   - "no" - ONLY when there is significant evidence supporting a negative or opposite result clearly in the context.
   - "maybe" - ONLY when:
    *The context explicitly discusses conditions where the answer varies.
    *The question includes multiple elements with different outcomes.

IMPORTANT:
-The context may contain texts unrelated to the question (distractors). Don’t be misled by texts that contain similar words but address different topics!
-ALWAYS include both <CoT> and <ANSWER>, YOU MUST use the numbers 1-4 and subtitles exactly as shown above.
-<ANSWER> must be based directly on your <CoT>.
-Avoid "maybe" when there is only slight or theoretical uncertainty. Use "yes" whenever there is significant evidence of positive outcomes, even if they aren’t universal.
-Ensure the conclusion is based on significant results and focus on relevant experimental or statistical evidence.
Use ONLY ONE quote, the most relevant one.
-<ANSWER> should be only "yes," "no," or "maybe" in English!
<|eot_id|>

<|start_header_id|>user<|end_header_id|>
Question: {query}\n
Context: {retrived_context}\n
The CoT should be in English and start <CoT> directly!The final answer MUST be a single word in English: "yes," "no," or "maybe.
<|eot_id|>

<|start_header_id|>\nassistant<|end_header_id|>"""
    response = generate_response(model, tokenizer, prompt, temperature=temperature)
    
    #debug the response
    try:
        assistant_response = response.split("\nassistant")[1].strip()
    except:
        print("Warning: Couldn't find assistant tag")
        assistant_response = response.strip()
    # Initialize default values
    print(assistant_response)
    cot = "None"
    answer = "invalid"
    # Then extract CoT and answer from assistant's response
    try:
        # First try to get both CoT and answer
        if "<CoT>" in assistant_response:
            # Split on <CoT> and look for the text before <ANSWER>
            cot_text = assistant_response.split("<CoT>")[1]
            # Remove any leading ":" if present
            cot = cot_text.lstrip(":\n").split("<ANSWER>")[0].strip()
        
        # If that fails, try to find just the answer
        if "<ANSWER>" in assistant_response:
            answer_text = assistant_response.split("<ANSWER>")[1]
            # Remove any leading ":" if present
            answer = answer_text.lstrip(":\n").strip().lower()
        
        # Final validation of answer
        if answer not in ["yes", "no", "maybe"]:
            print(f"Warning: Invalid answer format received: {answer}")
            answer = "invalid"
            
    except Exception as e:
        print(f"Error in response parsing: {str(e)}")
        print(f"Response was: {assistant_response}")
        cot = "None"
        answer = "invalid"

    print(f"CoT:{cot}")
    print(f"Final Answer:{answer}")
    #print(f"assistant full response: {assistant_response}")
    return answer, cot, retrived_context

# Main script with argument parsing
def main():
    parser = argparse.ArgumentParser(description="Generate answers and calculate BLEU score using fine-tuned LLM.")
    
    # Arguments for the dataset, model paths, and output file
    parser.add_argument("--dataset_dir", type=str, required=True, help="Hugging Face dataset name or path to local data")
    parser.add_argument("--base_model_dir", type=str, default= "base", required=True, help="Path to the base model (checkpoint folder)")
    parser.add_argument("--adapter_model_dir", type=str, required=True, help="Path to the LoRA adapter model folder")
    parser.add_argument("--output_file", type=str, required=True, help="Path to save the generated responses CSV")
    parser.add_argument("--max_examples", type=int, default=100, help="Max number of examples to process (default: 100)")
    parser.add_argument("--load_in_4bit", action="store_true", help="Load model in 4-bit precision")
    #parser.add_argument('--language', default='English', help='Language to process the questions in')
    parser.add_argument("--top_k", type=int, default=3, help="top k number in the RAG retrieval")
    parser.add_argument("--temperature", type=float, default=0.0, help="Temperature for model generation (default: 0.0)")
    
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("CUDA is not available. Please check your GPU setup.")
        return

    # Load the JSON data
    with open(args.dataset_dir, 'r') as f:
        dataset = json.load(f)
    
    print(f"Building index with {args.max_examples} samples...")
    index, sampled_data = build_index(dataset, num_samples=args.max_examples)

    # Load the dataset from hugging face
    #dataset = load_dataset(args.dataset_name, split="test")

    # Load the fine-tuned model
    if args.base_model_dir != "base":
        print("Loading fine-tuned model...")
        model, tokenizer = load_finetuned_model(args.base_model_dir, args.adapter_model_dir)
    else:
        print("Loading original model...")
        model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="unsloth/llama-3-8b-Instruct-bnb-4bit",
        max_seq_length= 2048,
        dtype=None,  # Auto-detect
        load_in_4bit=args.load_in_4bit)
        FastLanguageModel.for_inference(model)
    # Process questions and generate answers
    results = []
    
    print(f"Processing {len(sampled_data)} examples...")
    correct_count = 0
    total_count = 0
    for key, item in tqdm(sampled_data.items()):
        question = item['QUESTION']
        gold_answer = item['final_decision']
        oracle_context = item["CONTEXTS"]
        
        # Generate answer and CoT
        generated_answer, chain_of_thought, retrieved_context = retrieve_and_answer(index, model, tokenizer, question, args.top_k, temperature=args.temperature)
        # Determine if the generated answer is valid (i.e., valid CoT + valid Answer)
        is_valid = chain_of_thought != "None" and generated_answer in ["yes", "no", "maybe"]
        
        # Check if the answer matches the gold answer
        is_correct = generated_answer.lower() == gold_answer.lower() if is_valid else False
        total_count += 1
        if is_correct:
            correct_count += 1
        
        if total_count % 10 == 0:
            print(f"\nCurrent statistics:")
            print(f"Total processed: {total_count}")
            print(f"Correct answers: {correct_count}")
            print(f"Current raw accuracy: {correct_count/total_count:.4f}")
        results.append({
            'id': key,
            'question': question,
            'generated_answer': generated_answer,
            'gold_answer': gold_answer,
            'chain_of_thought': chain_of_thought,
            "oracle_context": oracle_context,
            "retrived context": retrieved_context,
            'correct': is_correct,  # If the answer is correct or not
            'valid': is_valid
        })

    # Calculate accuracy
    gold_answers = [r['gold_answer'] for r in results]
    predicted_answers = [r['generated_answer'] for r in results]

    accuracy = calculate_accuracy(gold_answers, predicted_answers)
    valid_accuracy = calculate_valid_accuracy(results)
    print(f"Raw Accuracy: {accuracy:.4f}")
    print(f"Valid Accuracy (valid CoT + Answer): {valid_accuracy:.4f}")

    for result in results:
        result['final_raw_accuracy'] = f"{accuracy:.4f}"
        result['final_valid_accuracy'] = f"{valid_accuracy:.4f}"

    # Save results to CSV
    with open(args.output_file, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['id', 'question', 'generated_answer', 'gold_answer', 'chain_of_thought', "oracle_context", "retrived context",'correct', 'valid', 'final_raw_accuracy', 'final_valid_accuracy']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    print(f"Results saved to {args.output_file}")

if __name__ == "__main__":
    main()