import torch
import wandb
import os
from datasets import load_dataset
from transformers import TrainingArguments
from trl import SFTTrainer
from unsloth import FastLanguageModel, is_bfloat16_supported
import argparse

#input arguments 
def get_args():
    parser = argparse.ArgumentParser(description="Train Llama-3 model")
    parser.add_argument("--model_name", type=str, default="unsloth/llama-3-8b-Instruct-bnb-4bit", help="Model name or path")
    parser.add_argument("--max_seq_length", type=int, default=2048, help="Maximum sequence length")
    parser.add_argument("--load_in_4bit", action="store_true", help="Load model in 4-bit precision")
    parser.add_argument("--data_path", type=str, required=True, help="Path to the training data")
    parser.add_argument("--output_dir", type=str, default="./outputs", help="Directory to save the model")
    parser.add_argument("--num_train_epochs", type=int, default=1, help="Number of training epochs")
    parser.add_argument("--per_device_train_batch_size", type=int, default=2, help="Batch size per device during training")
    parser.add_argument("--logging_steps", type=int, default=50, help="Number of steps between logging")  # <-- Add this
    parser.add_argument("--test_size", type=float, default=0.1, help="Size of test set")
    parser.add_argument("--eval_steps", type=int, default=10, help="Number of update steps between evaluations")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4, help="Number of updates steps to accumulate before performing a backward/update pass")
    parser.add_argument("--max_steps", type=int, default=None, help="Maximum number of training steps. Defaults to None for full epochs")
    parser.add_argument("--learning_rate", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--rank", type=int, default=8, help="Rank for LoRA")
    parser.add_argument("--alpha", type=int, default=16, help="Alpha for LoRA")
    parser.add_argument("--report_to", type=str, default="wandb", help="Report to")
    parser.add_argument("--wandb_project", type=str, default=None, help="wandb project name")
    parser.add_argument("--wandb_run_name", type=str, default=None, help="wandb run name")
    return parser.parse_args()

def main():
    args = get_args()

    # Load the model
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model_name,
        max_seq_length=args.max_seq_length,
        dtype=None,  # Auto-detect
        load_in_4bit=args.load_in_4bit,
    )

    # Add LoRA adapters
    model = FastLanguageModel.get_peft_model(
        model,
        r=args.rank,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_alpha=args.alpha,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=3407,
        use_rslora=False,
        loftq_config=None,
    )

    # Prepare the dataset
    prompt_template = """<|begin_of_text|>
<|start_header_id|>system<|end_header_id|>
You are an expert at answering questions. Answer user questions in the same language as the user language, based only on the context provided. Learn to distinguish useful context and ignore the distractor context. Provide step-by-step reasoning in a chain of thought (CoT), explaining which parts of the context are important, including relevant quotes, analyze the quote and end with a concise answer. Make sure to answer the question in the user language!
<|eot_id|>

<|start_header_id|>user<|end_header_id|>
Question: {question}

Context: {context}
<|eot_id|>

<|start_header_id|>assistant<|end_header_id|>
{answer}
<|eot_id|>"""

    def formatting_prompts_func(examples):
        texts = []
        for question, context, answer in zip(examples["question"], examples["context"], examples["cot_answer"]):
            text = prompt_template.format(
                question=question,
                context=context['sentences'][0],
                answer=answer
            )
            texts.append(text)
        return {"text": texts}

    dataset = load_dataset('json', data_files=args.data_path)
    dataset = dataset['train'].train_test_split(test_size=args.test_size, seed=42)

    train_dataset = dataset['train'].map(formatting_prompts_func, batched=True)
    test_dataset = dataset['test'].map(formatting_prompts_func, batched=True)

    # Set up the trainer
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        dataset_text_field="text",
        max_seq_length=args.max_seq_length,
        dataset_num_proc=2,
        packing=False,
        args=TrainingArguments(
            per_device_train_batch_size=args.per_device_train_batch_size,
            per_device_eval_batch_size=args.per_device_train_batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            warmup_steps=5,
            max_steps=args.max_steps if args.max_steps is not None else -1,
            learning_rate=args.learning_rate,
            fp16=not is_bfloat16_supported(),
            bf16=is_bfloat16_supported(),
            logging_steps=args.logging_steps,
            optim="adamw_8bit",
            weight_decay=0.01,
            lr_scheduler_type="linear",
            seed=3407,
            output_dir=args.output_dir,
            num_train_epochs=args.num_train_epochs,
            report_to=args.report_to,
            evaluation_strategy="steps",
            eval_steps=args.eval_steps,
            save_strategy="steps",
            save_steps=args.eval_steps,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
        ),
    )

    if args.report_to == "wandb" and args.wandb_project is not None:
        os.environ["WANDB_PROJECT"] = args.wandb_project

    # Train the model
    trainer.train()

    # Save the model
    trainer.save_model(args.output_dir)
    if args.report_to == "wandb":
        wandb.finish()

if __name__ == "__main__":
    main()