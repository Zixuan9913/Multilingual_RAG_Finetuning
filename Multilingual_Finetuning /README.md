## Fine-tuning LLAMA3-8b-model with generated data 

[Training Data](https://drive.google.com/file/d/18JCPMP8MRxOGL6CVMfv2U8fFQ5yckGRp/view?usp=drive_link)

Steps: 
1. Argument Setup - Parse command-line arguments for model configuration, dataset paths, and training parameters.
2. Model Loading - Load the pre-trained Llama-3 model with specified precision settings.
3. LoRA Configuration - Add Low-Rank Adaptation layers to specific model components for efficient fine-tuning.
4. Prompt Formatting - Define a template for structuring inputs with system instructions, user questions, and context.The language of the prompt should be the same as the training data. 
5. Dataset Preparation - Load and process the dataset, splitting it into training and testing portions.
6. Trainer Configuration - Set up the SFTTrainer with optimizers, learning rate, evaluation settings, and other parameters.
7. Training Execution - Train the model using the prepared datasets and training arguments.
8. Model Saving - Save the fine-tuned model to the specified output directory.
