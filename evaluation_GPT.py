import ast
import os
import torch
import glob
import logging
import pickle
import datasets
import pandas as pd
from tqdm import tqdm
from torch.utils.data import DataLoader
from transformers import AutoModelForSeq2SeqLM, AutoModelForCausalLM, AutoTokenizer, PLBartForCausalLM, \
            DataCollatorForSeq2Seq, DataCollatorForLanguageModeling


#########################
######## config #########
model_name = 'gpt2'
model_dir  = '/home/saranya/HDD18TB/RL/PyRL-text-to-testcase-main/save/APPS/gpt2-large/checkpoint-8257'
test_dataset_folder = '/home/saranya/HDD18TB/RL/PyRL-text-to-testcase-main/Dataset/Evaluation/Csv'  # Change to the folder path
input_column = 'description' 
output_column = 'test_case'
processed = False

output_dir = model_dir
output_suffix = '_testcase'

batch_size = 4
beam_size = 4
max_new_tokens = 500  # Overwrite max_length
max_length = 1024

gpu_num = '0'

output_parent_dir = '/home/saranya/HDD18TB/RL/PyRL-text-to-testcase-main/Output/Csv'  # Change this to the parent directory where you want to save the .txt files

#########################

def generate(model, input_ids, attention_mask, max_new_tokens, num_beams, test_method, **kwargs):
    with torch.no_grad():
        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pad_token_id=tokenizer.eos_token_id,
            max_new_tokens=max_new_tokens,
            num_return_sequences=num_beams,
            num_beams=num_beams,
            early_stopping=True,
            max_time=30
        )

        decoded_outputs = tokenizer.batch_decode(outputs, skip_special_tokens=True, clean_up_tokenization_spaces=False)

        processed_outputs = []
        for i in range(0, len(decoded_outputs), num_beams):
            batch_outputs = decoded_outputs[i:i+num_beams]
            processed_outputs.append(batch_outputs)

# Iterate through CSV files in the folder
for csv_file_path in glob.glob(os.path.join(test_dataset_folder, '*.csv')):
    # Load the CSV test dataset
    test_df = pd.read_csv(csv_file_path, delimiter='\t')

     # Create a directory for saving the output .txt files
    csv_file_name = os.path.splitext(os.path.basename(csv_file_path))[0]
    output_dir = os.path.join('/home/saranya/HDD18TB/RL/PyRL-text-to-testcase-main/Output/Csv', csv_file_name)
    # os.makedirs(output_dir, exist_ok=True)  # Ensure the directory exists


    # Load the trained model and initialize the tokenizer
    model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_dir, 
                                              padding_side='left',  # Set padding_side to 'left'
                                              truncation_side='left',  # Adjust truncation_side if needed
                                              do_lower_case=False
                                             )

    if model_name == 'gpt2':
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokenizer.pad_token = tokenizer.eos_token  # Set the padding token to the EOS token
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
        )
    # Tokenize the inputs for the test dataset
    test_inputs_text = list(test_df[input_column])
    test_inputs = tokenizer(test_inputs_text, padding=True, truncation=True, max_length=max_length)

    # Convert the tokenized inputs into a PyTorch dataset
    test_dataset = datasets.Dataset.from_dict({
        "input_ids": test_inputs["input_ids"],
        "attention_mask": test_inputs["attention_mask"],
    })

    # Create PyTorch dataloaders for the test dataset
    test_loader = DataLoader(test_dataset, batch_size=batch_size, collate_fn=DataCollatorForSeq2Seq(tokenizer) if 'bigscience/bloom-560m' in model_name else DataCollatorForLanguageModeling(tokenizer, mlm=False))

    # Generate predictions on the test dataset
    predictions = []
    raw_outputs = []
    skip_special_tokens = not (processed and model_name in ['gpt2', 'codegpt', 'pycoder'])

    for batch in tqdm(test_loader):
        with torch.no_grad():
            input_ids = batch["input_ids"].to('cpu')
            attention_mask = batch["attention_mask"].to('cpu')

            # Generate test case predictions
            outputs = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pad_token_id=tokenizer.eos_token_id,
                max_new_tokens=max_new_tokens,
                num_return_sequences=beam_size,
                num_beams=beam_size,
                early_stopping=True,
                max_time=30
            )

            # Post-process the generated outputs
            outputs = outputs[:, input_ids.shape[-1]:]  # Trim output to only generated tokens
            decoded_outputs = tokenizer.batch_decode(outputs, skip_special_tokens=skip_special_tokens, clean_up_tokenization_spaces=False)

            # Process decoded outputs to ensure correctness (modify this part as needed)
            processed_outputs = []
            for i in range(0, len(decoded_outputs), beam_size):
                batch_outputs = decoded_outputs[i:i+beam_size]
                processed_outputs.append(batch_outputs)  # Ensure this step correctly separates batches

            predictions.extend(processed_outputs)

            # Store raw outputs for debugging
            raw_outputs.append([outputs[i:i+beam_size] for i in range(0, len(outputs), beam_size)])

    first_predictions = [pred[0] for pred in predictions]
    output_txt_path = os.path.splitext(output_dir)[0] + '.txt'

    with open(output_txt_path, 'w') as f:
        for data in first_predictions:
          print(data)
          f.write(data + '\n')
