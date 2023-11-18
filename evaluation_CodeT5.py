import os
import torch
import glob
import datasets
import pickle
import pandas as pd
from tqdm import tqdm
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, DataCollatorForLanguageModeling, AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq

#########################
######## config #########
model_name = 'Salesforce/codet5-large'
model_dir  = '/home/saranya/HDD18TB/RL/PyRL-text-to-testcase-main/save/APPS/codet5-code-v2/checkpoint-183520'
test_dataset_folder = '/home/saranya/HDD18TB/RL/PyRL-text-to-testcase-main/Dataset/Test'
input_column = 'description' 
output_column = 'test_case'
processed = False

output_dir = '/home/saranya/HDD18TB/RL/PyRL-text-to-testcase-main/Output/Sample'
output_suffix = 'Test'

batch_size = 4
beam_size = 4
max_new_tokens = 500  # Overwrite max_length
max_length = 1024

gpu_num = '0'


# Initialize an empty DataFrame to store data from all CSV files
all_data = pd.DataFrame()

# List all CSV files in the directory
csv_files = [file for file in os.listdir(test_dataset_folder) if file.endswith(".csv")]

# Iterate through each CSV file and read it into a DataFrame
for csv_file in csv_files:
    file_path = os.path.join(test_dataset_folder, csv_file)
    data = pd.read_csv(file_path, delimiter='\t')
    all_data = all_data.append(data, ignore_index=True)

# Load the trained model and initialize the tokenizer
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_dir,
                                          padding_side='left',
                                          truncation_side='left',
                                          do_lower_case=False
                                          )

if model_name == 'gpt2':
    tokenizer.pad_token = tokenizer.eos_token

# Tokenize the inputs for the test dataset
test_inputs_text = list(all_data[input_column])
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
        outputs = outputs[:, input_ids.shape[-1]:]
        decoded_outputs = tokenizer.batch_decode(outputs, skip_special_tokens=skip_special_tokens, clean_up_tokenization_spaces=False)

        # Process decoded outputs to ensure correctness
        processed_outputs = []
        for i in range(0, len(decoded_outputs), beam_size):
            batch_outputs = decoded_outputs[i:i+beam_size]
            processed_outputs.append(batch_outputs)
        predictions.extend(processed_outputs)

        # Store raw outputs for debugging
        raw_outputs.append([outputs[i:i+beam_size] for i in range(0, len(outputs), beam_size)])

# Save predictions to text files with the same name as the input CSV files in the output folder
for csv_file, prediction in zip(csv_files, predictions):
    base_name = os.path.splitext(csv_file)[0]
    output_txt_path = os.path.join(output_dir, f"{base_name}{output_suffix}.txt")

    with open(output_txt_path, 'w') as f:
        for data in prediction:
            f.write(data + '\n')

# Continue with any further processing or evaluation steps as needed
