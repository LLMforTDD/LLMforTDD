import os
import csv
from transformers import T5ForConditionalGeneration, RobertaTokenizer
import argparse
from transformers import BloomTokenizerFast, get_scheduler
# from unixcoder import UniXcoder
from sklearn.model_selection import train_test_split
from transformers import AutoConfig, AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForCausalLM, \
                        GPT2Config, GPT2Model, RobertaModel, PLBartForCausalLM, \
                        Trainer, TrainingArguments, \
                        DataCollatorForSeq2Seq, DataCollatorForLanguageModeling

model_name = "bigscience/bloom-560m"
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = BloomTokenizerFast.from_pretrained(model_name)

# Specify the input folder containing multiple subfolders with CSV files
input_folder = "/home/saranya/HDD18TB/LLM/LLM-for-Test-Case-Generation/Evaluation_NoFineTuned_Prompts/Csv"  # Replace with the actual path to your input folder

# Specify the output folder for the generated text files
output_folder = "/home/saranya/HDD18TB/LLM/LLM-for-Test-Case-Generation/Output/Evaluation_NoFineTuned_Prompts/Bloom/Csv"  # Replace with the desired output folder path
os.makedirs(output_folder, exist_ok=True)

# Function to generate JUnit test cases and write to a text file
def generate_and_write_tests(csv_file_path, input_folder, output_folder):
    relative_path = os.path.relpath(csv_file_path, input_folder)
    output_path = os.path.join(output_folder, relative_path)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    method_descriptions = []
    with open(csv_file_path, 'r') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            if row:  # Check if the row is not empty
                method_descriptions.append(row[0])

    # output_file_path = os.path.join(output_path, f"{os.path.basename(csv_file_path)[:-4]}Test.txt")
    output_file_path = os.path.splitext(output_path)[0] + '.txt'
    # Ensure the directory structure exists
    os.makedirs(os.path.dirname(output_file_path), exist_ok=True)
    
    with open(output_file_path, 'w') as output_file:
        for method_description in method_descriptions:
            generated_code = model.generate(
                tokenizer.encode(method_description, return_tensors="pt"),
                max_length=1024,
                num_return_sequences=1,
                no_repeat_ngram_size=2,
                top_k=50,
                top_p=0.95,
                temperature=0.95,
            )

            generated_code = tokenizer.decode(generated_code[0], skip_special_tokens=True)
            output_file.write(f"Generated JUnit Test Case: {generated_code}\n\n")

    

# Iterate over CSV files in the input folder and its subfolders
for root, dirs, files in os.walk(input_folder):
    for file in files:
        if file.endswith('.csv'):
            csv_file_path = os.path.join(root, file)
            generate_and_write_tests(csv_file_path, input_folder, output_folder)
