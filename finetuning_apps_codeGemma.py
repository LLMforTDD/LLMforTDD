import pandas as pd
import numpy as np
import os
import logging
import torch
import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments, DataCollatorForLanguageModeling
from sklearn.model_selection import train_test_split

# Define the token if needed
token = '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'

# Initialize the CodeGemma model and tokenizer
checkpoint = "codegemma/CodeGemma-small"
tokenizer = AutoTokenizer.from_pretrained(checkpoint, revision="main", token=token)
model = AutoModelForCausalLM.from_pretrained(checkpoint, revision="main", token=token)

parser = argparse.ArgumentParser(description='Description of your script.')
parser.add_argument('--input', '-i', type=str, help='The path to the TSV file.')
args = parser.parse_args()

input_file = args.input

input_dir = '/home/saranya/HDD18TB/RL/PyRL-text-to-testcase-main/Dataset/Test/CSVFormat.csv'
input_column = 'description'
output_filename = '/home/saranya/HDD18TB/RL/PyRL-text-to-testcase-main/Output/Csv'  # Specify your output directory
output_suffix = '_testcase'
batch_size = 1

# Configure CUDA settings
device = "cuda"
tokenizer = AutoTokenizer.from_pretrained(checkpoint, cache_dir='/HDD18TB/saranya/LLM_HF')
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = 'left'
tokenizer.truncation_side = 'left'
model = AutoModelForCausalLM.from_pretrained(checkpoint,
                                             device_map="auto",
                                             load_in_8bit=True,
                                             cache_dir='/HDD18TB/saranya/LLM_HF')

model_name = 'codegemma/CodeGemma-small'
is_shuffle = True

train_dataset_path = input_file
output_column = 'test_case'
output_dir = 'save/APPS/codegemma/CodeGemma-small'

# Set environment variables
os.environ['TRANSFORMERS_NO_ADVISORY_WARNINGS'] = 'true'
os.environ["TOKENIZERS_PARALLELISM"] = "true"

# Set logging
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
log_file = output_dir + '/train.log'
logger = logging.getLogger(__name__)
logging.basicConfig(filename=log_file, filemode='a', format='%(asctime)s - %(levelname)s - %(name)s - %(message)s', datefmt='%m/%d/%Y %H:%M:%S', level=logging.INFO)
logger.addHandler(logging.StreamHandler())
logger.info(f"Is CUDA available: {torch.cuda.is_available()}")
logger.info(f"CUDA device: {torch.cuda.get_device_name(torch.cuda.current_device())}")

# Load the CSV
train_df = pd.read_csv(train_dataset_path, delimiter='\t', error_bad_lines=False, skip_blank_lines=True)
train_df, eval_df = train_test_split(train_df, test_size=0.1, random_state=42, shuffle=is_shuffle)

# Convert the tokenized inputs and outputs into a PyTorch dataset
class MyDataset(torch.utils.data.Dataset):
    def __init__(self, inputs, outputs):
        self.inputs = inputs
        self.outputs = outputs

    def __getitem__(self, idx):
        return {
            "input_ids": torch.tensor(self.inputs["input_ids"][idx]),
            "attention_mask": torch.tensor(self.inputs["attention_mask"][idx]),
            "labels": torch.tensor(self.outputs["input_ids"][idx]),
        }

    def __len__(self):
        return len(self.inputs["input_ids"])

# Tokenize the inputs and outputs
train_inputs = tokenizer(list(train_df[input_column]), padding=True, truncation=True, max_length=512)
train_outputs = tokenizer(list(train_df[output_column]), padding=True, truncation=True, max_length=512)
train_dataset = MyDataset(train_inputs, train_outputs)

eval_inputs = tokenizer(list(eval_df[input_column]), padding=True, truncation=True, max_length=512)
eval_outputs = tokenizer(list(eval_df[output_column]), padding=True, truncation=True, max_length=512)
eval_dataset = MyDataset(eval_inputs, eval_outputs)
logger.info('loaded dataset successfully.')

training_args = TrainingArguments(
    output_dir=output_dir,
    learning_rate=2e-5,
    weight_decay=0.01,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    gradient_accumulation_steps=4,
    num_train_epochs=20,
    logging_strategy="steps",
    logging_steps=500,
    save_strategy="epoch",
    eval_accumulation_steps=1,
    evaluation_strategy="epoch",
    warmup_steps=1000,
    fp16=True,
    save_total_limit=10,
    optim="adamw_torch",
    lr_scheduler_type="inverse_sqrt"
)
logger.info(f'''training_args: lr={training_args.learning_rate},
            batch_size={training_args.per_device_train_batch_size},
            epoch={training_args.num_train_epochs},
            gradient_accumulation_steps={training_args.gradient_accumulation_steps},
            warmup_steps={training_args.warmup_steps},
            weight_decay={training_args.weight_decay},
            optim={training_args.optim},
            lr_scheduler_type={training_args.lr_scheduler_type},
            fp16={training_args.fp16}
            ''')

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
    tokenizer=tokenizer,
)

trainer.train()
