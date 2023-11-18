import pandas as pd
import numpy as np
import os
import logging
import torch
import argparse
from transformers import RobertaTokenizer, get_scheduler
# from unixcoder import UniXcoder
from sklearn.model_selection import train_test_split
from transformers import AutoConfig, AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForCausalLM, \
                        GPT2Config, GPT2Model, RobertaModel, PLBartForCausalLM, \
                        Trainer, TrainingArguments, \
                        DataCollatorForSeq2Seq, DataCollatorForLanguageModeling, T5ForConditionalGeneration
from transformers import T5Tokenizer

# Initialize the tokenizer
tokenizer = T5Tokenizer.from_pretrained("t5-small")

parser = argparse.ArgumentParser(description='Description of your script.')

parser.add_argument('--input', '-i', type=str, help='The path to the TSV file.')
args = parser.parse_args()

input_file = args.input
#########################
######## config #########
# export CUDA_VISIBLE_DEVICES=0
model_name = 'codet5-large'
is_shuffle = True

train_dataset_path = input_file
# eval_dataset_path = 'dataset/APPS/apps_processed_eval.csv' # not use
input_column = 'description'
output_column = 'test_case'
output_dir = 'save/APPS/codet5-code-v2'

# gpu_num = '0'

#########################
model_name = model_name.lower()
base_model = {
    'pycoder': AutoModelForCausalLM,
    'codegpt': AutoModelForCausalLM,
    'transformers': GPT2Model,
    'gpt2': AutoModelForCausalLM,
    'codet5-large': AutoModelForSeq2SeqLM,
    'codet5-large': AutoModelForSeq2SeqLM,
    'codet5-large': AutoModelForSeq2SeqLM,
   
}
base_checkpoint = {
    'pycoder': 'Wannita/PyCoder',
    'codegpt': 'microsoft/CodeGPT-small-py',
    'transformers': 'gpt2_no_pretrain_weight',
    'gpt2': 'gpt2',
    'codet5-large': 'Salesforce/codet5-large',
    'codet5-large': 'Salesforce/codet5-large',
    'codet5-large': 'Salesforce/codet5-large',
    'unixcoder': 'microsoft/unixcoder-base',
    'plbart': 'uclanlp/plbart-base',
}
# os.environ["CUDA_VISIBLE_DEVICES"] = gpu_num
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
device = "cuda" 

# Initialize the tokenizer, model
if model_name == 'transformers':
    config = GPT2Config()
    tokenizer = AutoTokenizer.from_pretrained('gpt2', truncation_side='left', do_lower_case=False)
    tokenizer.pad_token = tokenizer.eos_token
    model = base_model[model_name](config)
elif model_name == 'unixcoder':
    config =AutoConfig.from_pretrained(base_checkpoint[model_name])
    config.is_decoder = True
    tokenizer = AutoTokenizer.from_pretrained(base_checkpoint[model_name], truncation_side='left', do_lower_case=False)
    encoder = RobertaModel.from_pretrained(base_checkpoint[model_name],config=config)
    model=Seq2Seq(encoder=encoder,decoder=encoder,config=config,
                  beam_size=5,max_length=512,
                  sos_id=tokenizer.cls_token_id,eos_id=[tokenizer.sep_token_id])
else:
    tokenizer = None
    if model_name == 'Salesforce/codet5-large':
        tokenizer = RobertaTokenizer.from_pretrained(model_name)
        tokenizer.padding_side = 'right'
        tokenizer.model_max_length = 256
        model = T5ForConditionalGeneration.from_pretrained(
            model_name,
        ).to('cuda')
    
logger.info('loaded model and tokenizer sucessfully.')

# Load the CSV
train_df = pd.read_csv(train_dataset_path, delimiter='\t', error_bad_lines=False, skip_blank_lines=True)

# eval_df = pd.read_csv(eval_dataset_path)
# print("null input:", len(train_df[input_column].dropna()))
# print("null output:", len(train_df[output_column].dropna()))
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
            # "decoder_input_ids": torch.tensor(self.outputs["input_ids"][idx]),
            # "decoder_attention_mask": torch.tensor(self.outputs["attention_mask"][idx]),
            "labels": torch.tensor(self.outputs["input_ids"][idx]),
        }

    def __len__(self):
        return len(self.inputs["input_ids"])

# Tokenize the inputs and outputs
# Convert the tokenized inputs and outputs into a PyTorch dataset
print(train_df.columns)
print(input_column)
print(output_column)
# Define the input column
input_column = 'description'  # Replace with the actual column name

# Tokenize the input
train_inputs = tokenizer(list(train_df[input_column]), padding=True, truncation=True, max_length=512)
train_outputs = tokenizer(list(train_df[output_column]), padding=True, truncation=True, max_length=512)
train_dataset = MyDataset(train_inputs, train_outputs)

eval_inputs = tokenizer(list(eval_df[input_column]), padding=True, truncation=True, max_length=512)
eval_outputs = tokenizer(list(eval_df[output_column]), padding=True, truncation=True, max_length=512)
eval_dataset = MyDataset(eval_inputs, eval_outputs)
logger.info('loaded dataset sucessfully.')

training_args = TrainingArguments(
    output_dir=output_dir,
    learning_rate=2e-5, #1.5e-5,
    weight_decay=0.01,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    gradient_accumulation_steps=4,
    # predict_with_generate=True,
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
    lr_scheduler_type = "inverse_sqrt"
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
    data_collator=DataCollatorForSeq2Seq(tokenizer) if 'codet5' in model_name else DataCollatorForLanguageModeling(tokenizer, mlm=False),
    tokenizer=tokenizer,
)

trainer.train()