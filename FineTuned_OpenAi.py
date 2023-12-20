import pandas as pd
import pickle
import json
import openai
import time
from tqdm import tqdm
import os.path
from tenacity import ( retry, stop_after_attempt, wait_random_exponential ) # for exponential backoff
import argparse

parser.add_argument('--train', '-t', type=str, help='The path to the train TSV file.')
parser.add_argument('--test', '-s', type=str, help='The path to the test TSV file.')
args = parser.parse_args()

train_file = args.train
test_file = args.test

input_column = 'Description'
output_column = 'test_case'

train_dataset_path = train_file
test_dataset_path = test_file

# Reading the train and test files
df_train = pd.read_csv(train_dataset_path,delimiter='\t')
df_test = pd.read_csv(test_dataset_path,delimiter='\t')

#Setting up the fine-tuning parameters
fine_tuning_data = []

for i in tqdm(range(len(df_train))):
    description_content = df_train[input_column][i]
    test_case = df_train[output_column][i]
    fine_tuning_data.append({
        "prompt": description_content,
        "completion": test_case
    })

# Fine-tuning the GPT-3.5 model
openai.api_key = json.load(open('access_token.json'))['openai_access_token']

fine_tuning_response = openai.CreateFineTuneJob.create(
    training_set=fine_tuning_data,
    model="gpt-3.5-turbo"
)

# Waiting for the fine-tuning job to complete
fine_tuning_job = openai.FineTuneJob.retrieve(fine_tuning_response["id"])
while fine_tuning_job["status"] != "completed":
    time.sleep(60)
    fine_tuning_job = openai.FineTuneJob.retrieve(fine_tuning_response["id"])

# Using the fine-tuned model to generate test cases
@retry(wait=wait_random_exponential(min=0.3, max=2), stop=stop_after_attempt(6))
def completion_with_backoff(**kwargs):
    return openai.ChatCompletion.create(**kwargs)

openai.api_key = key
openai_completions = []
curr_idx = 0
skip = False
if os.path.isfile('save/openai_gpt35_apps_testcase.json'):
    skip = True
    rfile = open('save/openai_gpt35_apps_testcase.json', "r")
    curr_idx = len(rfile.readlines())
    rfile.close()
file = open('save/openai_gpt35_apps_testcase.json', "a")
for i in tqdm(range(len(df_test))):
    if i < curr_idx:
        continue
    if skip:
        print(f'continue at idx: {i}')
        skip = False
        description_content = "Write a JUnit Test Case for the description provided : " + df_test[input_column][i]
    completion = completion_with_backoff(
        model=fine_tuning_job["engine"],
        messages=[
            {"role": "user", "content": description_content}
        ],
        max_tokens=300
    )
    openai_completions.append(completion)
    save_obj = {'idx': i, 'response': completion}
    json_str = json.dumps(save_obj)
    file.write(json_str + '\n')
    time.sleep(1)
file.close()
with open('save/openai_gpt35_apps_testcase.pkl', 'wb') as f:
    pickle.dump(openai_completions, f)
