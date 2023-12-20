import pandas as pd
import os
import pickle
import json
import openai
import time
from tenacity import retry, stop_after_attempt, wait_random_exponential

# Define the folder containing CSV files
folder_path = '/home/saranya/HDD18TB/RL/PyRL-text-to-testcase-main/Dataset/Evaluation_Dataset/Evaluation_NoFineTuned_Prompts/Lang/Lang_folder_6'

# Define the output folder for text files
output_folder = '/home/saranya/HDD18TB/RL/PyRL-text-to-testcase-main/Output/Evaluation_NoFineTuned_Prompts/GPT3.5/Lang'

# Reading API keys
api_key = json.load(open('access_token.json'))
key = api_key['openai_access_token']

@retry(wait=wait_random_exponential(min=0.3, max=2), stop=stop_after_attempt(6))
def completion_with_backoff(**kwargs):
    return openai.ChatCompletion.create(**kwargs)

openai.api_key = key

# Get a list of CSV files in the folder
csv_files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]

for csv_file in csv_files:
    # Read the CSV file
    df_test = pd.read_csv(os.path.join(folder_path, csv_file), delimiter='\t')

    # Define the output text file name
    output_filename = os.path.splitext(csv_file)[0] + '.txt'
    output_file_path = os.path.join(output_folder, output_filename)

    openai_completions = []

    input_column = 'description'
    # Open the output text file for writing
    with open(output_file_path, "a") as file:
        for i in range(len(df_test)):
            description_content = str(df_test[input_column][i])
            completion = completion_with_backoff(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "user", "content": description_content}
                ],
                max_tokens=300
            )
            openai_completions.append(completion)
            save_obj = {'idx': i, 'response': completion}
            json_str = json.dumps(save_obj)
            file.write(json_str + '\n')
            time.sleep(10)
