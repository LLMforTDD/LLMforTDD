import os
import json
import csv
import sys
import argparse

DEVICE = 'cuda'

def read_json_files(input_dir):
    for root, dirs, files in os.walk(input_dir):
        for file in files:
            if file.endswith('.json'):
                with open(os.path.join(root, file), 'r') as f:
                    data = json.load(f)
                yield data


def extract_fields(data):
    test_case = data['test_case']['body']
    focal_method = data['focal_method']['body']
    description = data['description']

    # Ignore if the description is empty
    if not description:
        return None

    return [description,focal_method, test_case]

def create_csv(fields, data, output_file):
    with open(output_file, 'w') as f:
        writer = csv.writer(f, delimiter='\t')
        writer.writerow(fields)
        for row in data:
            row_data = extract_fields(row)
            if row_data:
                writer.writerow(row_data)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Extracts the test_case, focal_method and description from a JSON file and creates a CSV file.')
    parser.add_argument('--input', '-i', type=str, help='The path to the JSON file.')
    parser.add_argument('--output', '-o', type=str, help='The path to the TSV file.')

    args = parser.parse_args()

    input_dir = args.input
    output_file = args.output

    data = read_json_files(input_dir)
    fields = ['description','focal_method','test_case']
    create_csv(fields, data, output_file)