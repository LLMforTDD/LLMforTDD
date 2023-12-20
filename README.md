# README

This is a sample README file.

# LLM-for-Test-Case-Generation

## Datasets

The dataset for are available in LLMforTDD.zip
* Evaluation_NoFineTuned_NoPrompts
* Evaluation_NoFineTuned_Prompts
* Evaluation_FineTuned_NoPrompts
* Evaluation_FineTuned_Prompts

Developing a BLOOM based model for the unit test cases generation

## To Install Requirement Files

```python
pip install -q petals datasets wandb scikit-learn
```

## To run the test training script for the BLOOM small model
```python
python training_script.py
```

## To run the test training script for Chat GPT 3.5 

```python
python OpenAi.py --input dataset.tsv
```

## To run the test training script for Finetuning model (code T5)

```python
python finetuning_apps_code.py --input dataset.tsv
```

## To run the test training script for PPO Trainging file

```python
python ppo_training_apps_code.py --input dataset.tsv
```


## To run the extraction file

```python
python extraction_dataset1.py --repo_lines txt.jsonl --grammar java-grammar.so --tmp /tmp/tmp --output /tmp/output1/

```

#### Args

The following **Args** can be passed to the parser:

* `--repo_lines`: The path to the JSON lines file containing the repos to analyze.
* `--grammar`: The filepath of the tree-sitter grammar.
* `--tmp`: The path to a temporary folder used for processing.
* `--output`: The path to the output folder.

For example, to analyze the repos in the `repos.jsonl` file using the `java-grammar.so` grammar and output the results to the `output` folder, you would use the following command:

#### Example

```
python extraction_dataset.py
    --repo_lines read.jsonl
    --grammar ./java-grammar.so
    --tmp /tmp/tmp/
    --output /tmp/output/
```

## To run Make Dataset file

This Python script extracts test cases from a directory of JSON files.

## Usage

To run the extraction script, you need to install the following dependencies:

* Python 3.7+
* json
* csv
* argparse

Once you have installed the dependencies, you can run the script as follows:


```python
python make_dataset.py --input input_dir --output output.tsv
```




The `input` argument specifies the path to the directory containing the JSON files. The `output` argument specifies the path to the TSV file to write the extracted test cases to.

For example, to extract the test cases from the directory `testcases` and write them to the file `output.tsv`, you would use the following command:


#### Arguments

The following arguments (**Args**) can be passed to the parser:

* `--input`: The path to the directory containing the JSON files.
* `--output`: The path to the TSV file to write the extracted test cases to.

## License

This project is licensed under the MIT License.


## Additional Information

The `make_dataset.py` script first reads the JSON files in the input directory. It then extracts the test cases from each JSON file and writes them to the output TSV file. The test cases are written to the TSV file in the following format:



test_case, focal_method, description



The `test_case` field contains the test case code. The `focal_method` field contains the name of the focal method in the test case. The `description` field contains the description of the test case.

## Formatting

The README.md file should be formatted using the following guidelines:

* Use a consistent font and font size.
* Use line breaks to separate paragraphs.
* Use bold and italics to highlight important text.
* Use numbered or bulleted lists to organize information.
* Use links to provide additional information.
