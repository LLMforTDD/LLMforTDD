import openai
import pickle
import openai
import json

def generate_junit_test_case(description):
    # Set your OpenAI API key here
    openai.api_key = "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"

    # Construct the prompt
    prompt = f"Create a JUnit test case for the following requirement:\n{description}"

    try:
        # Generate the test case using OpenAI's GPT-4
        response = openai.Completion.create(
            engine="text-davinci-004",  # You might need to change this based on available models
            prompt=prompt,
            max_tokens=150  # You can adjust this based on how lengthy the expected output is
        )

        # Extracting the text from the response
        test_case = response.choices[0].text.strip()
        return test_case
    except Exception as e:
        print(f"An error occurred: {e}")
        return None

# Example usage
description = "A Calculator class with a method add(int a, int b) that returns the sum of a and b."
test_case = generate_junit_test_case(description)
if test_case:
    print("Generated JUnit Test Case:\n", test_case)
