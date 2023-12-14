from openai import OpenAI
import os
from dotenv import load_dotenv
import json


def read_training_icl_json_file(file_path):
    """Reads the training data from the file"""
    with open(file_path, "r") as file:
        return [json.loads(line.strip()) for line in file]


def create_icl_prompt(translations, new_sentence):
    """Creates the prompt for the ICL model"""
    prompt = "Translate following sentences from English to French:\n"
    # Add the translations to the prompt that will be used to train the ICL model
    for pair in translations:
        prompt += f"English: {pair['english']}\nFrench: {pair['french']}\n\n"

    prompt += "User's sentence in English: " + new_sentence + "\nTranslated French sentence:"
    return prompt


def translate_sentence(new_sentence, translations_icl_file='./assets/translations_icl.jsonl'):
    '''Translate a sentence from English to French'''
    # Load the translations from the file
    translations = read_training_icl_json_file(translations_icl_file)
    # Create the prompt for the ICL model
    prompt = create_icl_prompt(translations, new_sentence)
    # Call the ICL model to translate the sentence
    response = client.completions.create(model="davinci-instruct-beta",
    prompt=prompt,
    max_tokens=100,
    temperature=0)

    # Return the translated sentence
    return response.choices[0].text.strip()


if __name__ == '__main__':
    # Load the environment variables
    load_dotenv()
    # Set the OpenAI API key
    client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
    
    # Translate a sentence from English to French
    new_sentence = input("Enter a sentence in English: ")
    print("Translated text: ", translate_sentence(new_sentence))
