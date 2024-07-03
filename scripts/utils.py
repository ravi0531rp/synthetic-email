import json
from langchain_openai import OpenAI
import os
import re
from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers import pipeline
from loguru import logger

os.environ["OPENAI_API_KEY"] = ""
device = 'cpu' # cuda

def get_models():
    llms = [OpenAI(temperature=temp) for temp in [0.2,0.4,0.6]]
    tokenizer = AutoTokenizer.from_pretrained("dslim/bert-large-NER")
    model = AutoModelForTokenClassification.from_pretrained("dslim/bert-large-NER")
    ner = pipeline("ner", model=model, tokenizer=tokenizer, device= device)
    
    prompt = """Parse the provided email data and generate the following information as a single line json:

* subject: The subject line of the provided email if present.
* timezone: The timezone of the provided email if available.
* length: The total number of words in the provided email body.
* year: The year extracted from the date header in the provided email.
* month: The month extracted from the date header in the provided email.
* recipients: The total number of recipients in the provided email.
* cc_participants: The number of participants in the CC field (0 if None) in the provided email.
* is_reply: A boolean value (0 or 1) indicating whether the email is a reply based on the presence of "Re:" in the provided email.
* summary: A concise summary of the email content, capturing key points like:
    * Reason for the email 
    * Actions proposed 
    * Any mentioned deadlines or timeframes.
* response: Generate an email response to the provided email in the same format as the input email from the perspective of the main recipient.

** For the response, please strictly adhere to the provided email format which contains Date, From, To, CC, Subject, Body.

**Provided Email:**

"""
    return llms, ner, prompt

def parse_email(email):
    email_dict = {}
    
    date_match = re.search(r"Date:\s*(.*)", email)
    from_match = re.search(r"From:\s*(.*)", email)
    to_match = re.search(r"To:\s*(.*)", email)
    subject_match = re.search(r"Subject:\s*(.*)", email)
    body_match = re.search(r"Body:\s*(.*)", email, re.DOTALL)
    
    email_dict['Date'] = date_match.group(1).strip() if date_match else ''
    email_dict['From'] = from_match.group(1).strip() if from_match else ''
    email_dict['To'] = to_match.group(1).strip() if to_match else ''
    email_dict['Subject'] = subject_match.group(1).strip() if subject_match else ''
    email_dict['Body'] = body_match.group(1).strip() if body_match else ''
    if email_dict['Subject'] in ['Body:', 'Re:']:
        email_dict['Subject'] = ''
        
    
    return email_dict

def count_recipients(text):
    return text.count("@")

def is_email_valid(email):
    if email is None or email == "":
        logger.error("Invalid Input.")
        return False
    recipients = parse_email(email)["To"]
    if not count_recipients(recipients):
        return False
    return True

def get_json(pred):
    pred = re.sub(r'(\n")', '"', pred)
    dictionary = json.loads(pred)
    return dictionary

def invoke_llm(llms, prompt, email_data, max_retries = 10):
    for llm in llms:
        for _ in range(max_retries):
            try:
                input_text = prompt + email_data
                llm_out = llm.invoke(input_text).lstrip("\n\n")
                llm_json = get_json(llm_out)
                llm_response = {"original_email": email_data, **llm_json}
                return llm_response
            except:
                pass
    return {}

def reconstruct(entities, original_sentence):
    reconstructed_entities = []
    current_entity = ""
    current_type = None
    start_index = None

    for e in entities:
        if e['entity'].startswith("B-"):
            if current_entity:
                reconstructed_entities.append({
                    "entity": current_type,
                    "word": current_entity,
                    "start": start_index,
                    "end": end_index
                })
            current_entity = e['word']
            current_type = e['entity'][2:]
            start_index = e['start']
            end_index = e['end']
        else:
            current_entity += e['word'].replace("##", "")
            end_index = e['end']

    if current_entity:
        reconstructed_entities.append({
            "entity": current_type,
            "word": current_entity,
            "start": start_index,
            "end": end_index
        })
    
    masked_sentence = original_sentence
    offset = 0

    for entity in reconstructed_entities:
        replacement = "<person>" if entity["entity"] == "PER" else "<org>"
        start = entity["start"] + offset
        end = entity["end"] + offset
        masked_sentence = masked_sentence[:start] + replacement + masked_sentence[end:]
        offset += len(replacement) - (end - start)
    
    masked_sentence = re.sub(r'(<person>)+', '<person>', masked_sentence)

    # Replace multiple adjacent <org> with a single <org>
    masked_sentence = re.sub(r'(<org>)+', '<org>', masked_sentence)


    return masked_sentence


def generate_masked_sentence(ner, response):
    ner_results = ner(response)
    return reconstruct(ner_results, response)

