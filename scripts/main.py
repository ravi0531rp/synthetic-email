import os
from loguru import logger
from utils import is_email_valid, invoke_llm, generate_masked_sentence, get_models


def process(llms, ner, email, prompt):
    
    if not is_email_valid(email):
        logger.error("Invalid Email. Please check again if it's empty or there's no recipient")
    
    output = invoke_llm(llms, prompt, email)
    logger.debug(output)
    masked_original_sentence = generate_masked_sentence(ner, output["original_email"])
    masked_response_sentence = generate_masked_sentence(ner, output["response"])
    logger.info("Masked Original Email")
    logger.debug(masked_original_sentence)
    logger.info("Masked Response Email")
    logger.debug(masked_response_sentence)

if __name__ == "__main__":
    llms, ner, prompt = get_models()
    input_email = "Date: Mon, 5 Feb 2001 03:26:00 -0800 (PST)\nFrom: errol.mclaughlin@enron.com\nTo: jeffrey.gossett@enron.com\nSubject: Re: G-Daily-Est book deals to be flipped updated list\nBody: \nLuchas has been working on them and will be finished within the hour.\nErrol"
    process(llms, ner, input_email, prompt)
