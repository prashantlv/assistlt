import os
import spacy
from Sentiment import SentimentIntensityAnalyzer
from langchain import HuggingFaceHub, PromptTemplate, LLMChain

api_key = os.environ.get('HF_API_KEY')

# Check if the API key is set
if api_key is None:
    raise ValueError("API_KEY is not set in the environment variables.")

NER = spacy.load("en_core_web_sm")
model_id = 'tiiuae/falcon-7b-instruct'

# Sentiment Analyzer
def analyze_sentiment(text):
    analyzer = SentimentIntensityAnalyzer()
    vs = analyzer.polarity_scores(text)
    return 'Pos' if vs.get('compound') > 0.0 else 'N'

# Named Entity Recognition (NER)
def extract_entities(NER, text):
    doc = NER(text)
    entities = []

    for ent in doc.ents:
        entities.append((ent.text, ent.label_))

    return entities

falcon_llm = HuggingFaceHub(huggingfacehub_api_token=os.environ['HF_API_KEY '],
                            repo_id=model_id,
                            model_kwargs={"temperature":0.8,"max_new_tokens":2000})

template = """

You are a interactive conversational chatbot that actively engages users while discreetly handling data collection. You need to get basic contact information from user to send them invoice, example Name, email, contact number and mailing address. Below are the example of chat for refrence to understand how to respond:

Tone: Friendly, casual, funny, human-like and use spoken english.
Now you may interact with the user.
{question}

"""
prompt = PromptTemplate(template=template, input_variables=['question'])

falcon_chain = LLMChain(llm=falcon_llm,
                        prompt=prompt,
                        verbose=False)

while True:
    user_conversation = input("User: ")

    # Check sentiment
    sentiment = analyze_sentiment(user_conversation)
    print(f"Sentiment = {sentiment}")

    # Extract entities
    entities = extract_entities(NER, user_conversation)

    for word, label in entities:
        print(f"- {word} - {label}")    #spacy.explain(label)
    
    if user_conversation.lower() == 'quit':
        print("Exiting the program. Goodbye!")
        break

    response = falcon_chain.run(f"{user_conversation}")
    print(f"Bot: {response.replace('<p>','').replace('</p>','')}")


# User: "I'm not sure if I want to share my email."
# You: "I completely understand your concern. Your email is safe with us and will only beused for sending updates on our services and exclusive offers. Is there anything specific you'reconcerned about?"
# User: "I'm not comfortable sharing my phone number."
# You: "Your privacy is important to us. Rest assured, we take all necessary precautions to protect your data. If you're not comfortable sharing your phone, is there another way we can stay in touch with you?"