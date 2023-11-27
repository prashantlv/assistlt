import os
import spacy
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from langchain import HuggingFaceHub, PromptTemplate, LLMChain

os.environ['API_KEY'] = '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'

model_id = 'tiiuae/falcon-7b-instruct'
analyzer = SentimentIntensityAnalyzer()

falcon_llm = HuggingFaceHub(huggingfacehub_api_token=os.environ['API_KEY'],
                            repo_id=model_id,
                            model_kwargs={"temperature":0.8,"max_new_tokens":2000})

template = """

You are a interactive conversational chatbot that actively engages users while discreetly handling data collection. You need to get basic contact information from user to send them invoice, example Name, email, contact number and mailing address. Below are the example of chat for refrence to understand how to respond:

User: "I'm not sure if I want to share my email."
Chatbot: "I completely understand your concern. Your email is safe with us and will only beused for sending updates on our services and exclusive offers. Is there anything specific you'reconcerned about?"
User: "I'm not comfortable sharing my phone number."
Chatbot: "Your privacy is important to us. Rest assured, we take all necessary precautions to protect your data. If you're not comfortable sharing your phone, is there another way we can stay in touch with you?"

Now you may interact with the user.
{question}

"""
prompt = PromptTemplate(template=template, input_variables=['question'])

falcon_chain = LLMChain(llm=falcon_llm,
                        prompt=prompt,
                        verbose=False)


while True:
    user_conversation = input("User: ")

    if user_conversation.lower() == 'quit':
        print("Exiting the program. Goodbye!")
        break

    response = falcon_chain.run(f"{user_conversation}")
    vs = analyzer.polarity_scores(user_conversation)
    print(f"score = {['Pos' if vs.get('compound') > 0.05 else 'N'}]")
    print(response)


