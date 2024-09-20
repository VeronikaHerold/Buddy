import torch
import json
from transformers import GPT2LMHeadModel, GPT2Tokenizer
# In responses.py
from models.ner_processing import extract_entities

def generate_response_with_ner(prompt):
    """
    Generiert eine Antwort unter Verwendung von NER.
    """
    entities = extract_entities(prompt)
    # Hier kannst du die extrahierten Entitäten nutzen, um die Antwort zu verbessern oder anzupassen
    # Beispiel: Eine spezielle Antwort generieren, wenn eine bestimmte Entität erkannt wird
    response = "Eine spezielle Antwort für " + entities[0][0]
    return response

# Modell und Tokenizer initialisieren
with open("config/config.json", "r") as config_file:
    config = json.load(config_file)

model_name = config["model_name"]
model = GPT2LMHeadModel.from_pretrained(model_name)
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model.eval()

# Funktion zur Generierung von Antworten
def generate_response(prompt):
    inputs = tokenizer.encode(prompt, return_tensors="pt")
    outputs = model.generate(inputs, max_length=100, do_sample=True, top_p=0.95, top_k=60)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response
