# ner_processing.py
import spacy

# Laden des vortrainierten SpaCy-Modells für NER (z.B. "de_core_news_sm" für Deutsch)
nlp = spacy.load("de_core_news_sm")

def extract_entities(text):
    """
    Extrahiert benannte Entitäten aus dem gegebenen Text.
    """
    doc = nlp(text)
    entities = []
    for ent in doc.ents:
        entities.append((ent.text, ent.label_))
    return entities
