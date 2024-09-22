import spacy
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
