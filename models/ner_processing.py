from typing import List, Tuple
import spacy
nlp = spacy.load("de_core_news_sm")

def extract_entities(text: str) -> List[Tuple[str, str]]:

    if not text: raise ValueError('Input text cannot be empty or None')
    
    try:
        doc = nlp(text)
    except Exception as e:
        raise ValueError('Error processing text with NLP model') from e
    
    entities = [(ent.text, ent.label_) for ent in doc.ents]
    return entities
