import torch
from models.ner_processing import extract_entities
from models.custom_model import Seq2SeqModel 

vocab = {"<PAD>": 0, "<UNK>": 1} 
id_to_word = {id: word for word, id in vocab.items()}


input_size = 100  
hidden_size = 256  
output_size = 100  

class CustomTokenizer:
    def __init__(self, vocab, max_length=100):
        self.vocab = vocab
        self.max_length = max_length

    def tokenize(self, text):
        """
        Tokenisiert den Text basierend auf dem Vokabular.
        """
        tokens = [self.vocab.get(word, self.vocab["<UNK>"]) for word in text.split()]
        return tokens

    def encode(self, text, return_tensors='pt'):
        """
        Tokenisiert den Text und fügt Padding hinzu, um die Länge anzupassen.
        """
        if not isinstance(text, str):
            raise ValueError("Text muss ein String sein.")

        tokens = self.tokenize(text)
        tokens = tokens[:self.max_length] 
        tokens += [self.vocab["<PAD>"]] * (self.max_length - len(tokens))
        
        if return_tensors == 'pt':
            return torch.tensor(tokens, dtype=torch.long).unsqueeze(0) 
        
        return tokens 
    
tokenizer = CustomTokenizer(vocab, max_length=100) 

class ResponseGenerator:
    def __init__(self, model, tokenizer, id_to_word):
        self.model = model
        self.tokenizer = tokenizer
        self.id_to_word = id_to_word

    def decode_output(self, output):
        output_ids = output.argmax(dim=-1).squeeze().tolist()
        decoded_words = [self.id_to_word.get(token_id, "<UNK>") for token_id in output_ids]
        decoded_sentence = ' '.join(decoded_words).replace("<PAD>", "").strip()
        return decoded_sentence



from models.ner_processing import extract_entities

def generate_response_with_ner(question):
    """
    Generiert eine Antwort und extrahiert benannte Entitäten mit NER.
    """
    entities = extract_entities(question)
    tokenized_input = tokenizer.encode(question)
    tokenized_input = tokenized_input.clone().detach()  
    tokenized_input = tokenized_input.unsqueeze(0)  
    tokenized_input = tokenized_input.float()
    with torch.no_grad():
        generated_response = model(tokenized_input)  
    if entities:
        entity_text = ', '.join([ent[0] for ent in entities])
        generated_response = f"{generated_response} Ich habe folgende Entitäten erkannt: {entity_text}."
    
    return generated_response

model = Seq2SeqModel(input_size=input_size, hidden_size=hidden_size, output_size=output_size)
response_generator = ResponseGenerator(model, tokenizer, id_to_word)
