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
        tokens = [self.vocab.get(word, self.vocab["<UNK>"]) for word in text.split()]
        return tokens

    def encode(self, text, return_tensors='pt'):
        if not isinstance(text, str):
            raise ValueError("Text muss ein String sein.")

        tokens = self.tokenize(text)
        tokens = tokens[:self.max_length]
        tokens += [self.vocab["<PAD>"]] * (self.max_length - len(tokens))
        
        if return_tensors == 'pt':
            return torch.tensor(tokens, dtype=torch.float32).unsqueeze(0)
        
        return tokens

tokenizer = CustomTokenizer(vocab, max_length=100)

class ResponseGenerator:
    def __init__(self, model, tokenizer, id_to_word):
        self.model = model
        self.tokenizer = tokenizer
        self.id_to_word = id_to_word

    def decode_output(self, output):
        output_ids = output.argmax(dim=-1).squeeze().tolist()
        if isinstance(output_ids, int):
            output_ids = [output_ids]
        decoded_words = [self.id_to_word.get(token_id, "<UNK>") for token_id in output_ids]
        decoded_sentence = ' '.join(decoded_words).replace("<PAD>", "").strip()
        return decoded_sentence

def generate_response_with_ner(question, response_generator):
    entities = extract_entities(question)
    tokenized_input = response_generator.tokenizer.encode(question)
    
    with torch.no_grad():
        tokenized_input = tokenized_input.float() 
        generated_response_tensor = response_generator.model(tokenized_input)
    
    generated_response = response_generator.decode_output(generated_response_tensor)
    
    if entities:
        entity_text = ', '.join([ent[0] for ent in entities])
        generated_response = f"{generated_response} Ich habe folgende Entitäten erkannt: {entity_text}."
    
    return generated_response

def create_response_generator():
    model = Seq2SeqModel(input_size=input_size, hidden_size=hidden_size, output_size=output_size)
    return ResponseGenerator(model, tokenizer, id_to_word)

def save_response_to_file(response, file_path):
    with open(file_path, 'a', encoding='utf-8') as file:
        file.write(response + '\n')