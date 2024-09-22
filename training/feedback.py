import logging
import os
from typing import List, Dict
from functools import lru_cache
import json
import random
from nltk.corpus import wordnet
from models.ner_processing import extract_entities
from _main.utils import speak


def load_feedback_data(filepath: str) -> dict:
    if os.path.exists(filepath):
        try:
            with open(filepath, "r") as feedback_file:
                feedback_data = json.load(feedback_file)
                return feedback_data
        except json.JSONDecodeError:
            logging.error(f"Error decoding JSON file {filepath}.")
            return {}
    else:
        logging.error(f"Feedback-Datei {filepath} nicht gefunden.")
        return {}

def append_feedback_data(new_feedback, filepath: str):
    feedback_data = load_feedback_data(filepath)
    feedback_data.append(new_feedback)
    with open(filepath, "w") as feedback_file:
        json.dump(feedback_data, feedback_file)

@lru_cache
def check_file_existence(filepath):
    return os.path.exists(filepath)

def archive_feedback(filename, archive_filename, max_entries=200):
    try:
        data = []
        if check_file_existence(filename) and os.stat(filename).st_size > 0:
            with open(filename, "r", encoding="utf-8") as file:
                data = json.load(file)
        if len(data) > max_entries:
            archive_data = []
            if check_file_existence(archive_filename):
                with open(archive_filename, "r", encoding="utf-8") as archive_file:
                    archive_data = json.load(archive_file)
            archive_data.extend(data[:len(data) - max_entries])
            with open(archive_filename, "w", encoding="utf-8") as archive_file:
                json.dump(archive_data, archive_file, indent=4, ensure_ascii=False)
            with open(filename, "w", encoding="utf-8") as file:
                json.dump(data[-max_entries:], file, indent=4, ensure_ascii=False)
            logging.info(f"Feedback archiviert. Ältere Daten in {archive_filename} verschoben.")
    except (OSError, ValueError) as e:
        logging.error(f"Error occurred: {e}")

def save_feedback(entry: dict, theme: str, filename_format: str = "data/{}/{}_feedback_data.json") -> None:
    feedback_file_path = filename_format.format(theme, theme)
    archive_file_path = filename_format.format(theme, f"{theme}_archive")
    archive_feedback(feedback_file_path, archive_file_path)  
    try:
        if os.path.exists(feedback_file_path) and os.stat(feedback_file_path).st_size > 0:
            with open(feedback_file_path, "r+", encoding="utf-8") as file:
                data = json.load(file)
                data.append(entry)
                file.seek(0)
                json.dump(data, file, indent=4, ensure_ascii=False)
        else:
            with open(feedback_file_path, "w", encoding="utf-8") as file:
                json.dump([entry], file, indent=4, ensure_ascii=False)
        logging.info(f"Feedback in {feedback_file_path} gespeichert!")
    except Exception as e:
        logging.error(f"Fehler beim Speichern des Feedbacks: {e}")

def augment_with_synonyms(words: List[str]) -> str:
    new_text = []
    for word in words:
        synonyms = wordnet.synsets(word)
        if synonyms:
            synonym = random.choice(synonyms).lemmas()[0].name()
            new_text.append(synonym)
        else:
            new_text.append(word)
    return " ".join(new_text)

def prepare_data_with_feedback(training_data_file: str, feedback_file: str) -> List[Dict[str, str]]:
    if os.path.exists(training_data_file):
        with open(training_data_file, "r", encoding="utf-8") as file:
            training_data = json.load(file)
    else:
        logging.error(f"Keine Trainingsdaten gefunden.")
        training_data = []

    if os.path.exists(feedback_file):
        with open(feedback_file, "r", encoding="utf-8") as file:
            feedback_data = json.load(file)
    else:
        logging.error(f"Keine Feedbackdaten gefunden.")
        feedback_data = []

    combined_data = training_data + feedback_data
    augmented_data = []
    for entry in combined_data:
        augmented_data.append(entry)
        augmented_entry = {
            "input": augment_with_synonyms(entry["input"].split()),
            "output": entry["output"]
        }
        augmented_data.append(augmented_entry)
    return augmented_data
