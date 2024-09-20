import os
import json
import random
import difflib
import random
from nltk.corpus import wordnet
from models.ner_processing import extract_entities

def analyze_feedback_with_ner(feedback_text):
    """
    Analysiert das Feedback unter Verwendung von NER.
    """
    entities = extract_entities(feedback_text)
    # Hier könntest du die extrahierten Entitäten weiter verarbeiten oder analysieren
    return entities

def archive_feedback(filename="data/feedback_data.json", archive_filename="data/feedback_archive.json", max_entries=100):
    """
    Archiviert ältere Feedback-Daten, wenn die Datei eine bestimmte Größe überschreitet.
    """
    if os.path.exists(filename) and os.stat(filename).st_size > 0:
        with open(filename, "r", encoding="utf-8") as file:
            data = json.load(file)
        
        # Prüfen, ob die Anzahl der Einträge das Limit überschreitet
        if len(data) > max_entries:
            archive_data = []
            
            # Wenn es bereits archivierte Daten gibt, diese laden
            if os.path.exists(archive_filename):
                with open(archive_filename, "r", encoding="utf-8") as archive_file:
                    archive_data = json.load(archive_file)
            
            # Älteste Einträge archivieren
            archive_data.extend(data[:len(data) - max_entries])
            with open(archive_filename, "w", encoding="utf-8") as archive_file:
                json.dump(archive_data, archive_file, indent=4, ensure_ascii=False)
            
            # Feedback-Datei kürzen
            with open(filename, "w", encoding="utf-8") as file:
                json.dump(data[-max_entries:], file, indent=4, ensure_ascii=False)

            print(f"Feedback archiviert. Ältere Daten in {archive_filename} verschoben.")
            
def save_feedback(entry, filename="data/feedback_data.json"):
    """
    Speichert das Feedback und archiviert ältere Daten.
    """
    archive_feedback()  # Feedback archivieren, falls nötig

    if os.path.exists(filename) and os.stat(filename).st_size > 0:
        try:
            with open(filename, "r+", encoding="utf-8") as file:
                data = json.load(file)
                data.append(entry)
                file.seek(0)
                json.dump(data, file, indent=4, ensure_ascii=False)
        except json.JSONDecodeError:
            print(f"Ungültige Daten in {filename}. Datei wird neu erstellt.")
            with open(filename, "w", encoding="utf-8") as file:
                json.dump([entry], file, indent=4, ensure_ascii=False)
    else:
        with open(filename, "w", encoding="utf-8") as file:
            json.dump([entry], file, indent=4, ensure_ascii=False)

    print(f"Feedback in {filename} gespeichert!")

def augment_with_synonyms(text):
    """
    Ersetzt zufällige Wörter durch Synonyme, um das Training zu erweitern.
    """
    words = text.split()
    new_text = []
    
    for word in words:
        synonyms = wordnet.synsets(word)
        if synonyms:
            synonym = random.choice(synonyms).lemmas()[0].name()
            new_text.append(synonym)
        else:
            new_text.append(word)
    
    return " ".join(new_text)

def load_feedback_data(filepath="data/feedback_data.json"):
    try:
        with open(filepath, "r") as feedback_file:
            feedback_data = json.load(feedback_file)
        return feedback_data
    except FileNotFoundError:
        print(f"Feedback-Datei {filepath} nicht gefunden.")
        return []
    
# Augmentierung während der Datenaufbereitung in prepare_data_with_feedback:
def prepare_data_with_feedback(training_data_file="data/training_data.json", feedback_file="data/feedback_data.json"):
    """
    Bereitet Trainingsdaten vor und kombiniert sie mit Feedback-Daten. 
    Falls keine Dateien vorhanden sind, werden Standardfragen verwendet. 
    Außerdem wird eine Datenaugmentierung durch Synonym-Ersetzungen durchgeführt.
    """
    # Standarddaten, falls keine Trainingsdaten vorhanden sind
    default_training_data = [
        {"input": "Was ist das Kreuzprodukt?", "output": "A X B - Die beiden Vektoren definieren ein Parallelogramm."}
    ]
    
    # Versuche, die Trainingsdaten zu laden, oder verwende Standarddaten
    try:
        with open(training_data_file, "r", encoding="utf-8") as file:
            training_data = json.load(file)
    except FileNotFoundError:
        print(f"Keine Trainingsdaten gefunden. Standardfragen werden genutzt.")
        training_data = default_training_data

    # Versuche, Feedbackdaten zu laden, oder verwende eine leere Liste
    try:
        with open(feedback_file, "r", encoding="utf-8") as file:
            feedback_data = json.load(file)
    except FileNotFoundError:
        print(f"Keine Feedbackdaten gefunden.")
        feedback_data = []

    # Kombiniere Trainingsdaten mit Feedback-Daten
    combined_data = training_data + feedback_data

    # Augmentiere Daten durch Synonym-Ersetzungen
    augmented_data = []
    for entry in combined_data:
        augmented_data.append(entry)
        augmented_entry = {
            "input": augment_with_synonyms(entry["input"]),
            "output": entry["output"]
        }
        augmented_data.append(augmented_entry)

    return augmented_data
def buddy_mode(training_data_file="data/training_data.json"):
    """
    Buddy-Mode: Die KI stellt Fragen, der Benutzer antwortet, und es wird überprüft,
    ob die Antwort richtig ist. Das Feedback wird basierend auf der Antwortähnlichkeit gegeben, mit Tipp-Funktion.
    """
    try:
        with open(training_data_file, "r", encoding="utf-8") as file:
            training_data = json.load(file)
    except FileNotFoundError:
        print(f"Keine Trainingsdaten in {training_data_file} gefunden.")
        return
    
    while True:
        question_entry = random.choice(training_data)
        question = question_entry["input"]
        correct_answer = question_entry["output"]

        print(f"Frage: {question}")
        user_answer = input("Deine Antwort (oder 'hint' für einen Tipp, 'exit' zum Beenden): ").strip()

        if user_answer.lower() == 'exit':
            print("Buddy-Mode beendet.")
            break

        if user_answer.lower() == 'hint':
            print(f"Tipp: Die Antwort beginnt mit: {correct_answer[0]}")
            continue

        # Antworten vergleichen
        similarity = difflib.SequenceMatcher(None, correct_answer.lower(), user_answer.lower()).ratio()

        # Feedback geben
        if similarity == 1.0:
            print("Deine Antwort ist 100% korrekt!")
            feedback_message = "100% korrekt"
        elif similarity > 0.85:
            print("Deine Antwort ist fast ganz richtig!")
            feedback_message = "Fast ganz richtig"
        elif similarity > 0.5:
            print("Deine Antwort ist teilweise richtig.")
            feedback_message = "Teilweise richtig"
        else:
            print(f"Leider ist deine Antwort falsch. Die richtige Antwort wäre: {correct_answer}")
            feedback_message = "Falsch"

        # Benutzer-Feedback erfassen (Skala von 1 bis 5)
        while True:
            try:
                feedback = int(input("Wie bewertest du die Antwort? (1 = schlecht, 5 = perfekt): "))
                if 1 <= feedback <= 5:
                    break
                else:
                    print("Bitte eine Zahl zwischen 1 und 5 eingeben.")
            except ValueError:
                print("Ungültige Eingabe. Bitte eine Zahl zwischen 1 und 5 eingeben.")

        # Feedback speichern
        entry = {"input": question, "output": correct_answer, "similarity": feedback_message, "reward": feedback}
        save_feedback(entry)

