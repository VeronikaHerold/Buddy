import random
import difflib
import json
import re

def buddy_mode(training_data_file="data/{}/training_data.json", theme=""):
    """
    Buddy-Mode mit zwei Modi: 'Fragen' und 'Test'.
    Im Fragen-Modus stellt die KI Fragen und gibt Feedback.
    Im Test-Modus bekommt der Benutzer Multiple-Choice-Fragen, die bewertet werden.
    """
    try:
        with open(training_data_file, "r", encoding="utf-8") as file:
            training_data = json.load(file)
    except FileNotFoundError:
        print(f"Keine Trainingsdaten in {training_data_file} gefunden.")
        return

    while True:
        mode = input("Wähle den Modus ('fragen', 'test', 'exit'): ").lower()

        if mode == 'exit':
            print("Buddy-Mode beendet.")
            break
        elif mode == 'fragen':
            fragen_modus(training_data, theme)
        elif mode == 'test':
            test_modus(training_data, theme)
        else:
            print("Ungültige Eingabe. Bitte wähle 'fragen', 'test' oder 'exit'.")


def fragen_modus(training_data, theme):
    """Stellt dem Benutzer Fragen und gibt Feedback basierend auf der Antwortähnlichkeit."""
    from training.feedback import save_feedback
    while True:
        question_entry = random.choice(training_data)
        question = question_entry["input"]
        correct_answer = question_entry["output"]

        print(f"Frage: {question}")
        user_answer = input("Deine Antwort (oder 'hint' für einen Tipp, 'exit' zum Beenden): ").strip()

        if user_answer.lower() == 'exit':
            print("Fragen-Modus beendet.")
            break

        if user_answer.lower() == 'hint':
            print(f"Tipp: Die Antwort beginnt mit: {correct_answer[0]}")
            continue

        if normalize_text(user_answer) == normalize_text(correct_answer):
            print("Deine Antwort ist korrekt!")
        else:
            print(f"Leider falsch. Die richtige Antwort wäre: {correct_answer}")

        while True:
            feedback = input("Wie bewertest du die Antwort? (1 = schlecht, 5 = perfekt): ")
            if feedback in {'1', '2', '3', '4', '5'}:
                feedback = int(feedback)
                break
            else:
                print("Ungültige Eingabe. Bitte gib eine Zahl zwischen 1 und 5 ein.")
        entry = {"input": question, "output": correct_answer, "reward": feedback}
        save_feedback(entry, theme)

def test_modus(training_data, theme):
    """Der Test-Modus stellt dem Benutzer Single- und Multiple-Choice-Fragen."""
    score = 0
    total_fragen = len(training_data)

    print(f"Test für das Thema: {theme}")

    for frage_entry in training_data:
        frage = frage_entry["input"]
        korrekt_antworten = frage_entry["output"].split(', ')
        print(f"\nFrage: {frage}")

        antwort_optionen = generate_answer_options(korrekt_antworten)
        for idx, option in enumerate(antwort_optionen, start=1):
            print(f"{idx}) {option}")

        user_antworten = input("Gib die richtigen Antwortnummern ein (z.B. '1' oder '1,2'): ").split(',')

        user_antworten = [antwort_optionen[int(num.strip()) - 1] for num in user_antworten]

        if set(user_antworten) == set(korrekt_antworten):
            print("Richtig!")
            score += 1
        else:
            print(f"Leider falsch. Die richtige(n) Antwort(en): {', '.join(korrekt_antworten)}")

    print(f"\nTest beendet! Du hast {score} von {total_fragen} Fragen richtig beantwortet.")
    note = calculate_grade(score, total_fragen)
    print(f"Deine Note: {note}")

def generate_answer_options(korrekt_antworten):
    """Erzeugt Antwortoptionen inklusive der korrekten Antworten."""
    alle_antworten = set(korrekt_antworten + ["Falsch1", "Falsch2", "Falsch3"])
    return list(alle_antworten)

def calculate_grade(score, total_fragen):
    """Berechnet die Note basierend auf dem Ergebnis des Tests."""
    prozent = (score / total_fragen) * 100
    if prozent >= 90:
        return "1"
    elif prozent >= 80:
        return "2"
    elif prozent >= 70:
        return "3"
    elif prozent >= 60:
        return "4"
    else:
        return "5"

def normalize_text(text):
    return re.sub(r'\s+', ' ', text.strip().lower())