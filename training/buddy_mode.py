import random
import re
from models.responses import create_response_generator, generate_response_with_ner, save_response_to_file
from training.feedback import prepare_data_with_feedback
from _main.utils import speak

def buddy_mode(training_data_file, feedback_data_file, theme):
    while True:
        mode = input("Wähle den Modus ('fragen', 'test', 'exit'): ").lower()

        if mode == 'exit':
            message = "Buddy-Mode beendet."
            speak(message)
            print(message)
            break
        elif mode in {'fragen', 'test'}:
            try:
                training_data = prepare_data_with_feedback(training_data_file, feedback_data_file)
            except FileNotFoundError as e:
                message = str(e)
                speak(message)
                print(message)
                return

            if mode == 'fragen':
                question_mode(training_data, theme)
            elif mode == 'test':
                test_mode(training_data, theme)
        else:
            message = "Ungültige Eingabe. Bitte wähle 'fragen', 'test' oder 'exit'."
            speak(message)
            print(message)

def question_mode(training_data, theme):
    from training.feedback import save_feedback

    response_generator = create_response_generator()

    while True:
        question_entry = random.choice(training_data)
        question = question_entry["input"]
        correct_answer = question_entry["output"]
        message = f"Frage: {question}"
        speak(message)
        print(message)
        while True:
            user_answer = input("Deine Antwort (oder 'hint' für einen Tipp, 'exit' zum Beenden): ").strip()
            if user_answer.lower() == 'exit':
                message = "Fragen-Modus beendet."
                speak(message)
                print(message)
                return
            if user_answer.lower() == 'hint':
                message = f"Tipp: Die Antwort beginnt mit: {correct_answer[0]}"
                speak(message)
                print(message)
                continue
            ai_response = generate_response_with_ner(question, response_generator)
            save_response_to_file(ai_response, 'responses.txt') 
            message = f"Buddy:"
            speak(message)
            print(message)
            if normalize_text(user_answer) == normalize_text(correct_answer):
                message = "Deine Antwort ist korrekt!"
                speak(message)
                print(message)
            else:
                message = f"Leider falsch. Die richtige Antwort wäre: {correct_answer}"
                speak(message)
                print(message)
            break
        while True:
            feedback = input("Wie bewertest du die Antwort? (1 = schlecht, 5 = perfekt): ")
            if feedback in {'1', '2', '3', '4', '5'}:
                feedback = int(feedback)
                break
            else:
                message = "Ungültige Eingabe. Bitte gib eine Zahl zwischen 1 und 5 ein."
                speak(message)
                print(message)
        entry = {"input": question, "output": correct_answer, "reward": feedback}
        save_feedback(entry, theme)    
def test_mode(training_data, theme):
    score = 0
    total_questions = len(training_data)
    message = f"Test für das Thema: {theme}"
    speak(message)
    print(message)
    for question_entry in training_data:
        question = question_entry["input"]
        correct_answers = question_entry["output"].split(', ')
        message = f"\nFrage: {question}"
        speak(message)
        print(message)
        answer_options = generate_answer_options(correct_answers, training_data)
        for idx, option in enumerate(answer_options, start=1):
            message = f"{idx}) {option}"
            speak(message)
            print(message)
        user_answers = input("Gib die richtigen Antwortnummern ein (z.B. '1' 2' '3' '4' usw.): ").split(',')
        user_answers = [num.strip() for num in user_answers if num.strip().isdigit()]
        try:
            user_answers = [answer_options[int(num) - 1] for num in user_answers]
        except (ValueError, IndexError):
            message = "Ungültige Eingabe. Bitte gib gültige Antwortnummern ein."
            speak(message)
            print(message)
            continue
        if set(user_answers) == set(correct_answers):
            message = "Richtig!"
            speak(message)
            print(message)
            score += 1
        else:
            message = f"Leider falsch. Die richtige(n) Antwort(en): {', '.join(correct_answers)}"
            speak(message)
            print(message)
    message = f"\nTest beendet! Du hast {score} von {total_questions} Fragen richtig beantwortet."
    speak(message)
    print(message)
    grade = calculate_grade(score, total_questions)
    message = f"Deine Note: {grade}"
    speak(message)
    print(message)

def generate_answer_options(correct_answers, training_data):

    other_answers = [entry["output"] for entry in training_data if entry["output"] not in correct_answers]
    other_answers = list(set(other_answers))  
    random.shuffle(other_answers)
    wrong_answers = other_answers[:3] 
    all_answers = set(correct_answers + wrong_answers)
    return list(all_answers)

def calculate_grade(score, total_questions):

    percentage = (score / total_questions) * 100
    if percentage >= 90:
        return "1"
    elif percentage >= 80:
        return "2"
    elif percentage >= 70:
        return "3"
    elif percentage >= 60:
        return "4"
    else:
        return "5"

def normalize_text(text):
    return re.sub(r'\s+', ' ', text.strip().lower())

