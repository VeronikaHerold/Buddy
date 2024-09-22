import json
from _main.utils import speak, prompt_input


def training_mode(training_data_file="data/training_data.json", theme=""):
    while True:
        question = prompt_input("Gib eine Frage ein (oder 'exit' zum Beenden): ").strip()
        if question.lower() == 'exit':
            break
        
        if not question:
            message = "Die Frage darf nicht leer sein. Versuche es erneut."
            speak(message)
            print(message)
            continue

        answer = prompt_input("Gib die richtige Antwort ein: ").strip()
        if not answer:
            message = "Die Antwort darf nicht leer sein. Versuche es erneut."
            speak(message)
            print(message)
            continue

        entry = {"input": question, "output": answer}

        try:
            with open(training_data_file, "r+", encoding="utf-8") as file:
                data = json.load(file)
                data.append(entry)
                file.seek(0)
                json.dump(data, file, indent=4, ensure_ascii=False)
            message = f"Die Frage-Antwort-Paar wurde in {training_data_file} gespeichert!"
            speak(message)
            print(message)

        except FileNotFoundError:
            message = f"Datei {training_data_file} nicht gefunden. Eine neue Datei wird erstellt."
            speak(message)
            print(message)
            with open(training_data_file, "w", encoding="utf-8") as file:
                json.dump([entry], file, indent=4, ensure_ascii=False)

        except json.JSONDecodeError:
            message = f"Ungültige Daten in {training_data_file}. Datei wird neu erstellt."
            speak(message)
            print(message)
            with open(training_data_file, "w", encoding="utf-8") as file:
                json.dump([entry], file, indent=4, ensure_ascii=False)

        except Exception as e:
            message = f"Fehler beim Speichern der Frage-Antwort-Paare: {e}"
            speak(message)
            print(message)