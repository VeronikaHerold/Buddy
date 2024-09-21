import json

def training_mode(training_data_file="data/training_data.json", theme=""):
    while True:
        question = input("Gib eine Frage ein (oder 'exit' zum Beenden): ").strip()
        if question.lower() == 'exit':
            break

        if not question:
            print("Die Frage darf nicht leer sein. Versuche es erneut.")
            continue

        answer = input("Gib die richtige Antwort ein: ").strip()
        if not answer:
            print("Die Antwort darf nicht leer sein. Versuche es erneut.")
            continue

        entry = {"input": question, "output": answer}

        try:
            with open(training_data_file, "r+", encoding="utf-8") as file:
                data = json.load(file)
                data.append(entry)
                file.seek(0)
                json.dump(data, file, indent=4, ensure_ascii=False)
            print(f"Die Frage-Antwort-Paar wurde in {training_data_file} gespeichert!")

        except FileNotFoundError:
            print(f"Datei {training_data_file} nicht gefunden. Eine neue Datei wird erstellt.")
            with open(training_data_file, "w", encoding="utf-8") as file:
                json.dump([entry], file, indent=4, ensure_ascii=False)

        except json.JSONDecodeError:
            print(f"Ungültige Daten in {training_data_file}. Datei wird neu erstellt.")
            with open(training_data_file, "w", encoding="utf-8") as file:
                json.dump([entry], file, indent=4, ensure_ascii=False)

        except Exception as e:
            print(f"Fehler beim Speichern der Frage-Antwort-Paare: {e}")
