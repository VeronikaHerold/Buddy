import json

def training_mode(filename="data/training_data.json"):
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
            with open(filename, "r+", encoding="utf-8") as file:
                data = json.load(file)
                data.append(entry)
                file.seek(0)
                json.dump(data, file, indent=4, ensure_ascii=False)
        except FileNotFoundError:
            with open(filename, "w", encoding="utf-8") as file:
                json.dump([entry], file, indent=4, ensure_ascii=False)

        print(f"Die Frage-Antwort-Paar wurde in {filename} gespeichert!")
