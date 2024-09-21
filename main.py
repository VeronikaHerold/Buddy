from training.training import training_mode
from training.fine_tune import fine_tune_gpt2_with_feedback
from training.feedback import buddy_mode

def main():
    print("Willkommen zum KI-Lern-Buddy!")
    
    try:
        theme = input("Wähle ein Thema oder erstelle ein neues: ").strip()
        theme_file = f"data/{theme}_training_data.json"
    except KeyboardInterrupt:
        print("\nProgramm wurde durch Benutzerabbruch (Strg+C) beendet.")
        exit(0)  
    theme_file = f"data/{theme}_training_data.json"
    
    while True:
        mode = input("Wähle einen Modus ('training', 'buddy', 'exit'): ").strip().lower()

        if mode == 'training':
            training_mode(theme_file)
        elif mode == 'buddy':
            buddy_mode(theme_file)
        elif mode == 'exit':
            print("Auf Wiedersehen!")
            break
        else:
            print("Ungültige Eingabe. Bitte 'training', 'buddy' oder 'exit' eingeben.")

if __name__ == "__main__":
    main()
