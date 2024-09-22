from _main.theme_manager import create_theme_directory
from training.training import training_mode
from training.buddy_mode import buddy_mode
from _main.utils import remind_user, reset_timer
import threading
from _main.utils import speak

def main():
    reminder_thread = threading.Thread(target=remind_user, daemon=True)
    reminder_thread.start()
    
    welcome_message = "Willkommen zum KI-Lern-Buddy!"
    print(welcome_message)
    speak(welcome_message)
    
    try:
        theme_prompt = "Wähle ein Thema oder erstelle ein neues: "
        speak(theme_prompt)
        theme = input(theme_prompt).strip()
        speak(f"Du hast das Thema {theme} gewählt.")
        feedback_file_path, training_file_path = create_theme_directory(theme)
        reset_timer()  
    except KeyboardInterrupt:
        exit_message = "\nProgramm wurde durch Benutzerabbruch (Strg+C) beendet."
        print(exit_message)
        speak(exit_message)
        exit(0)
    
    while True:
        mode_prompt = "Wähle einen Modus ('training', 'buddy', 'exit'): "
        speak(mode_prompt)
        mode = input(mode_prompt).strip().lower()
        reset_timer()  

        if mode == 'training':
            training_message = "Du hast den Trainingsmodus gewählt."
            print(training_message)
            speak(training_message)
            training_mode(training_file_path, theme)
        elif mode == 'buddy':
            buddy_message = "Du hast den Buddy-Modus gewählt."
            print(buddy_message)
            speak(buddy_message)
            buddy_mode(training_file_path, feedback_file_path, theme) 
        elif mode == 'exit':
            goodbye_message = "Auf Wiedersehen!"
            print(goodbye_message)
            speak(goodbye_message)
            break
        else:
            invalid_input_message = "Ungültige Eingabe, verwende die genauen Begriffe."
            print(invalid_input_message)
            speak(invalid_input_message)

if __name__ == "__main__":
    main()