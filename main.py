from _main.theme_manager import create_theme_directory
from training.training import training_mode
from training.buddy_mode import buddy_mode
from _main.utils import remind_user, reset_timer
import tkinter as tk
import threading

def main():
    reminder_thread = threading.Thread(target=remind_user, daemon=True)
    reminder_thread.start()
    
    print("Willkommen zum KI-Lern-Buddy!")
    
    try:
        theme = input("Wähle ein Thema oder erstelle ein neues: ").strip()
        feedback_file_path, training_file_path = create_theme_directory(theme)
        reset_timer()  
    except KeyboardInterrupt:
        print("\nProgramm wurde durch Benutzerabbruch (Strg+C) beendet.")
        exit(0)
    
    while True:
        mode = input("Wähle einen Modus ('training', 'buddy', 'exit'): ").strip().lower()
        reset_timer()  

        if mode == 'training':
            training_mode(training_file_path, theme)
        elif mode == 'buddy':
            buddy_mode(training_file_path, theme) 
        elif mode == 'exit':
            print("Auf Wiedersehen!")
            break
        else:
            print("Ungültige Eingabe. Bitte 'training', 'buddy' oder 'exit' eingeben.")
            print("Hinweis: Verwende die genauen Begriffe")

if __name__ == "__main__":
    main()
