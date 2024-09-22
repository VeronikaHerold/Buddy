import tkinter as tk
import time
import winsound
import pyttsx3

tts_engine = pyttsx3.init()

def speak(text):
    tts_engine.say(text)
    tts_engine.runAndWait()
def prompt_input(prompt_text):
    speak(prompt_text)
    return input(prompt_text).strip()


TIMEOUT = 10 * 60
last_activity_time = time.time()

def remind_user():
    global last_activity_time
    while True:
        time_elapsed = time.time() - last_activity_time
        if time_elapsed >= TIMEOUT:
            show_popup()
        time.sleep(60)

def show_popup():

    winsound.Beep(1000, 500)
    root = tk.Tk()
    root.withdraw()
    tk.messagebox.showwarning("Erinnerung", "Hör auf zu zocken, lern jetzt weiter!")
    root.destroy()

def reset_timer():
    global last_activity_time
    last_activity_time = time.time()
