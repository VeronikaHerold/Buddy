import tkinter as tk
import threading
import time
import winsound

# Timer (15 Minuten)
TIMEOUT = 15 * 60
last_activity_time = time.time()

def remind_user():
    """Erinnert den Benutzer alle 15 Minuten."""
    global last_activity_time
    while True:
        time_elapsed = time.time() - last_activity_time
        if time_elapsed >= TIMEOUT:
            show_popup()
        time.sleep(60)

def show_popup():
    """Zeigt ein Popup an, um den Benutzer zu erinnern."""
    winsound.Beep(1000, 500)
    root = tk.Tk()
    root.withdraw()
    tk.messagebox.showwarning("Erinnerung", "Hör auf zu zocken, lern jetzt weiter!")
    root.destroy()

def reset_timer():
    """Setzt den Timer zurück."""
    global last_activity_time
    last_activity_time = time.time()
