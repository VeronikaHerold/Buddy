import os

def create_theme_directory(theme):
    """Erstellt einen Unterordner für das Thema und die erforderlichen Dateien."""
    theme_directory = f"data/{theme}"
    os.makedirs(theme_directory, exist_ok=True)
    
    # Erstelle die beiden Dateien für Feedback und Training
    feedback_file_path = os.path.join(theme_directory, f"{theme}_feedback_data.json")
    training_file_path = os.path.join(theme_directory, f"{theme}_training_data.json")
    
    if not os.path.exists(feedback_file_path):
        with open(feedback_file_path, 'w', encoding='utf-8') as f:
            f.write('[]')  # Leere JSON-Liste
    
    if not os.path.exists(training_file_path):
        with open(training_file_path, 'w', encoding='utf-8') as f:
            f.write('[]')  # Leere JSON-Liste

    return feedback_file_path, training_file_path
