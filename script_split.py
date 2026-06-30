import os
import pandas as pd
from sklearn.model_selection import train_test_split

print("FASE 1: Ricerca Dati PROFONDA in LOCALE \n")

DATASET_DIR = './waste_type_identification/waste_type_identification' 
OUTPUT_CSV = './waste_type_identification/train_val_split.csv'

if not os.path.exists(DATASET_DIR):
    raise FileNotFoundError(f"ERRORE: La cartella non si trova all'indirizzo '{DATASET_DIR}'.")

print(f" Directory trovata: {DATASET_DIR}")

class_mapping = {
    'battery': 0, 'clothing': 1, 'glass': 2, 'metal': 3,
    'organic': 4, 'papery': 5, 'plastic': 6, 'undifferentiated': 7
}

data = []
print("Esplorazione profonda delle cartelle in corso...")

for class_name in os.listdir(DATASET_DIR):
    class_dir = os.path.join(DATASET_DIR, class_name)
    
    if os.path.isdir(class_dir) and class_name.lower() in class_mapping:
        label = class_mapping[class_name.lower()]
        
        # IL CAMBIAMENTO È QUI: os.walk esplora tutte le sottocartelle
        for root_dir, dirs, files in os.walk(class_dir):
            for file_name in files:
                # Controlliamo estensioni comuni (aggiunto anche webp per sicurezza)
                if file_name.lower().endswith(('.png', '.jpg', '.jpeg', '.webp')):
                    file_path = os.path.join(root_dir, file_name)
                    data.append({'filepath': file_path, 'label': label, 'class_name': class_name.capitalize()})

df = pd.DataFrame(data)

if len(df) == 0:
    raise ValueError(f" CRITICO: La cartella {DATASET_DIR} è vuota o non ci sono immagini.")

print(f" Trovate {len(df)} immagini in totale esplorando tutte le sottocartelle.")

# SPLIT E SALVATAGGIO CSV
print("Esecuzione dello split (80% train, 20% validation) in corso ...")
train_df, val_df = train_test_split(df, test_size=0.20, stratify=df['label'], random_state=42)

train_df = train_df.copy(); train_df['split'] = 'train'
val_df = val_df.copy(); val_df['split'] = 'val'

final_df = pd.concat([train_df, val_df])
final_df.to_csv(OUTPUT_CSV, index=False)
print(f" Split completato. File salvato in: {OUTPUT_CSV}")

print("\nDistribuzione reale delle classi nel Training Set:")
print(train_df['class_name'].value_counts())