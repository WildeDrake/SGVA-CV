import os
import numpy as np
import data_process as dp

# --------------------- Configuración --------------------- #
BASE_DATASET = "../../split_dataset"
OUTPUT_DIR = "../preprocessed_dataset"
WINDOW_SIZE = 52   # longitud de ventana usada en DIVA
os.makedirs(OUTPUT_DIR, exist_ok=True)

SPLITS = ["training", "validation", "testing", "cross_subject"]
HANDS = ["LEFT", "RIGHT"]


# --------------------- Función para procesar un split --------------------- #
def process_split(split_name):
    list_total_user_data = []
    list_total_user_labels = []

    # Recorrer sujetos
    subjects = sorted([s for s in os.listdir(BASE_DATASET) if s.startswith("s")])
    for subj in subjects:
        subj_path = os.path.join(BASE_DATASET, subj)
        split_path = os.path.join(subj_path, split_name)

        subj_data = []
        subj_labels = []
        for hand in HANDS:
            hand_path = os.path.join(split_path, hand)
            if not os.path.exists(hand_path):
                continue
            for gesture_folder in os.listdir(hand_path):
                gesture_path = os.path.join(hand_path, gesture_folder)
                label = dp.label_indicator(gesture_folder)
                for file in os.listdir(gesture_path):
                    file_path = os.path.join(gesture_path, file)
                    data = dp.txt2array(file_path)
                    data = dp.preprocessing(data)
                    start, end = dp.detect_muscle_activity(data)
                    start, end = int(start), int(end)
                    activation = data[:, start:end]
                    # Ventanas deslizantes
                    for i in range(0, activation.shape[1] - WINDOW_SIZE + 1):
                        window = activation[:, i:i + WINDOW_SIZE]
                        subj_data.append(window.astype(np.float32))
                        subj_labels.append(label)
        if subj_data:  # solo agregar si hay datos
            list_total_user_data.append(subj_data)
            list_total_user_labels.append(subj_labels)
    
    return list_total_user_data, list_total_user_labels

# --------------------- Guardar datasets --------------------- #
def save_all_splits():
    for split in SPLITS:
        data, labels = process_split(split)

        output_file = os.path.join(OUTPUT_DIR, f"{split}.npy")
        np.save(output_file, {
            "list_total_user_data": data,
            "list_total_user_labels": labels
        }, allow_pickle=True)
        total_windows = sum(len(u) for u in data)
        print(f"Guardado {split} con {len(data)} sujetos y {total_windows} ventanas en {output_file}")

# --------------------- Main --------------------- #
if __name__ == "__main__":
    save_all_splits()