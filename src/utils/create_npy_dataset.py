import os
import numpy as np
import utils.data_process as dp

# --------------------- Configuración --------------------- #
WINDOW_SIZE = 52

SPLITS = ["training", "validation", "testing", "cross_subject"]
HANDS = ["LEFT", "RIGHT"]
PATHOLOGIES = ["Healthy", "DMD", "Neuropathy", "Parkinson", "Stroke", "ALS", "Artifact"]


# --------------------- Función para procesar un split + patología --------------------- #
def process_split(split_name, pathology_name, BASE_DATASET = "../../split_dataset"):

    # Healthy → carpeta "all"
    pathology_folder = "all" if pathology_name == "Healthy" else pathology_name

    list_total_user_data = []
    list_total_user_labels = []

    subjects = sorted([s for s in os.listdir(BASE_DATASET) if s.startswith("s")])

    for subj in subjects:
        split_path = os.path.join(BASE_DATASET, subj, split_name)
        pathology_path = os.path.join(split_path, pathology_folder)

        if not os.path.exists(pathology_path):
            continue

        subj_data = []
        subj_labels = []

        for hand in HANDS:
            hand_path = os.path.join(pathology_path, hand)
            if not os.path.exists(hand_path):
                continue

            for gesture_folder in os.listdir(hand_path):
                gesture_path = os.path.join(hand_path, gesture_folder)
                if not os.path.isdir(gesture_path):
                    continue

                for file in os.listdir(gesture_path):
                    file_path = os.path.join(gesture_path, file)
                    if not file_path.endswith(".txt"):
                        continue

                    # Leer EMG
                    data = dp.txt2array(file_path)
                    if data is None or data.size == 0:
                        continue

                    data = dp.preprocessing(data)

                    # detectar EMG activo
                    start, end = dp.detect_muscle_activity(data)
                    start, end = int(start), int(end)

                    if end - start < WINDOW_SIZE:
                        continue

                    activation = data[:, start:end]

                    # Ventanas deslizantes
                    for i in range(0, activation.shape[1] - WINDOW_SIZE + 1):
                        window = activation[:, i:i + WINDOW_SIZE].astype(np.float32)

                        # Label dinámico
                        label = dp.label_indicator(gesture_folder, emg_data=window)
                        if label is None:
                            continue

                        subj_data.append(window)
                        subj_labels.append(label)

        if subj_data:
            list_total_user_data.append(subj_data)
            list_total_user_labels.append(subj_labels)

    return list_total_user_data, list_total_user_labels


# --------------------- Guardar archivos por patología --------------------- #
def save_all_splits(BASE_DATASET = "../../split_dataset", OUTPUT_DIR = "../preprocessed_dataset"):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    for split in SPLITS:

        split_out_path = os.path.join(OUTPUT_DIR, split)
        os.makedirs(split_out_path, exist_ok=True)

        print(f"\n========== Procesando split: {split} ==========")

        for pathology in PATHOLOGIES:
            print(f"  → Patología: {pathology}")

            data, labels = process_split(split, pathology, BASE_DATASET=BASE_DATASET)

            out_file = os.path.join(split_out_path, f"{pathology}.npy")

            np.save(
                out_file,
                {
                    "list_total_user_data": data,
                    "list_total_user_labels": labels
                },
                allow_pickle=True
            )

            total_windows = sum(len(u) for u in data)
            print(f"      Guardado {pathology} con {len(data)} sujetos y {total_windows} ventanas")


# --------------------- Main --------------------- #
if __name__ == "__main__":
    save_all_splits()
