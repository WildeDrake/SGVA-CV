import os
import shutil
import random

# ------------------ Configuración ------------------ #
ROOT_DIR = "../../dataset"
OUTPUT_DIR = "../../split_dataset"

TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
TEST_RATIO = 0.15
TEST_CROSS_SUBJECT = 2  # número de sujetos para cross-subject hold-out

# ------------------ Funciones ------------------ #
def split_subject_data(subject_path, subject_name, cross_subject=False):
    """
    Divide los archivos de un sujeto en training, validation y testing.
    Si cross_subject=True, todos los archivos van a cross_subject/
    """
    all_dir = os.path.join(subject_path, "all")
    # Carpeta base de salida para este sujeto
    out_subject_dir = os.path.join(OUTPUT_DIR, subject_name)
    if cross_subject:
        subsets_to_create = ["cross_subject"]
    else:
        subsets_to_create = ["training", "validation", "testing"]
    # Limpiar carpetas de salida si existen
    for subset in subsets_to_create:
        subset_dir = os.path.join(out_subject_dir, subset)
        if os.path.exists(subset_dir):
            shutil.rmtree(subset_dir)
    # Procesar manos
    for hand in ["LEFT", "RIGHT"]:
        hand_path = os.path.join(all_dir, hand)
        if not os.path.exists(hand_path):
            continue
        files = [f for f in os.listdir(hand_path) if f.endswith(".txt")]
        # Agrupar archivos por gesto
        gesture_dict = {}
        for f in files:
            gesture = f.split("-")[0]
            gesture_dict.setdefault(gesture, []).append(f)
        # Dividir y copiar archivos
        for gesture, gesture_files in gesture_dict.items():
            random.shuffle(gesture_files)
            n = len(gesture_files)
            if cross_subject:
                subsets = {"cross_subject": gesture_files}  # todo a cross_subject
            else:
                n_train = int(n * TRAIN_RATIO)
                n_val = int(n * VAL_RATIO)
                subsets = {
                    "training": gesture_files[:n_train],
                    "validation": gesture_files[n_train:n_train + n_val],
                    "testing": gesture_files[n_train + n_val:]
                }
            # Copiar archivos
            for subset, subset_files in subsets.items():
                dest_dir = os.path.join(out_subject_dir, subset, hand, gesture)
                os.makedirs(dest_dir, exist_ok=True)
                for f in subset_files:
                    src = os.path.join(hand_path, f)
                    dst = os.path.join(dest_dir, f)
                    shutil.copy2(src, dst)

def main():
    random.seed(33)
    # Listar sujetos y ordenar
    subjects = sorted([s for s in os.listdir(ROOT_DIR) if s.startswith("s")])
    # Seleccionar sujetos hold-out para cross-subject
    cross_subjects = subjects[-TEST_CROSS_SUBJECT:]
    train_subjects = [s for s in subjects if s not in cross_subjects]
    print(f"Sujetos para training/val/test: {train_subjects}")
    print(f"Sujetos hold-out cross-subject: {cross_subjects}")
    # Dividir sujetos de training
    for subj in train_subjects:
        split_subject_data(os.path.join(ROOT_DIR, subj), subj, cross_subject=False)
    # Dividir sujetos hold-out en cross_subject
    for subj in cross_subjects:
        split_subject_data(os.path.join(ROOT_DIR, subj), subj, cross_subject=True)

if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    main()
