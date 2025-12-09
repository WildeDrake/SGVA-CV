import os
import shutil
import random

TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
TEST_RATIO = 0.15
TEST_CROSS_SUBJECT = 2  # número de sujetos para cross-subject hold-out

PATHOLOGIES = ["all", "DMD", "Neuropathy", "Parkinson", "Stroke", "ALS", "Artifact"]


def split_subject_data(subject_path, subject_name, OUTPUT_DIR, cross_subject=False):

    out_subject_dir = os.path.join(OUTPUT_DIR, subject_name)

    # Qué subsets crear
    if cross_subject:
        subsets_to_create = ["cross_subject"]
    else:
        subsets_to_create = ["training", "validation", "testing"]

    # Limpiar carpeta de salida si existe
    for subset in subsets_to_create:
        subset_dir = os.path.join(out_subject_dir, subset)
        if os.path.exists(subset_dir):
            shutil.rmtree(subset_dir)

    # Primero hacer SPLIT SOLO DESDE ALL
    all_splits = {}

    pathology_dir_all = os.path.join(subject_path, "all")
    if not os.path.exists(pathology_dir_all):
        return

    for hand in ["LEFT", "RIGHT"]:
        hand_path = os.path.join(pathology_dir_all, hand)
        if not os.path.exists(hand_path):
            continue

        files = [f for f in os.listdir(hand_path) if f.endswith(".txt")]

        # Agrupar en ALL por gesto
        gesture_dict = {}
        for f in files:
            gesture = f.split("-")[0]
            gesture_dict.setdefault(gesture, []).append(f)

        # Generar split según ALL
        for gesture, gesture_files in gesture_dict.items():
            random.shuffle(gesture_files)
            n = len(gesture_files)

            if cross_subject:
                subsets = {"cross_subject": gesture_files}
            else:
                n_train = int(n * TRAIN_RATIO)
                n_val = int(n * VAL_RATIO)
                subsets = {
                    "training": gesture_files[:n_train],
                    "validation": gesture_files[n_train:n_train + n_val],
                    "testing": gesture_files[n_train + n_val:]
                }

            all_splits[(hand, gesture)] = subsets

    # Copiar cada patología usando el split de ALL
    for pathology in PATHOLOGIES:
        pathology_dir = os.path.join(subject_path, pathology)
        if not os.path.exists(pathology_dir):
            continue

        print(f"> Procesando {subject_name}/{pathology}")

        for hand in ["LEFT", "RIGHT"]:
            hand_path = os.path.join(pathology_dir, hand)
            if not os.path.exists(hand_path):
                continue

            gestures = set([f.split("-")[0] for f in os.listdir(hand_path) if f.endswith(".txt")])

            for gesture in gestures:
                # Split definido por ALL
                if (hand, gesture) not in all_splits:
                    continue

                subsets = all_splits[(hand, gesture)]

                for subset, subset_files in subsets.items():
                    dest_dir = os.path.join(out_subject_dir, subset, pathology, hand, gesture)
                    os.makedirs(dest_dir, exist_ok=True)

                    for f in subset_files:
                        src = os.path.join(hand_path, f)
                        if os.path.exists(src):  # por si esa patología no tiene ese archivo
                            dst = os.path.join(dest_dir, f)
                            shutil.copy2(src, dst)


def split_dataset(ROOT_DIR, OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    random.seed(33)

    subjects = sorted([s for s in os.listdir(ROOT_DIR) if s.startswith("s")])
    cross_subjects = subjects[-TEST_CROSS_SUBJECT:]
    train_subjects = [s for s in subjects if s not in cross_subjects]

    print(f"Sujetos para training/val/test: {train_subjects}")
    print(f"Sujetos hold-out cross-subject: {cross_subjects}")

    # Training/val/test con split por all
    for subj in train_subjects:
        split_subject_data(
            os.path.join(ROOT_DIR, subj),
            subj,
            OUTPUT_DIR=OUTPUT_DIR,
            cross_subject=False
        )

    # Cross-subject igual copia todo a un único subset
    for subj in cross_subjects:
        split_subject_data(
            os.path.join(ROOT_DIR, subj),
            subj,
            OUTPUT_DIR=OUTPUT_DIR,
            cross_subject=True
        )


if __name__ == "__main__":
    split_dataset(ROOT_DIR="../../dataset", OUTPUT_DIR="../../split_dataset")
