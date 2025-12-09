from utils.generate_pathological_dataset import create_pathological_dataset
from utils.split_dataset import split_dataset
from utils.create_npy_dataset import save_all_splits   

create_pathological_dataset(path="../dataset")
split_dataset(ROOT_DIR = "../dataset", OUTPUT_DIR = "../split_dataset")
save_all_splits(BASE_DATASET = "../split_dataset", OUTPUT_DIR = "./preprocessed_dataset")