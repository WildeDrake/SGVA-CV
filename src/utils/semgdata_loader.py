import os
import numpy as np
import torch
import torch.utils.data as data_utils


# Carga datos sEMG preprocesados desde archivos .npy
class semgdata_load(data_utils.Dataset):
    """
    root: Ruta a la carpeta que contiene los archivos preprocesados (.npy)
    split: Permite seleccionar splits ('training', 'validation', 'testing', 'cross_subject')
    y elegir qué sujetos cargar (útil para Leave-One-Subject-Out o cross-subject holdout).
    subjects: Lista de índices de sujetos a cargar. Si None, se cargan todos.
    transform: Transformaciones adicionales a aplicar a los datos.
    """
    def __init__(self, root, split="training", subjects=None, transform=None):
        self.root = os.path.expanduser(root)
        self.split = split
        self.subjects = subjects
        self.transform = transform
        # Cargar archivo .npy correspondiente
        data_path = os.path.join(self.root, f"{split}.npy")
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"No se encontró el archivo {data_path}")
        # Cargar datos
        dataset = np.load(data_path, allow_pickle=True).item()
        all_data = dataset["list_total_user_data"]
        all_labels = dataset["list_total_user_labels"]
        # Si se especifican sujetos, filtrar
        if subjects is not None:
            all_data = [all_data[i] for i in subjects]
            all_labels = [all_labels[i] for i in subjects]
        # Concatenar todos los sujetos seleccionados
        self.data = np.concatenate(all_data)
        self.labels = np.concatenate(all_labels)
        # Crear etiquetas de dominio (sujeto)
        self.domains = []
        for i, subj_labels in enumerate(all_labels):
            self.domains.extend([i] * len(subj_labels))
        self.domains = np.array(self.domains)
        # Convertir a tensores
        self.data = torch.tensor(self.data, dtype=torch.float32).unsqueeze(1)  # (N,1,8,52)
        self.labels = torch.tensor(self.labels, dtype=torch.long)
        self.domains = torch.tensor(self.domains, dtype=torch.long)
        # Convertir etiquetas y dominios a one-hot
        num_classes = self.labels.max().item() + 1
        num_domains = len(all_data)
        self.labels = torch.eye(num_classes)[self.labels]
        self.domains = torch.eye(num_domains)[self.domains]

    # Longitud del dataset
    def __len__(self):
        return len(self.labels)

    # Obtener muestra
    def __getitem__(self, index):
        x = self.data[index]
        y = self.labels[index]
        d = self.domains[index]
        if self.transform is not None:
            x = self.transform(x)
        return x, y, d