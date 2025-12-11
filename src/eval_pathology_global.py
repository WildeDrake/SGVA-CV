import argparse
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import itertools 

# Definición de las Clases
PATHOLOGY_MAP = {
    "Healthy": 0, "DMD": 1, "Neuropathy": 2, "Parkinson": 3, 
    "Stroke": 4, "ALS": 5, "Artifact": 6 
}
NUM_CLASSES = len(PATHOLOGY_MAP)

# -------------------------- 1. Modelo: BeefyMLP --------------------------
class BeefyMLP(nn.Module):
    """
    MLP con arquitectura densa, Batch Normalization y Dropout.
    Optimizado para capturar diferencias de amplitud y textura en señales rectificadas.
    """
    def __init__(self, input_dim, num_classes):
        super(BeefyMLP, self).__init__()
        self.flatten = nn.Flatten()
        
        # Arquitectura: 1024 -> 512 -> 256
        self.fc1 = nn.Linear(input_dim, 1024)
        self.bn1 = nn.BatchNorm1d(1024)
        
        self.fc2 = nn.Linear(1024, 512)
        self.bn2 = nn.BatchNorm1d(512)
        
        self.fc3 = nn.Linear(512, 256)
        self.bn3 = nn.BatchNorm1d(256)
        
        self.fc_out = nn.Linear(256, num_classes)
        
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.25)

    def forward(self, x):
        x = self.flatten(x) 
        x = self.relu(self.bn1(self.fc1(x)))
        x = self.dropout(x)
        x = self.relu(self.bn2(self.fc2(x)))
        x = self.dropout(x)
        x = self.relu(self.bn3(self.fc3(x)))
        x = self.fc_out(x)
        return x

# -------------------------- 2. Data Loader Global + Rectificación --------------------------
def load_global_pathology_data(generated_dir, batch_size, test_split):
    data_list, labels_list = [], []
    
    print(f"--- Preparando datos para Clasificacion de Patologia Global ---")

    for name, label_id in PATHOLOGY_MAP.items():
        file_path = os.path.join(generated_dir, f"generated_{name}.npy")
        if not os.path.exists(file_path):
            print(f"Advertencia: Archivo {name} no encontrado en {generated_dir}")
            continue
            
        try:
            data_dict = np.load(file_path, allow_pickle=True).item()
            X_raw = data_dict["X"]
            Y_raw = data_dict["C"] # Target: Patología
            
            # Promedio de canales si es necesario
            if X_raw.ndim == 4 and X_raw.shape[1] > 1:
                X_raw = np.mean(X_raw, axis=1, keepdims=True)
                
            data_list.append(X_raw)
            labels_list.append(Y_raw)
        except Exception as e:
            print(f"Error procesando {name}: {e}")

    if not data_list: 
        raise FileNotFoundError("No se encontraron datos validos para entrenamiento.")

    X = np.concatenate(data_list, axis=0)
    Y = np.concatenate(labels_list, axis=0)

    # RECTIFICACIÓN DE SEÑAL
    # Convierte valores negativos a positivos para analizar magnitud/energía
    print("Aplicando Rectificacion de Senal (np.abs)...")
    X = np.abs(X)

    # Estandarización
    scaler = StandardScaler()
    original_shape = X.shape 
    X_flat = X.reshape(X.shape[0], -1) 
    X_scaled = scaler.fit_transform(X_flat)
    X = X_scaled.reshape(original_shape)

    # Split Train/Test
    np.random.seed(42)
    p = np.random.permutation(X.shape[0])
    X, Y = X[p], Y[p]
    
    split_idx = int(X.shape[0] * (1 - test_split))
    
    X_train, X_test = X[:split_idx], X[split_idx:]
    Y_train, Y_test = Y[:split_idx], Y[split_idx:]

    # Conversión a Tensores
    train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(Y_train, dtype=torch.long))
    test_dataset = TensorDataset(torch.tensor(X_test, dtype=torch.float32), torch.tensor(Y_test, dtype=torch.long))

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # Cálculo dinámico de dimensiones
    input_dim = np.prod(X_train.shape[1:])
    
    print(f"Datos cargados exitosamente.")
    print(f"Total muestras: {X.shape[0]} (Train: {len(X_train)}, Test: {len(X_test)})")
    print(f"Dimension de entrada (aplanada): {input_dim}")
    
    return train_loader, test_loader, input_dim, X_test, Y_test

# -------------------------- 3. Funciones de Visualización --------------------------
def plot_training_curves(train_losses, train_accs, test_accs, filename):
    epochs = range(1, len(train_losses) + 1)
    
    plt.figure(figsize=(12, 5))
    
    # Loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, 'r-', label='Train Loss')
    plt.title('Evolucion de Perdida (Loss)')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_accs, 'g-', label='Train Acc')
    plt.plot(epochs, test_accs, 'b-', label='Test Acc')
    plt.title('Evolucion de Precision (Accuracy)')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

def plot_confusion_matrix(cm, classes, filename, normalize=False, title='Matriz de Confusion'):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        title += " (Normalizada)"
        fmt = '.2f'
    else:
        fmt = 'd'

    plt.figure(figsize=(10, 8))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Reds)
    plt.title(title)
    plt.colorbar()
    
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt), horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    
    plt.ylabel('Etiqueta Verdadera')
    plt.xlabel('Prediccion del Modelo')
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

# -------------------------- 4. Funciones de Entrenamiento --------------------------
def train_model(model, train_loader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for X, Y in train_loader:
        X, Y = X.to(device), Y.to(device)
        optimizer.zero_grad()
        outputs = model(X)
        loss = criterion(outputs, Y)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * X.size(0)
        _, predicted = torch.max(outputs.data, 1)
        total += Y.size(0)
        correct += (predicted == Y).sum().item()
        
    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc

def evaluate_model(model, test_loader, device):
    model.eval()
    correct = 0
    total = 0
    all_preds, all_true = [], []
    
    with torch.no_grad():
        for X, Y in test_loader:
            X, Y = X.to(device), Y.to(device)
            outputs = model(X)
            _, predicted = torch.max(outputs.data, 1)
            
            total += Y.size(0)
            correct += (predicted == Y).sum().item()
            
            all_preds.extend(predicted.cpu().numpy())
            all_true.extend(Y.cpu().numpy())
            
    return correct / total, np.array(all_true), np.array(all_preds)

# -------------------------- 5. Main Execution --------------------------
def main(args):
    device = torch.device("cuda" if args.use_cuda and torch.cuda.is_available() else "cpu")
    print(f"Dispositivo: {device}")
    
    # 1. Cargar Datos
    train_loader, test_loader, input_dim, _, _ = load_global_pathology_data(
        args.generated_dir, args.batch_size, args.test_split
    )
    
    # 2. Inicializar Modelo
    model = BeefyMLP(input_dim, NUM_CLASSES).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()
    
    # Historial para gráficos
    history_loss = []
    history_train_acc = []
    history_test_acc = []
    
    best_acc = 0.0
    save_path = os.path.join(args.outpath, "best_pathology_model.pth")
    
    print("\n--- Iniciando Entrenamiento Clasificador de Patologias ---")
    
    for epoch in range(1, args.epochs + 1):
        loss, train_acc = train_model(model, train_loader, optimizer, criterion, device)
        test_acc, y_true, y_pred = evaluate_model(model, test_loader, device)
        
        history_loss.append(loss)
        history_train_acc.append(train_acc)
        history_test_acc.append(test_acc)
        
        print(f"Epoch {epoch}/{args.epochs} - Loss: {loss:.4f}, Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}")
        
        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), save_path)
            
    print(f"\nEntrenamiento finalizado. Mejor Accuracy Test: {best_acc:.4f}")
    
    # --- Generación de Reportes Gráficos ---
    print("Generando graficos de evaluacion...")
    
    # 1. Curvas de Entrenamiento
    plot_training_curves(history_loss, history_train_acc, history_test_acc,
                         os.path.join(args.outpath, "pathology_training_curves.png"))
    
    # 2. Matriz de Confusión (Cargamos el mejor modelo para esto)
    model.load_state_dict(torch.load(save_path))
    _, y_true_best, y_pred_best = evaluate_model(model, test_loader, device)
    
    cm = confusion_matrix(y_true_best, y_pred_best)
    class_names = list(PATHOLOGY_MAP.keys())
    
    # Guardar versión conteo absoluto
    plot_confusion_matrix(cm, class_names, 
                          os.path.join(args.outpath, "cm_pathology_absolute.png"), 
                          normalize=False)
    
    # Guardar versión normalizada (porcentajes)
    plot_confusion_matrix(cm, class_names, 
                          os.path.join(args.outpath, "cm_pathology_normalized.png"), 
                          normalize=True)
                          
    print(f"Resultados guardados en: {args.outpath}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Global Pathology Classifier Evaluation")
    parser.add_argument('--generated-dir', type=str, default='./generated_data')
    parser.add_argument('--outpath', type=str, default='./saved_model/eval_pathology')
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch-size', type=int, default=512)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--test-split', type=float, default=0.2)
    parser.add_argument('--use-cuda', action='store_true', default=True)
    
    args = parser.parse_args()
    os.makedirs(args.outpath, exist_ok=True)
    main(args)