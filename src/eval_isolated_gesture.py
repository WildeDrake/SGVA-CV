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

# -------------------------- 1. Modelo: MLP (BeefyMLP) --------------------------
class BeefyMLP(nn.Module):
    """
    MLP ancho con Batch Normalization y Dropout.
    Diseñado para capturar patrones de energía en señales rectificadas.
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


# -------------------------- 2. Data Loader Especializado --------------------------
def load_isolated_data(generated_dir, batch_size, test_split, gesture_id):
    data_list, labels_list = [], []
    all_raw_gesture_labels = [] 
    
    print(f"--- Preparando data para PATOLOGIA (Solo Gesto {gesture_id}) ---")

    for name in PATHOLOGY_MAP.keys():
        file_path = os.path.join(generated_dir, f"generated_{name}.npy")
        if not os.path.exists(file_path):
            print(f"Advertencia: Archivo {name} no encontrado.")
            continue
            
        data_dict = np.load(file_path, allow_pickle=True).item()
        X_raw = data_dict["X"]
        Y_raw = data_dict["C"] # Siempre cargamos Patología
        
        # Guardar etiquetas de gesto para filtrar posteriormente
        all_raw_gesture_labels.append(data_dict["Y"])
        
        # Promedio de canales si es necesario
        if X_raw.ndim == 4 and X_raw.shape[1] > 1:
            X_raw = np.mean(X_raw, axis=1, keepdims=True)
            
        data_list.append(X_raw)
        labels_list.append(Y_raw)

    if not data_list: 
        raise FileNotFoundError("No se encontraron datos en el directorio especificado.")

    X = np.concatenate(data_list, axis=0)
    Y = np.concatenate(labels_list, axis=0)
    all_Y_gestures = np.concatenate(all_raw_gesture_labels, axis=0)

    # 1. FILTRADO (Aislamiento por Gesto)
    print(f"Total muestras brutas: {X.shape[0]}. Filtrando por Gesto ID: {gesture_id}...")
    mask = (all_Y_gestures == gesture_id)
    X = X[mask]
    Y = Y[mask]
    print(f"Muestras restantes tras filtrado: {X.shape[0]}")

    if X.shape[0] == 0:
        raise ValueError(f"No hay muestras para el gesto {gesture_id}. Verifica los datos generados.")

    # 2. RECTIFICACIÓN (Valor Absoluto)
    # Fundamental para analizar la energía de la señal independientemente de la fase
    X = np.abs(X)

    # 3. ESTANDARIZACIÓN
    scaler = StandardScaler()
    original_shape = X.shape 
    X_flat = X.reshape(X.shape[0], -1) 
    X_scaled = scaler.fit_transform(X_flat)
    X = X_scaled.reshape(original_shape)

    # 4. SPLIT
    np.random.seed(42)
    p = np.random.permutation(X.shape[0])
    X, Y = X[p], Y[p]
    
    split_idx = int(X.shape[0] * (1 - test_split))
    
    X_train, X_test = X[:split_idx], X[split_idx:]
    Y_train, Y_test = Y[:split_idx], Y[split_idx:]

    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    Y_train_tensor = torch.tensor(Y_train, dtype=torch.long)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    Y_test_tensor = torch.tensor(Y_test, dtype=torch.long)

    train_dataset = TensorDataset(X_train_tensor, Y_train_tensor)
    test_dataset = TensorDataset(X_test_tensor, Y_test_tensor)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # Cálculo dinámico de la dimensión de entrada aplanada
    input_dim = np.prod(X_train.shape[1:])
    num_classes = len(PATHOLOGY_MAP)
    
    return train_loader, test_loader, input_dim, num_classes, X_test, Y_test

# -------------------------- Funciones de Visualización --------------------------
def plot_training_curves(train_losses, train_accs, test_accs, filename):
    epochs = range(1, len(train_losses) + 1)
    
    plt.figure(figsize=(12, 5))
    
    # Gráfico de Pérdida (Loss)
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, 'r-', label='Train Loss')
    plt.title('Evolucion de Perdida (Loss)')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Gráfico de Precisión (Accuracy)
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_accs, 'g-', label='Train Accuracy')
    plt.plot(epochs, test_accs, 'b-', label='Test Accuracy')
    plt.title('Evolucion de Precision (Accuracy)')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
    print(f"Grafico de entrenamiento guardado en: {filename}")

def plot_confusion_matrix(cm, classes, filename, normalize=False, title='Matriz de Confusion'):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        title += " (Normalizada)"
    
    plt.figure(figsize=(10, 8))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(title)
    plt.colorbar()
    
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt), horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    
    plt.ylabel('Etiqueta Verdadera')
    plt.xlabel('Prediccion del Modelo')
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

# -------------------------- 3. Training Loop --------------------------

def train_model(model, train_loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    correct_predictions = 0
    total_samples = 0
    
    for X, Y in train_loader:
        X, Y = X.to(device), Y.to(device)
        optimizer.zero_grad()
        outputs = model(X)
        loss = criterion(outputs, Y)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        correct_predictions += (predicted == Y).sum().item()
        total_samples += Y.size(0)
        
    avg_loss = total_loss / len(train_loader)
    avg_acc = correct_predictions / total_samples
    return avg_loss, avg_acc

def evaluate_model(model, test_loader, device):
    model.eval()
    all_preds, all_true = [], []
    correct = 0
    total = 0
    with torch.no_grad():
        for X, Y in test_loader:
            X, Y = X.to(device), Y.to(device)
            outputs = model(X)
            _, predicted = torch.max(outputs.data, 1)
            
            all_preds.append(predicted.cpu().numpy())
            all_true.append(Y.cpu().numpy())
            
            correct += (predicted == Y).sum().item()
            total += Y.size(0)
            
    return correct / total, np.concatenate(all_true), np.concatenate(all_preds)

# -------------------------- 4. Main Execution --------------------------

def main(args):
    device = torch.device("cuda" if args.use_cuda and torch.cuda.is_available() else "cpu")
    print(f"Dispositivo de entrenamiento: {device}")
    
    # 1. Cargar Data
    train_loader, test_loader, input_dim, num_classes, _, _ = load_isolated_data(
        args.generated_dir, args.batch_size, args.test_split, args.gesture_id
    )
    
    print(f"Dimension de entrada detectada: {input_dim}")
    
    # 2. Inicializar Modelo
    model = BeefyMLP(input_dim, num_classes).to(device)
    
    if args.load_model:
        if os.path.exists(args.load_model):
            print(f"Cargando checkpoint: {args.load_model}")
            model.load_state_dict(torch.load(args.load_model))
        else:
            print(f"Advertencia: No se encontro el archivo {args.load_model}, iniciando desde cero.")

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    best_acc = 0.0
    patience_counter = 0
    save_path = os.path.join(args.outpath, f"best_isolated_G{args.gesture_id}.pth")
    
    # Listas para historial
    history_loss = []
    history_train_acc = []
    history_test_acc = []
    
    print(f"\n--- Iniciando Entrenamiento (Gesto {args.gesture_id}) ---")
    
    for epoch in range(1, args.epochs + 1):
        loss, train_acc = train_model(model, train_loader, optimizer, criterion, device)
        test_acc, _, _ = evaluate_model(model, test_loader, device)

        # Guardar en historial
        history_loss.append(loss)
        history_train_acc.append(train_acc)
        history_test_acc.append(test_acc)

        print(f"Epoch {epoch}/{args.epochs} - Loss: {loss:.4f}, Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}")
        
        if test_acc > best_acc:
            best_acc = test_acc
            patience_counter = 0
            torch.save(model.state_dict(), save_path)
        else:
            patience_counter += 1
            if patience_counter >= args.early_stop_patience:
                print(f"Detencion temprana (Early Stopping) en epoch {epoch}. Mejor Accuracy: {best_acc:.4f}")
                break
    
    # Generar graficos
    plot_training_curves(history_loss, history_train_acc, history_test_acc, 
                         os.path.join(args.outpath, f"history_G{args.gesture_id}.png"))

    # Evaluacion Final y Matrices
    print(f"\nGenerando matrices de confusion...")
    model.load_state_dict(torch.load(save_path)) # Cargar el mejor modelo guardado
    _, y_true, y_pred = evaluate_model(model, test_loader, device)
    
    class_names = list(PATHOLOGY_MAP.keys())
    cm = confusion_matrix(y_true, y_pred)
    
    plot_confusion_matrix(cm, class_names, 
                          os.path.join(args.outpath, f"cm_G{args.gesture_id}_norm.png"), 
                          normalize=True, title=f"Matriz Normalizada Gesto {args.gesture_id}")
    
    plot_confusion_matrix(cm, class_names, 
                          os.path.join(args.outpath, f"cm_G{args.gesture_id}_raw.png"), 
                          normalize=False, title=f"Matriz Conteo Gesto {args.gesture_id}")
    
    print(f"Archivos guardados en: {args.outpath}")
    print(f"RESULTADO FINAL GESTO {args.gesture_id}: {best_acc:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--generated-dir', type=str, default='./generated_data')
    parser.add_argument('--outpath', type=str, default='./saved_model/eval_gestures_isolated')
    parser.add_argument('--load-model', type=str, default=None, help="Ruta al .pth para continuar entrenando")
    parser.add_argument('--gesture-id', type=int, default=0, help="ID del gesto a aislar (default 0)")
    
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--early-stop-patience', type=int, default=15)
    parser.add_argument('--batch-size', type=int, default=512)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--test-split', type=float, default=0.2)
    parser.add_argument('--use-cuda', action='store_true', default=True)
    
    args = parser.parse_args()
    os.makedirs(args.outpath, exist_ok=True)
    main(args)