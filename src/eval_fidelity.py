import argparse
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import itertools

# -------------------------- 0. Configuración Global --------------------------
PATHOLOGY_MAP = {
    "Healthy": 0, "DMD": 1, "Neuropathy": 2, "Parkinson": 3, 
    "Stroke": 4, "ALS": 5, "Artifact": 6 
}

# Rutas (Ajustar si es necesario)
ROOT_SYNTHETIC_ORIGINAL = "./preprocessed_dataset/training"
ROOT_GENERATED_data = "./generated_data"

# -------------------------- 1. Definición del Modelo --------------------------
class SimpleMLP(nn.Module):
    def __init__(self, input_dim, num_classes=2):
        super(SimpleMLP, self).__init__()
        self.flatten = nn.Flatten()
        
        self.main = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.flatten(x)
        return self.main(x)

# -------------------------- 2. Funciones de Entrenamiento --------------------------
def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for inputs, labels in loader:
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * inputs.size(0)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
    return running_loss / total, correct / total

def evaluate_epoch(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return correct / total

def get_predictions(model, loader, device):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    return np.array(all_labels), np.array(all_preds)

# -------------------------- 3. Visualización --------------------------
def plot_confusion_matrix(cm, classes, output_path, title='Matriz de Confusion Fidelidad'):
    plt.figure(figsize=(8, 6))
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    plt.imshow(cm_normalized, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(title)
    plt.colorbar()
    
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    thresh = cm_normalized.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm_normalized[i, j], '.2f'),
                 horizontalalignment="center",
                 color="white" if cm_normalized[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('Etiqueta Verdadera')
    plt.xlabel('Prediccion del Modelo')
    plt.savefig(output_path)
    plt.close()

def plot_training_curves(train_losses, train_accs, test_accs, output_path):
    epochs = range(1, len(train_losses) + 1)
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, 'r-', label='Train Loss')
    plt.title('Evolucion de Loss')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_accs, 'g-', label='Train Acc')
    plt.plot(epochs, test_accs, 'b-', label='Test Acc')
    plt.axhline(y=0.5, color='gray', linestyle='--', label='Ideal (0.5)')
    plt.title('Evolucion de Accuracy (C2ST)')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

# -------------------------- 4. Carga de Datos (CORREGIDA) --------------------------
def normalize_shape(X):
    """
    Fuerza a que el array sea 3D (N, H, W).
    Si viene 4D (N, C, H, W), elimina el canal mediante promedio o squeeze.
    """
    if X.ndim == 4:
        # Caso (N, C, H, W)
        if X.shape[1] > 1:
            # Si tiene múltiples canales, hacemos promedio
            return np.mean(X, axis=1) # -> (N, H, W)
        else:
            # Si tiene 1 canal, hacemos squeeze
            return np.squeeze(X, axis=1) # -> (N, H, W)
    elif X.ndim == 3:
        # Caso (N, H, W) - Ya está bien
        return X
    else:
        # Caso raro, tratar de aplanar todo menos el batch
        print(f"⚠️ Advertencia: Dimensiones extrañas detectadas {X.shape}. Intentando reshape forzado.")
        return X.reshape(X.shape[0], -1) 

def load_fidelity_data(generated_dir, original_dir, batch_size=256, test_split=0.2):
    all_original = []
    all_generated = []
    
    print("--- Cargando datos para Evaluacion de Fidelidad ---")

    # 1. Cargar datos ORIGINALES
    for pat_name in PATHOLOGY_MAP.keys():
        original_path = os.path.join(original_dir, f"{pat_name}.npy")
        if os.path.exists(original_path):
            try:
                data_dict = np.load(original_path, allow_pickle=True).item()
                # Concatenamos usuarios
                X_part = np.concatenate(data_dict["list_total_user_data"], axis=0)
                
                # 🚨 NORMALIZACIÓN FORZADA A 3D
                X_part = normalize_shape(X_part)
                
                all_original.append(X_part)
            except Exception as e:
                print(f"Saltando original {pat_name}: {e}")

    # 2. Cargar datos GENERADOS
    for pat_name in PATHOLOGY_MAP.keys():
        generated_path = os.path.join(generated_dir, f"generated_{pat_name}.npy")
        if os.path.exists(generated_path):
            try:
                data_dict = np.load(generated_path, allow_pickle=True).item()
                X_part = data_dict["X"]
                
                # 🚨 NORMALIZACIÓN FORZADA A 3D
                X_part = normalize_shape(X_part)
                
                all_generated.append(X_part)
            except Exception as e:
                print(f"Saltando generado {pat_name}: {e}")
    
    if not all_original or not all_generated:
        raise FileNotFoundError("Error Critico: No se cargaron datos suficientes.")

    # Concatenar listas (Ahora seguro porque todo es 3D)
    X_original_full = np.concatenate(all_original, axis=0)
    X_generated_full = np.concatenate(all_generated, axis=0)

    # Balanceo estricto
    min_samples = min(len(X_original_full), len(X_generated_full))
    print(f"Muestras totales disponibles - Orig: {len(X_original_full)}, Gen: {len(X_generated_full)}")
    print(f"Recortando ambas clases a: {min_samples}")

    idx_orig = np.random.choice(len(X_original_full), min_samples, replace=False)
    idx_gen = np.random.choice(len(X_generated_full), min_samples, replace=False)

    X_original_balanced = X_original_full[idx_orig]
    X_generated_balanced = X_generated_full[idx_gen]

    # Concatenación final (Aquí fallaba antes)
    X_concat = np.concatenate([X_original_balanced, X_generated_balanced], axis=0)
    Y_concat = np.concatenate([np.zeros(min_samples), np.ones(min_samples)], axis=0).astype(np.int64)

    # Shuffle
    p = np.random.permutation(X_concat.shape[0])
    X_concat, Y_concat = X_concat[p], Y_concat[p]
    
    # Split
    split_idx = int(X_concat.shape[0] * (1 - test_split))
    
    X_train, X_test = X_concat[:split_idx], X_concat[split_idx:]
    Y_train, Y_test = Y_concat[:split_idx], Y_concat[split_idx:]

    # Tensores
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    Y_train_tensor = torch.tensor(Y_train, dtype=torch.long)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    Y_test_tensor = torch.tensor(Y_test, dtype=torch.long)

    train_dataset = TensorDataset(X_train_tensor, Y_train_tensor)
    test_dataset = TensorDataset(X_test_tensor, Y_test_tensor)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # Input dim es el producto de H*W (ya que no hay canales)
    input_dim = np.prod(X_train_tensor.shape[1:])
    return train_loader, test_loader, input_dim

# -------------------------- 5. Main --------------------------
def main_fidelity(args):
    device = torch.device("cuda" if args.use_cuda and torch.cuda.is_available() else "cpu")
    print(f"Dispositivo: {device}")
    os.makedirs(args.out_dir, exist_ok=True)
    
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    try:
        train_loader, test_loader, input_dim = load_fidelity_data(
            ROOT_GENERATED_data, ROOT_SYNTHETIC_ORIGINAL, args.batch_size, args.test_split
        )
    except Exception as e:
        print(f"\n❌ Error cargando datos: {e}")
        # Imprimir traceback para debug si es necesario
        import traceback
        traceback.print_exc()
        return

    model = SimpleMLP(input_dim, num_classes=2).to(device) 
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()
    
    print("\n--- Iniciando Entrenamiento Detector de Fidelidad (C2ST) ---")
    
    best_accuracy = 0.0
    history_loss, history_train_acc, history_test_acc = [], [], []
    
    for epoch in range(1, args.epochs + 1):
        loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device)
        test_acc = evaluate_epoch(model, test_loader, device)
        
        history_loss.append(loss)
        history_train_acc.append(train_acc)
        history_test_acc.append(test_acc)

        print(f"Epoch {epoch}/{args.epochs} - Loss: {loss:.4f}, Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}")
        
        if test_acc > best_accuracy:
            best_accuracy = test_acc

    # Generación de Reportes
    print("\nGenerando graficos...")
    plot_training_curves(history_loss, history_train_acc, history_test_acc, 
                         os.path.join(args.out_dir, "fidelity_training_curves.png"))

    y_true, y_pred = get_predictions(model, test_loader, device)
    cm = confusion_matrix(y_true, y_pred)
    plot_confusion_matrix(cm, ["Original", "Generado"], 
                          os.path.join(args.out_dir, "fidelity_confusion_matrix.png"))

    print(f"\nRESULTADOS FINALES EN: {args.out_dir}")
    print(f"Precision Final (Test): {best_accuracy:.4f}")
    
    print("-" * 50)
    print("INTERPRETACION (C2ST):")
    if best_accuracy <= 0.60:
         print(f"🟢 [ALTA]: {best_accuracy:.2f} ≈ 0.5. Datos indistinguibles.")
    elif best_accuracy <= 0.75:
         print(f"🟡 [MEDIA]: {best_accuracy:.2f}. Detectable pero similar.")
    else:
         print(f"🔴 [BAJA]: {best_accuracy:.2f} >> 0.5. Facilmente distinguible.")
    print("-" * 50)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--test-split', type=float, default=0.2)
    parser.add_argument('--use-cuda', action='store_true', default=True)
    parser.add_argument('--out-dir', type=str, default='./saved_model/eval_fidelity')
    
    args = parser.parse_args()
    main_fidelity(args)