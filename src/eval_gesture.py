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
GESTURE_CLASSES = 5 

# -------------------------- 1. Modelo: SimpleConvNet --------------------------
class SimpleConvNet(nn.Module):
    def __init__(self, input_dim_dummy, num_classes):
        super(SimpleConvNet, self).__init__()
        
        # Bloque 1
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2) 

        # Bloque 2
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.flatten = nn.Flatten()
        # Nota: Estas dimensiones dependen del tamaño de entrada. 
        # Asegurarse que la entrada sea compatible (ej. 1x1x52x8 o similar)
        self.fc_input_dim = 32 * 2 * 13 
        
        self.fc1 = nn.Linear(self.fc_input_dim, 128)
        self.relu3 = nn.ReLU()
        self.dropout = nn.Dropout(0.4) 
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool1(self.relu1(self.bn1(self.conv1(x))))
        x = self.pool2(self.relu2(self.bn2(self.conv2(x))))
        x = self.flatten(x)
        x = self.relu3(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# -------------------------- 2. Data Loader (Solo Gestos) --------------------------
def load_gesture_data(generated_dir, batch_size, test_split):
    data_list, labels_list = [], []
    
    print(f"\n--- Preparando data para GESTOS (Pureza del Disentanglement) ---")

    for name in PATHOLOGY_MAP.keys():
        file_path = os.path.join(generated_dir, f"generated_{name}.npy")
        if not os.path.exists(file_path):
            continue
            
        data_dict = np.load(file_path, allow_pickle=True).item()
        X_raw = data_dict["X"]
        Y_raw = data_dict["Y"] # 🚨 Extraemos Y (Gesto)
        
        if X_raw.ndim == 4 and X_raw.shape[1] > 1:
            X_raw = np.mean(X_raw, axis=1, keepdims=True)
            
        data_list.append(X_raw)
        labels_list.append(Y_raw)

    if not data_list: raise FileNotFoundError("No data found.")

    X = np.concatenate(data_list, axis=0)
    Y = np.concatenate(labels_list, axis=0)

    # Estandarización
    scaler = StandardScaler()
    original_shape = X.shape 
    X_flat = X.reshape(X.shape[0], -1) 
    X_scaled = scaler.fit_transform(X_flat)
    X = X_scaled.reshape(original_shape)

    # Split
    np.random.seed(42)
    p = np.random.permutation(X.shape[0])
    X, Y = X[p], Y[p]
    
    split_idx = int(X.shape[0] * (1 - test_split))
    
    X_train, X_test = X[:split_idx], X[split_idx:]
    Y_train, Y_test = Y[:split_idx], Y[split_idx:]

    # Tensores
    train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(Y_train, dtype=torch.long))
    test_dataset = TensorDataset(torch.tensor(X_test, dtype=torch.float32), torch.tensor(Y_test, dtype=torch.long))

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    print(f"Total muestras: {X.shape[0]}. Clases de Gesto: {GESTURE_CLASSES}")
    return train_loader, test_loader, X_test, Y_test

# -------------------------- 3. Training Utils --------------------------
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
    all_preds, all_true = [], []
    with torch.no_grad():
        for X, Y in test_loader:
            X, Y = X.to(device), Y.to(device)
            outputs = model(X)
            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == Y).sum().item()
            all_preds.append(predicted.cpu().numpy())
            all_true.append(Y.cpu().numpy())
    return correct / len(test_loader.dataset), np.concatenate(all_true), np.concatenate(all_preds)

# -------------------------- 4. Visualization Utils --------------------------
def plot_cm(cm, classes, filename):
    """Genera y guarda la matriz de confusión normalizada."""
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Greens)
    plt.title('Matriz de Confusión (Gestos)')
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], '.2f'), horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    
    plt.tight_layout()
    plt.ylabel('Verdadera')
    plt.xlabel('Predicha')
    plt.savefig(filename)
    plt.close()

def plot_training_curves(train_losses, train_accs, test_accs, filename):
    """Genera y guarda las curvas de pérdida y accuracy."""
    epochs = range(1, len(train_losses) + 1)
    
    plt.figure(figsize=(12, 5))
    
    # Subplot 1: Loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, 'r-', label='Train Loss')
    plt.title('Evolución de Pérdida (Loss)')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    # Subplot 2: Accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_accs, 'g-', label='Train Acc')
    plt.plot(epochs, test_accs, 'b-', label='Test Acc')
    plt.title('Evolución de Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

# -------------------------- 5. Main --------------------------
def main(args):
    device = torch.device("cuda" if args.use_cuda and torch.cuda.is_available() else "cpu")
    print(f"Dispositivo de entrenamiento: {device}")
    
    train_loader, test_loader, _, _ = load_gesture_data(args.generated_dir, args.batch_size, args.test_split)
    
    model = SimpleConvNet(None, GESTURE_CLASSES).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()
    
    # Listas para guardar historial
    history_train_loss = []
    history_train_acc = []
    history_test_acc = []
    
    best_acc = 0.0
    print("\n--- Iniciando Entrenamiento de Reconocimiento de Gesto ---")
    
    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_model(model, train_loader, optimizer, criterion, device)
        test_acc, y_true, y_pred = evaluate_model(model, test_loader, device)
        
        # Guardar métricas
        history_train_loss.append(train_loss)
        history_train_acc.append(train_acc)
        history_test_acc.append(test_acc)
        
        print(f"Epoch {epoch}/{args.epochs} - Loss: {train_loss:.4f} - Train Acc: {train_acc:.4f} - Test Acc: {test_acc:.4f}")
        
        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), os.path.join(args.outpath, "best_gesture_model.pth"))
    
    # --- Generación de Gráficos ---
    print("\nGenerando gráficos de resultados...")
    
    # 1. Matriz de Confusión
    cm = confusion_matrix(y_true, y_pred)
    plot_cm(cm, [f"G{i}" for i in range(GESTURE_CLASSES)], os.path.join(args.outpath, "cm_gesture.png"))
    
    # 2. Curvas de Entrenamiento
    plot_training_curves(history_train_loss, history_train_acc, history_test_acc, os.path.join(args.outpath, "training_curves_gesture.png"))
    
    print(f"Archivos guardados en: {args.outpath}")
    print(f"RESULTADO FINAL GESTO (Best Test Acc): {best_acc:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--generated-dir', type=str, default='./generated_data')
    parser.add_argument('--outpath', type=str, default='./saved_model/eval_gestures')
    parser.add_argument('--epochs', type=int, default=20) 
    parser.add_argument('--batch-size', type=int, default=512)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--test-split', type=float, default=0.2)
    parser.add_argument('--use-cuda', action='store_true', default=True)
    
    args = parser.parse_args()
    os.makedirs(args.outpath, exist_ok=True)
    main(args)