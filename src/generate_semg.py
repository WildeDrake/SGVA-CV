import argparse
import os
import numpy as np
import torch
# No se necesita h5py en esta versión
from model_diva import DIVA
from utils.semgdata_loader import load_split, PATHOLOGY_MAP, C_DIM

# -------------------------- Configuración --------------------------
ROOT_preprocessed = "./preprocessed_dataset"
TOTAL_GESTURES = 5
BASE_SUBJECT_NAME = "Healthy" 

# -------------------------- Funciones de Utilidad --------------------------
def extract_latents(loader, model, device):
    """ 
    Extrae las medias latentes (mu) para zs y zy, agrupadas por Gesto (Y).
    Esta función NO se modificó durante la optimización de RAM.
    """
    model.eval()
    latents_by_gesture = {i: {'zs': [], 'zy': [], 'y': []} for i in range(TOTAL_GESTURES)}
    
    with torch.no_grad():
        for x, y, d, c in loader: 
            x, y, d, c = x.to(device), y.to(device), d.to(device), c.to(device) 
            mu_zd_loc, _ = model.qzd(x) 
            mu_zy_loc, _ = model.qzy(x) 
            y_labels = y.argmax(dim=1).cpu().numpy()
            
            for i in range(len(y_labels)):
                gesture_id = y_labels[i]
                latents_by_gesture[gesture_id]['zs'].append(mu_zd_loc[i:i+1].cpu().numpy())
                latents_by_gesture[gesture_id]['zy'].append(mu_zy_loc[i:i+1].cpu().numpy())
                latents_by_gesture[gesture_id]['y'].append(gesture_id)
                
    for i in range(TOTAL_GESTURES):
        if latents_by_gesture[i]['zs']:
            latents_by_gesture[i]['zs'] = np.concatenate(latents_by_gesture[i]['zs'], axis=0)
            latents_by_gesture[i]['zy'] = np.concatenate(latents_by_gesture[i]['zy'], axis=0)
            latents_by_gesture[i]['y'] = np.array(latents_by_gesture[i]['y'])
        else:
            latents_by_gesture[i]['zs'] = np.empty((0, model.zd_dim), dtype=np.float32)
            latents_by_gesture[i]['zy'] = np.empty((0, model.zy_dim), dtype=np.float32)
            latents_by_gesture[i]['y'] = np.empty(0, dtype=np.int64)
            
    return latents_by_gesture


def generate_signals(model, device, latents_by_gesture, target_pathology_id, num_samples, gen_batch_size):
    """
    Genera señales con mini-batching y balanceo forzado por Gesto.
    (La función de guardado en RAM, pero optimizada para 1 canal).
    """
    model.eval()
    all_x_generated, all_y_labels, all_c_labels = [], [], [] 

    # Inicialización de C_tensor_base (corregida)
    C_tensor_base = torch.zeros(1, C_DIM).to(device)
    C_tensor_base[0, target_pathology_id] = 1.0
    
    # ... (Lógica de muestreo y batching idéntica a generate_signals_RAM_test) ...
    
    samples_per_gesture = int(num_samples / TOTAL_GESTURES)
    total_generated = samples_per_gesture * TOTAL_GESTURES
    
    # print(f"    Total generado (balanceado): {total_generated}. Muestras por Gesto: {samples_per_gesture}.")
    
    num_batches = int(np.ceil(total_generated / gen_batch_size))
    samples_remaining_by_gesture = {i: samples_per_gesture for i in range(TOTAL_GESTURES)}

    print(f"    Dividiendo {total_generated} muestras en {num_batches} lotes de tamaño {gen_batch_size}...")

    current_idx = 0
    for i in range(num_batches):
        batch_size_needed = min(gen_batch_size, total_generated - current_idx)

        zs_batch, zy_batch, y_batch = [], [], []
        samples_per_gesture_in_batch_target = int(np.ceil(batch_size_needed / TOTAL_GESTURES))
        
        for gesture_id in range(TOTAL_GESTURES):
            num_to_sample = min(samples_per_gesture_in_batch_target, samples_remaining_by_gesture[gesture_id])
            if num_to_sample <= 0: continue 

            base_z = latents_by_gesture[gesture_id]
            N_base_samples = len(base_z['zs'])
            if N_base_samples == 0: continue
            
            indices = np.random.choice(N_base_samples, size=num_to_sample, replace=True)
            
            zs_batch.append(base_z['zs'][indices])
            zy_batch.append(base_z['zy'][indices])
            y_batch.append(base_z['y'][indices])
            
            samples_remaining_by_gesture[gesture_id] -= num_to_sample

        if not zs_batch: break
            
        zs_sample = torch.tensor(np.concatenate(zs_batch, axis=0), dtype=torch.float32).to(device)
        zy_sample = torch.tensor(np.concatenate(zy_batch, axis=0), dtype=torch.float32).to(device)
        y_labels_batch = np.concatenate(y_batch, axis=0)
        
        final_batch_size = zs_sample.size(0)

        zx_dim = model.zx_dim
        zx_sample = torch.randn(final_batch_size, zx_dim).to(device)
        C_batch = C_tensor_base.repeat(final_batch_size, 1)

        with torch.no_grad():
            x_gen = model.px(zs_sample, zx_sample, zy_sample, C_batch)
        
        # Guardado en RAM (Optimizada a 1 canal)
        x_gen_1ch = x_gen.mean(dim=1, keepdim=True) 
        all_x_generated.append(x_gen_1ch.cpu().numpy())
        
        all_y_labels.append(y_labels_batch)
        all_c_labels.append(np.full(final_batch_size, target_pathology_id, dtype=np.int64))

        current_idx += final_batch_size
        if device.type == 'cuda': torch.cuda.empty_cache()
            
    # PASO FINAL: CONCATENACIÓN MASIVA EN RAM (PUNTO DE FALLO POTENCIAL)
    # print("--- CONCATENANDO EN RAM ---") # Quitamos los logs de alerta
    X_gen = np.concatenate(all_x_generated, axis=0)
    Y_labels = np.concatenate(all_y_labels, axis=0)
    C_labels = np.concatenate(all_c_labels, axis=0)
    # print("--- CONCATENACIÓN EXITOSA. Guardando... ---")
    
    return X_gen, Y_labels, C_labels, current_idx


# -------------------------- Main Execution --------------------------
def main_generation(args):
    device = torch.device("cuda" if (not args.no_cuda and torch.cuda.is_available()) else "cpu")
    
    # 1. Cargar el modelo Fine-Tuned
    print(f"Cargando modelo Fine-Tuned desde: {args.model_path}")
    model = DIVA(args).to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.eval()

    # 2. Extraer Latentes de Base (Sanos)
    print("Extrayendo latentes de la base Healthy y agrupando por Gesto...")
    base_loader = load_split(ROOT_preprocessed, "training", BASE_SUBJECT_NAME, batch_size=args.batch_size, shuffle=False)
    latents_by_gesture = extract_latents(base_loader, model, device)
    
    N_base_total = sum(len(d['zs']) for d in latents_by_gesture.values())
    
    os.makedirs(args.generated_dir, exist_ok=True)
    
    print(f"\nIniciando Generación Condicional de {len(PATHOLOGY_MAP)} Patologías...")

    # 4. Bucle de Generación por Patología
    for pat_name, pat_id in PATHOLOGY_MAP.items():
        print(f"  → Generando señales para: {pat_name} (ID: {pat_id})")
        
        num_samples = int(N_base_total * args.gen_factor)
        output_path = os.path.join(args.generated_dir, f"generated_{pat_name}.npy") 
        
        # 🚨 LLamada a la función de generación (sin el sufijo _RAM_test)
        X_gen, Y_labels, C_labels, total_generated = generate_signals(
            model, device, latents_by_gesture, pat_id, num_samples, args.gen_batch_size
        )
        
        # 🚨 GUARDADO FINAL USANDO NUMPY
        np.save(
            output_path, 
            {"X": X_gen, "Y": Y_labels, "C": C_labels}, 
            allow_pickle=True
        )
        print(f"  → Guardado {total_generated} muestras en {output_path}")

    print("\n¡Generación Condicional Completada!")


# -------------------------- 5. Argparse --------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Conditional sEMG Generation (DIVA)")
    parser.add_argument('--model-path', type=str, default='./saved_model/diva_best_seed0.model',
                        help="Ruta al modelo DIVA Fine-Tuned (.model)")
    parser.add_argument('--generated-dir', type=str, default='./generated_data', # Cambio de carpeta
                        help="Ruta para guardar las señales generadas (.npy)")
    parser.add_argument('--batch-size', type=int, default=512)
    parser.add_argument('--no-cuda', action='store_true', default=False)
    
    parser.add_argument('--gen-factor', type=float, default=1.5,
                        help="Factor multiplicador para el número de muestras a generar.")
    parser.add_argument('--gen-batch-size', type=int, default=256, 
                        help="Tamaño de lote usado durante la generación (mini-batch).")
    
    # Parámetros del modelo (deben coincidir con el entrenamiento)
    parser.add_argument('--c-dim', type=int, default=C_DIM)
    parser.add_argument('--d-dim', type=int, default=10)
    parser.add_argument('--x-dim', type=int, default=416)
    parser.add_argument('--y-dim', type=int, default=TOTAL_GESTURES)
    parser.add_argument('--zd-dim', type=int, default=128)
    parser.add_argument('--zx-dim', type=int, default=128)
    parser.add_argument('--zy-dim', type=int, default=128)
    parser.add_argument('--aux_loss_multiplier_y', type=float, default=3500.)
    parser.add_argument('--aux_loss_multiplier_d', type=float, default=2000.)
    parser.add_argument('--beta_d', type=float, default=1.)
    parser.add_argument('--beta_x', type=float, default=1.)
    parser.add_argument('--beta_y', type=float, default=1.)
    
    args = parser.parse_args()
    main_generation(args)