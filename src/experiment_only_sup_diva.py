import argparse
import os
import time
import random
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
import torch.utils.data as data_utils
import torch.serialization
import matplotlib.pyplot as plt
from model_diva import DIVA
from utils.semgdata_loader import semgdata_load

TOTAL_SUBJECTS = 12
CROSS_SUBJECT = 2
TRAIN_SUBJECTS = TOTAL_SUBJECTS - CROSS_SUBJECT
GESTURES = 5

# ------------------------------ Utils ------------------------------ #
def save_reconstructions(model, out_dir, x_batch, epoch, max_images=8):
    """
    Intenta pedir reconstrucciones al modelo y guardarlas como una imagen tipo "heatmap".
    Si el modelo no implementa un método de reconstrucción directo, se salva el input como sanity check.
    """
    os.makedirs(out_dir, exist_ok=True)
    try:
        model.eval()
        with torch.no_grad():
            # intentar llamados comunes (ajusta según la API real si es distinta)
            if hasattr(model, "reconstruct"):
                recon = model.reconstruct(x_batch.to(next(model.parameters()).device))
            elif hasattr(model, "forward_reconstruct"):
                recon = model.forward_reconstruct(x_batch.to(next(model.parameters()).device))
            else:
                # fallback: usar identity (guardamos input)
                recon = x_batch.to(next(model.parameters()).device)

            recon = recon.detach().cpu().numpy()
    except Exception as e:
        # si falla, guardamos input para inspección
        recon = x_batch.detach().cpu().numpy()

    # guardamos hasta max_images ejemplos
    n = min(max_images, recon.shape[0])
    for i in range(n):
        arr = recon[i].squeeze()  # (1,8,52) -> (8,52) or (8,52)
        if arr.ndim == 3:
            arr = arr[0]  # if shape (1,8,52)
        plt.figure(figsize=(6,2))
        plt.imshow(arr, aspect='auto', origin='lower')
        plt.colorbar()
        plt.title(f'epoch{epoch}_img{i}')
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f'epoch{epoch}_img{i}.png'))
        plt.close()


def compute_accuracy_from_logits(pred_logits, y_onehot):
    """
    pred_logits: tensor (B, C) or probabilities/logits
    y_onehot: tensor (B, C) one-hot
    returns float accuracy (0..1)
    """
    pred_labels = pred_logits.argmax(dim=1)
    true_labels = y_onehot.argmax(dim=1)
    correct = (pred_labels == true_labels).sum().item()
    return correct / float(pred_labels.size(0))


# Entrenamiento por época
def train_one_epoch(train_loader, model, optimizer, device):
    model.train()
    total_loss = 0.0
    total_class_y_loss = 0.0
    num_batches = 0
    for (x, y, d) in train_loader:
        x = x.to(device)
        y = y.to(device)
        d = d.to(device)
        optimizer.zero_grad()
        loss, class_y_loss, zd_q, zy_q, zx_q, d_target, y_target = model.loss_function(d, x, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        total_class_y_loss += class_y_loss.item() if hasattr(class_y_loss, 'item') else float(class_y_loss)
        num_batches += 1
    if num_batches == 0:
        return 0.0, 0.0
    return total_loss / num_batches, total_class_y_loss / num_batches


def evaluate(loader, model, device):
    model.eval()
    acc_y = []
    acc_d = []
    with torch.no_grad():
        for (x, y, d) in loader:
            x = x.to(device)
            y = y.to(device)
            d = d.to(device)
            pred_d, pred_y = model.classifier(x)
            # pred_y and pred_d likely logits; y,d are one-hot
            acc_y.append(compute_accuracy_from_logits(pred_y, y))
            acc_d.append(compute_accuracy_from_logits(pred_d, d))
    if len(acc_y) == 0:
        return 0.0, 0.0
    return float(np.mean(acc_d)), float(np.mean(acc_y))


# ------------------------------ Main ------------------------------ #
def main(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    device = torch.device("cuda" if (not args.no_cuda and torch.cuda.is_available()) else "cpu")
    kwargs = {'num_workers': 4, 'pin_memory': True} if device.type == 'cuda' else {}

    # Cargar info sobre cuántos sujetos hay en el split 'training'
    training_npy = os.path.join(args.preprocessed_root, "training.npy")
    if not os.path.exists(training_npy):
        raise FileNotFoundError(f"{training_npy} not found. Asegúrate de haber corrido el preprocesado.")

    # leer número de sujetos
    tmp = np.load(training_npy, allow_pickle=True).item()
    all_data = tmp["list_total_user_data"]
    num_subjects = len(all_data)
    print(f"🔎 Encontrados {num_subjects} sujetos en 'training' split.")

    results_all_seeds = []

    # Repeticiones por semilla (como en el original)
    for seed in range(args.seed_repeats):
        print("="*60)
        print(f"RUN seed={seed}")
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        # Guardar resultados por sujeto
        seed_results = []
        # Leave-One-Subject-Out (LOSO)
        for test_idx in range(num_subjects):
            print(f"\n--- LOSO test subject = {test_idx} ---")
            ## 🔥 Justo antes de crear el modelo dentro del loop LOSO
            train_subjects = [i for i in range(num_subjects) if i != test_idx]
            test_subjects  = [test_idx]

            # Ajuste dinámico del d_dim según sujetos de entrenamiento
            args.d_dim = len(train_subjects)  # <- esto corrige el error mat1 x mat2

            # DataLoaders
            train_ds = semgdata_load(root=args.preprocessed_root, split="training", subjects=train_subjects, transform=None)
            val_ds   = semgdata_load(root=args.preprocessed_root, split="validation", subjects=train_subjects, transform=None)
            test_ds  = semgdata_load(root=args.preprocessed_root, split="training", subjects=test_subjects, transform=None)

            train_loader = data_utils.DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, **kwargs)
            val_loader   = data_utils.DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, **kwargs)
            test_loader  = data_utils.DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, **kwargs)

            # Build model and optimizer
            model = DIVA(args).to(device)
            optimizer = optim.Adam(model.parameters(), lr=args.lr)

            # Early stopping vars
            best_val_acc = -1.0
            best_val_loss = float('inf')
            early_counter = 0
            best_model_path = os.path.join(args.outdir, f"diva_seed{seed}_loso_test{test_idx}.model")

            # Logging per epoch
            logs = []

            # Training loop
            for epoch in range(1, args.epochs + 1):
                # Warm-up betas: linear schedule from min_beta -> max_beta over args.warmup epochs
                if args.warmup > 0 and epoch <= args.warmup:
                    frac = epoch / float(max(1, args.warmup))
                    model.beta_d = args.min_beta + frac * (args.max_beta - args.min_beta)
                    model.beta_x = args.min_beta + frac * (args.max_beta - args.min_beta)
                    model.beta_y = args.min_beta + frac * (args.max_beta - args.min_beta)
                else:
                    model.beta_d = args.max_beta
                    model.beta_x = args.max_beta
                    model.beta_y = args.max_beta

                # Train epoch
                t0 = time.time()
                print(f"--- Epoch {epoch} ---")
                train_loss, train_class_y_loss = train_one_epoch(train_loader, model, optimizer, device)
                t1 = time.time()

                # Evaluate on validation (and training for bookkeeping)
                val_acc_d, val_acc_y = evaluate(val_loader, model, device)
                train_acc_d, train_acc_y = evaluate(train_loader, model, device)

                log = {
                    "seed": seed,
                    "test_subject": test_idx,
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "train_class_y_loss": train_class_y_loss,
                    "train_acc_y": train_acc_y,
                    "train_acc_d": train_acc_d,
                    "val_acc_y": val_acc_y,
                    "val_acc_d": val_acc_d,
                    "time_epoch_s": t1-t0,
                    "beta": model.beta_y
                }
                logs.append(log)

                # Print progress
                print(f"[s{seed} t{test_idx}] epoch {epoch:03d} | tr_acc_y {train_acc_y:.4f} val_acc_y {val_acc_y:.4f} | tr_loss {train_loss:.4f} | beta {model.beta_y:.3f}")

                # Save reconstructions occasionally
                if epoch % args.recon_every == 0:
                    try:
                        save_reconstructions(model, out_dir=os.path.join(args.outdir, "recons", f"seed{seed}_test{test_idx}"), x_batch=next(iter(train_loader))[0], epoch=epoch, max_images=6)
                    except Exception as e:
                        # no block on errors here
                        print("⚠️ save_reconstructions failed:", e)

                # Early stopping & checkpointing based on val accuracy (y)
                # prefer higher val_acc_y; if tie, smaller val loss
                if val_acc_y > best_val_acc or (val_acc_y == best_val_acc and train_class_y_loss < best_val_loss):
                    best_val_acc = val_acc_y
                    best_val_loss = train_class_y_loss
                    early_counter = 0
                    # save model
                    torch.save(model.state_dict(), best_model_path)
                    print(f"✅ New best model saved (val_acc_y={best_val_acc:.4f}) -> {best_model_path}")
                else:
                    early_counter += 1
                    if early_counter >= args.early_stop_patience:
                        print(f"⏱ Early stopping triggered (no improv for {args.early_stop_patience} epochs)")
                        break


            # After training -> load best model (if exists)
            if os.path.exists(best_model_path):
                model.load_state_dict(torch.load(best_model_path, map_location=device))
            else:
                print("⚠️ Best model not found, using last model from training.")

            # Evaluate on test subjec
            model.eval()
            test_acc_d, test_acc_y = evaluate(test_loader, model, device)
            print(f"*** Final test accuracy (subject {test_idx}) - y: {test_acc_y:.4f}, d: {test_acc_d:.4f}")

            # Save per-subject logs to Excel
            df_logs = pd.DataFrame(logs)
            out_xls = os.path.join(args.outdir, f"logs_seed{seed}_test{test_idx}.xlsx")
            df_logs.to_excel(out_xls, index=False)
            print(f"📄 Saved per-epoch logs to {out_xls}")

            seed_results.append({
                "seed": seed,
                "test_subject": test_idx,
                "test_acc_y": test_acc_y,
                "test_acc_d": test_acc_d,
                "best_val_acc_y": best_val_acc,
                "best_val_loss": best_val_loss,
                "n_epochs_trained": epoch
            })

        # Save seed summary
        df_seed = pd.DataFrame(seed_results)
        out_seed_xls = os.path.join(args.outdir, f"summary_seed{seed}.xlsx")
        df_seed.to_excel(out_seed_xls, index=False)
        print(f"📄 Saved LOSO seed summary to {out_seed_xls}")
        results_all_seeds.append(seed_results)

    # Save final aggregated results
    all_df = pd.DataFrame([item for seed_res in results_all_seeds for item in seed_res])
    all_xls = os.path.join(args.outdir, "final_loso_results.xlsx")
    all_df.to_excel(all_xls, index=False)
    print(f"\n🏁 LOSO experiment finished. Final results saved to {all_xls}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--no-cuda', action='store_true', default=False)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--seed-repeats', type=int, default=3, help="cuántas repeticiones con distintas seeds (original=10)")
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--outdir', type=str, default='./saved_model')
    parser.add_argument('--preprocessed-root', type=str, default='./preprocessed_dataset', help="carpeta con training.npy, validation.npy, testing.npy, cross_subject.npy")
    parser.add_argument('--d-dim', type=int, default=TRAIN_SUBJECTS)
    parser.add_argument('--y-dim', type=int, default=GESTURES)
    parser.add_argument('--zd-dim', type=int, default=128)
    parser.add_argument('--zx-dim', type=int, default=128)
    parser.add_argument('--x-dim', type=int, default=416, help='dimensión total de entrada (por ejemplo 8*52)')
    parser.add_argument('--zy-dim', type=int, default=128)
    parser.add_argument('--aux_loss_multiplier_y', type=float, default=3500.)
    parser.add_argument('--aux_loss_multiplier_d', type=float, default=2000.)
    parser.add_argument('--beta_d', type=float, default=1.)
    parser.add_argument('--beta_x', type=float, default=1.)
    parser.add_argument('--beta_y', type=float, default=1.)
    parser.add_argument('--warmup', type=int, default=100)
    parser.add_argument('--min_beta', type=float, default=0.0)
    parser.add_argument('--max_beta', type=float, default=3.0)
    parser.add_argument('--recon_every', type=int, default=20, help="guardar reconstructions cada N epochs")
    parser.add_argument('--early_stop_patience', type=int, default=10)
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    main(args)
