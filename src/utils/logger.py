import os
import csv
import matplotlib.pyplot as plt
import json

class TrainerLogger:
    def __init__(self, outdir, log_name="metrics"):
        self.outdir = outdir
        os.makedirs(outdir, exist_ok=True)
        self.csv_path = os.path.join(outdir, f"{log_name}.csv")
        self.plot_path = os.path.join(outdir, f"{log_name}.png")
        self.metrics = []

        # Crear CSV vacío si no existe
        if not os.path.exists(self.csv_path):
            with open(self.csv_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    "epoch",
                    "train_loss", "train_class_y_loss", "train_loss_y", "train_loss_d",
                    "val_loss", "val_class_y_loss", "val_loss_y", "val_loss_d",
                    "cross_loss_y", "cross_loss_d"
                ])

    def log_epoch(self, epoch, train_loss, train_class_y_loss, train_loss_y, train_loss_d,
                  val_loss=None, val_class_y_loss=None, val_loss_y=None, val_loss_d=None,
                  cross_loss_y=None, cross_loss_d=None):
        row = [epoch, train_loss, train_class_y_loss, train_loss_y, train_loss_d,
               val_loss, val_class_y_loss, val_loss_y, val_loss_d,
               cross_loss_y, cross_loss_d]
        self.metrics.append(row)

        # Guardar en CSV
        with open(self.csv_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(row)

        # Actualizar gráfico
        self._plot_metrics()

    def _plot_metrics(self):
        if len(self.metrics) == 0:
            return

        epochs = [m[0] for m in self.metrics]
        train_loss = [m[2] for m in self.metrics]
        val_loss = [m[6] for m in self.metrics if m[6] is not None]
        cross_loss = [m[9] for m in self.metrics if m[9] is not None]

        plt.figure(figsize=(8,5))
        plt.plot(epochs, train_loss, label="Train loss")
        if val_loss:
            plt.plot(epochs[:len(val_loss)], val_loss, label="Val loss")
        if cross_loss:
            plt.plot(epochs[:len(cross_loss)], cross_loss, label="Cross loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(self.plot_path)
        plt.close()
