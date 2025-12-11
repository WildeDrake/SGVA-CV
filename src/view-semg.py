import numpy as np
import matplotlib.pyplot as plt
import os

# --- CONFIGURACIÓN ---
# Ruta a tu archivo (ajusta esto según dónde guardes este script)
FILE_PATH = '../dataset/s1/all/LEFT/Fist-1.txt' 

def plot_8_channels(file_path):
    if not os.path.exists(file_path):
        print(f"Error: No se encontró el archivo en: {file_path}")
        return

    # 1. Cargar los datos
    # np.loadtxt detecta automáticamente los espacios como separadores
    try:
        data = np.loadtxt(file_path)
    except Exception as e:
        print(f"Error leyendo el archivo: {e}")
        return

    # Verificación rápida de forma
    rows, cols = data.shape
    print(f"Datos cargados: {rows} muestras temporales, {cols} canales.")

    if cols != 8:
        print("Advertencia: El archivo no parece tener 8 columnas.")

    # 2. Configurar la gráfica (8 filas, 1 columna)
    # sharex=True hace que al hacer zoom en uno, se muevan todos a la vez (muy útil)
    fig, axes = plt.subplots(nrows=8, ncols=1, figsize=(10, 12), sharex=True)
    
    # Título General
    fig.suptitle(f'Señales EMG - {os.path.basename(file_path)}', fontsize=16)

    # 3. Graficar cada canal
    for i in range(8):
        # Seleccionamos la columna i (data[:, i])
        signal = data[:, i]
        
        # Graficamos en el subplot correspondiente
        # Color 'k' es negro, lw es el grosor de línea
        axes[i].plot(signal, color='#1f77b4', linewidth=0.8) 
        
        # Estética
        axes[i].set_ylabel(f'CH {i+1}', rotation=0, labelpad=20, fontweight='bold')
        axes[i].grid(True, alpha=0.3)
        
        # Opcional: Fijar límites Y si quieres que todos se vean a la misma escala
        # axes[i].set_ylim(-128, 127) # Si es de 8 bits signed

    # Etiqueta del eje X solo en el último gráfico
    axes[-1].set_xlabel('Tiempo (Muestras)')

    # Ajustar espacios para que no se solapen
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    # Guardar o Mostrar
    # plt.savefig('emg_plot.png', dpi=300) # Descomenta para guardar
    plt.show()

if __name__ == "__main__":
    plot_8_channels(FILE_PATH)