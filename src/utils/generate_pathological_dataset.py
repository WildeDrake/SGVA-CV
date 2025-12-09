import os
import glob
import numpy as np


#  Cargar archivos .txt de EMG
def load_emg_txt(path):
    return np.loadtxt(path, dtype=int)

#  Guardar archivos .txt de EMG
def save_emg_txt(path, matrix):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    temp = []
    for row in matrix:
        line = "\t".join(str(int(v)) for v in row) + "\t"
        temp.append(line)
    with open(path, "w") as f:
        f.write("\n".join(temp))


#  Transformaciones patológicas
def apply_pathology(matrix, pathology_name):
    mat = matrix.astype(float)

    # Aplicar modificaciones de DMD
    if pathology_name == "DMD":
        # 1. Estadísticas del archivo
        abs_mat = np.abs(mat)
        mean_amp = abs_mat.mean()
        max_amp = abs_mat.max()
        min_amp = abs_mat.min()
        range_amp = max_amp - min_amp
        range_amp = max(range_amp, 1e-8)
        #   Severidad proporcional al nivel de activación original
        #   Señales más fuertes -> mayor caída relativa
        severity = mean_amp / range_amp
        severity = np.clip(severity, 0.3, 0.85)
        # 2. Decaimiento progresivo
        #   Los últimos frames estarán 40–60% más deteriorados
        final_drop = np.random.uniform(0.4, 0.6)
        decay_curve = np.linspace(1.0, final_drop * severity, mat.shape[0]).reshape(-1, 1)
        mat = mat * decay_curve
        # 3. Ruido miopático proporcional
        #   Ruido = 5%–15% de la amplitud media local
        noise_scale = np.random.uniform(0.05, 0.15) * mean_amp
        noise = np.random.normal(0, noise_scale, mat.shape)
        mat = mat + noise
        # 4. degeneración de fibras
        drop_mask = np.random.rand(*mat.shape) < 0.02   # 2% de los valores
        mat[drop_mask] *= np.random.uniform(0.1, 0.4)   # caída abrupta
        gap_mask = np.random.rand(*mat.shape) < 0.005   # 0.5%
        mat[gap_mask] = 0                               # ausencia súbita de activación
        # 5. Suavizado leve de cada canal
        kernel = np.array([0.2, 0.6, 0.2])
        for c in range(mat.shape[1]):
            mat[:, c] = np.convolve(mat[:, c], kernel, mode="same")

    #  Aplicar modificaciones de Neuropatía
    elif pathology_name == "Neuropathy":
        # 1. Estadísticas generales
        mean_amp = np.mean(mat)
        max_amp = np.max(mat)
        # 2. Reducción de amplitud por pérdida de unidades motoras
        #   Reducción aleatoria entre 30–60% según severidad simulada
        reduction_factor = np.random.uniform(0.4, 0.7)
        mat = mat * reduction_factor
        # 3. Variabilidad no-lineal (debilidad irregular)
        #   Se modula cada fila con factores diferentes
        variability = np.random.normal(1.0, 0.15, (mat.shape[0], 1))
        mat = mat * variability
        # 4. Ruido spiky simulando descargas espontáneas
        #   Se generan spikes raros pero intensos
        spikes = np.zeros_like(mat)
        num_spikes = max(1, mat.shape[0] // 20)  # 5% de las muestras
        spike_positions = np.random.choice(mat.shape[0], num_spikes, replace=False)
        for pos in spike_positions:
            spike = np.random.uniform(0.5 * mean_amp, max_amp)  # amplitud alta
            spikes[pos] = spike * (np.random.rand(mat.shape[1]) - 0.5) * 2
        mat = mat + spikes
        # 5. Crear silencios parciales
        silent_fraction = 0.1  # 10% del registro
        silent_rows = np.random.choice(mat.shape[0], int(mat.shape[0] * silent_fraction), replace=False)
        mat[silent_rows] *= np.random.uniform(0.05, 0.2)  # casi apagado

    #  Aplicar modificaciones de Stroke
    elif pathology_name == "Stroke":
        # 1. Estadísticas del archivo original
        mean_amp = np.mean(mat)
        std_amp = np.std(mat)
        max_amp = np.max(mat)
        # 2. Reducción de amplitud (fuerza reducida post-ACV)
        reduction_factor = np.random.uniform(0.4, 0.7)
        mat = mat * reduction_factor
        # 3. Co-contracción: mezcla moderada entre canales
        if mat.shape[1] > 1:
            # Mezcla suave (20–50% entre canales vecinos)
            mix_strength = np.random.uniform(0.2, 0.5)
            mixed = mat[:, :-1] * (1 - mix_strength) + mat[:, 1:] * mix_strength
            mat[:, :-1] = mixed
        # 4. Burst tones: activaciones bruscas involuntarias
        burst_positions = np.random.choice(
            mat.shape[0],
            size=max(1, mat.shape[0] // 30),  # ~3% de las filas
            replace=False
        )
        for pos in burst_positions:
            burst_amp = np.random.uniform(0.4 * mean_amp, 0.8 * max_amp)
            mat[pos] += burst_amp * (np.random.rand(mat.shape[1]) - 0.5) * 2
        # 5. Rigidez muscular / espasticidad: suavizado leve
        kernel = np.array([0.2, 0.6, 0.2])
        for c in range(mat.shape[1]):
            mat[:, c] = np.convolve(mat[:, c], kernel, mode="same")
        # 6. Agregar ruido leve (temblores o variaciones irregulares)
        noise = np.random.normal(0, std_amp * 0.05, mat.shape)
        mat = mat + noise

    #  Aplicar modificaciones de Parkinson
    elif pathology_name == "Parkinson":
        # 1. Estadísticas originales
        mean_amp = np.mean(mat)
        std_amp = np.std(mat)
        # 2. Temblor principal 4–6 Hz
        tremor_freq = np.random.uniform(4, 6)
        t = np.linspace(0, 2 * np.pi, mat.shape[0])
        # Amplitud del temblor escalada al nivel del archivo
        tremor_amp = std_amp * np.random.uniform(0.4, 1.0)
        tremor = tremor_amp * np.sin(t * tremor_freq)
        # Añadir el temblor a todos los canales
        mat += tremor.reshape(-1, 1)
        # 3. Variación aleatoria del temblor (temblor no perfecto)
        jitter = np.random.normal(0, tremor_amp * 0.1, mat.shape[0])
        mat += jitter.reshape(-1, 1)
        # 4. Bursts (típicos en temblor parkinsoniano en EMG)
        num_bursts = max(1, mat.shape[0] // 200)  # 0.5% del largo
        burst_pos = np.random.choice(mat.shape[0], size=num_bursts, replace=False)
        for pos in burst_pos:
            burst_strength = np.random.uniform(0.3, 0.8) * tremor_amp
            mat[pos] += burst_strength * (np.random.rand(mat.shape[1]) - 0.5) * 2
        # 5. Co-contracción: mezcla entre canales
        if mat.shape[1] > 1:
            mix = np.random.uniform(0.1, 0.3)
            mat[:, :-1] = mat[:, :-1] * (1 - mix) + mat[:, 1:] * mix
        # 6. Rigidez muscular: suavizado leve
        kernel = np.array([0.25, 0.5, 0.25])
        for c in range(mat.shape[1]):
            mat[:, c] = np.convolve(mat[:, c], kernel, mode="same")
        # 7. Pequeño ruido adicional
        noise = np.random.normal(0, std_amp * 0.02, mat.shape)
        mat += noise

    #  Aplicar modificaciones de ALS (Esclerosis Lateral Amiotrófica)
    elif pathology_name == "ALS":
        # 1. Estadísticas
        abs_mat = np.abs(mat)
        mean_amp = abs_mat.mean()
        max_amp = abs_mat.max()
        std_amp = np.std(mat)
        # 2. Pérdida progresiva de unidades motoras
        #   Reducción fuerte pero irregular
        reduction_curve = np.linspace(
            np.random.uniform(0.7, 0.9),        # inicio casi normal
            np.random.uniform(0.2, 0.4),        # final muy debilitado
            mat.shape[0]
        ).reshape(-1, 1)
        mat = mat * reduction_curve
        # 3. Fasciculaciones
        num_fasc = max(3, mat.shape[0] // 40)  # ~2.5% filas
        fasc_positions = np.random.choice(mat.shape[0], num_fasc, replace=False)
        for pos in fasc_positions:
            fasc_amp = np.random.uniform(0.3 * mean_amp, max_amp)
            mat[pos] += fasc_amp * (np.random.rand(mat.shape[1]) - 0.5) * 2
        # 4. Reinnervation jitter (no lineal, caótico)
        jitter = np.random.normal(1.0, 0.25, (mat.shape[0], 1))
        mat = mat * jitter
        # 5. Debilidad + ruido leve final
        noise = np.random.normal(0, std_amp * 0.03, mat.shape)
        mat += noise
        # 6. Suavizado leve (recrutamiento lento)
        kernel = np.array([0.25, 0.5, 0.25])
        for c in range(mat.shape[1]):
            mat[:, c] = np.convolve(mat[:, c], kernel, mode="same")


    #  Aplicar modificaciones de Artifact (ruido por movimiento)
    elif pathology_name == "Artifact":
        # 1. Estadísticas base
        mean_amp = np.mean(np.abs(mat))
        std_amp = np.std(mat)
        # 2. Motion artifacts: offsets repentinos
        num_jumps = max(1, mat.shape[0] // 200)
        jump_positions = np.random.choice(mat.shape[0], num_jumps, replace=False)
        for pos in jump_positions:
            jump = np.random.uniform(-1.5*mean_amp, 1.5*mean_amp)
            mat[pos:] += jump   # se arrastra el offset
        # 3. Line noise 50–60 Hz
        freq = np.random.uniform(48, 62)
        t = np.linspace(0, 2*np.pi, mat.shape[0])
        line_noise = (mean_amp * 0.15) * np.sin(freq * t)
        mat += line_noise.reshape(-1, 1)
        # 4. Pulsos bruscos tipo cable suelto
        num_spikes = max(1, mat.shape[0] // 150)
        spike_positions = np.random.choice(mat.shape[0], num_spikes, replace=False)
        for pos in spike_positions:
            spike = np.random.uniform(0.5*mean_amp, 2*mean_amp)
            mat[pos] += spike * (np.random.rand(mat.shape[1]) - 0.5) * 2
        # 5. Ruido blanco fuerte
        white_noise = np.random.normal(0, std_amp * 0.4, mat.shape)
        mat += white_noise
        # 6. Clip para evitar explosiones de amplitud
        max_clip = np.percentile(np.abs(mat), 99)
        mat = np.clip(mat, -max_clip, max_clip)

    else:
        raise ValueError(f"Patología desconocida: {pathology_name}")
    return np.round(mat).astype(int)


#  Procesar un sujeto individual
def process_subject(subject_path, pathology_name):
    all_path = os.path.join(subject_path, "all")
    if not os.path.exists(all_path):
        return
    pathology_path = os.path.join(subject_path, pathology_name)
    os.makedirs(pathology_path, exist_ok=True)
    # Recorrer LEFT y RIGHT
    for hand in ["LEFT", "RIGHT"]:
        hand_path = os.path.join(all_path, hand)
        if not os.path.exists(hand_path):
            continue
        # Crear carpeta destino para LEFT/RIGHT
        out_hand_dir = os.path.join(pathology_path, hand)
        os.makedirs(out_hand_dir, exist_ok=True)
        # Recorrer todos los .txt del gesto
        txt_files = glob.glob(os.path.join(hand_path, "*.txt"))
        for txt_path in txt_files:
            filename = os.path.basename(txt_path)
            # cargar, modificar y guardar
            mat = load_emg_txt(txt_path)
            mat_p = apply_pathology(mat, pathology_name)
            save_emg_txt(os.path.join(out_hand_dir, filename), mat_p)

#  Procesar el dataset completo
def process_dataset(root_dir, pathology_name):
    subjects = sorted([s for s in os.listdir(root_dir) if s.startswith("s")])
    print(f"Procesando sujetos: {subjects}")
    for subj in subjects:
        process_subject(os.path.join(root_dir, subj), pathology_name)
    print("  Dataset patológico generado correctamente.")


#  Crear datasets patológicos
def create_pathological_dataset(path="../../dataset"):
    process_dataset(path, "DMD")
    process_dataset(path, "Neuropathy")
    process_dataset(path, "Stroke")
    process_dataset(path, "Parkinson")
    process_dataset(path, "Artifact")
    process_dataset(path, "ALS")


# main
if __name__ == "__main__":
    create_pathological_dataset()