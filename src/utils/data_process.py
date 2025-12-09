import numpy as np
import pandas as pd
from scipy import signal
import logging


#--------------------- Procesamiento de datos de archivos txt a matrices numpy ---------------------#
# leer txt sin modificar en la carpeta all
def txt2array(txt_path):
    """
    :param txt_path:    ruta específica de un solo archivo txt
    :return:            matriz preprocesada de dimensión 2 <class 'np.ndarray'> del archivo txt de entrada
    """
    table_file = pd.read_table(txt_path, header=None)
    txt_file = table_file.iloc[:, :-1]
    txt_array = txt_file.values
    return txt_array


# --------------------- Preprocesamiento de datos EMG --------------------#
def preprocessing(data):
    """
    Preprocesa una matriz EMG leída del .txt.
    - Mantiene la amplitud real.
    - Rectifica (abs), transpone a (channels, samples).
    - Filtra con lowpass Butterworth (orden 4, wn=0.05).
    - Retorna array (channels, samples), dtype float32.
    """
    # Asegurar numpy array float
    data = np.asarray(data, dtype=float)

    # Rectificar
    arr = np.abs(data)

    # Transponer (channels, samples)
    if arr.shape[0] < arr.shape[-1] and arr.shape[0] <= 8:
        arr_proc = arr
    else:
        arr_proc = arr.T

    # Filtrado (lowpass)
    wn = 0.05
    order = 4
    try:
        b, a = signal.butter(order, wn, btype='low')
        arr_filtered = signal.filtfilt(b, a, arr_proc, axis=1)
    except Exception:
        arr_filtered = arr_proc

    return arr_filtered.astype(np.float32)


# --------------------- Detección de la región de actividad muscular --------------------#
def detect_muscle_activity(emg_data,
                           fs=200,
                           rms_win_ms=50,
                           rms_step_ms=10,
                           energy_threshold_factor=3.0,
                           percentile_threshold=75,
                           min_activation_length=40,
                           max_gap_ms=100,
                           padding_ms=100):
    """
    Detecta regiones activas con un método RMS sliding + umbral adaptativo.
    Devuelve índices (start, end) en muestras (0-indexed).
    Parámetros ajustables:
    - fs: frecuencia muestreo (Hz)
    - rms_win_ms: ventana RMS en ms
    - rms_step_ms: paso RMS en ms
    - energy_threshold_factor: factor multiplicador sobre la mediana/mean para threshold
    - percentile_threshold: fallback percentile del vector RMS para threshold si es más conservador
    - min_activation_length: length mínima en muestras (si el segmento es más corto, asumimos toda la señal)
    - max_gap_ms: gaps menores a esto se unen
    - padding_ms: añadir padding antes/después de cada segmento (más realista para DIVA)
    """
    # emg_data: (channels, samples)
    emg = np.asarray(emg_data, dtype=float)
    if emg.ndim != 2:
        raise ValueError("emg_data debe ser (channels, samples)")

    n_channels, n_samples = emg.shape
    if n_samples <= 0:
        return 0, 0

    # 1) calcular RMS combinado (suma de RMS por canal)
    win = int(rms_win_ms * fs / 1000)
    step = int(rms_step_ms * fs / 1000)
    win = max(win, 4)
    step = max(step, 1)
    # calcular RMS por canal con convolución
    squared = emg ** 2
    kernel = np.ones(win)
    rms_per_channel = np.zeros_like(squared)
    for ch in range(squared.shape[0]):
        rms_per_channel[ch] = np.sqrt(
            signal.convolve(
                squared[ch],
                kernel / win,
                mode='same'
            )
        )
    # suma o mean across channels
    energy = rms_per_channel.mean(axis=0)

    # 2) Umbral adaptativo: combinación de factores
    global_mean = np.mean(energy)
    global_median = np.median(energy)
    perc = np.percentile(energy, percentile_threshold)
    # threshold candidate basado en factor y en percentil
    thr1 = global_median * energy_threshold_factor
    thr2 = perc
    threshold = max(thr1, thr2, global_mean * 0.5 * energy_threshold_factor)
    # si la energía global es muy baja, bajamos el umbral relativo para no descartar todo
    if global_median < 1e-6:
        threshold = max(threshold * 0.1, 1e-6)

    # 3) Binarizar energy por threshold
    active_mask = energy > threshold

    # 4) Densificar el mask usando convolution
    max_gap = int(max_gap_ms * fs / 1000)
    if max_gap > 0:
        # cerrar gaps pequeños: si en window of size max_gap existan 1
        fill_kernel = np.ones(max_gap)
        fill_conv = signal.convolve(active_mask.astype(int), fill_kernel, mode='same')
        active_mask = fill_conv > 0

    # 5) Encontrar bordes
    dif = np.diff(np.concatenate(([0], active_mask.astype(int), [0])))
    starts = np.where(dif == 1)[0]
    ends = np.where(dif == -1)[0]  # end indices (exclusive)
    # si no hay segmentos detectados -> devolver toda la señal
    if len(starts) == 0:
        # fallback: devolver toda la señal
        start_idx = 0
        end_idx = n_samples - 1
        return int(start_idx), int(end_idx)

    # 6) Unir segmentos cercanos
    merged = []
    cur_s, cur_e = starts[0], ends[0]
    for s, e in zip(starts[1:], ends[1:]):
        if s <= cur_e + max_gap:
            cur_e = e
        else:
            merged.append((cur_s, cur_e))
            cur_s, cur_e = s, e
    merged.append((cur_s, cur_e))

    # 7) seleccionar el segmento más largo
    best_seg = max(merged, key=lambda se: se[1] - se[0])
    start_idx, end_idx = best_seg

    # 8) añadir padding
    pad = int(padding_ms * fs / 1000)
    start_idx = max(0, start_idx - pad)
    end_idx = min(n_samples - 1, end_idx + pad)

    # 9) garantizar longitud mínima
    if (end_idx - start_idx) < min_activation_length:
        # fallback: devolver toda la señal, más seguro para patologías
        start_idx = 0
        end_idx = n_samples - 1

    return int(start_idx), int(end_idx)



def label_indicator(path, emg_data=None):
    label = None
    if 'Fist' in path:
        label = 0
    elif 'Open' in path:
        label = 1
    elif 'Tap' in path:
        label = 2
    elif 'WaveIn' in path:
        label = 3
    elif 'WaveOut' in path:
        label = 4
    return label