"""
music_data_utils.py

Módulo de utilidades para el procesamiento de archivos MIDI enfocado en Deep Learning.
Provee funciones para:
1. Parsear archivos MIDI a una representación de eventos textuales (tokens).
2. Aplicar cuantización híbrida (binaria/ternaria) para limpiar el ritmo.
3. Realizar Data Augmentation mediante transposición inteligente.
4. Decodificar secuencias de tokens nuevamente a archivos MIDI escuchables.

Formato de Token: "ON=pitch1,pitch2;OFF=pitch3;DUR=tiempo"
"""

from music21 import midi, note, chord, stream, tempo, instrument
from collections import defaultdict
from typing import List, Dict, Tuple, Any
import warnings
import statistics


def snap_to_grid(duration):
    """
    Aplica una Cuantización Inteligente Híbrida a una duración dada.
    
    El algoritmo compara el error de ajuste contra dos grillas:
    1. Binaria: Base 1/32 (Fusas).
    2. Ternaria: Base 1/48 (Tresillos de Fusa).
    
    Args:
        duration (float): Duración del evento en 'quarter lengths' (tiempos de negra).

    Returns:
        float: La duración cuantizada al grid que produzca el menor error. 
               Retorna 0.0 si la duración es menor al umbral de ruido (0.04).
    """
    
    # Grid A: Binario (Base 1/32 - Fusa)
    grid_binary = 0.125
    steps_bin = round(duration / grid_binary)
    snapped_bin = steps_bin * grid_binary
    error_bin = abs(duration - snapped_bin)
    
    # Grid B: Ternario (Base 1/48 - Tresillo de Fusa)
    grid_ternary = 1.0 / 12.0 
    steps_ter = round(duration / grid_ternary)
    snapped_ter = steps_ter * grid_ternary
    error_ter = abs(duration - snapped_ter)
    
    # Umbral de seguridad:
    if duration < 0.04: # Menos de media fusa de tresillo
        return 0.0

    # Gana el que tenga menos error
    if error_bin <= error_ter:
        return round(snapped_bin, 3)
    else:
        return round(snapped_ter, 3)


def _extract_raw_events(score) -> List[Tuple[float, float, List[int]]]:
    """
    Extrae una lista lineal de eventos musicales (Notas y Acordes) desde un Score de music21.
    
    Args:
        score (music21.stream.Score): El objeto partitura parseado previamente.

    Returns:
        List[Tuple[float, float, List[int]]]: Lista de tuplas. Cada tupla contiene:
            - offset (float): Tiempo de inicio.
            - duration (float): Duración del evento.
            - pitches (List[int]): Lista de números MIDI (0-127) activos en ese evento.
    """
    raw_events = []
    flat_score = score.flatten()
    elements = [e for e in flat_score if isinstance(e, (chord.Chord, note.Note))]
    
    for element in elements:
        pitches = []
        if isinstance(element, chord.Chord):
            pitches = [p.midi for p in element.pitches]
        elif isinstance(element, note.Note):
            pitches = [element.pitch.midi]
            
        offset = float(element.offset)
        duration = float(element.quarterLength)
        raw_events.append((offset, duration, pitches))
        
    return raw_events


def _get_pitch_median(raw_events) -> int:
    """
    Calcula la mediana de altura (pitch) de todos los eventos para determinar el centro tonal aproximado.
    Útil para estrategias de Data Augmentation.

    Args:
        raw_events (List[Tuple]): Lista de eventos crudos obtenida de `_extract_raw_events`.

    Returns:
        int: El valor MIDI mediano. Retorna 60 (Do central) si la lista está vacía.
    """
    all_pitches = []
    for _, _, pitches in raw_events:
        all_pitches.extend(pitches)
    
    if not all_pitches:
        return 60 # Default seguro si el midi está vacío
        
    return int(statistics.median(all_pitches))


def _generate_tokens_from_events(raw_events, transpose_semitones=0) -> List[str]:
    """
    Convierte una lista de eventos crudos en una secuencia de tokens de texto estructurados.
    
    Lógica de procesamiento:
      - Transposición: Aplica desplazamiento de semitonos según el parámetro indicado.
      - Reataque: Emite nuevamente el token al detectar un pitch ON sin un OFF previo. Prioriza eventos ON sobre OFF en el mismo instante temporal.
      - Máximo sustain: Procesa únicamente el OFF correspondiente al último reataque en notas repetidas.
      - Delta Time: Calcula el tiempo (DUR) hasta el siguiente evento cuantizado.

    Args:
        raw_events (List[Tuple]): Lista de eventos (offset, duration, pitches).
        transpose_semitones (int): Cantidad de semitonos para transportar la secuencia. Default 0.

    Returns:
        List[str]: Lista de tokens en formato 'ON=...;OFF=...;DUR=...'.
    """
    start_map = defaultdict(list)
    end_map = defaultdict(list)
    
    # 1. Mapeo con Transposición
    for offset, duration, pitches in raw_events:
        end_time = offset + duration
        
        valid_pitches = []
        for p in pitches:
            new_p = p + transpose_semitones
            # Protección de rango MIDI (0-127)
            if 0 <= new_p <= 127:
                valid_pitches.append(new_p)
        
        # Si quedó al menos una nota válida en el evento, lo registramos
        if valid_pitches:
            for p in valid_pitches:
                start_map[offset].append(p)
                end_map[end_time].append(p)

    # 2. Bucle Principal de Tokenización
    all_times = sorted(list(set(start_map.keys()) | set(end_map.keys())))
    if not all_times:
        return []

    tokens = []
    note_state = defaultdict(lambda: {'count': 0})
    prev_time = all_times[0] 

    # Acumuladores de eventos pendientes para el próximo token
    pending_on = set()
    pending_off = set()

    # Iteramos sobre cada punto temporal donde pasa algo
    for i, t in enumerate(all_times):
        # A. Procesar OFFs en este tiempo t
        ending_notes = end_map.get(t, [])
        for p in ending_notes:
            note_state[p]['count'] -= 1
            if note_state[p]['count'] == 0: # Si el contador llega a 0, la nota realmente se apagó
                pending_off.add(p)
                
        # B. Procesar ONs en este tiempo t
        starting_notes = start_map.get(t, [])
        for p in starting_notes:
            note_state[p]['count'] += 1
            pending_on.add(p)

        # C. Limpieza de redundancia
        # Si una nota se apaga y se prende en el mismo instante, el ON tiene prioridad
        intersection = pending_on.intersection(pending_off)
        for p in intersection:
            pending_off.remove(p)

        # D. Calcular Duración hacia el siguiente evento
        is_last_event = (i == len(all_times) - 1)
        if not is_last_event:
            next_t = all_times[i+1]
            raw_dur = next_t - prev_time # Calculamos desde el último punto emitido
            dur = snap_to_grid(raw_dur)
        else:
            dur = 0.0

        # E. Emisión de Token
        # Emitimos si hay duración válida o si es el último evento
        if (dur > 0.0) or is_last_event:
            if pending_on or pending_off or dur > 0: # Solo emitimos si efectivamente hay contenido
                if not is_last_event:
                     str_on = ",".join(map(str, sorted(list(pending_on)))) if pending_on else "_"
                else:
                     str_on = "_" 

                str_off = ",".join(map(str, sorted(list(pending_off)))) if pending_off else "_"
                
                tokens.append(f"ON={str_on};OFF={str_off};DUR={dur:.3f}") # Crear token
                
                # Resetear acumuladores y avanzar el reloj
                pending_on = set()
                pending_off = set()
                if not is_last_event:
                    prev_time = next_t
                    
    return tokens


# --- Funciones Principales ---

def parse_midi_to_tokens(
    midi_path: str, 
    augment: bool = False
) -> List[Tuple[List[str], Dict[str, Any]]]:
    """
    Procesa un archivo MIDI y lo convierte en secuencias de tokens.
    
    Args:
        midi_path (str): Ruta del archivo MIDI a leer.
        augment (bool): Genera variaciones de la pieza transportándola a 12 tonalidades distintas, buscando centrarse alrededor del Do central (MIDI 60) e incluyendo la tonalidad original.

    Returns:
        List[Tuple[List[str], Dict[str, Any]]]: Una lista donde cada elemento es una tupla:
            - tokens (List[str]): La secuencia de eventos textuales.
            - metadata (Dict): Información extra, en este caso el 'tempo'.
            
        Con augment=False, la lista contiene un único elemento (tonalidad original).
        Con augment=True, retorna múltiples variaciones válidas dentro del rango MIDI.
    """
    # 1. Carga en crudo
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=midi.translate.TranslateWarning)
            mf = midi.MidiFile()
            mf.open(midi_path)
            mf.read()
            mf.close()
            score = midi.translate.midiFileToStream(mf, quantizePost=False) # Leemos sin cuantizar aquí para tener la data cruda real

    except Exception as e:
        print(f"❌ Error leyendo MIDI {midi_path}: {e}")
        return []

    # 2. Extracción de Datos Crudos
    raw_events = _extract_raw_events(score)

    # Metadata
    meta = {'tempo': 120}
    tempos = score.flatten().getElementsByClass(tempo.MetronomeMark)
    if tempos:
        meta['tempo'] = int(tempos[0].number)

    # 3. Generación
    if not augment:
        tokens = _generate_tokens_from_events(raw_events, transpose_semitones=0)
        return [(tokens, meta)]
    else:
        # Caso Data Augmentation: 12 Keys
        augmented_data = []

        # 1. Calcular Mediana
        median_pitch = _get_pitch_median(raw_events)
        
        # 2. Calcular Desplazamiento 
        shift_ideal = 60 - median_pitch # Ideal para centrar en Middle C (60)
        shift = max(-6, min(5, shift_ideal))  # Aseguramos que el shift central esté entre -6 y +5.
        
        # 3. Crear Ventana de 12 semitonos
        shifts = range(shift - 5, shift + 7)  # Privilegiamos ligeramente hacia arriba

        for semitone in shifts: 
            tokens = _generate_tokens_from_events(raw_events, transpose_semitones=semitone)
            if tokens: # Solo si la transposición fue válida y generó tokens
                augmented_data.append((tokens, meta))
        
        return augmented_data        


def tokens_to_midi(tokens: List[str], metadata: Dict[str, Any], output_path: str):
    """
    Reconstruye un archivo MIDI a partir de una secuencia de tokens.
    
    Incluye post-procesamiento para evitar errores comunes de generación:
      - Recorte de notas: Limita la duración máxima a `MAX_NOTE_DURATION` (8.0 tiempos) para controlar notas excesivamente largas.
      - Reataques: Fuerza el cierre de notas activas al recibir un nuevo comando ON para el mismo pitch.

    Args:
        tokens (List[str]): Secuencia de tokens a decodificar.
        metadata (Dict[str, Any]): Diccionario con metadatos, espera la clave 'tempo'.
        output_path (str): Ruta donde se guardará el archivo .mid generado.
    """
    print(f"🔨 Construyendo MIDI en: {output_path} ...")
    
    MAX_NOTE_DURATION = 8.0 # Límite máximo de duración para cualquier nota

    # 1. Deserialización
    current_time = 0.0
    note_state = defaultdict(lambda: {'start': None}) # Inicio actual
    collected_notes = [] # (pitch, start, duration)

    for i, token in enumerate(tokens):
        try:
            parts = token.split(';')
            on_part = parts[0].split('=')[1]
            off_part = parts[1].split('=')[1]
            dur_part = float(parts[2].split('=')[1])

            # A. OFF (Cerrar notas explícitamente)
            if off_part != '_':
                off_pitches = [int(x) for x in off_part.split(',')]
                for pitch in off_pitches:
                    state = note_state[pitch]
                    if state['start'] is not None: # Si hay inicio regristrado
                          raw_duration = current_time - state['start']
                          if raw_duration > 0:
                              final_dur = min(raw_duration, MAX_NOTE_DURATION)
                              collected_notes.append((pitch, state['start'], final_dur))
                    state['start'] = None # Apagado

            # B. ON (Abrir notas y gestionar Re-ataques)
            if on_part != '_':
                on_pitches = [int(x) for x in on_part.split(',')]
                for pitch in on_pitches:
                    state = note_state[pitch]
                    if state['start'] is not None: # Si ya sonaba, cerramos la anterior (Reataque)
                        raw_duration = current_time - state['start']
                        if raw_duration > 0:
                            final_dur = min(raw_duration, MAX_NOTE_DURATION)
                            collected_notes.append((pitch, state['start'], final_dur))
                    state['start'] = current_time # Iniciamos la nueva nota

            current_time += dur_part  # Avanzamos el tiempo

        except Exception as e:
            print(f"⚠️ Error token #{i}: {token} -> {e}")

    # 2. Limpieza final (Notas que nunca recibieron OFF)
    for pitch, state in note_state.items():
        if state['start'] is not None:
            raw_duration = current_time - state['start']
            final_dur = min(raw_duration, MAX_NOTE_DURATION)
            if final_dur > 0:
                collected_notes.append((pitch, state['start'], final_dur))
    
    # 3. Escritura con music21
    
    collected_notes.sort(key=lambda x: x[1]) # Importante para music21

    s = stream.Score()
    p = stream.Part()
    
    bpm = metadata.get('tempo', 120)
    p.insert(0, tempo.MetronomeMark(number=bpm))
    p.insert(0, instrument.Piano())
    
    for pitch, start, dur in collected_notes:
        n = note.Note(pitch)
        n.quarterLength = dur
        p.insert(start, n)

    s.append(p)
    try:
        s.write('midi', fp=output_path)
        print(f"✅ Archivo guardado correctamente.")
    except Exception as e:
        print(f"❌ Error escribiendo archivo: {e}")