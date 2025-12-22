# 🎵 Music Generation with an LSTM Net

> Fusing Final Fantasy VII and X Music with LSTM Neural Networks.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jdavibedoya/ff-music-lstm/blob/main/FF_Music_LSTM.ipynb)
![Python](https://img.shields.io/badge/Python-3.12-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)

## 📖 Project Description
This project explores the intersection of **Artificial Intelligence** and **Music Composition**. I built an **LSTM (Long Short-Term Memory)** network that learns musical grammar and stylistic patterns from MIDI files to generate note-by-note compositions, handling both pitch and duration.

While the architecture is flexible enough to train models on any MIDI dataset, the primary goal was to create a coherent fusion between two iconic themes: *"Aerith's Theme"* (FF VII) and *"To Zanarkand"* (FF X).

<p align="center">
  <img src="https://raw.githubusercontent.com/jdavibedoya/ff-music-lstm/main/images/ffvii-x.png" width="500" style="border-radius: 10px;">
  <br>
  <em>Artistic fusion of FF VII and FF X</em>
</p>

---

## 🎧 Listen to Generated Samples
Below are two examples of model-generated compositions initialized with hybrid seeds at different temperatures (creativity levels).

> **Production Note:** The generated MIDI was rendered in a DAW (**Ableton Live**) with *Sound Magic Piano One* and *Valhalla Vintage Verb* to achieve a professional aesthetic.

[▶️ **Listen on Project Website**](https://jdavibedoya.github.io/projects/ff-music-lstm/) or download here:

| Seed / Temperature | Audio | Description |
| :--- | :---: | :--- |
| **Seed 2 (Temp 0.75)** | [⬇️ **Download MP3**](https://raw.githubusercontent.com/jdavibedoya/ff-music-lstm/main/audio/Session_Seed2_Temp0.75.mp3) | More conservative and melodic generation. |
| **Seed 1 (Temp 1.25)** | [⬇️ **Download MP3**](https://raw.githubusercontent.com/jdavibedoya/ff-music-lstm/main/audio/Session_Seed1_Temp1.25.mp3) | Higher creative risk and rhythmic variation. |

---

## 🛠️ Technical Details

### Dataset and Preprocessing
* **Sources:** Manually curated MIDI files.
* **Data Augmentation:** Implemented a tonal transposition pipeline to enrich the dataset, allowing the model to learn intervallic relationships across different keys.
* **Custom Musical Grammar:** Designed a tokenization system to interpret music as a sequence of events:
    * `NOTE_ON`: Note start.
    * `NOTE_OFF`: Note end.
    * `DUR`: Temporal duration.

### Model Architecture
* **Network:** LSTM (Long Short-Term Memory) optimized for temporal sequences.
* **Custom Module:** Developed the `music_data_utils.py` library to abstract the complexity of MIDI processing, tokenization, and decoding.
* **Generation:** Implemented a **hybrid seed search** heuristic to initialize generation with coherent musical contexts.

<p align="center">
  <img src="https://raw.githubusercontent.com/jdavibedoya/ff-music-lstm/main/images/model_train.png" width="500" style="border-radius: 10px;">
  <br>
  <em>Model Architecture (Training)</em>
</p>

### Tech Stack
* **Core:** `Python`, `NumPy`, `Collections`
* **Deep Learning:** `TensorFlow`, `Keras`
* **Audio & Music Theory:** `music21`, `pretty_midi`
* **Visualization:** `Matplotlib`

---

## 📢 Notes & Acknowledgments

* **Inspiration:** This project is inspired by the *Jazz Improvisation with LSTM* lab from the **Sequence Models** course (DeepLearning.AI). However, the technical implementation, data preprocessing, and architecture presented here are original adaptations for this specific use case.
* **Language:** Documentation and code comments are in **Spanish**, while naming conventions strictly follow **English** industry standards.

---
*Developed by [David Bedoya](https://github.com/jdavibedoya)*
