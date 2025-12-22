# 🎵 FF-Music-LSTM: Algorithmic Music Generation

> **Style Fusion:** A Recurrent Neural Network (LSTM) trained to compose original music based on the *Final Fantasy* aesthetic.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jdavibedoya/ff-music-lstm/blob/main/FF_Music_LSTM.ipynb)
![Python](https://img.shields.io/badge/Python-3.12-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)

## 📖 Project Description
This project explores the intersection of music and Artificial Intelligence using an **LSTM (Long Short-Term Memory)** network. The model learns musical grammar and stylistic patterns from MIDI files to generate note-by-note compositions, managing both pitch and duration.

Although the project architecture is flexible and allows training models with any MIDI dataset, the main objective of this experiment was to achieve a coherent fusion between two iconic themes from the saga: *"Aerith's Theme"* (FF VII) and *"To Zanarkand"* (FF X).

<p align="center">
  <img src="https://raw.githubusercontent.com/jdavibedoya/ff-music-lstm/main/images/ffvii-x.png" width="500" style="border-radius: 10px;">
  <br>
  <em>Artistic fusion of FF VII and FF X</em>
</p>

---

## 🎧 Listen to Generated Samples
Below are two examples of compositions extended by the model from hybrid seeds, using different temperatures (creativity levels).

> **Production Note:** The network-generated MIDI was rendered externally in a DAW (**Ableton Live**) using *Sound Magic Piano One* for timbre and *Valhalla Vintage Verb* for spatiality, aiming for a professional aesthetic.

| Seed / Temperature | Audio | Description |
| :--- | :---: | :--- |
| **Seed 2 (Temp 0.75)** | [▶️ **Listen to Audio**](https://raw.githubusercontent.com/jdavibedoya/ff-music-lstm/main/wav/Session_Seed2_Temp0.75.wav) | More conservative and melodic generation. |
| **Seed 1 (Temp 1.25)** | [▶️ **Listen to Audio**](https://raw.githubusercontent.com/jdavibedoya/ff-music-lstm/main/wav/Session_Seed1_Temp1.25.wav) | Higher creative risk and rhythmic variation. |

---

## 🛠️ Technical Details

### Dataset and Preprocessing
* **Sources:** Manually curated MIDI files from *Final Fantasy VII* and *X*.
* **Data Augmentation:** A tonal transposition pipeline was implemented to enrich the dataset, allowing the model to learn intervallic relationships across different keys.
* **Custom Musical Grammar:** A custom tokenization system was designed to interpret music as a sequence of events:
    * `NOTE_ON`: Note start.
    * `NOTE_OFF`: Note end.
    * `DUR`: Temporal duration.

### Model Architecture
* **Network:** LSTM (Long Short-Term Memory) optimized for temporal sequences.
* **Custom Module:** The `music_data_utils.py` library was developed to abstract the complexity of MIDI processing, tokenization, and decoding.
* **Generation:** Implementation of a **hybrid seed search** heuristic to initialize generation with coherent musical contexts.

### Tech Stack
The project was built using:
* **Core:** `Python`, `NumPy`, `Collections`
* **Deep Learning:** `TensorFlow`, `Keras`
* **Audio & Music Theory:** `music21`, `pretty_midi`
* **Visualization:** `Matplotlib`

---

## 📢 Notes & Acknowledgments

* **Inspiration:** This project is inspired by the *Jazz Improvisation with LSTM* lab from the **Sequence Models** course (part of the *Deep Learning Specialization* by DeepLearning.AI). However, the technical implementation, data preprocessing, and architecture presented here are original and adapted for this specific use case.
* **Language:** The documentation and code comments are in **Spanish**; however, naming conventions for variables, functions, and classes follow **English** industry standards.

---
*Developed by [David Bedoya](https://github.com/jdavibedoya)*
