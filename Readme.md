# Dual Speaker Embedding for Cross-Lingual Text-to-Speech

This is a Python implementation of a Dual Speaker Embedding TTS (DSE-TTS) system for cross-lingual text-to-speech (CTTS). The DSE-TTS system uses two different embeddings to address the challenges of CTTS, namely retaining speaker timbres and eliminating accents from the speaker's first language. One embedding is fed to the acoustic model to learn the linguistic speaking style, while the other one is integrated into the vocoder to mimic the target speaker's timbre.

---

<p align="center"> <img src="https://img.shields.io/badge/Python-3.6%20or%20later-blue?style=for-the-badge" alt="Python 3.6 or later"> <img src="https://img.shields.io/badge/PyTorch-1.7%20or%20later-orange?style=for-the-badge" alt="PyTorch 1.7 or later"> <img src="https://img.shields.io/badge/NumPy-green?style=for-the-badge" alt="NumPy"> <img src="https://img.shields.io/badge/Matplotlib-(optional)-yellow?style=for-the-badge" alt="Matplotlib (optional)"> </p>

---



[Implemented by @codemaya]

## Open for contributions

Raise issues for more.

## Features

![arch](https://github.com/collabora/spear-tts-pytorch/assets/64596494/b7df1ade-9f91-40bf-ae0a-92e55df2a4e8)



- 🎤 Dual speaker embedding is used to model linguistic characteristics and speaker timbre separately, helping to retain the speaker's timbre in synthesized speech and eliminating accents in cross-lingual synthesis.

- 📜 Text encoder (txt2vec) is used to generate VQ acoustic features from input text, which are then fed into the acoustic model.

- 🎧 Acoustic model predicts acoustic features from text and speaker embeddings, allowing the system to
 model the linguistic speaking style and the target speaker's timbre separately.

- 🎶 Vocoder generates the waveform from acoustic features and speaker embeddings, producing high-quality synthesized speech.

- 🔊 Waveform generator converts vocoder features to audio waveform, allowing the synthesized speech to be played back.

- 🏅 DSE-TTS system demonstrates highly competitive naturalness among publicly available TTS systems.

- 👍 DSE-TTS system shows significantly better cross-lingual synthesis compared to state-of-the-art SANE-TTS.

- 🔍 The DSE-TTS system is composed of an acoustic model, txt2vec, and a vocoder, vec2wav.

- 🤖 The DSE-TTS system is implemented in Python using PyTorch, making it easy to integrate with other PyTorch-based models.

- 🌟 The DSE-TTS system is a state-of-the-art approach to cross-lingual speech synthesis, with potential applications in various fields such as language learning and assistive technology.

---

## Dependencies
- Python 3.6 or later
- PyTorch 1.7 or later
- NumPy
- Matplotlib (optional)

## Usage
1. Clone the repository and navigate to the root directory.

2. Train the DSE-TTS system:
   - Modify the hyperparameters in `train.py` as needed.
   - Run `python train.py` to start training.
   
3. Generate speech:
   - Modify the hyperparameters in `inference.py` as needed.
   - Run `python inference.py` to generate speech from input text.

## References

```
@inproceedings{DSETTS,
  title = {DSE-TTS: Dual Speaker Embedding for Cross-Lingual Text-to-Speech},
  author = {sen.liu, cantabile kwok, duchenpeng, chenxie95, kai.yu},
  url = {https://arxiv.org/abs/2306.14145},
  publisher = {arXiv},
  month = {june},
  year = {2023},
}
```
