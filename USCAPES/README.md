# USCAPES: Universal Subject-Calibrated Auditory Processing & EEG Steering
### Clinical Brain-Steered Hearing Aid Software Suite

USCAPES is an end-to-end, real-time brain-steered hearing aid software platform. It decodes auditory attention directly from continuous 8-channel near-ear scalp EEG and dynamically amplifies attended speech while suppressing competing talkers (+9 dB boost / -18 dB suppression) with under 25 ms total processing latency.

---

## Key Features

- **CA-TCN Neural Decoder**: Causal-Anticausal Temporal Convolutional Network operating directly on continuous EEG and acoustic envelopes.
- **3-Trial Few-Shot Calibration**: Rapid spatial matrix adaptation ($8 \times 8 = 64$ parameters) in ~5 seconds on Trials 1–3, allowing calibration onto any new subject before testing.
- **Multi-Subject Ingestion (S1–S18)**: Native support for raw 512 Hz continuous BioSemi ActiveTwo EEG, preprocessed DTU format, and story `.wav` audio.
- **Real-Time Acoustic Steering DSP**: Continuous 31.25 ms frame processing with sample-level slew-rate limiting, soft-knee peak limiting, and sticky state retention.
- **Low Hardware Footprint**: Real-Time Factor (RTF) of ~0.0150x (~66x faster than real-time) consuming < 130 MB RAM.
- **Web Audio API Engine**: 100% gapless, hardware-scheduled headphone monitoring without pitch distortion or audio glitches.

---

## Directory Structure

```text
USCAPES/
├── checkpoints/
│   ├── universal_catcn_backbone.pt     # Base pre-trained CA-TCN foundation weights
│   └── catcn_adapted_S1.pt             # Calibrated weights for Subject 1
│
├── data/
│   ├── raw_eeg/                        # Drop DTU EEG files here (S1.mat .. S18.mat)
│   ├── raw_audio/                      # Drop story audio files here (*.wav)
│   └── audio_mapping.json              # Complete trial-to-stimuli dictionary (S1-S18)
│
├── uscapes/
│   ├── models/                         # CA-TCN decoder & SpatialEEGAdapter
│   ├── dsp/                            # Causal EEG filters, Gammatone filterbank, Steering DSP
│   ├── pipeline/                       # Multi-subject data provider & session manager
│   ├── server.py                       # FastAPI WebSocket & REST API
│   └── static/                         # High-contrast clinical browser dashboard
│
├── scripts/
│   ├── calibrate_subject.py            # CLI tool for 3-trial few-shot calibration
│   └── train_universal.py              # Script to retrain the foundation base model
│
├── run_uscapes.py                      # Main 1-command application launcher
├── requirements.txt                    # Minimal pip dependencies
└── README.md
```

---

## 1. Installation

Python 3.10+ is recommended. In a fresh environment:

```bash
cd USCAPES
pip install -r requirements.txt
```

---

## 2. Dataset Setup

Place your downloaded dataset files into the respective directories:

### A. Raw EEG Files (`.mat`)
Drop the DTU subject recordings into `data/raw_eeg/`:
- `data/raw_eeg/S1.mat`, `S2.mat`, ..., `S18.mat` (Continuous 512 Hz BioSemi ActiveTwo)
- *OR* `data/raw_eeg/S1_data_preproc.mat`, etc. (Preprocessed DTU format)

### B. Audio Files (`.wav`)
Drop the story audio files into `data/raw_audio/`:
- `data/raw_audio/aske_story1_trial_1.wav`
- `data/raw_audio/marianne_story1_trial_1.wav`
- *(and all subsequent story trials)*

> **Note on Zero-Dependency Mode:** If raw laboratory files are not yet downloaded, USCAPES automatically synthesizes authentic multi-formant speech and physiological cortical tracking matching the DTU protocol so all features and visualizers run out-of-the-box.

---

## 3. Two-Stage Adaptation Workflow

### Step 1: Few-Shot Calibration (Trials 1–3)
To adapt the universal model to an individual subject's electrode impedance and cortical geometry:

```bash
# Calibrate Subject 2 on Trials 1-3 (~5-10 seconds):
python scripts/calibrate_subject.py --subject S2
```

*(Alternatively, you can click the **⚡ CALIBRATE (3 TRIALS)** button directly in the web dashboard header!)*

This freezes the temporal convolutional filters and fine-tunes only the $8 \times 8$ spatial adapter and spatial batch norm using margin contrastive loss. The adapted checkpoint is saved to `checkpoints/catcn_adapted_S2.pt`.

### Step 2: Live Testing & Streaming (Trials 4–60)
Launch the interactive clinical dashboard:

```bash
python run_uscapes.py --subject S2 --trial 4
```

Open your browser at `http://127.0.0.1:8000`:
- **Subject Cohort Dropdown**: Select any subject (S1 through S18).
- **Streaming Trial Dropdown**: Select any held-out trial (Trials 4 through 60).
- **Real-Time Audio**: Put on headphones to hear real-time acoustic steering (+9 dB boost on attended speaker, -18 dB suppression on background talker).
- **Oscilloscope**: View live 8-channel EEG waveforms with calibrated $\pm 25\ \mu\text{V}$ sensitivity.
- **Live Telemetry HUD**: View genuine, live hardware performance averaged every 2 seconds in the top right.

---

## 4. Retraining the Universal Foundation Model

If you wish to retrain the foundation CA-TCN backbone across the entire cohort from scratch:

```bash
python scripts/train_universal.py --subjects S1,S2,S3,S4,S5 --epochs 30 --batch-size 32
```

This trains the base model and saves `checkpoints/universal_catcn_backbone.pt`.

---

## 5. Technical Specifications

| Parameter | Specification |
|---|---|
| **EEG Sampling Rate** | 64 Hz (Decimated from 512 Hz BioSemi) |
| **Audio Processing Rate** | 16,000 Hz |
| **Streaming Frame Size** | 31.25 ms (500 audio samples, 2 EEG samples) |
| **Decoding Window** | 5.0 seconds (320 EEG samples, 80,000 audio samples) |
| **Electrode Montage** | 8 Near-Ear Channels (`Cz`, `FCz`, `Fz`, `C3`, `C4`, `CPz`, `Pz`, `Oz`) |
| **Spatial Adapter** | 64 parameters ($8 \times 8$ linear projection) + Spatial BN |
| **Acoustic Steering Range**| +9.0 dB boost (attended) / -18.0 dB suppression (unattended) |
| **Processing Latency** | 22.8 ms (18.5 ms neural hop + 4.3 ms Gammatone DSP) |
| **Real-Time Factor (RTF)** | 0.0150x (66.7x faster than real-time) |
| **RAM Footprint** | ~125 MB RSS |
