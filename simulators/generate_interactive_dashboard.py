import argparse
import sys
import json
from pathlib import Path
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

VERIFY_ROOT = REPO_ROOT / "scripts" / "verify_baseline"
if str(VERIFY_ROOT) not in sys.path:
    sys.path.insert(0, str(VERIFY_ROOT))

from src.streaming.pipeline import StreamingAADPipeline
from scripts.verify_baseline.models.catcn import CATCNDirectDecoder
from scripts.verify_baseline.training.montages import MONTAGES
from scripts.verify_baseline.training.train_matchnet_wavlm import get_mapping_data, prepare_dataset, FS
from scripts.verify_baseline.baselines.ridge_aad import load_subject_examples, subject_files

HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Real-Time Neural BCI Hearing Aid Dashboard | DTU AAD</title>
<style>
  :root {
    --bg-primary: #07090e;
    --bg-card: rgba(15, 23, 42, 0.75);
    --border-card: rgba(56, 189, 248, 0.18);
    --border-glow: rgba(56, 189, 248, 0.35);
    --cyan: #00f2fe;
    --blue: #38bdf8;
    --orange: #fb923c;
    --amber: #f59e0b;
    --green: #10b981;
    --red: #ef4444;
    --purple: #a855f7;
    --text-main: #f8fafc;
    --text-muted: #94a3b8;
  }
  * { box-sizing: border-box; margin: 0; padding: 0; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif; }
  body {
    background: var(--bg-primary);
    color: var(--text-main);
    min-height: 100vh;
    padding: 16px;
    overflow-x: hidden;
  }
  .dashboard {
    max-width: 1400px;
    margin: 0 auto;
    display: flex;
    flex-direction: column;
    gap: 16px;
  }
  /* HEADER */
  .header {
    background: var(--bg-card);
    border: 1px solid var(--border-card);
    border-radius: 12px;
    padding: 14px 20px;
    display: flex;
    justify-content: space-between;
    align-items: center;
    backdrop-filter: blur(12px);
    box-shadow: 0 4px 24px rgba(0, 0, 0, 0.4);
  }
  .logo-group { display: flex; align-items: center; gap: 14px; }
  .logo-icon {
    width: 38px; height: 38px; border-radius: 8px;
    background: linear-gradient(135deg, #00f2fe, #4facfe);
    display: flex; align-items: center; justify-content: center;
    box-shadow: 0 0 16px rgba(0, 242, 254, 0.4);
  }
  .logo-text h1 { font-size: 1.15rem; font-weight: 700; letter-spacing: 0.5px; }
  .logo-text span { font-size: 0.78rem; color: var(--blue); letter-spacing: 1px; text-transform: uppercase; }
  .header-badges { display: flex; gap: 12px; align-items: center; }
  .badge {
    padding: 5px 12px; border-radius: 20px; font-size: 0.75rem; font-weight: 600;
    display: flex; align-items: center; gap: 6px; letter-spacing: 0.5px;
  }
  .badge-live { background: rgba(16, 185, 129, 0.15); color: var(--green); border: 1px solid rgba(16, 185, 129, 0.3); }
  .pulse-dot { width: 7px; height: 7px; border-radius: 50%; background: var(--green); box-shadow: 0 0 8px var(--green); animation: pulse 1.5s infinite; }
  @keyframes pulse { 0%, 100% { opacity: 1; transform: scale(1); } 50% { opacity: 0.4; transform: scale(0.8); } }
  .badge-patient { background: rgba(56, 189, 248, 0.12); color: var(--blue); border: 1px solid rgba(56, 189, 248, 0.25); }

  /* MAIN GRID */
  .grid-top {
    display: grid;
    grid-template-columns: 1.1fr 1.2fr 0.9fr;
    gap: 16px;
  }
  @media (max-width: 1024px) { .grid-top { grid-template-columns: 1fr; } }
  .card {
    background: var(--bg-card);
    border: 1px solid var(--border-card);
    border-radius: 12px;
    padding: 18px;
    backdrop-filter: blur(12px);
    display: flex;
    flex-direction: column;
    gap: 14px;
    position: relative;
    box-shadow: 0 4px 20px rgba(0,0,0,0.3);
  }
  .card-header {
    display: flex; justify-content: space-between; align-items: center;
    border-bottom: 1px solid rgba(255,255,255,0.06); padding-bottom: 8px;
  }
  .card-title { font-size: 0.85rem; font-weight: 700; color: var(--text-muted); text-transform: uppercase; letter-spacing: 1px; }

  /* SPATIAL ACOUSTIC HEAD */
  .spatial-container {
    display: flex; flex-direction: column; align-items: center; gap: 12px;
    position: relative; padding: 10px 0;
  }
  .head-stage {
    width: 260px; height: 210px; position: relative;
    display: flex; align-items: center; justify-content: center;
  }
  .speaker-card {
    position: absolute; width: 85px; padding: 8px 6px; border-radius: 8px;
    text-align: center; font-size: 0.75rem; font-weight: 700;
    transition: all 0.3s ease;
  }
  .speaker-a { left: 0px; top: 35px; border: 1px solid rgba(56, 189, 248, 0.4); background: rgba(56, 189, 248, 0.08); }
  .speaker-b { right: 0px; top: 35px; border: 1px solid rgba(251, 146, 60, 0.4); background: rgba(251, 146, 60, 0.08); }
  .speaker-gain { font-size: 0.95rem; margin-top: 4px; font-weight: 800; }
  .active-beam-a { box-shadow: 0 0 20px rgba(0, 242, 254, 0.6); background: rgba(0, 242, 254, 0.22); border-color: var(--cyan); }
  .active-beam-b { box-shadow: 0 0 20px rgba(251, 146, 60, 0.6); background: rgba(251, 146, 60, 0.22); border-color: var(--orange); }

  /* ATTENTION COMPASS / BIPOLAR METER */
  .compass-wrapper { width: 100%; display: flex; flex-direction: column; gap: 6px; }
  .compass-labels { display: flex; justify-content: space-between; font-size: 0.75rem; font-weight: 700; }
  .compass-bar {
    width: 100%; height: 16px; background: rgba(15, 23, 42, 0.8);
    border: 1px solid rgba(255,255,255,0.1); border-radius: 8px; position: relative;
    overflow: hidden;
  }
  .compass-center { position: absolute; left: 50%; top: 0; width: 2px; height: 100%; background: rgba(255,255,255,0.3); z-index: 1; }
  .compass-pointer {
    position: absolute; top: 2px; height: 12px; width: 18px; border-radius: 4px;
    background: linear-gradient(135deg, #00f2fe, #38bdf8);
    box-shadow: 0 0 10px #00f2fe; transition: left 0.15s ease-out;
    transform: translateX(-50%);
  }

  /* NEURAL DECISION HERO */
  .decision-hero {
    display: flex; flex-direction: column; align-items: center; justify-content: center;
    padding: 16px; background: rgba(0, 0, 0, 0.35); border-radius: 12px; border: 1px solid rgba(255,255,255,0.05);
    text-align: center; gap: 8px;
  }
  .decision-state {
    font-size: 1.45rem; font-weight: 800; letter-spacing: 1px;
    padding: 6px 20px; border-radius: 24px; transition: all 0.3s ease;
  }
  .state-A { background: rgba(56, 189, 248, 0.2); color: var(--cyan); border: 1px solid var(--cyan); box-shadow: 0 0 20px rgba(0, 242, 254, 0.3); }
  .state-B { background: rgba(251, 146, 60, 0.2); color: var(--orange); border: 1px solid var(--orange); box-shadow: 0 0 20px rgba(251, 146, 60, 0.3); }
  .state-UNCERTAIN { background: rgba(168, 85, 247, 0.15); color: var(--purple); border: 1px solid var(--purple); }

  .metrics-row {
    display: grid; grid-template-columns: repeat(3, 1fr); gap: 10px; width: 100%;
  }
  .metric-box {
    background: rgba(0, 0, 0, 0.25); border-radius: 8px; padding: 10px 8px; text-align: center;
    border: 1px solid rgba(255,255,255,0.05);
  }
  .metric-label { font-size: 0.68rem; color: var(--text-muted); text-transform: uppercase; font-weight: 600; margin-bottom: 4px; }
  .metric-val { font-size: 1.1rem; font-weight: 800; font-family: 'Courier New', monospace; }

  /* TELEMETRY ENGINE CARD */
  .telemetry-list { display: flex; flex-direction: column; gap: 10px; }
  .telem-row {
    display: flex; justify-content: space-between; align-items: center;
    padding: 8px 12px; border-radius: 8px; background: rgba(0, 0, 0, 0.25);
    font-size: 0.8rem; border-left: 3px solid var(--blue);
  }
  .telem-val { font-weight: 700; font-family: 'Courier New', monospace; }

  /* OSCILLOSCOPE SECTION */
  .bottom-section {
    display: grid; grid-template-columns: 1fr; gap: 16px;
  }
  .canvas-card {
    background: var(--bg-card); border: 1px solid var(--border-card);
    border-radius: 12px; padding: 16px; backdrop-filter: blur(12px);
    display: flex; flex-direction: column; gap: 10px;
  }
  .oscilloscope-canvas {
    width: 100%; height: 180px; background: #04060a; border-radius: 8px;
    border: 1px solid rgba(0, 242, 254, 0.15);
  }
  .chart-canvas {
    width: 100%; height: 140px; background: #04060a; border-radius: 8px;
    border: 1px solid rgba(56, 189, 248, 0.15);
  }

  /* CONTROL BAR */
  .controls-bar {
    background: rgba(15, 23, 42, 0.9);
    border: 1px solid var(--border-glow);
    border-radius: 12px;
    padding: 12px 20px;
    display: flex;
    align-items: center;
    gap: 18px;
    box-shadow: 0 -4px 30px rgba(0, 0, 0, 0.6);
    position: sticky;
    bottom: 12px;
    z-index: 100;
  }
  .btn-play {
    width: 44px; height: 44px; border-radius: 50%; border: none;
    background: linear-gradient(135deg, #00f2fe, #38bdf8);
    color: #000; font-size: 1.1rem; font-weight: 800; cursor: pointer;
    display: flex; align-items: center; justify-content: center;
    box-shadow: 0 0 16px rgba(0, 242, 254, 0.5);
    transition: transform 0.1s ease;
  }
  .btn-play:hover { transform: scale(1.05); }
  .timeline-slider {
    flex: 1; height: 6px; -webkit-appearance: none; appearance: none;
    background: rgba(255,255,255,0.15); border-radius: 3px; outline: none;
  }
  .timeline-slider::-webkit-slider-thumb {
    -webkit-appearance: none; appearance: none; width: 16px; height: 16px;
    border-radius: 50%; background: var(--cyan); cursor: pointer; box-shadow: 0 0 8px var(--cyan);
  }
  .speed-pills { display: flex; gap: 6px; }
  .speed-btn {
    padding: 4px 10px; border-radius: 6px; background: rgba(255,255,255,0.06);
    border: 1px solid rgba(255,255,255,0.1); color: var(--text-muted);
    font-size: 0.75rem; font-weight: 600; cursor: pointer;
  }
  .speed-btn.active { background: rgba(56, 189, 248, 0.25); color: var(--cyan); border-color: var(--cyan); }
  .time-display { font-family: 'Courier New', monospace; font-size: 0.95rem; font-weight: 700; min-width: 95px; }
</style>
</head>
<body>

<div class="dashboard">

  <!-- HEADER -->
  <header class="header">
    <div class="logo-group">
      <div class="logo-icon">
        <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="#000" stroke-width="2.5"><path d="M12 2a3 3 0 0 0-3 3v7a3 3 0 0 0 6 0V5a3 3 0 0 0-3-3Z"/><path d="M19 10v2a7 7 0 0 1-14 0v-2"/><line x1="12" y1="19" x2="12" y2="22"/></svg>
      </div>
      <div class="logo-text">
        <h1>NEURAL BCI HEARING AID</h1>
        <span>Decoupled Real-Time Streaming Architecture</span>
      </div>
    </div>
    <div class="header-badges">
      <div class="badge badge-patient">PATIENT: __SUBJECT__ (Trial __TRIAL_IDX__)</div>
      <div class="badge badge-patient">MONTAGE: near_ear_expanded (8-Ch)</div>
      <div class="badge badge-live">
        <div class="pulse-dot"></div>
        LIVE 64 Hz HARDWARE REPLAY
      </div>
    </div>
  </header>

  <!-- TOP GRID: 3 PANELS -->
  <div class="grid-top">

    <!-- 1. SPATIAL ACOUSTIC HEAD & GAINS -->
    <div class="card">
      <div class="card-header">
        <span class="card-title">Bilateral Hearing Aid Steering</span>
        <span style="font-size: 0.7rem; color: var(--text-muted);">Decoupled Loop (< 10ms)</span>
      </div>
      <div class="spatial-container">
        <div class="head-stage">
          <!-- Speaker A (Left) -->
          <div id="cardSpeakerA" class="speaker-card speaker-a">
            <div style="color: var(--cyan);">SPEAKER A</div>
            <div id="txtGainA" class="speaker-gain" style="color: var(--cyan);">-0.0 dB</div>
            <div style="font-size: 0.65rem; color: var(--text-muted);">ATTENDED</div>
          </div>

          <!-- Head SVG Vector -->
          <svg width="120" height="135" viewBox="0 0 120 135">
            <!-- Ears -->
            <ellipse id="earLeft" cx="15" cy="68" rx="8" ry="16" fill="rgba(56,189,248,0.4)" stroke="#38bdf8" stroke-width="1.5"/>
            <ellipse id="earRight" cx="105" cy="68" rx="8" ry="16" fill="rgba(251,146,60,0.4)" stroke="#fb923c" stroke-width="1.5"/>
            <!-- Head Outline -->
            <path d="M 25 68 C 25 25, 95 25, 95 68 C 95 105, 75 125, 60 125 C 45 125, 25 105, 25 68 Z" fill="#0f172a" stroke="rgba(255,255,255,0.3)" stroke-width="2"/>
            <!-- Nose -->
            <path d="M 57 32 L 60 22 L 63 32" stroke="rgba(255,255,255,0.4)" stroke-width="2" fill="none"/>
            <!-- 8 Peri-Auricular Electrodes -->
            <circle cx="28" cy="55" r="3" fill="#00f2fe"/><circle cx="28" cy="80" r="3" fill="#00f2fe"/>
            <circle cx="22" cy="68" r="3" fill="#00f2fe"/><circle cx="36" cy="68" r="3" fill="#00f2fe"/>
            <circle cx="92" cy="55" r="3" fill="#fb923c"/><circle cx="92" cy="80" r="3" fill="#fb923c"/>
            <circle cx="98" cy="68" r="3" fill="#fb923c"/><circle cx="84" cy="68" r="3" fill="#fb923c"/>
          </svg>

          <!-- Speaker B (Right) -->
          <div id="cardSpeakerB" class="speaker-card speaker-b">
            <div style="color: var(--orange);">SPEAKER B</div>
            <div id="txtGainB" class="speaker-gain" style="color: var(--orange);">-6.0 dB</div>
            <div style="font-size: 0.65rem; color: var(--text-muted);">SUPPRESSED</div>
          </div>
        </div>

        <!-- Bipolar Steering Compass -->
        <div class="compass-wrapper">
          <div class="compass-labels">
            <span style="color: var(--cyan);">◀ Speaker A</span>
            <span id="txtHorizonScore" style="color: var(--text-muted); font-family: monospace;">S_t: 0.00</span>
            <span style="color: var(--orange);">Speaker B ▶</span>
          </div>
          <div class="compass-bar">
            <div class="compass-center"></div>
            <div id="compassPointer" class="compass-pointer" style="left: 50%;"></div>
          </div>
        </div>
      </div>
    </div>

    <!-- 2. NEURAL ATTENTION DECISION HUD -->
    <div class="card">
      <div class="card-header">
        <span class="card-title">Neural Attention State Engine</span>
        <span id="txtHysteresis" style="font-size: 0.7rem; color: var(--green);">EMA + Hysteresis (N=2)</span>
      </div>

      <div class="decision-hero">
        <div style="font-size: 0.72rem; color: var(--text-muted); text-transform: uppercase;">Active Audio Routing Target</div>
        <div id="badgeDecision" class="decision-state state-UNCERTAIN">INITIALIZING...</div>
        <div id="txtConfidence" style="font-size: 0.85rem; font-weight: 700; color: var(--blue);">Confidence: --%</div>
      </div>

      <div class="metrics-row">
        <div class="metric-box">
          <div class="metric-label">Neural Logit A</div>
          <div id="valLogitA" class="metric-val" style="color: var(--cyan);">--</div>
        </div>
        <div class="metric-box">
          <div class="metric-label">Neural Logit B</div>
          <div id="valLogitB" class="metric-val" style="color: var(--orange);">--</div>
        </div>
        <div class="metric-box">
          <div class="metric-label">Margin (Δ_t)</div>
          <div id="valDelta" class="metric-val" style="color: var(--text-main);">--</div>
        </div>
      </div>
    </div>

    <!-- 3. DECOUPLED ARCHITECTURE TELEMETRY -->
    <div class="card">
      <div class="card-header">
        <span class="card-title">Edge Hardware Telemetry</span>
        <span style="font-size: 0.7rem; color: var(--blue);">TorchScript JIT</span>
      </div>
      <div class="telemetry-list">
        <div class="telem-row">
          <span>BCI Compute Latency (T_comp)</span>
          <span id="valComputeMs" class="telem-val" style="color: var(--green);">8.9 ms</span>
        </div>
        <div class="telem-row">
          <span>Step Budget Utilization (500ms)</span>
          <span id="valBudgetPct" class="telem-val" style="color: var(--green);">1.78%</span>
        </div>
        <div class="telem-row">
          <span>Digital Audio Mixing Time</span>
          <span class="telem-val" style="color: var(--cyan);">0.0072 ms</span>
        </div>
        <div class="telem-row">
          <span>Continuous Switches Observed</span>
          <span id="valSwitches" class="telem-val" style="color: var(--purple);">0</span>
        </div>
        <div class="telem-row">
          <span>Anti-Chatter Hysteresis Status</span>
          <span class="telem-val" style="color: var(--blue);">LOCKED (|S_t| > 0.25)</span>
        </div>
      </div>
    </div>

  </div>

  <!-- BOTTOM SECTION: OSCILLOSCOPE & LOGIT CHART -->
  <div class="bottom-section">
    <!-- Live 8-Ch EEG Oscilloscope -->
    <div class="canvas-card">
      <div class="card-header">
        <span class="card-title">Live 8-Channel Peri-Auricular EEG Oscilloscope (64 Hz)</span>
        <span style="font-size: 0.7rem; color: var(--cyan);">T7, T8, TP7, TP8, CP5, CP6, FC5, FC6</span>
      </div>
      <canvas id="eegCanvas" class="oscilloscope-canvas"></canvas>
    </div>

    <!-- Neural Logits Trajectory Chart -->
    <div class="canvas-card">
      <div class="card-header">
        <span class="card-title">Neural Cross-Correlation Logit Trajectory (Logit A vs Logit B)</span>
        <span style="font-size: 0.7rem; color: var(--text-muted);">Decision Window: 5.0 s (320 samples)</span>
      </div>
      <canvas id="chartCanvas" class="chart-canvas"></canvas>
    </div>
  </div>

  <!-- FLOATING CONTROL BAR -->
  <div class="controls-bar">
    <button id="btnPlayPause" class="btn-play">▶</button>
    <div class="time-display">
      <span id="txtCurTime" style="color: var(--cyan);">00:00.0</span> / <span id="txtTotalTime" style="color: var(--text-muted);">00:50.0</span>
    </div>
    <input type="range" id="timelineSlider" class="timeline-slider" min="0" max="1000" value="0">
    <div class="speed-pills">
      <button class="speed-btn" data-speed="0.5">0.5x</button>
      <button class="speed-btn active" data-speed="1.0">1.0x</button>
      <button class="speed-btn" data-speed="2.0">2.0x</button>
      <button class="speed-btn" data-speed="4.0">4.0x</button>
    </div>
  </div>

</div>

<script>
// INJECTED DATA FROM PYTHON
const TELEMETRY_DATA = __TELEMETRY_JSON__;
const EEG_DATA = __EEG_JSON__;
const TOTAL_DURATION = __TOTAL_DURATION__;
const FS = 64.0;

// STATE VARIABLES
let isPlaying = false;
let playbackSpeed = 1.0;
let currentTime = 0.0;
let lastFrameTime = null;
let animationFrameId = null;

// DOM ELEMENTS
const btnPlayPause = document.getElementById("btnPlayPause");
const timelineSlider = document.getElementById("timelineSlider");
const txtCurTime = document.getElementById("txtCurTime");
const txtTotalTime = document.getElementById("txtTotalTime");
const badgeDecision = document.getElementById("badgeDecision");
const txtConfidence = document.getElementById("txtConfidence");
const valLogitA = document.getElementById("valLogitA");
const valLogitB = document.getElementById("valLogitB");
const valDelta = document.getElementById("valDelta");
const valComputeMs = document.getElementById("valComputeMs");
const valBudgetPct = document.getElementById("valBudgetPct");
const valSwitches = document.getElementById("valSwitches");
const txtGainA = document.getElementById("txtGainA");
const txtGainB = document.getElementById("txtGainB");
const cardSpeakerA = document.getElementById("cardSpeakerA");
const cardSpeakerB = document.getElementById("cardSpeakerB");
const compassPointer = document.getElementById("compassPointer");
const txtHorizonScore = document.getElementById("txtHorizonScore");
const eegCanvas = document.getElementById("eegCanvas");
const chartCanvas = document.getElementById("chartCanvas");
const speedButtons = document.querySelectorAll(".speed-btn");

// FORMAT HELPERS
function formatTime(sec) {
  const m = Math.floor(sec / 60);
  const s = Math.floor(sec % 60);
  const ms = Math.floor((sec % 1) * 10);
  return `${String(m).padStart(2, '0')}:${String(s).padStart(2, '0')}.${ms}`;
}

txtTotalTime.innerText = formatTime(TOTAL_DURATION);

// FIND CURRENT TELEMETRY STEP
function getActiveTelemetry(t) {
  if (t < TELEMETRY_DATA[0].t) return TELEMETRY_DATA[0];
  for (let i = TELEMETRY_DATA.length - 1; i >= 0; i--) {
    if (t >= TELEMETRY_DATA[i].t) return TELEMETRY_DATA[i];
  }
  return TELEMETRY_DATA[0];
}

// UPDATE UI
function updateUI(t) {
  txtCurTime.innerText = formatTime(t);
  timelineSlider.value = (t / TOTAL_DURATION) * 1000;

  const cur = getActiveTelemetry(t);
  if (!cur) return;

  // Decision Badge
  badgeDecision.className = `decision-state state-${cur.stream}`;
  badgeDecision.innerText = cur.stream === 'UNCERTAIN' ? 'SEARCHING / UNCERTAIN' : `ATTENDING: SPEAKER ${cur.stream}`;
  txtConfidence.innerText = `Confidence: ${(cur.conf * 100).toFixed(1)}%`;

  // Logits
  valLogitA.innerText = (cur.logit_a >= 0 ? '+' : '') + cur.logit_a.toFixed(2);
  valLogitB.innerText = (cur.logit_b >= 0 ? '+' : '') + cur.logit_b.toFixed(2);
  valDelta.innerText = (cur.delta >= 0 ? '+' : '') + cur.delta.toFixed(2);

  // Gains & Spatial Head
  txtGainA.innerText = `${cur.ga_db.toFixed(1)} dB`;
  txtGainB.innerText = `${cur.gb_db.toFixed(1)} dB`;
  
  if (cur.stream === 'A') {
    cardSpeakerA.className = "speaker-card speaker-a active-beam-a";
    cardSpeakerB.className = "speaker-card speaker-b";
  } else if (cur.stream === 'B') {
    cardSpeakerA.className = "speaker-card speaker-a";
    cardSpeakerB.className = "speaker-card speaker-b active-beam-b";
  } else {
    cardSpeakerA.className = "speaker-card speaker-a";
    cardSpeakerB.className = "speaker-card speaker-b";
  }

  // Compass Pointer
  const clamped = Math.max(-1.0, Math.min(1.0, cur.smooth));
  const pointerLeftPct = ((1.0 - clamped) / 2.0) * 100;
  compassPointer.style.left = `${pointerLeftPct}%`;
  txtHorizonScore.innerText = `S_t: ${(cur.smooth >= 0 ? '+' : '') + cur.smooth.toFixed(2)}`;

  // Telemetry Engine
  valComputeMs.innerText = `${cur.compute_ms.toFixed(1)} ms`;
  valBudgetPct.innerText = `${((cur.compute_ms / 500.0) * 100).toFixed(2)}%`;
  
  // Count switches up to t
  let swCount = 0;
  for (let item of TELEMETRY_DATA) {
    if (item.t <= t && item.switched) swCount++;
  }
  valSwitches.innerText = swCount;

  // Render Canvas
  renderEEG(t);
  renderChart(t);
}

// EEG OSCILLOSCOPE RENDERER
function renderEEG(t) {
  const ctx = eegCanvas.getContext("2d");
  const w = eegCanvas.width = eegCanvas.clientWidth;
  const h = eegCanvas.height = eegCanvas.clientHeight;
  ctx.clearRect(0, 0, w, h);

  const windowSec = 4.0; // Show 4 seconds of rolling EEG
  const endSample = Math.floor(t * FS);
  const startSample = Math.max(0, endSample - Math.floor(windowSec * FS));
  const numSamples = endSample - startSample;
  if (numSamples < 2) return;

  const nCh = EEG_DATA.length;
  const chHeight = h / nCh;

  ctx.lineWidth = 1.3;
  ctx.strokeStyle = "#00f2fe";

  for (let ch = 0; ch < nCh; ch++) {
    const raw = EEG_DATA[ch];
    const yCenter = (ch + 0.5) * chHeight;

    // Draw baseline grid
    ctx.strokeStyle = "rgba(255,255,255,0.04)";
    ctx.beginPath();
    ctx.moveTo(0, yCenter);
    ctx.lineTo(w, yCenter);
    ctx.stroke();

    // Draw EEG Waveform
    ctx.strokeStyle = ch % 2 === 0 ? "#00f2fe" : "#38bdf8";
    ctx.beginPath();
    for (let i = 0; i < numSamples; i++) {
      const sIdx = startSample + i;
      const x = (i / (windowSec * FS)) * w;
      const val = raw[sIdx] || 0;
      const y = yCenter - (val * (chHeight * 0.38));
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    }
    ctx.stroke();

    // Channel label
    ctx.fillStyle = "rgba(255,255,255,0.3)";
    ctx.font = "9px monospace";
    ctx.fillText(`CH${ch+1}`, 8, yCenter - 4);
  }
}

// LOGIT TRAJECTORY CHART RENDERER
function renderChart(t) {
  const ctx = chartCanvas.getContext("2d");
  const w = chartCanvas.width = chartCanvas.clientWidth;
  const h = chartCanvas.height = chartCanvas.clientHeight;
  ctx.clearRect(0, 0, w, h);

  const yCenter = h / 2;
  ctx.strokeStyle = "rgba(255,255,255,0.1)";
  ctx.beginPath(); ctx.moveTo(0, yCenter); ctx.lineTo(w, yCenter); ctx.stroke();

  // Draw Thresholds
  const scaleY = (h * 0.35) / 2.0; // +/- 2.0 logit range
  ctx.strokeStyle = "rgba(56,189,248,0.25)";
  ctx.setLineDash([3, 3]);
  ctx.beginPath(); ctx.moveTo(0, yCenter - 0.25 * scaleY); ctx.lineTo(w, yCenter - 0.25 * scaleY); ctx.stroke();
  ctx.strokeStyle = "rgba(251,146,60,0.25)";
  ctx.beginPath(); ctx.moveTo(0, yCenter + 0.25 * scaleY); ctx.lineTo(w, yCenter + 0.25 * scaleY); ctx.stroke();
  ctx.setLineDash([]);

  // Plot Logit A & Logit B lines up to TOTAL_DURATION
  ctx.lineWidth = 2.0;
  
  // Logit A
  ctx.strokeStyle = "#00f2fe";
  ctx.beginPath();
  for (let i = 0; i < TELEMETRY_DATA.length; i++) {
    const item = TELEMETRY_DATA[i];
    const x = (item.t / TOTAL_DURATION) * w;
    const y = yCenter - (item.logit_a * scaleY);
    if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
  }
  ctx.stroke();

  // Logit B
  ctx.strokeStyle = "#fb923c";
  ctx.beginPath();
  for (let i = 0; i < TELEMETRY_DATA.length; i++) {
    const item = TELEMETRY_DATA[i];
    const x = (item.t / TOTAL_DURATION) * w;
    const y = yCenter - (item.logit_b * scaleY);
    if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
  }
  ctx.stroke();

  // Current Time Cursor
  const curX = (t / TOTAL_DURATION) * w;
  ctx.strokeStyle = "#ffffff";
  ctx.lineWidth = 1.5;
  ctx.beginPath(); ctx.moveTo(curX, 0); ctx.lineTo(curX, h); ctx.stroke();
}

// MAIN ANIMATION LOOP
function tick(timestamp) {
  if (!lastFrameTime) lastFrameTime = timestamp;
  const deltaSec = (timestamp - lastFrameTime) / 1000.0;
  lastFrameTime = timestamp;

  if (isPlaying) {
    currentTime += deltaSec * playbackSpeed;
    if (currentTime >= TOTAL_DURATION) {
      currentTime = TOTAL_DURATION;
      pause();
    }
    updateUI(currentTime);
  }

  animationFrameId = requestAnimationFrame(tick);
}

function play() {
  isPlaying = true;
  btnPlayPause.innerText = "⏸";
  lastFrameTime = null;
}

function pause() {
  isPlaying = false;
  btnPlayPause.innerText = "▶";
  lastFrameTime = null;
}

btnPlayPause.addEventListener("click", () => {
  if (isPlaying) pause(); else play();
});

timelineSlider.addEventListener("input", (e) => {
  currentTime = (e.target.value / 1000) * TOTAL_DURATION;
  updateUI(currentTime);
});

speedButtons.forEach(btn => {
  btn.addEventListener("click", () => {
    speedButtons.forEach(b => b.classList.remove("active"));
    btn.classList.add("active");
    playbackSpeed = parseFloat(btn.dataset.speed);
  });
});

// START
updateUI(0.0);
animationFrameId = requestAnimationFrame(tick);
setTimeout(play, 600); // Auto-play after 600ms
</script>
</body>
</html>
"""

def generate_interactive_ui(
    subject: str = "S1_data_preproc",
    montage: str = "near_ear_expanded",
    trial_idx: int = 0,
    checkpoint: str = "",
    output_html: str = "bci_realtime_dashboard.html"
):
    print("=" * 85)
    print("  GENERATING INTERACTIVE REAL-TIME BCI DASHBOARD")
    print(f"  Subject: {subject} | Montage: {montage} | Trial: {trial_idx}")
    print("=" * 85)

    # 1. Setup Model
    montage_channels = MONTAGES[montage]
    n_ch = len(montage_channels)
    model = CATCNDirectDecoder(eeg_channels=n_ch, audio_channels=1, hidden_dim=64, max_lag_samples=8)
    
    ckpt_path = checkpoint
    if not ckpt_path:
        for candidate in ["/kaggle/working/catcn_deployment_weights.pt", "/kaggle/working/catcn_universal_model.pt"]:
            if Path(candidate).exists():
                ckpt_path = candidate
                break
        
    if ckpt_path and Path(ckpt_path).exists():
        print(f"[MODEL] Loading trained checkpoint from: {ckpt_path}")
        state = torch.load(ckpt_path, map_location="cpu")
        model.load_state_dict(state.get("model_state_dict", state.get("state_dict", state)))
        print("[MODEL] Checkpoint loaded successfully!")
    else:
        print("[MODEL WARNING] Running with initialized weights.")
        
    model.eval()

    # 2. Load Genuine DTU Data
    files = subject_files()
    target_files = [f for f in files if f.stem == subject or f.stem.split("_")[0] == subject.split("_")[0]]
    if not target_files:
        raise FileNotFoundError(f"Could not find DTU subject file for {subject} in DATA_DIR.")
        
    mapping, envelopes = get_mapping_data("gammatone")
    test_exs = list(load_subject_examples(target_files[0]))
    _, YA_all, YB_all = prepare_dataset(test_exs, montage_channels, 1.0, 6.0, subject, mapping, envelopes)
    
    t_idx = min(trial_idx, len(test_exs) - 1, len(YA_all) - 1)
    raw_eeg = test_exs[t_idx].eeg[:, montage_channels].astype(np.float32)
    ya = YA_all[t_idx].mean(axis=0).squeeze() if YA_all[t_idx].ndim > 1 else YA_all[t_idx].squeeze()
    yb = YB_all[t_idx].mean(axis=0).squeeze() if YB_all[t_idx].ndim > 1 else YB_all[t_idx].squeeze()
    
    min_len = min(len(raw_eeg), len(ya), len(yb))
    raw_eeg = raw_eeg[:min_len]
    ya = ya[:min_len]
    yb = yb[:min_len]
    total_sec = float(min_len / FS)

    # 3. Stream Telemetry
    pipeline = StreamingAADPipeline(
        model=model,
        n_eeg_channels=n_ch,
        fs=FS,
        raw_audio_input=False,
        window_sec=5.0,
        step_sec=0.5,
        engine_mode="torchscript",
        decision_alpha=0.7,
        decision_threshold=0.25,
        n_confirm=2,
        boost_db=6.0
    )

    telemetry_list = []
    chunk_samples = 16
    idx = 0
    while idx < min_len:
        end_idx = min(idx + chunk_samples, min_len)
        telem = pipeline.feed_sample_block(raw_eeg[idx:end_idx], ya[idx:end_idx], yb[idx:end_idx])
        if telem is not None:
            ga_db = float(20.0 * np.log10(max(1e-3, telem["gain_a"])))
            gb_db = float(20.0 * np.log10(max(1e-3, telem["gain_b"])))
            telemetry_list.append({
                "t": round(float(telem["timestamp_sec"]), 2),
                "logit_a": round(float(telem["logit_a"]), 3),
                "logit_b": round(float(telem["logit_b"]), 3),
                "delta": round(float(telem["raw_delta"]), 3),
                "smooth": round(float(telem["smoothed_score"]), 3),
                "stream": str(telem["attended_stream"]),
                "conf": round(float(telem["confidence"]), 3),
                "ga_db": round(ga_db, 1),
                "gb_db": round(gb_db, 1),
                "compute_ms": round(float(telem["compute_ms"]), 2),
                "switched": bool(telem["switched"])
            })
        idx = end_idx

    # Normalize EEG waveforms for oscilloscope display (8 channels x min_len)
    eeg_normalized = []
    for c in range(n_ch):
        ch_sig = raw_eeg[:, c]
        ch_std = np.std(ch_sig) + 1e-12
        ch_norm = (ch_sig - np.mean(ch_sig)) / ch_std
        eeg_normalized.append([round(float(v), 2) for v in ch_norm])

    # 4. Inject into HTML Template
    html_content = HTML_TEMPLATE
    html_content = html_content.replace("__SUBJECT__", subject)
    html_content = html_content.replace("__TRIAL_IDX__", str(t_idx))
    html_content = html_content.replace("__TOTAL_DURATION__", f"{total_sec:.1f}")
    html_content = html_content.replace("__TELEMETRY_JSON__", json.dumps(telemetry_list))
    html_content = html_content.replace("__EEG_JSON__", json.dumps(eeg_normalized))

    out_file = Path(output_html)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        f.write(html_content)

    print(f"\n[DASHBOARD READY] Successfully written interactive dashboard to:")
    print(f"  -> {out_file.resolve()} ({out_file.stat().st_size / 1024:.1f} KB)")
    print("\nTo render in Kaggle Notebook, run:")
    print("  from IPython.display import IFrame, display")
    print(f"  display(IFrame('{out_file.name}', width='100%', height=900))")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Interactive Real-Time BCI Dashboard")
    parser.add_argument("--subject", type=str, default="S1_data_preproc", help="DTU Subject name")
    parser.add_argument("--montage", type=str, default="near_ear_expanded", help="Electrode montage")
    parser.add_argument("--trial_idx", type=int, default=0, help="Trial index to visualize")
    parser.add_argument("--checkpoint", type=str, default="", help="Path to model checkpoint")
    parser.add_argument("--output_html", type=str, default="/kaggle/working/bci_realtime_dashboard.html", help="Output HTML path")
    args = parser.parse_args()

    generate_interactive_ui(
        subject=args.subject,
        montage=args.montage,
        trial_idx=args.trial_idx,
        checkpoint=args.checkpoint,
        output_html=args.output_html
    )
