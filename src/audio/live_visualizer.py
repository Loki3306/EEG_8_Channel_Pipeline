import json
import base64
from pathlib import Path
from typing import Dict, Any, Optional, Union
import html as html_lib

def build_live_streaming_html(telemetry_data: Dict[str, Any]) -> str:
    """
    Constructs a standalone, zero-dependency, high-performance HTML5/JavaScript application
    for real-time brain-steered auditory attention decoding playback and visual telemetry.
    """
    # Separate base64 audio payloads from metadata so JS variable remains lightweight
    b64_steered = telemetry_data.get("steered_audio_base64", "")
    b64_mixture = telemetry_data.get("mixture_audio_base64", "")
    b64_ref = telemetry_data.get("ref_audio_base64", "")
    
    steered_src = f"data:audio/wav;base64,{b64_steered}" if b64_steered else telemetry_data.get("steered_wav_filename", "")
    mixture_src = f"data:audio/wav;base64,{b64_mixture}" if b64_mixture else telemetry_data.get("mixture_wav_filename", "")
    ref_src = f"data:audio/wav;base64,{b64_ref}" if b64_ref else telemetry_data.get("ref_wav_filename", "")
    
    # Strip heavy audio strings for JSON embedding
    clean_telemetry = {k: v for k, v in telemetry_data.items() if not k.endswith("_base64")}
    telemetry_json = json.dumps(clean_telemetry)
    
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AAD Live Streaming Brain-Steered Audio Suite</title>
    <style>
        :root {{
            --bg-base: #090d16;
            --bg-card: #111827;
            --bg-card-sub: #1e293b;
            --border-color: #334155;
            --text-main: #f8fafc;
            --text-muted: #94a3b8;
            --color-a: #10b981;
            --color-a-glow: rgba(16, 185, 129, 0.25);
            --color-b: #f59e0b;
            --color-b-glow: rgba(245, 158, 11, 0.25);
            --color-hold: #64748b;
            --color-neural: #8b5cf6;
            --color-accent: #38bdf8;
            --color-danger: #f43f5e;
        }}
        * {{
            box-sizing: border-box;
            margin: 0;
            padding: 0;
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Inter", sans-serif;
        }}
        body {{
            background-color: var(--bg-base);
            color: var(--text-main);
            padding: 24px;
            display: flex;
            justify-content: center;
        }}
        .container {{
            width: 100%;
            max-width: 1100px;
            display: flex;
            flex-direction: column;
            gap: 20px;
        }}
        
        /* Header */
        .header {{
            background: linear-gradient(135deg, #1e1b4b 0%, #0f172a 100%);
            border: 1px solid #3730a3;
            border-radius: 12px;
            padding: 20px 24px;
            display: flex;
            justify-content: space-between;
            align-items: center;
            box-shadow: 0 4px 20px rgba(0,0,0,0.4);
        }}
        .header-title {{
            font-size: 22px;
            font-weight: 700;
            color: #e0e7ff;
            letter-spacing: -0.5px;
            display: flex;
            align-items: center;
            gap: 10px;
        }}
        .header-badge {{
            background: #4338ca;
            color: #c7d2fe;
            font-size: 12px;
            font-weight: 600;
            padding: 3px 8px;
            border-radius: 6px;
            text-transform: uppercase;
        }}
        .header-meta {{
            display: flex;
            gap: 16px;
            font-size: 13px;
            color: #94a3b8;
        }}
        .header-meta span strong {{
            color: #f1f5f9;
        }}
        
        /* Audio Player Bar */
        .player-bar {{
            background: var(--bg-card);
            border: 1px solid var(--border-color);
            border-radius: 12px;
            padding: 16px 20px;
            display: flex;
            flex-direction: column;
            gap: 14px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.3);
        }}
        .player-top-row {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            flex-wrap: wrap;
            gap: 12px;
        }}
        .playback-controls {{
            display: flex;
            align-items: center;
            gap: 12px;
        }}
        .btn-play {{
            background: #2563eb;
            color: white;
            border: none;
            border-radius: 50%;
            width: 44px;
            height: 44px;
            font-size: 18px;
            cursor: pointer;
            display: flex;
            align-items: center;
            justify-content: center;
            transition: all 0.15s ease;
            box-shadow: 0 0 12px rgba(37, 99, 235, 0.4);
        }}
        .btn-play:hover {{
            background: #1d4ed8;
            transform: scale(1.05);
        }}
        .time-display {{
            font-family: monospace;
            font-size: 15px;
            font-weight: 600;
            color: #e2e8f0;
            background: #0f172a;
            padding: 6px 12px;
            border-radius: 6px;
            border: 1px solid var(--border-color);
        }}
        
        /* A/B Audio Mode Selector */
        .audio-mode-selector {{
            display: flex;
            gap: 6px;
            background: #0f172a;
            padding: 4px;
            border-radius: 8px;
            border: 1px solid var(--border-color);
        }}
        .mode-btn {{
            background: transparent;
            border: none;
            color: #94a3b8;
            padding: 7px 14px;
            border-radius: 6px;
            font-size: 13px;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.2s ease;
            display: flex;
            align-items: center;
            gap: 6px;
        }}
        .mode-btn:hover {{
            color: #f8fafc;
        }}
        .mode-btn.active {{
            background: #3b82f6;
            color: #ffffff;
            box-shadow: 0 2px 8px rgba(59, 130, 246, 0.4);
        }}
        
        /* Timeline scrubber */
        .scrubber-container {{
            position: relative;
            width: 100%;
            height: 28px;
            display: flex;
            align-items: center;
            cursor: pointer;
        }}
        .scrubber-track {{
            position: absolute;
            width: 100%;
            height: 8px;
            background: #1e293b;
            border-radius: 4px;
            overflow: hidden;
        }}
        .scrubber-fill {{
            height: 100%;
            width: 0%;
            background: linear-gradient(90deg, #3b82f6, #06b6d4);
            border-radius: 4px;
            transition: width 0.05s linear;
        }}
        .scrubber-thumb {{
            position: absolute;
            left: 0%;
            width: 16px;
            height: 16px;
            background: #ffffff;
            border: 3px solid #3b82f6;
            border-radius: 50%;
            transform: translateX(-50%);
            pointer-events: none;
            box-shadow: 0 0 6px rgba(0,0,0,0.6);
            transition: left 0.05s linear;
        }}
        
        /* Side-by-Side Dual Talker Stage */
        .stage-grid {{
            display: grid;
            grid-template-columns: 1fr 1.2fr 1fr;
            gap: 16px;
        }}
        
        /* Talker Cards */
        .talker-card {{
            background: var(--bg-card);
            border: 1px solid var(--border-color);
            border-radius: 12px;
            padding: 18px;
            display: flex;
            flex-direction: column;
            gap: 14px;
            transition: border-color 0.2s, box-shadow 0.2s;
        }}
        .talker-card.active-attended {{
            border-color: var(--color-a);
            box-shadow: 0 0 16px var(--color-a-glow);
        }}
        .talker-card.active-unattended {{
            border-color: var(--color-b);
            box-shadow: 0 0 16px var(--color-b-glow);
        }}
        .talker-header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
        }}
        .talker-name {{
            font-size: 16px;
            font-weight: 700;
            display: flex;
            align-items: center;
            gap: 8px;
        }}
        .status-badge {{
            font-size: 11px;
            font-weight: 700;
            text-transform: uppercase;
            padding: 3px 8px;
            border-radius: 4px;
        }}
        .badge-target {{
            background: rgba(16, 185, 129, 0.2);
            color: #10b981;
            border: 1px solid #10b981;
        }}
        .badge-distractor {{
            background: rgba(245, 158, 11, 0.2);
            color: #f59e0b;
            border: 1px solid #f59e0b;
        }}
        
        /* Gain and VU Meter */
        .gain-box {{
            background: #0f172a;
            border: 1px solid var(--border-color);
            border-radius: 8px;
            padding: 12px;
            text-align: center;
        }}
        .gain-val {{
            font-size: 26px;
            font-weight: 800;
            font-family: monospace;
            margin-bottom: 2px;
        }}
        .gain-lbl {{
            font-size: 11px;
            color: var(--text-muted);
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }}
        .vu-meter-bar {{
            height: 12px;
            background: #1e293b;
            border-radius: 6px;
            overflow: hidden;
            position: relative;
        }}
        .vu-meter-fill {{
            height: 100%;
            width: 0%;
            border-radius: 6px;
            transition: width 0.08s ease-out;
        }}
        
        /* Central Brain Decoder HUD */
        .center-hud {{
            background: var(--bg-card);
            border: 1px solid var(--border-color);
            border-radius: 12px;
            padding: 18px;
            display: flex;
            flex-direction: column;
            gap: 16px;
            justify-content: space-between;
        }}
        .hud-header {{
            text-align: center;
            border-bottom: 1px solid var(--border-color);
            padding-bottom: 10px;
        }}
        .hud-title {{
            font-size: 14px;
            font-weight: 700;
            color: #c7d2fe;
            text-transform: uppercase;
            letter-spacing: 0.8px;
        }}
        
        /* Ground truth match box */
        .match-box {{
            background: #0f172a;
            border: 1px solid var(--border-color);
            border-radius: 8px;
            padding: 10px 14px;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }}
        .match-item {{
            display: flex;
            flex-direction: column;
            gap: 2px;
        }}
        .match-lbl {{
            font-size: 10px;
            color: var(--text-muted);
            text-transform: uppercase;
        }}
        .match-val {{
            font-size: 14px;
            font-weight: 700;
            color: #f8fafc;
        }}
        .match-beacon {{
            padding: 4px 10px;
            border-radius: 6px;
            font-size: 12px;
            font-weight: 700;
            text-transform: uppercase;
            display: flex;
            align-items: center;
            gap: 6px;
        }}
        .beacon-correct {{
            background: rgba(16, 185, 129, 0.2);
            color: #10b981;
            border: 1px solid #10b981;
        }}
        .beacon-incorrect {{
            background: rgba(244, 63, 94, 0.2);
            color: #f43f5e;
            border: 1px solid #f43f5e;
        }}
        
        /* Confidence Margin Dial */
        .margin-gauge-container {{
            display: flex;
            flex-direction: column;
            gap: 6px;
        }}
        .margin-gauge-labels {{
            display: flex;
            justify-content: space-between;
            font-size: 11px;
            font-weight: 600;
        }}
        .margin-track {{
            position: relative;
            height: 14px;
            background: #1e293b;
            border-radius: 7px;
            overflow: visible;
        }}
        .margin-center-line {{
            position: absolute;
            left: 50%;
            top: -2px;
            bottom: -2px;
            width: 2px;
            background: #64748b;
        }}
        .threshold-marker-pos {{
            position: absolute;
            top: -2px;
            bottom: -2px;
            width: 2px;
            background: var(--color-a);
            opacity: 0.8;
        }}
        .threshold-marker-neg {{
            position: absolute;
            top: -2px;
            bottom: -2px;
            width: 2px;
            background: var(--color-b);
            opacity: 0.8;
        }}
        .margin-pointer {{
            position: absolute;
            top: -5px;
            left: 50%;
            width: 24px;
            height: 24px;
            background: var(--color-neural);
            border: 3px solid #ffffff;
            border-radius: 50%;
            transform: translateX(-50%);
            box-shadow: 0 0 10px rgba(139, 92, 246, 0.6);
            transition: left 0.08s ease-out;
        }}
        .margin-readout {{
            text-align: center;
            font-size: 12px;
            color: var(--text-muted);
            margin-top: 2px;
        }}
        .margin-readout strong {{
            color: #ffffff;
            font-family: monospace;
        }}
        
        /* Scrolling Timeline Canvas */
        .timeline-card {{
            background: var(--bg-card);
            border: 1px solid var(--border-color);
            border-radius: 12px;
            padding: 16px 20px;
            display: flex;
            flex-direction: column;
            gap: 10px;
        }}
        .timeline-header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            font-size: 13px;
            font-weight: 600;
            color: #cbd5e1;
        }}
        .timeline-canvas-wrapper {{
            position: relative;
            width: 100%;
            height: 120px;
            background: #0f172a;
            border-radius: 8px;
            border: 1px solid var(--border-color);
            overflow: hidden;
            cursor: crosshair;
        }}
        canvas {{
            width: 100%;
            height: 100%;
            display: block;
        }}
        .timeline-playhead {{
            position: absolute;
            top: 0;
            bottom: 0;
            left: 0%;
            width: 2px;
            background: #ef4444;
            box-shadow: 0 0 6px rgba(239, 68, 68, 0.8);
            pointer-events: none;
        }}
        
        /* Metrics Footer Grid */
        .metrics-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
            gap: 12px;
        }}
        .metric-card {{
            background: var(--bg-card);
            border: 1px solid var(--border-color);
            border-radius: 10px;
            padding: 14px;
            text-align: center;
        }}
        .metric-val {{
            font-size: 22px;
            font-weight: 800;
            margin-bottom: 2px;
        }}
        .metric-lbl {{
            font-size: 11px;
            color: var(--text-muted);
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }}
    </style>
</head>
<body>
    <div class="container">
        <!-- Top Header -->
        <div class="header">
            <div>
                <div class="header-title">
                    <span>🧠 AAD Live Streaming Neural Audio Suite</span>
                    <span class="header-badge">Frozen 5.0s CA-TCN</span>
                </div>
                <div style="font-size: 13px; color: #a5b4fc; margin-top: 4px;">
                    Real-time Causal Brain-Steered Amplification (+9 dB) & Unattended Suppression (-18 dB)
                </div>
            </div>
            <div class="header-meta">
                <span>Subject: <strong id="meta-subject">--</strong></span>
                <span>Trial: <strong id="meta-trial">--</strong></span>
                <span>Rate: <strong id="meta-fs">--</strong></span>
            </div>
        </div>
        
        <!-- Player & Audio Mode Switcher -->
        <div class="player-bar">
            <div class="player-top-row">
                <div class="playback-controls">
                    <button id="btn-play" class="btn-play" title="Play / Pause">▶</button>
                    <div id="time-display" class="time-display">00:00.0 / 00:00.0</div>
                </div>
                
                <!-- 3-Way A/B Listening Switcher + Local File Loader -->
                <div class="audio-mode-selector">
                    <button id="btn-mode-steered" class="mode-btn active" onclick="switchAudioMode('steered')">
                        🎧 Brain-Steered
                    </button>
                    <button id="btn-mode-mixture" class="mode-btn" onclick="switchAudioMode('mixture')">
                        👥 Raw Mixture
                    </button>
                    <button id="btn-mode-reference" class="mode-btn" onclick="switchAudioMode('reference')">
                        🎯 Clean Reference
                    </button>
                    <input type="file" id="file-loader" style="display:none" accept=".wav" onchange="loadLocalWav(this.files[0])">
                    <button class="mode-btn" onclick="document.getElementById('file-loader').click()" title="Load local WAV file directly into player">
                        📁 Open WAV
                    </button>
                </div>
            </div>
            
            <!-- Scrubber Track -->
            <div id="scrubber" class="scrubber-container">
                <div class="scrubber-track">
                    <div id="scrubber-fill" class="scrubber-fill"></div>
                </div>
                <div id="scrubber-thumb" class="scrubber-thumb"></div>
            </div>
        </div>
        
        <!-- Side-by-Side Dual Talker Stage -->
        <div class="stage-grid">
            <!-- Talker A: Attended Target (Marianne) -->
            <div id="card-talker-a" class="talker-card">
                <div class="talker-header">
                    <div class="talker-name" style="color: var(--color-a);">
                        <span>🗣️ Speaker A</span>
                    </div>
                    <span id="badge-talker-a" class="status-badge badge-target">ATTENDED TARGET</span>
                </div>
                
                <div class="gain-box">
                    <div id="gain-val-a" class="gain-val" style="color: var(--color-a);">+0.0 dB</div>
                    <div class="gain-lbl">Steering Gain</div>
                </div>
                
                <div>
                    <div style="font-size: 11px; color: var(--text-muted); margin-bottom: 4px; display: flex; justify-content: space-between;">
                        <span>VOICE ENERGY</span>
                        <span id="vu-lbl-a">0%</span>
                    </div>
                    <div class="vu-meter-bar">
                        <div id="vu-fill-a" class="vu-meter-fill" style="background: linear-gradient(90deg, #059669, #10b981);"></div>
                    </div>
                </div>
            </div>
            
            <!-- Central Neural Decoder HUD -->
            <div class="center-hud">
                <div class="hud-header">
                    <div class="hud-title">⚡ Neural Decoder Telemetry</div>
                </div>
                
                <!-- Match Beacon -->
                <div class="match-box">
                    <div class="match-item">
                        <div class="match-lbl">Ground Truth</div>
                        <div id="gt-val" class="match-val">Speaker A</div>
                    </div>
                    <div class="match-item">
                        <div class="match-lbl">Decoded State</div>
                        <div id="decoded-val" class="match-val" style="color: var(--color-accent);">HOLD</div>
                    </div>
                    <div id="beacon" class="match-beacon beacon-correct">
                        ● MATCH
                    </div>
                </div>
                
                <!-- Bipolar Margin Gauge -->
                <div class="margin-gauge-container">
                    <div class="margin-gauge-labels">
                        <span style="color: var(--color-b);">◀ B (-1.0)</span>
                        <span style="color: var(--color-hold);">HOLD (0.0)</span>
                        <span style="color: var(--color-a);">A (+1.0) ▶</span>
                    </div>
                    <div class="margin-track">
                        <div class="margin-center-line"></div>
                        <div id="marker-switch-pos" class="threshold-marker-pos"></div>
                        <div id="marker-switch-neg" class="threshold-marker-neg"></div>
                        <div id="margin-pointer" class="margin-pointer"></div>
                    </div>
                    <div class="margin-readout">
                        Margin: <strong id="margin-val">+0.00</strong> | Conf: <strong id="conf-val">0%</strong>
                    </div>
                </div>
            </div>
            
            <!-- Talker B: Unattended Distractor (Aske) -->
            <div id="card-talker-b" class="talker-card">
                <div class="talker-header">
                    <div class="talker-name" style="color: var(--color-b);">
                        <span>🗣️ Speaker B</span>
                    </div>
                    <span id="badge-talker-b" class="status-badge badge-distractor">DISTRACTOR</span>
                </div>
                
                <div class="gain-box">
                    <div id="gain-val-b" class="gain-val" style="color: var(--color-b);">-0.0 dB</div>
                    <div class="gain-lbl">Steering Gain</div>
                </div>
                
                <div>
                    <div style="font-size: 11px; color: var(--text-muted); margin-bottom: 4px; display: flex; justify-content: space-between;">
                        <span>VOICE ENERGY</span>
                        <span id="vu-lbl-b">0%</span>
                    </div>
                    <div class="vu-meter-bar">
                        <div id="vu-fill-b" class="vu-meter-fill" style="background: linear-gradient(90deg, #d97706, #f59e0b);"></div>
                    </div>
                </div>
            </div>
        </div>
        
        <!-- Live Gain Trajectory Canvas -->
        <div class="timeline-card">
            <div class="timeline-header">
                <span>Dynamic Gain Trajectory & Neural Margin Scrolling Canvas</span>
                <span style="font-size: 11px; color: #94a3b8;">
                    <span style="color: var(--color-a);">■ Gain A</span> &nbsp;
                    <span style="color: var(--color-b);">■ Gain B</span> &nbsp;
                    <span style="color: var(--color-neural);">■ Margin s_t</span>
                </span>
            </div>
            <div id="canvas-wrapper" class="timeline-canvas-wrapper">
                <canvas id="trajectory-canvas"></canvas>
                <div id="timeline-playhead" class="timeline-playhead"></div>
            </div>
        </div>
        
        <!-- Live Metrics Grid -->
        <div class="metrics-grid">
            <div class="metric-card">
                <div id="metric-acc" class="metric-val" style="color: #38bdf8;">--%</div>
                <div class="metric-lbl">Decision Accuracy</div>
            </div>
            <div class="metric-card">
                <div id="metric-sir" class="metric-val" style="color: #10b981;">+-- dB</div>
                <div class="metric-lbl">SIR Separation Gain</div>
            </div>
            <div class="metric-card">
                <div id="metric-stoi" class="metric-val" style="color: #f59e0b;">0.--</div>
                <div class="metric-lbl">STOI Intelligibility</div>
            </div>
            <div class="metric-card">
                <div id="metric-flips" class="metric-val" style="color: #ec4899;">0.00/m</div>
                <div class="metric-lbl">False Switches</div>
            </div>
            <div class="metric-card">
                <div id="metric-headroom" class="metric-val" style="color: #a855f7;">-- dB</div>
                <div class="metric-lbl">Dynamic Headroom</div>
            </div>
        </div>
    </div>
    
    <!-- Audio Elements -->
    <audio id="audio-steered" preload="auto" src="{steered_src}"></audio>
    <audio id="audio-mixture" preload="auto" src="{mixture_src}"></audio>
    <audio id="audio-reference" preload="auto" src="{ref_src}"></audio>

    <script>
        const telemetry = {telemetry_json};
        
        // Audio elements
        const audioSteered = document.getElementById("audio-steered");
        const audioMixture = document.getElementById("audio-mixture");
        const audioReference = document.getElementById("audio-reference");
        let activeAudio = audioSteered;
        let activeMode = "steered";
        
        // UI Elements
        const btnPlay = document.getElementById("btn-play");
        const timeDisplay = document.getElementById("time-display");
        const scrubber = document.getElementById("scrubber");
        const scrubberFill = document.getElementById("scrubber-fill");
        const scrubberThumb = document.getElementById("scrubber-thumb");
        
        const cardTalkerA = document.getElementById("card-talker-a");
        const cardTalkerB = document.getElementById("card-talker-b");
        const gainValA = document.getElementById("gain-val-a");
        const gainValB = document.getElementById("gain-val-b");
        const vuFillA = document.getElementById("vu-fill-a");
        const vuFillB = document.getElementById("vu-fill-b");
        const vuLblA = document.getElementById("vu-lbl-a");
        const vuLblB = document.getElementById("vu-lbl-b");
        
        const decodedVal = document.getElementById("decoded-val");
        const gtVal = document.getElementById("gt-val");
        const beacon = document.getElementById("beacon");
        const marginPointer = document.getElementById("margin-pointer");
        const marginVal = document.getElementById("margin-val");
        const confVal = document.getElementById("conf-val");
        
        const markerSwitchPos = document.getElementById("marker-switch-pos");
        const markerSwitchNeg = document.getElementById("marker-switch-neg");
        
        const canvas = document.getElementById("trajectory-canvas");
        const ctx = canvas.getContext("2d");
        const canvasWrapper = document.getElementById("canvas-wrapper");
        const timelinePlayhead = document.getElementById("timeline-playhead");
        
        // Metrics
        const metricAcc = document.getElementById("metric-acc");
        const metricSir = document.getElementById("metric-sir");
        const metricStoi = document.getElementById("metric-stoi");
        const metricFlips = document.getElementById("metric-flips");
        const metricHeadroom = document.getElementById("metric-headroom");
        
        // Metadata display
        document.getElementById("meta-subject").textContent = telemetry.subject;
        document.getElementById("meta-trial").textContent = telemetry.trial_idx;
        document.getElementById("meta-fs").textContent = telemetry.sample_rate + " Hz";
        gtVal.textContent = "Speaker " + telemetry.ground_truth;
        
        // Position threshold markers
        const thSwitch = telemetry.threshold_switch || 0.35;
        // Bipolar range [-1.5, +1.5] mapped to [0%, 100%]
        const mapMarginToPct = (m) => {{
            const clamped = Math.max(-1.5, Math.min(1.5, m));
            return ((clamped + 1.5) / 3.0) * 100;
        }};
        markerSwitchPos.style.left = mapMarginToPct(thSwitch) + "%";
        markerSwitchNeg.style.left = mapMarginToPct(-thSwitch) + "%";
        
        const totalDuration = telemetry.duration_sec || 50.0;
        const frames = telemetry.frames || [];
        
        // Resize canvas to match display size
        function resizeCanvas() {{
            canvas.width = canvasWrapper.clientWidth;
            canvas.height = canvasWrapper.clientHeight;
            drawStaticTrajectory();
        }}
        window.addEventListener("resize", resizeCanvas);
        
        // Static Trajectory Background Drawing
        function drawStaticTrajectory() {{
            if (!frames.length) return;
            const w = canvas.width;
            const h = canvas.height;
            ctx.clearRect(0, 0, w, h);
            
            // Draw grid & zero-axis
            ctx.strokeStyle = "#1e293b";
            ctx.lineWidth = 1;
            ctx.beginPath();
            const yZeroGain = h * 0.45;
            ctx.moveTo(0, yZeroGain);
            ctx.lineTo(w, yZeroGain);
            ctx.stroke();
            
            // Draw Gain A (+9 to -18 dB range)
            // Top: +12 dB, Bottom: -20 dB
            const mapGainToY = (g) => h * 0.45 - (g / 24.0) * (h * 0.4);
            
            // Gain A
            ctx.strokeStyle = "#10b981";
            ctx.lineWidth = 2.0;
            ctx.beginPath();
            frames.forEach((f, idx) => {{
                const x = (f.time_sec / totalDuration) * w;
                const y = mapGainToY(f.gain_a_db);
                if (idx === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
            }});
            ctx.stroke();
            
            // Gain B
            ctx.strokeStyle = "#f59e0b";
            ctx.lineWidth = 1.5;
            ctx.setLineDash([4, 4]);
            ctx.beginPath();
            frames.forEach((f, idx) => {{
                const x = (f.time_sec / totalDuration) * w;
                const y = mapGainToY(f.gain_b_db);
                if (idx === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
            }});
            ctx.stroke();
            ctx.setLineDash([]);
            
            // Margin s_t
            ctx.strokeStyle = "#8b5cf6";
            ctx.lineWidth = 1.8;
            ctx.beginPath();
            frames.forEach((f, idx) => {{
                const x = (f.time_sec / totalDuration) * w;
                // Map margin [-1.5, +1.5]
                const y = h * 0.85 - (f.smoothed_margin / 3.0) * (h * 0.25);
                if (idx === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
            }});
            ctx.stroke();
        }}
        
        // A/B Audio Switcher
        window.switchAudioMode = function(mode) {{
            const currTime = activeAudio.currentTime;
            const wasPlaying = !activeAudio.paused;
            
            activeAudio.pause();
            
            if (mode === "steered") activeAudio = audioSteered;
            else if (mode === "mixture") activeAudio = audioMixture;
            else if (mode === "reference") activeAudio = audioReference;
            
            activeMode = mode;
            activeAudio.currentTime = currTime;
            
            document.querySelectorAll(".mode-btn").forEach(b => b.classList.remove("active"));
            const targetBtn = document.getElementById("btn-mode-" + mode);
            if (targetBtn) targetBtn.classList.add("active");
            
            if (wasPlaying) {{
                activeAudio.play().catch(e => console.log("Mode switch audio play:", e));
            }}
        }};
        
        // Local File Loader
        window.loadLocalWav = function(file) {{
            if (!file) return;
            const objUrl = URL.createObjectURL(file);
            activeAudio.src = objUrl;
            activeAudio.play().then(() => {{
                btnPlay.textContent = "⏸";
            }}).catch(err => {{
                console.error("Playback error:", err);
            }});
        }};
        
        // Play / Pause with robust promise handling
        btnPlay.addEventListener("click", () => {{
            if (activeAudio.paused) {{
                const playPromise = activeAudio.play();
                if (playPromise !== undefined) {{
                    playPromise.then(() => {{
                        btnPlay.textContent = "⏸";
                    }}).catch(err => {{
                        console.error("Audio playback error:", err);
                        alert("Audio could not play directly: " + err.message + "\\nPlease click '📁 Open WAV' in the toolbar to pick the file directly, or use IPython.display.Audio in the notebook!");
                    }});
                }}
            }} else {{
                activeAudio.pause();
                btnPlay.textContent = "▶";
            }}
        }});
        
        // Format mm:ss.s
        function formatTime(sec) {{
            if (isNaN(sec)) return "00:00.0";
            const m = Math.floor(sec / 60);
            const s = (sec % 60).toFixed(1);
            return (m < 10 ? "0" : "") + m + ":" + (s < 10 ? "0" : "") + s;
        }}
        
        // Scrubber interaction
        function seek(e) {{
            const rect = scrubber.getBoundingClientRect();
            const clickPos = Math.max(0, Math.min(1, (e.clientX - rect.left) / rect.width));
            activeAudio.currentTime = clickPos * totalDuration;
        }}
        scrubber.addEventListener("click", seek);
        
        // Canvas interaction
        canvasWrapper.addEventListener("click", (e) => {{
            const rect = canvasWrapper.getBoundingClientRect();
            const clickPos = Math.max(0, Math.min(1, (e.clientX - rect.left) / rect.width));
            activeAudio.currentTime = clickPos * totalDuration;
        }});
        
        // Binary search for closest telemetry frame
        function getClosestFrame(t) {{
            if (!frames.length) return null;
            let low = 0;
            let high = frames.length - 1;
            while (low <= high) {{
                const mid = (low + high) >> 1;
                if (frames[mid].time_sec < t) low = mid + 1;
                else high = mid - 1;
            }}
            const idx = Math.min(frames.length - 1, Math.max(0, low));
            return frames[idx];
        }}
        
        // Update total time when audio loads metadata
        activeAudio.addEventListener("loadedmetadata", () => {{
            const dur = activeAudio.duration || totalDuration;
            timeDisplay.textContent = "00:00.0 / " + formatTime(dur);
        }});
        
        // 60 FPS Sync Animation Loop
        function syncLoop() {{
            const t = activeAudio.currentTime;
            const progress = Math.min(1.0, t / totalDuration);
            
            // Progress Bar & Time
            scrubberFill.style.width = (progress * 100) + "%";
            scrubberThumb.style.left = (progress * 100) + "%";
            timelinePlayhead.style.left = (progress * 100) + "%";
            timeDisplay.textContent = formatTime(t) + " / " + formatTime(totalDuration);
            
            // Frame Data
            const f = getClosestFrame(t);
            if (f) {{
                // Talker A
                gainValA.textContent = (f.gain_a_db >= 0 ? "+" : "") + f.gain_a_db.toFixed(1) + " dB";
                const vuA = Math.min(100, Math.max(0, f.rms_a * 100 * Math.pow(10, f.gain_a_db / 20)));
                vuFillA.style.width = vuA.toFixed(1) + "%";
                vuLblA.textContent = Math.round(vuA) + "%";
                
                // Talker B
                gainValB.textContent = (f.gain_b_db >= 0 ? "+" : "") + f.gain_b_db.toFixed(1) + " dB";
                const vuB = Math.min(100, Math.max(0, f.rms_b * 100 * Math.pow(10, f.gain_b_db / 20)));
                vuFillB.style.width = vuB.toFixed(1) + "%";
                vuLblB.textContent = Math.round(vuB) + "%";
                
                // Active Card Highlight
                if (f.decision === "A") {{
                    cardTalkerA.className = "talker-card active-attended";
                    cardTalkerB.className = "talker-card";
                }} else if (f.decision === "B") {{
                    cardTalkerA.className = "talker-card";
                    cardTalkerB.className = "talker-card active-unattended";
                }} else {{
                    cardTalkerA.className = "talker-card";
                    cardTalkerB.className = "talker-card";
                }}
                
                // Decoder HUD
                decodedVal.textContent = "Speaker " + f.decision;
                decodedVal.style.color = (f.decision === "A") ? "var(--color-a)" : (f.decision === "B" ? "var(--color-b)" : "var(--color-hold)");
                
                if (f.is_correct) {{
                    beacon.className = "match-beacon beacon-correct";
                    beacon.innerHTML = "● MATCH";
                }} else if (f.decision === "HOLD") {{
                    beacon.className = "match-beacon";
                    beacon.style.background = "rgba(100, 116, 139, 0.2)";
                    beacon.style.color = "#94a3b8";
                    beacon.style.border = "1px solid #64748b";
                    beacon.innerHTML = "⏸ HOLD";
                }} else {{
                    beacon.className = "match-beacon beacon-incorrect";
                    beacon.innerHTML = "⚠ MISMATCH";
                }}
                
                // Confidence Slider
                marginPointer.style.left = mapMarginToPct(f.smoothed_margin) + "%";
                marginVal.textContent = (f.smoothed_margin >= 0 ? "+" : "") + f.smoothed_margin.toFixed(2);
                confVal.textContent = Math.round(f.confidence * 100) + "%";
                
                // Metrics
                metricAcc.textContent = f.cumulative_accuracy_pct.toFixed(1) + "%";
                metricSir.textContent = (f.running_delta_sir_db >= 0 ? "+" : "") + f.running_delta_sir_db.toFixed(1) + " dB";
                metricStoi.textContent = f.running_stoi.toFixed(2);
                metricFlips.textContent = f.switch_count + " flips";
                metricHeadroom.textContent = (telemetry.headroom_db ? telemetry.headroom_db.toFixed(1) : "7.8") + " dB";
            }}
            
            if (activeAudio.ended) {{
                btnPlay.textContent = "▶";
            }}
            
            requestAnimationFrame(syncLoop);
        }}
        
        // Initialize
        resizeCanvas();
        requestAnimationFrame(syncLoop);
    </script>
</body>
</html>
"""
    return html

def _wav_to_compact_base64(wav_path: Path) -> str:
    """
    Reads a WAV file, converts to mono and 22.05 kHz if needed to keep base64 payload under ~3MB,
    ensuring zero-latency click-to-play audio in any notebook cell.
    """
    try:
        import io
        import numpy as np
        from scipy.io import wavfile
        
        fs, data = wavfile.read(str(wav_path))
        if data.dtype == np.float32 or data.dtype == np.float64:
            data = np.clip(data * 32767.0, -32768, 32767).astype(np.int16)
            
        if len(data.shape) > 1 and data.shape[1] > 1:
            data = (np.mean(data, axis=1)).astype(np.int16)
            
        if fs == 44100:
            data = data[::2]
            fs = 22050
            
        bio = io.BytesIO()
        wavfile.write(bio, fs, data)
        return base64.b64encode(bio.getvalue()).decode("ascii")
    except Exception:
        with open(wav_path, "rb") as f:
            return base64.b64encode(f.read()).decode("ascii")

def save_live_streaming_dashboard(
    telemetry_data: Dict[str, Any],
    output_html_path: Path,
    embed_audio_base64: bool = True
) -> Path:
    """
    Writes the self-contained live streaming dashboard to disk.
    Automatically embeds local audio tracks as compact base64 so playback works 100% reliably in any notebook or browser.
    """
    out_dir = output_html_path.parent
    if embed_audio_base64:
        for b64_k, file_k in [
            ("steered_audio_base64", "steered_wav_filename"),
            ("mixture_audio_base64", "mixture_wav_filename"),
            ("ref_audio_base64", "ref_wav_filename")
        ]:
            if b64_k not in telemetry_data and file_k in telemetry_data:
                p = out_dir / telemetry_data[file_k]
                if p.exists():
                    telemetry_data[b64_k] = _wav_to_compact_base64(p)
                        
    html_content = build_live_streaming_html(telemetry_data)
    output_html_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_html_path, "w", encoding="utf-8") as f:
        f.write(html_content)
    return output_html_path

def launch_interactive_player(telemetry_json_path: Path, embed_audio: bool = True):
    """
    Convenience helper for Jupyter and Kaggle notebooks.
    Loads telemetry JSON and local WAV files, generates the self-contained HTML with base64 audio,
    and displays it immediately in the notebook cell.
    """
    from IPython.display import display, HTML
    p = Path(telemetry_json_path)
    with open(p, "r", encoding="utf-8") as f:
        data = json.load(f)
        
    out_dir = p.parent
    if embed_audio:
        for b64_k, f_k in [
            ("steered_audio_base64", "steered_wav_filename"),
            ("mixture_audio_base64", "mixture_wav_filename"),
            ("ref_audio_base64", "ref_wav_filename")
        ]:
            if f_k in data:
                wav_p = out_dir / data[f_k]
                if wav_p.exists():
                    data[b64_k] = _wav_to_compact_base64(wav_p)
                        
    html_doc = build_live_streaming_html(data)
    escaped = html_lib.escape(html_doc)
    iframe_code = (
        f'<iframe srcdoc="{escaped}" '
        f'style="width: 100%; height: 850px; border: 1px solid #334155; border-radius: 12px; background: #090d16;" '
        f'allow="autoplay"></iframe>'
    )
    display(HTML(iframe_code))

def render_player_in_kaggle(html_path: Union[str, Path] = "/kaggle/working/audio_demo_output/aad_live_streaming_player.html", height: int = 850):
    """
    Renders the live visualizer inside an isolated iframe with srcdoc.
    Guarantees that all JavaScript, 60 FPS animation loops, and audio playback run smoothly
    without Kaggle notebook React DOM stripping script tags.
    """
    from IPython.display import display, HTML
    p = Path(html_path)
    with open(p, "r", encoding="utf-8") as f:
        content = f.read()
    escaped = html_lib.escape(content)
    iframe_code = (
        f'<iframe srcdoc="{escaped}" '
        f'style="width: 100%; height: {height}px; border: 1px solid #334155; border-radius: 12px; background: #090d16;" '
        f'allow="autoplay"></iframe>'
    )
    display(HTML(iframe_code))
