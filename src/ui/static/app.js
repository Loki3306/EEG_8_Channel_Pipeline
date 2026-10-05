/**
 * NeuroSteer Clinical Suite — Client Application Logic
 * 
 * Features:
 * - Real-time WebSocket telemetry ingestion
 * - Web Audio API continuous PCM streaming for headphone monitoring
 * - 60 FPS Hi-DPI 8-Channel EEG Oscilloscope Canvas
 * - Real-Time CA-TCN Attention Decoder needle gauge & telemetry monitors
 */

class NeuroSteerApp {
    constructor() {
        this.ws = null;
        this.isPlaying = false;
        this.currentSubject = "S1";
        this.currentTrial = 4;
        this.listeningMode = "steered";
        this.totalDuration = 50.0;
        this.currentTime = 0.0;

        // Web Audio API
        this.audioCtx = null;
        this.nextAudioTime = 0;
        this.sampleRate = 16000;

        // EEG Rolling Oscilloscope Data
        this.channelNames = ["Cz", "FCz", "Fz", "C3", "C4", "CPz", "Pz", "Oz"];
        this.eegHistoryLength = 128; // ~2 seconds of rolling EEG at 64 Hz
        this.eegBuffers = Array.from({ length: 8 }, () => new Array(this.eegHistoryLength).fill(0));

        // DOM Elements
        this.initDOMElements();
        this.initCanvas();
        this.bindEvents();
        this.loadSubjects();
        this.checkDeviceStatus();
        this.connectWebSocket();
    }

    initDOMElements() {
        this.connectionBadge = document.getElementById("connectionBadge");
        this.connectionText = document.getElementById("connectionText");
        this.deviceStatusText = document.getElementById("deviceStatusText");

        this.subjectSelect = document.getElementById("subjectSelect");
        this.trialSelect = document.getElementById("trialSelect");
        this.sourceSelect = document.getElementById("sourceSelect");

        this.playBtn = document.getElementById("playBtn");
        this.playIcon = document.getElementById("playIcon");
        this.playBtnText = document.getElementById("playBtnText");
        this.resetBtn = document.getElementById("resetBtn");

        this.currentTimeEl = document.getElementById("currentTime");
        this.totalTimeEl = document.getElementById("totalTime");
        this.timeScrubber = document.getElementById("timeScrubber");
        this.scrubberFill = document.getElementById("scrubberFill");

        this.listeningModesGroup = document.getElementById("listeningModesGroup");

        // Decoder Elements
        this.decisionBadge = document.getElementById("decisionBadge");
        this.meterNeedle = document.getElementById("meterNeedle");
        this.currentMarginReadout = document.getElementById("currentMarginReadout");
        this.groundTruthTarget = document.getElementById("groundTruthTarget");
        this.decoderConfidence = document.getElementById("decoderConfidence");
        this.accuracyIndicator = document.getElementById("accuracyIndicator");

        // Gain Elements
        this.gainReadoutA = document.getElementById("gainReadoutA");
        this.gainReadoutB = document.getElementById("gainReadoutB");
        this.gainFillA = document.getElementById("gainFillA");
        this.gainFillB = document.getElementById("gainFillB");

        // Telemetry Elements
        this.dspTimeVal = document.getElementById("dspTimeVal");
        this.gpuTimeVal = document.getElementById("gpuTimeVal");
        this.cpuLoadVal = document.getElementById("cpuLoadVal");
        this.rtfVal = document.getElementById("rtfVal");

        this.channelLegend = document.getElementById("channelLegend");
        this.eegCanvas = document.getElementById("eegCanvas");
        this.ctx = this.eegCanvas.getContext("2d");
    }

    initCanvas() {
        const dpr = window.devicePixelRatio || 1;
        const rect = this.eegCanvas.getBoundingClientRect();
        this.eegCanvas.width = rect.width * dpr;
        this.eegCanvas.height = rect.height * dpr;
        this.ctx.scale(dpr, dpr);
        this.canvasWidth = rect.width;
        this.canvasHeight = rect.height;

        // Render legend
        this.channelLegend.innerHTML = "";
        this.channelNames.forEach((ch, idx) => {
            const item = document.createElement("span");
            item.className = "legend-item";
            item.textContent = `Ch ${idx + 1}: ${ch}`;
            this.channelLegend.appendChild(item);
        });

        this.drawEmptyOscilloscope();
    }

    bindEvents() {
        window.addEventListener("resize", () => this.initCanvas());

        this.playBtn.addEventListener("click", () => this.togglePlay());
        this.resetBtn.addEventListener("click", () => this.sendReset());

        this.timeScrubber.addEventListener("input", (e) => {
            const sec = parseFloat(e.target.value);
            this.sendSeek(sec);
        });

        this.subjectSelect.addEventListener("change", (e) => {
            this.currentSubject = e.target.value;
            this.loadTrials(this.currentSubject);
            this.sendSetSubject(this.currentSubject);
        });

        this.trialSelect.addEventListener("change", (e) => {
            this.currentTrial = parseInt(e.target.value, 10);
            this.sendSetTrial(this.currentTrial);
        });

        this.sourceSelect.addEventListener("change", (e) => {
            this.sendSetSource(e.target.value);
        });

        // Listening mode tabs
        const modeButtons = this.listeningModesGroup.querySelectorAll(".mode-tab");
        modeButtons.forEach(btn => {
            btn.addEventListener("click", () => {
                modeButtons.forEach(b => {
                    b.classList.remove("active");
                    b.querySelector(".tab-indicator").textContent = "○";
                });
                btn.classList.add("active");
                btn.querySelector(".tab-indicator").textContent = "●";

                this.listeningMode = btn.dataset.mode;
                this.sendListeningMode(this.listeningMode);
            });
        });
    }

    // =========================================================================
    // API Queries
    // =========================================================================
    async loadSubjects() {
        try {
            const res = await fetch("/api/subjects");
            const data = await res.json();
            if (data.status === "ok" && data.subjects) {
                this.subjectSelect.innerHTML = "";
                data.subjects.forEach(sub => {
                    const opt = document.createElement("option");
                    opt.value = sub.subject_id;
                    opt.textContent = `${sub.label} (${sub.win_rate}% Win Rate | Margin: +${sub.mean_margin})`;
                    if (sub.subject_id === this.currentSubject) opt.selected = true;
                    this.subjectSelect.appendChild(opt);
                });
                this.loadTrials(this.currentSubject);
            }
        } catch (e) {
            console.error("Failed to load subjects:", e);
        }
    }

    async loadTrials(subjectId) {
        try {
            const res = await fetch(`/api/trials/${subjectId}`);
            const data = await res.json();
            if (data.status === "ok" && data.trials) {
                this.trialSelect.innerHTML = "";
                data.trials.forEach(tr => {
                    const opt = document.createElement("option");
                    opt.value = tr.trial_id;
                    opt.textContent = tr.label;
                    if (tr.trial_id === this.currentTrial) opt.selected = true;
                    this.trialSelect.appendChild(opt);
                });
            }
        } catch (e) {
            console.error("Failed to load trials:", e);
        }
    }

    async checkDeviceStatus() {
        try {
            const res = await fetch("/api/device_status");
            const data = await res.json();
            if (data.status === "ok") {
                this.deviceStatusText.textContent = data.mode;
                if (data.respeaker_connected) {
                    this.deviceStatusText.style.color = "#15803d";
                }
            }
        } catch (e) {
            console.error("Device status check failed:", e);
        }
    }

    // =========================================================================
    // WebSocket Streaming Connection
    // =========================================================================
    connectWebSocket() {
        const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
        const wsUrl = `${protocol}//${window.location.host}/ws/stream`;
        this.ws = new WebSocket(wsUrl);

        this.ws.onopen = () => {
            this.connectionBadge.className = "status-indicator";
            this.connectionBadge.querySelector(".status-dot").style.backgroundColor = "#059669";
            this.connectionText.textContent = "SYSTEM CONNECTED";
            this.connectionText.style.color = "#065f46";
        };

        this.ws.onmessage = (event) => {
            try {
                const data = jsonParse(event.data);
                if (data.type === "tick") {
                    this.handleTick(data);
                } else if (data.type === "end_of_trial") {
                    this.handleEndOfTrial();
                }
            } catch (err) {
                console.error("WS Parse Error:", err);
            }
        };

        this.ws.onclose = () => {
            this.connectionBadge.className = "status-indicator";
            this.connectionBadge.querySelector(".status-dot").style.backgroundColor = "#dc2626";
            this.connectionText.textContent = "DISCONNECTED (RECONNECTING)";
            this.connectionText.style.color = "#991b1b";
            setTimeout(() => this.connectWebSocket(), 2000);
        };
    }

    sendWs(msg) {
        if (this.ws && this.ws.readyState === WebSocket.OPEN) {
            this.ws.send(JSON.stringify(msg));
        }
    }

    togglePlay() {
        this.initAudioContext();
        if (!this.isPlaying) {
            this.sendWs({ action: "play" });
            this.isPlaying = true;
            this.playBtn.classList.add("paused");
            this.playIcon.textContent = "⏸";
            this.playBtnText.textContent = "PAUSE SIMULATION";
        } else {
            this.sendWs({ action: "pause" });
            this.isPlaying = false;
            this.playBtn.classList.remove("paused");
            this.playIcon.textContent = "▶";
            this.playBtnText.textContent = "RESUME SIMULATION";
        }
    }

    sendReset() {
        this.sendWs({ action: "reset" });
        this.isPlaying = false;
        this.playBtn.classList.remove("paused");
        this.playIcon.textContent = "▶";
        this.playBtnText.textContent = "START SIMULATION";
        this.currentTime = 0.0;
        this.timeScrubber.value = 0;
        this.updateTimeDisplay(0.0);
        this.resetOscilloscope();
        this.resetGauges();
    }

    sendSeek(timeSec) {
        this.sendWs({ action: "seek", time_sec: timeSec });
        this.currentTime = timeSec;
        this.updateTimeDisplay(timeSec);
    }

    sendSetSubject(subjectId) {
        this.sendWs({ action: "set_subject", subject_id: subjectId });
        this.sendReset();
    }

    sendSetTrial(trialId) {
        this.sendWs({ action: "set_trial", trial_id: trialId });
        this.sendReset();
    }

    sendListeningMode(mode) {
        this.sendWs({ action: "set_listening_mode", mode: mode });
    }

    sendSetSource(source) {
        this.sendWs({ action: "set_source", mode: source });
    }

    // =========================================================================
    // Real-Time Frame Ingestion (31.25 ms Tick)
    // =========================================================================
    handleTick(data) {
        this.currentTime = data.time_sec;
        this.timeScrubber.value = this.currentTime;
        this.updateTimeDisplay(this.currentTime);

        // 1. Play Audio Chunk
        if (data.audio_b64) {
            this.enqueueAudioChunk(data.audio_b64);
        }

        // 2. Push EEG Sample to Rolling Buffer
        if (data.eeg_sample && data.eeg_sample.length === 8) {
            for (let ch = 0; ch < 8; ch++) {
                this.eegBuffers[ch].shift();
                this.eegBuffers[ch].push(data.eeg_sample[ch]);
            }
            this.drawOscilloscope();
        }

        // 3. Update Attention Decoder Gauge
        this.updateDecoderGauge(data);

        // 4. Update Gain Steering Meters
        this.updateGainMeters(data);

        // 5. Update Telemetry Numbers
        this.updateTelemetry(data);
    }

    handleEndOfTrial() {
        this.isPlaying = false;
        this.playBtn.classList.remove("paused");
        this.playIcon.textContent = "▶";
        this.playBtnText.textContent = "REPLAY TRIAL";
    }

    updateTimeDisplay(sec) {
        const m = Math.floor(sec / 60);
        const s = Math.floor(sec % 60);
        const ms = Math.floor((sec % 1) * 10);
        this.currentTimeEl.textContent = `${String(m).padStart(2, '0')}:${String(s).padStart(2, '0')}.${ms}`;
    }

    // =========================================================================
    // Web Audio API Headphone Streamer
    // =========================================================================
    initAudioContext() {
        if (!this.audioCtx) {
            const AudioContextClass = window.AudioContext || window.webkitAudioContext;
            this.audioCtx = new AudioContextClass({ sampleRate: this.sampleRate });
            this.nextAudioTime = this.audioCtx.currentTime + 0.05;
        } else if (this.audioCtx.state === "suspended") {
            this.audioCtx.resume();
        }
    }

    enqueueAudioChunk(base64Data) {
        if (!this.audioCtx) return;

        const binary = atob(base64Data);
        const len = binary.length / 2;
        const int16Array = new Int16Array(len);
        for (let i = 0; i < len; i++) {
            int16Array[i] = binary.charCodeAt(i * 2) | (binary.charCodeAt(i * 2 + 1) << 8);
        }

        const floatArray = new Float32Array(len);
        for (let i = 0; i < len; i++) {
            floatArray[i] = int16Array[i] / 32768.0;
        }

        const buffer = this.audioCtx.createBuffer(1, len, this.sampleRate);
        buffer.copyToChannel(floatArray, 0);

        const source = this.audioCtx.createBufferSource();
        source.buffer = buffer;
        source.connect(this.audioCtx.destination);

        const now = this.audioCtx.currentTime;
        if (this.nextAudioTime < now) {
            this.nextAudioTime = now + 0.02;
        }

        source.start(this.nextAudioTime);
        this.nextAudioTime += buffer.duration;
    }

    // =========================================================================
    // 8-Channel EEG Canvas Oscilloscope Renderer
    // =========================================================================
    drawEmptyOscilloscope() {
        const ctx = this.ctx;
        const w = this.canvasWidth;
        const h = this.canvasHeight;

        ctx.clearRect(0, 0, w, h);
        ctx.fillStyle = "#ffffff";
        ctx.fillRect(0, 0, w, h);

        const chHeight = h / 8;
        ctx.lineWidth = 1;

        for (let ch = 0; ch < 8; ch++) {
            const baseY = chHeight * (ch + 0.5);

            // Channel baseline grid
            ctx.strokeStyle = "#f1f5f9";
            ctx.beginPath();
            ctx.moveTo(0, baseY);
            ctx.lineTo(w, baseY);
            ctx.stroke();

            // Channel tag
            ctx.font = "10px monospace";
            ctx.fillStyle = "#94a3b8";
            ctx.fillText(this.channelNames[ch], 8, baseY - 4);
        }
    }

    resetOscilloscope() {
        for (let ch = 0; ch < 8; ch++) {
            this.eegBuffers[ch].fill(0);
        }
        this.drawEmptyOscilloscope();
    }

    drawOscilloscope() {
        const ctx = this.ctx;
        const w = this.canvasWidth;
        const h = this.canvasHeight;

        ctx.clearRect(0, 0, w, h);
        ctx.fillStyle = "#ffffff";
        ctx.fillRect(0, 0, w, h);

        const chHeight = h / 8;
        const nPoints = this.eegHistoryLength;
        const stepX = w / (nPoints - 1);
        const microvoltScale = (chHeight * 0.45) / 50.0; // 50 uV full height scale

        for (let ch = 0; ch < 8; ch++) {
            const baseY = chHeight * (ch + 0.5);

            // Subtle baseline guideline
            ctx.strokeStyle = "#e2e8f0";
            ctx.lineWidth = 1;
            ctx.beginPath();
            ctx.moveTo(0, baseY);
            ctx.lineTo(w, baseY);
            ctx.stroke();

            // Channel Label
            ctx.font = "bold 10px monospace";
            ctx.fillStyle = "#64748b";
            ctx.fillText(this.channelNames[ch], 8, baseY - 4);

            // Physiological EEG voltage trace
            ctx.strokeStyle = "#2563eb"; // Clinical Medical Blue
            ctx.lineWidth = 1.3;
            ctx.beginPath();

            const buf = this.eegBuffers[ch];
            for (let i = 0; i < nPoints; i++) {
                const x = i * stepX;
                const uV = buf[i];
                const y = baseY - (uV * microvoltScale);
                if (i === 0) {
                    ctx.moveTo(x, y);
                } else {
                    ctx.lineTo(x, y);
                }
            }
            ctx.stroke();
        }
    }

    // =========================================================================
    // Decoder Gauge & Confidence
    // =========================================================================
    updateDecoderGauge(data) {
        const margin = data.smoothed_margin || 0.0;
        // Map margin [-1.5, +1.5] to meter percentage [0%, 100%]
        const pct = Math.max(0, Math.min(100, 50 + ((margin / 1.5) * 50)));
        this.meterNeedle.style.left = `${pct}%`;

        const signStr = margin >= 0 ? `+${margin.toFixed(2)}` : margin.toFixed(2);
        this.currentMarginReadout.textContent = `m_t = ${signStr}`;

        // Decision state badge
        const decision = data.decision || "HOLD";
        this.decisionBadge.className = "state-badge";
        if (decision.includes("A")) {
            this.decisionBadge.classList.add("speaker-a");
            this.decisionBadge.textContent = "ATTENDING MARIANNE (SPEAKER A)";
        } else if (decision.includes("B")) {
            this.decisionBadge.classList.add("speaker-b");
            this.decisionBadge.textContent = "ATTENDING ASKE (SPEAKER B)";
        } else {
            this.decisionBadge.classList.add("hold");
            this.decisionBadge.textContent = "HYSTERESIS HOLD (UNCERTAIN)";
        }

        // Ground truth comparison
        const attended = data.attended_speaker === "A" ? "Speaker A (Marianne)" : "Speaker B (Aske)";
        this.groundTruthTarget.textContent = attended;

        // Sigmoid Confidence (50% = neutral, >85% = strong lock)
        const conf = (1.0 / (1.0 + Math.exp(-3.0 * Math.abs(margin)))) * 100.0;
        this.decoderConfidence.textContent = `${conf.toFixed(1)}%`;

        // Tracking accuracy indicator
        if (data.is_correct) {
            this.accuracyIndicator.className = "stat-val correct";
            this.accuracyIndicator.textContent = "✓ LOCKED ON TARGET";
        } else {
            this.accuracyIndicator.className = "stat-val incorrect";
            this.accuracyIndicator.textContent = "⚠ ADAPTING / SEARCHING";
        }
    }

    resetGauges() {
        this.meterNeedle.style.left = "50%";
        this.currentMarginReadout.textContent = "m_t = +0.00";
        this.decisionBadge.className = "state-badge";
        this.decisionBadge.textContent = "STANDBY";
        this.gainReadoutA.textContent = "+0.0 dB";
        this.gainReadoutB.textContent = "-0.0 dB";
        this.gainFillA.style.width = "50%";
        this.gainFillB.style.width = "50%";
    }

    // =========================================================================
    // Gain Meters
    // =========================================================================
    updateGainMeters(data) {
        const gA = data.gain_a_db || 0.0;
        const gB = data.gain_b_db || 0.0;

        this.gainReadoutA.textContent = gA >= 0 ? `+${gA.toFixed(1)} dB` : `${gA.toFixed(1)} dB`;
        this.gainReadoutB.textContent = gB >= 0 ? `+${gB.toFixed(1)} dB` : `${gB.toFixed(1)} dB`;

        // Map range [-18 dB, +9 dB] (total 27 dB span) to 0% - 100%
        const pctA = Math.max(0, Math.min(100, ((gA + 18.0) / 27.0) * 100));
        const pctB = Math.max(0, Math.min(100, ((gB + 18.0) / 27.0) * 100));

        this.gainFillA.style.width = `${pctA}%`;
        this.gainFillB.style.width = `${pctB}%`;
    }

    // =========================================================================
    // Telemetry Numbers
    // =========================================================================
    updateTelemetry(data) {
        if (data.dsp_latency_us !== undefined) {
            this.dspTimeVal.textContent = `${data.dsp_latency_us.toFixed(1)} µs`;
        }
        if (data.gpu_latency_ms !== undefined) {
            this.gpuTimeVal.textContent = `${data.gpu_latency_ms.toFixed(2)} ms`;
        }
        if (data.cpu_load_pct !== undefined) {
            this.cpuLoadVal.textContent = `${data.cpu_load_pct.toFixed(2)}%`;
        }
        if (data.rtf !== undefined) {
            this.rtfVal.textContent = `${data.rtf.toFixed(4)}x`;
        }
    }
}

function jsonParse(str) {
    try {
        return JSON.parse(str);
    } catch {
        return {};
    }
}

// Initialize on DOM Ready
document.addEventListener("DOMContentLoaded", () => {
    window.neuroSteerApp = new NeuroSteerApp();
});
