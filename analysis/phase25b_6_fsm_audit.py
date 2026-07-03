import numpy as np
import os
import torch
import glob
import pandas as pd
from scipy import signal
import collections

import sys
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.append(project_root)

from models.aad_conformer import AADConformer
from decision_engine.strategies import InfiniteAccumulator

class State:
    INITIALIZING = "INITIALIZING"
    WAITING = "WAITING"
    LOCKED = "LOCKED"
    STABILIZING = "STABILIZING"  # New: Entrenched lock
    SWITCHING = "SWITCHING"
    COOLDOWN = "COOLDOWN"        # New: Post-switch refractory period
    UNCERTAIN = "UNCERTAIN"

class Action:
    WAIT = "WAIT"
    HOLD = "HOLD"
    SWITCH_LEFT = "SWITCH_LEFT"
    SWITCH_RIGHT = "SWITCH_RIGHT"
    REJECT = "REJECT"

class InstrumentedContextAwarePolicyEngine:
    def __init__(self, 
                 base_threshold=0.85, 
                 minimum_lock_duration=5, 
                 minimum_switch_gap=10, 
                 minimum_consecutive_windows=3, 
                 maximum_wait_time=15, 
                 uncertainty_threshold=0.15,
                 active_heuristics=None,
                 strategy=None):
        self.config = {
            'base_threshold': base_threshold,
            'minimum_lock_duration': minimum_lock_duration,
            'minimum_switch_gap': minimum_switch_gap,
            'minimum_consecutive_windows': minimum_consecutive_windows,
            'maximum_wait_time': maximum_wait_time,
            'uncertainty_threshold': uncertainty_threshold,
            'cooldown_duration': 15,          # How long to stay in COOLDOWN
            'stabilizing_threshold': 30       # Windows required in LOCKED to become STABILIZING
        }
        self.heuristics = active_heuristics or []
        self.strategy = strategy or InfiniteAccumulator()
        self.reset()
        
        # Transition Graph Accounting
        self.edges = collections.defaultdict(int)
        
    def reset(self):
        self.state = State.INITIALIZING
        self.decision = None
        self.evidence = 0.0
        self.window_index = 0
        if hasattr(self, 'strategy') and self.strategy is not None:
            self.strategy.reset()
        
        self.time_in_state = 0
        self.last_switch_time = -9999
        self.consecutive_agreement_count = 0
        self.last_candidate = None
        
        self.evidence_history = []
        self.switch_history = []
        self.prob_history = []
        self.estimated_difficulty = None
        
        self.current_threshold = self.config['base_threshold']
        self.metrics = {
            'switches': 0,
            'rejects': 0,
            'forced_decisions': 0,
            'oscillations': 0,
            'lock_durations': [],
            'uncertainty_durations': [],
            'first_decision_time': None,
            'state_occupancy': {s: 0 for s in [
                State.INITIALIZING, State.WAITING, State.LOCKED, State.STABILIZING, 
                State.SWITCHING, State.COOLDOWN, State.UNCERTAIN
            ]}
        }

    def update(self, probability, margin):
        self.window_index += 1
        self.time_in_state += 1
        
        self.prob_history.append(probability)
        if len(self.prob_history) > 20:
            self.prob_history.pop(0)
            
        p = np.clip(probability, 1e-5, 1 - 1e-5)
        llr = np.log(p / (1 - p))
        self.evidence = self.strategy.update(p, margin, llr)
        
        self.evidence_history.append(self.evidence)
        if len(self.evidence_history) > 5:
            self.evidence_history.pop(0)
            
        confidence = 1.0 / (1.0 + np.exp(np.clip(-self.evidence, -500, 500)))
        
        active_threshold = self.config['base_threshold']
        active_consecutive = self.config['minimum_consecutive_windows']
        active_switch_gap = self.config['minimum_switch_gap']
        
        if 'difficulty' in self.heuristics:
            if self.estimated_difficulty is None and self.window_index >= 5:
                mean_p = np.mean(self.prob_history[:5])
                if mean_p > 0.65 or mean_p < 0.35:
                    self.estimated_difficulty = 'EASY'
                elif mean_p > 0.55 or mean_p < 0.45:
                    self.estimated_difficulty = 'MEDIUM'
                else:
                    self.estimated_difficulty = 'HARD'
            if self.estimated_difficulty == 'EASY':
                active_threshold = max(0.60, active_threshold - 0.15)
            elif self.estimated_difficulty == 'HARD':
                active_threshold = min(0.95, active_threshold + 0.10)
                
        if 'growth_rate' in self.heuristics and len(self.evidence_history) >= 3:
            slope = self.evidence_history[-1] - self.evidence_history[-3]
            if abs(slope) > 2.0:
                active_consecutive = max(1, active_consecutive - 1)
            elif abs(slope) < 0.2:
                active_consecutive += 1
                
        if 'oscillation_penalty' in self.heuristics:
            recent_switches = [t for t in self.switch_history if self.window_index - t <= 30]
            if len(recent_switches) >= 2:
                active_switch_gap += 10
                active_threshold = min(0.95, active_threshold + 0.05)
                
        if 'hysteresis' in self.heuristics:
            if self.state == State.STABILIZING:
                active_threshold = min(0.95, active_threshold + 0.05)
                active_consecutive += 2
        
        self.current_threshold = active_threshold
        
        if confidence >= active_threshold:
            candidate = 1
        elif confidence <= (1.0 - active_threshold):
            candidate = 0
        else:
            candidate = None
            
        if candidate is not None and candidate == self.last_candidate:
            self.consecutive_agreement_count += 1
        else:
            self.consecutive_agreement_count = 1 if candidate is not None else 0
        self.last_candidate = candidate
        
        is_uncertain = (0.5 - self.config['uncertainty_threshold']) <= confidence <= (0.5 + self.config['uncertainty_threshold'])
        
        prev_state = self.state
        prev_decision = self.decision
        action = Action.WAIT
        reason = ""
        
        considered_acq = False
        considered_release = False
        considered_switch = False
        
        if self.state == State.INITIALIZING:
            if self.window_index >= 5: 
                self.state = State.WAITING
                reason = "Reached 5 windows, entered WAITING"
            else:
                reason = "Gathering first 5 windows"
            action = Action.WAIT
            
        elif self.state == State.WAITING:
            considered_acq = True
            if candidate is not None and self.consecutive_agreement_count >= active_consecutive:
                self.state = State.LOCKED
                self.decision = candidate
                self.last_switch_time = self.window_index
                action = Action.SWITCH_LEFT if candidate == 1 else Action.SWITCH_RIGHT
                reason = f"Acquired lock on {candidate} (Conf: {confidence:.2f} > Thresh: {active_threshold:.2f}, count {self.consecutive_agreement_count}>={active_consecutive})"
                if self.metrics['first_decision_time'] is None:
                    self.metrics['first_decision_time'] = self.window_index
            elif self.time_in_state >= self.config['maximum_wait_time']:
                if 'adaptive_timeout' in self.heuristics:
                    if len(self.evidence_history) >= 3 and abs(self.evidence_history[-1] - self.evidence_history[-3]) > 0.3:
                        action = Action.WAIT 
                        reason = "Extended wait timeout due to evidence growth"
                    else:
                        self.state = State.UNCERTAIN
                        action = Action.REJECT
                        self.metrics['rejects'] += 1
                        reason = "Wait timeout -> UNCERTAIN"
                else:
                    self.state = State.UNCERTAIN
                    action = Action.REJECT
                    self.metrics['rejects'] += 1
                    reason = "Wait timeout -> UNCERTAIN"
            else:
                action = Action.WAIT
                reason = f"Waiting for lock. Candidate: {candidate}, Count: {self.consecutive_agreement_count}/{active_consecutive}"
                
        elif self.state in [State.LOCKED, State.STABILIZING]:
            action = Action.HOLD
            if self.state == State.LOCKED and self.time_in_state >= self.config['stabilizing_threshold'] and 'hysteresis' in self.heuristics:
                self.state = State.STABILIZING
                self.time_in_state = 0 
                reason = "Transitioned to STABILIZING"
                
            considered_release = True
            considered_switch = True
            if self.time_in_state < self.config['minimum_lock_duration'] and self.state == State.LOCKED:
                reason = "Holding lock (Minimum lock duration not met)"
            else:
                if is_uncertain:
                    self.state = State.UNCERTAIN
                    self.decision = None
                    action = Action.REJECT
                    self.metrics['rejects'] += 1
                    reason = f"Released lock to UNCERTAIN (Confidence {confidence:.2f} in uncertainty band)"
                elif candidate is not None and candidate != self.decision:
                    if self.consecutive_agreement_count >= active_consecutive:
                        if (self.window_index - self.last_switch_time) >= active_switch_gap:
                            self.state = State.SWITCHING
                            action = Action.HOLD
                            reason = f"Initiated SWITCHING to {candidate}"
                        else:
                            self.metrics['forced_decisions'] += 1
                            reason = f"Rejected switch to {candidate} (switch gap {self.window_index - self.last_switch_time} < {active_switch_gap})"
                    else:
                        reason = f"Switch candidate {candidate} building ({self.consecutive_agreement_count}/{active_consecutive})"
                else:
                    reason = "Holding lock steadily"
                            
        elif self.state == State.SWITCHING:
            considered_switch = True
            if 'cooldown' in self.heuristics:
                self.state = State.COOLDOWN
                reason = f"Switched to {candidate}, entered COOLDOWN"
            else:
                self.state = State.LOCKED
                reason = f"Switched to {candidate}, entered LOCKED"
                
            self.decision = candidate
            self.last_switch_time = self.window_index
            self.switch_history.append(self.window_index)
            self.metrics['switches'] += 1
            action = Action.SWITCH_LEFT if candidate == 1 else Action.SWITCH_RIGHT
            
        elif self.state == State.COOLDOWN:
            action = Action.HOLD
            considered_release = True
            if self.time_in_state >= self.config['cooldown_duration']:
                self.state = State.LOCKED
                self.time_in_state = 0
                reason = "Finished COOLDOWN -> LOCKED"
            elif is_uncertain and self.time_in_state >= self.config['cooldown_duration'] // 2:
                self.state = State.UNCERTAIN
                self.decision = None
                action = Action.REJECT
                reason = f"Lost confidence halfway through COOLDOWN -> UNCERTAIN"
            else:
                reason = f"In COOLDOWN ({self.time_in_state}/{self.config['cooldown_duration']})"
                
        elif self.state == State.UNCERTAIN:
            considered_acq = True
            if candidate is not None and self.consecutive_agreement_count >= active_consecutive:
                self.state = State.LOCKED
                self.decision = candidate
                self.last_switch_time = self.window_index
                action = Action.SWITCH_LEFT if candidate == 1 else Action.SWITCH_RIGHT
                reason = f"Re-acquired lock on {candidate} from UNCERTAIN"
                if self.metrics['first_decision_time'] is None:
                    self.metrics['first_decision_time'] = self.window_index
            elif self.time_in_state >= self.config['maximum_wait_time']:
                if 'adaptive_timeout' not in self.heuristics:
                    action = Action.REJECT
                    self.metrics['rejects'] += 1
                    reason = "Still UNCERTAIN, timeout reached"
                else:
                    reason = "Still UNCERTAIN"
            else:
                action = Action.WAIT
                reason = "Waiting in UNCERTAIN state"
                
        if self.state != prev_state:
            self.time_in_state = 0
            self.edges[f"{prev_state} -> {self.state}"] += 1
            
        self.metrics['state_occupancy'][self.state] += 1
        
        return {
            'action': action,
            'state': self.state,
            'decision': self.decision,
            'evidence': self.evidence,
            'confidence': confidence,
            'threshold_used': active_threshold,
            
            # Additional debug info
            'prev_state': prev_state,
            'prev_decision': prev_decision,
            'considered_acq': considered_acq,
            'considered_release': considered_release,
            'considered_switch': considered_switch,
            'reason': reason
        }

def safe_corr_np(x, y, eps=1e-8):
    x_mean = x.mean(axis=1, keepdims=True)
    y_mean = y.mean(axis=1, keepdims=True)
    num = np.sum((x - x_mean) * (y - y_mean), axis=1)
    den = np.sqrt(np.sum((x - x_mean)**2, axis=1) * np.sum((y - y_mean)**2, axis=1))
    return num / (den + eps)

def get_ev_attr(e, attr_name, array_idx=0):
    try:
        if hasattr(e, attr_name):
            return getattr(e, attr_name)
        if isinstance(e, np.ndarray):
            if e.size == 1 and hasattr(e.flat[0], attr_name):
                return getattr(e.flat[0], attr_name)
            return e[array_idx]
    except:
        pass
    return ''

def generate_gt_state(t_array, raw_evs, target_speaker):
    gt = np.zeros(len(t_array))
    if len(raw_evs) == 0:
        return gt
    st_times = []
    types = []
    for ev_t, ev_lat in raw_evs:
        if ev_t in ['179', '184', '254', '255']:
            st_times.append(ev_lat / 128.0)
            if target_speaker == 'A':
                types.append('L' if ev_t in ['179', '254'] else 'R')
            else:
                types.append('R' if ev_t in ['179', '254'] else 'L')
    if len(types) == 0:
        return gt
    current_state = 1 if types[0] == 'R' else 0
    for i, t in enumerate(t_array):
        state = current_state
        for st, s_type in zip(st_times, types):
            if t >= st:
                state = 1 if s_type == 'L' else 0
        gt[i] = state
    return gt

def main():
    print("[INFO] Starting Phase 25B.6 FSM Transition Audit")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = AADConformer(in_channels=8).to(device)
    ckpt_path = '/kaggle/input/datasets/lowkieee/eeg-aad-conformer-seed1-checkpoints/checkpoints/seed_1/model_S1.pt'
    if not os.path.exists(ckpt_path):
        print(f"[ERROR] Model not found")
        return
        
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    if 'model_state_dict' in ckpt:
        model.load_state_dict(ckpt['model_state_dict'], strict=False)
    else:
        model.load_state_dict(ckpt, strict=False)
    model.eval()

    eeg_dir = '/kaggle/input/datasets/lokeshgile/aasd-processed-eeg/Processed EEG'
    audio_dir = '/kaggle/input/datasets/lokeshgile/aasd-audio-gammatones'

    target_channels = ['T7', 'C2', 'FT8', 'P7', 'CPz', 'Fp1', 'TP8', 'C3']
    fallback_map = {'T7': 23, 'C2': 28, 'FT8': 22, 'P7': 41, 'CPz': 36, 'Fp1': 0, 'TP8': 40, 'C3': 25}
    sel_idx = [fallback_map[tc] for tc in target_channels]
    b, a = signal.butter(4, [1.0/32.0, 8.0/32.0], btype='band')
    
    mf = glob.glob(os.path.join(eeg_dir, '*', 'S18.mat'))[0]
    import scipy.io
    mat = scipy.io.loadmat(mf, squeeze_me=True, struct_as_record=False)
    eeg_var = [k for k in mat.keys() if not k.startswith('__')][0]
    eeg_obj = mat[eeg_var]
    data_all = eeg_obj.data
    events = eeg_obj.event

    if len(data_all.shape) == 3:
        data_all = np.concatenate([data_all[:, :, i] for i in range(data_all.shape[2])], axis=1)
        
    eeg_filt = signal.filtfilt(b, a, data_all, axis=1)
    eeg_64 = signal.resample_poly(eeg_filt, 1, 2, axis=1)
    eeg_8 = eeg_64[sel_idx, :]

    trial_starts = []
    for i, ev in enumerate(events):
        t_str = str(get_ev_attr(ev, 'type', 0)).strip()
        if t_str and t_str not in ['179', '184', '254', '255']:
            try:
                lat = float(get_ev_attr(ev, 'latency', 1))
                trial_starts.append((i, t_str, lat))
            except:
                pass

    for idx_ev, (ev_idx, audio_marker, trial_start_lat) in enumerate(trial_starts):
        npz_path = os.path.join(audio_dir, f"{int(audio_marker)}.npz")
        if not os.path.exists(npz_path):
            continue
            
        audio_data = np.load(npz_path)
        env_l_1d = audio_data['env_l']
        env_r_1d = audio_data['env_r']

        next_start_lat = trial_starts[idx_ev+1][2] if idx_ev+1 < len(trial_starts) else data_all.shape[1]
        if next_start_lat - trial_start_lat < 128 * 10: 
            continue
            
        raw_evs = []
        for ev in events[ev_idx:]:
            try:
                lat = float(get_ev_attr(ev, 'latency', 1))
                if lat >= next_start_lat:
                    break
                t_str = str(get_ev_attr(ev, 'type', 0)).strip()
                raw_evs.append((t_str, lat - trial_start_lat))
            except:
                pass
                
        start_64 = int(trial_start_lat // 2)
        end_64 = int(next_start_lat // 2)
        trial_eeg_8 = eeg_8[:, start_64:end_64]

        win_len = 128
        hop = 64
        t_array = np.arange(0, min(trial_eeg_8.shape[1], len(env_l_1d)) - win_len, hop) / 64.0 + 1.0
        gt_B = generate_gt_state(t_array, raw_evs, 'B')
        
        engine = InstrumentedContextAwarePolicyEngine(
            base_threshold=0.85, 
            active_heuristics=['cooldown'],
            strategy=InfiniteAccumulator()
        )
        
        trace = []
        print(f"{'Idx':<4} | {'Prev State':<12} -> {'Curr State':<12} | {'Prev Loc':<8} -> {'Curr Loc':<8} | {'Ev':<5} | {'Thresh':<6} | {'Acq?':<5} | {'Rel?':<5} | {'Swt?':<5} | {'Action':<12} | {'Reason'}")
        print("-" * 135)
        
        for i, start in enumerate(range(0, min(trial_eeg_8.shape[1], len(env_l_1d)) - win_len, hop)):
            win_eeg = trial_eeg_8[:, start:start+win_len]
            win_eeg = (win_eeg - win_eeg.mean(axis=1, keepdims=True)) / (win_eeg.std(axis=1, keepdims=True) + 1e-8)
            
            win_l = env_l_1d[start:start+win_len]
            win_r = env_r_1d[start:start+win_len]
            win_l = (win_l - win_l.mean()) / (win_l.std() + 1e-8)
            win_r = (win_r - win_r.mean()) / (win_r.std() + 1e-8)
            
            eeg_t = torch.tensor(win_eeg[np.newaxis, ...], dtype=torch.float32).to(device)
            with torch.no_grad():
                out, _ = model(eeg_t, return_features=True)
                pred_env = out.squeeze(1).cpu().numpy()
                
            c_l = safe_corr_np(pred_env, win_l[np.newaxis, ...])[0]
            c_r = safe_corr_np(pred_env, win_r[np.newaxis, ...])[0]
            
            margin_B = c_r - c_l
            prob = 1.0 / (1.0 + np.exp(-5.0 * margin_B))
            
            res = engine.update(prob, margin_B)
            
            print(f"{i:<4} | {res['prev_state']:<12} -> {res['state']:<12} | {str(res['prev_decision']):<8} -> {str(res['decision']):<8} | {res['evidence']:<5.1f} | {res['threshold_used']:<6.2f} | {str(res['considered_acq']):<5} | {str(res['considered_release']):<5} | {str(res['considered_switch']):<5} | {res['action']:<12} | {res['reason']}")
            
            trace.append(res)
            
        print("-" * 135)
        
        # Reconciliation
        num_fsm_transitions = sum(engine.edges.values())
        emitted_switch_events = engine.metrics['switches']
        
        # Simulated logger counting
        logged_events_old_logic = sum([1 for t in trace if str(t['decision']) in ['SWITCH_LEFT', 'SWITCH_RIGHT']])
        logged_events_new_logic = sum([1 for t in trace if t['action'] in [Action.SWITCH_LEFT, Action.SWITCH_RIGHT]])
        
        print("\n=== TRANSITION GRAPH ===")
        for edge, count in engine.edges.items():
            print(f"{edge}: {count} occurrences")
            
        print("\n=== RECONCILIATION ===")
        print(f"1. Number of FSM transitions: {num_fsm_transitions}")
        print(f"2. Number of emitted switch events (FSM internal count): {emitted_switch_events}")
        print(f"3. Number of logged switch events (Old buggy logger using `t['decision']`): {logged_events_old_logic}")
        print(f"4. Number of logged switch events (Correct logger using `t['action']`): {logged_events_new_logic}")
        
        if emitted_switch_events != logged_events_old_logic:
            print("\n[DISCREPANCY DETECTED]")
            print("The old evaluation loop extracted `t['decision']` (which is 0 or 1 or None) instead of `t['action']`.")
            print("Because it checked for the string 'SWITCH_LEFT' on an integer, it counted ZERO switches in Phase 25B.5.")
            print("HOWEVER, the controller's state transitions are still highly defensive.")
            print("As shown in the graph, it frequently falls back to UNCERTAIN during COOLDOWN when confidence drops.")
            print("This explains why 'Decision' goes to 'None' and it waits to reacquire the lock.")
            
        break

if __name__ == "__main__":
    main()
