import numpy as np

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

class ContextAwarePolicyEngine:
    """
    Context-aware decision policy engine that dynamically modulates FSM constraints
    using live signal context (evidence growth, oscillation history, lock duration).
    """
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
        
        if strategy is None:
            from .strategies import InfiniteAccumulator
            self.strategy = InfiniteAccumulator()
        else:
            self.strategy = strategy
            
        self.reset()
        
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
        
        # Context tracking
        self.evidence_history = []
        self.switch_history = []
        self.prob_history = []
        self.estimated_difficulty = None
        
        # Dynamic State Variables
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
        self.prob_history.append(probability)
        if len(self.prob_history) > 20:
            self.prob_history.pop(0)
            
        # 1. Update Evidence
        p = np.clip(probability, 1e-5, 1 - 1e-5)
        llr = np.log(p / (1 - p))
        self.evidence = self.strategy.update(p, margin, llr)
        self.evidence_history.append(self.evidence)
        if len(self.evidence_history) > 5:
            self.evidence_history.pop(0)
            
        # 2. Bounded Confidence
        confidence = 1.0 / (1.0 + np.exp(np.clip(-self.evidence, -500, 500)))
        
        # 3. Process Heuristics (Dynamic Constraints)
        active_threshold = self.config['base_threshold']
        active_consecutive = self.config['minimum_consecutive_windows']
        active_switch_gap = self.config['minimum_switch_gap']
        
        # --- Heuristic: Difficulty Scaling ---
        if 'difficulty' in self.heuristics:
            if self.estimated_difficulty is None and self.window_index >= 5:
                # Estimate difficulty from first 5 windows (Phase 16B logic)
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
                
        # --- Heuristic: Evidence Growth Rate ---
        if 'growth_rate' in self.heuristics and len(self.evidence_history) >= 3:
            slope = self.evidence_history[-1] - self.evidence_history[-3]
            # High slope means rapid confirmation
            if abs(slope) > 2.0:
                active_consecutive = max(1, active_consecutive - 1)
            elif abs(slope) < 0.2:
                # Floundering, be careful
                active_consecutive += 1
                
        # --- Heuristic: Oscillation Penalty ---
        if 'oscillation_penalty' in self.heuristics:
            # Count switches in last 30 windows
            recent_switches = [t for t in self.switch_history if self.window_index - t <= 30]
            if len(recent_switches) >= 2:
                active_switch_gap += 10
                active_threshold = min(0.95, active_threshold + 0.05)
                
        # --- Heuristic: Hysteresis (Lock Entrenchment) ---
        if 'hysteresis' in self.heuristics:
            if self.state == State.STABILIZING:
                active_threshold = min(0.95, active_threshold + 0.05)
                active_consecutive += 2
        
        self.current_threshold = active_threshold
        
        # 4. Identify Candidate
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
        
        # 5. State Machine
        prev_state = self.state
        action = Action.WAIT
        reason = ""
        
        if self.state == State.INITIALIZING:
            if self.window_index >= 5: # Always wait 5 for difficulty check parity
                self.state = State.WAITING
            action = Action.WAIT
            
        elif self.state == State.WAITING:
            if candidate is not None and self.consecutive_agreement_count >= active_consecutive:
                self.state = State.LOCKED
                self.decision = candidate
                self.last_switch_time = self.window_index
                action = Action.SWITCH_LEFT if candidate == 1 else Action.SWITCH_RIGHT
                if self.metrics['first_decision_time'] is None:
                    self.metrics['first_decision_time'] = self.window_index
            elif self.time_in_state >= self.config['maximum_wait_time']:
                if 'adaptive_timeout' in self.heuristics:
                    # In adaptive timeout, we don't automatically reject if evidence is growing slowly but consistently
                    if len(self.evidence_history) >= 3 and abs(self.evidence_history[-1] - self.evidence_history[-3]) > 0.3:
                        action = Action.WAIT # extend timeout
                    else:
                        self.state = State.UNCERTAIN
                        action = Action.REJECT
                        self.metrics['rejects'] += 1
                else:
                    self.state = State.UNCERTAIN
                    action = Action.REJECT
                    self.metrics['rejects'] += 1
            else:
                action = Action.WAIT
                
        elif self.state in [State.LOCKED, State.STABILIZING]:
            action = Action.HOLD
            
            if self.state == State.LOCKED and self.time_in_state >= self.config['stabilizing_threshold'] and 'hysteresis' in self.heuristics:
                self.state = State.STABILIZING
                self.time_in_state = 0 # reset time to track stabilizing duration
                
            if self.time_in_state < self.config['minimum_lock_duration'] and self.state == State.LOCKED:
                pass # Ignore fluctuations
            else:
                if is_uncertain:
                    self.state = State.UNCERTAIN
                    self.decision = None
                    action = Action.REJECT
                    self.metrics['rejects'] += 1
                elif candidate is not None and candidate != self.decision:
                    if self.consecutive_agreement_count >= active_consecutive:
                        if (self.window_index - self.last_switch_time) >= active_switch_gap:
                            self.state = State.SWITCHING
                            action = Action.HOLD
                        else:
                            self.metrics['forced_decisions'] += 1
                            
        elif self.state == State.SWITCHING:
            if 'cooldown' in self.heuristics:
                self.state = State.COOLDOWN
            else:
                self.state = State.LOCKED
                
            self.decision = candidate
            self.last_switch_time = self.window_index
            self.switch_history.append(self.window_index)
            self.metrics['switches'] += 1
            action = Action.SWITCH_LEFT if candidate == 1 else Action.SWITCH_RIGHT
            
        elif self.state == State.COOLDOWN:
            action = Action.HOLD
            # In cooldown, we ignore all counter-evidence to prevent ping-pong
            if self.time_in_state >= self.config['cooldown_duration']:
                self.state = State.LOCKED
                self.time_in_state = 0
            elif is_uncertain and self.time_in_state >= self.config['cooldown_duration'] // 2:
                # If we lose confidence halfway through cooldown, bail to uncertain
                self.state = State.UNCERTAIN
                self.decision = None
                action = Action.REJECT
                self.metrics['rejects'] += 1
                
        elif self.state == State.UNCERTAIN:
            action = Action.REJECT
            if candidate is not None and self.consecutive_agreement_count >= active_consecutive:
                self.state = State.LOCKED
                self.decision = candidate
                self.last_switch_time = self.window_index
                self.switch_history.append(self.window_index) # Treat recovery as a switch
                action = Action.SWITCH_LEFT if candidate == 1 else Action.SWITCH_RIGHT
                
        # Metrics Updates
        if self.state == prev_state:
            self.time_in_state += 1
        else:
            if prev_state in [State.LOCKED, State.STABILIZING]:
                self.metrics['lock_durations'].append(self.time_in_state)
            elif prev_state == State.UNCERTAIN:
                self.metrics['uncertainty_durations'].append(self.time_in_state)
            
            if prev_state in [State.LOCKED, State.STABILIZING] and self.state in [State.LOCKED, State.SWITCHING]:
                if self.time_in_state < self.config['minimum_switch_gap'] * 2:
                    self.metrics['oscillations'] += 1
                    
            self.time_in_state = 1
            
        self.metrics['state_occupancy'][self.state] += 1
        
        return {
            'state': self.state,
            'decision': self.decision,
            'action': action,
            'confidence': confidence,
            'evidence': self.evidence,
            'threshold_used': active_threshold,
            'consecutive_used': active_consecutive
        }

    def statistics(self):
        if self.state in [State.LOCKED, State.STABILIZING]:
            self.metrics['lock_durations'].append(self.time_in_state)
        elif self.state == State.UNCERTAIN:
            self.metrics['uncertainty_durations'].append(self.time_in_state)
            
        avg_lock = np.mean(self.metrics['lock_durations']) if self.metrics['lock_durations'] else 0
        avg_uncertain = np.mean(self.metrics['uncertainty_durations']) if self.metrics['uncertainty_durations'] else 0
        
        total_steps = sum(self.metrics['state_occupancy'].values())
        occupancy_pct = {k: v / total_steps for k, v in self.metrics['state_occupancy'].items()} if total_steps > 0 else {}
        
        return {
            'latency': self.metrics['first_decision_time'],
            'switches': self.metrics['switches'],
            'rejects': self.metrics['rejects'],
            'forced_decisions': self.metrics['forced_decisions'],
            'oscillations': self.metrics['oscillations'],
            'avg_lock_duration': avg_lock,
            'avg_uncertain_duration': avg_uncertain,
            'state_occupancy': occupancy_pct
        }
