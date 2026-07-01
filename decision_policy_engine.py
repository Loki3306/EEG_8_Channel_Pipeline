import numpy as np

class State:
    INITIALIZING = "INITIALIZING"
    WAITING = "WAITING"
    LOCKED = "LOCKED"
    SWITCHING = "SWITCHING"
    UNCERTAIN = "UNCERTAIN"

class Action:
    WAIT = "WAIT"
    HOLD = "HOLD"
    SWITCH_LEFT = "SWITCH_LEFT"    # Decision 1
    SWITCH_RIGHT = "SWITCH_RIGHT"  # Decision 0
    REJECT = "REJECT"

class DecisionPolicyEngine:
    def __init__(self, 
                 confidence_threshold=0.85, 
                 minimum_lock_duration=5, 
                 minimum_switch_gap=10, 
                 minimum_consecutive_windows=3, 
                 maximum_wait_time=15, 
                 uncertainty_threshold=0.15):
        self.config = {
            'confidence_threshold': confidence_threshold,
            'minimum_lock_duration': minimum_lock_duration,
            'minimum_switch_gap': minimum_switch_gap,
            'minimum_consecutive_windows': minimum_consecutive_windows,
            'maximum_wait_time': maximum_wait_time,
            'uncertainty_threshold': uncertainty_threshold
        }
        self.reset()
        
    def reset(self):
        self.state = State.INITIALIZING
        self.decision = None
        self.evidence = 0.0
        self.window_index = 0
        
        self.time_in_state = 0
        self.last_switch_time = -9999
        self.consecutive_agreement_count = 0
        self.last_candidate = None
        
        self.metrics = {
            'switches': 0,
            'rejects': 0,
            'forced_decisions': 0,
            'oscillations': 0,
            'lock_durations': [],
            'uncertainty_durations': [],
            'first_decision_time': None,
            'state_occupancy': {s: 0 for s in [State.INITIALIZING, State.WAITING, State.LOCKED, State.SWITCHING, State.UNCERTAIN]}
        }
        
    def update(self, probability, margin):
        self.window_index += 1
        
        # 1. Update Evidence (Causal LLR formulation)
        p = np.clip(probability, 1e-5, 1 - 1e-5)
        llr = np.log(p / (1 - p))
        self.evidence += llr
        
        # 2. Convert evidence to bounded confidence [0, 1]
        confidence = 1.0 / (1.0 + np.exp(-self.evidence))
        
        # 3. Identify Candidate Decision
        if confidence >= self.config['confidence_threshold']:
            candidate = 1
        elif confidence <= (1.0 - self.config['confidence_threshold']):
            candidate = 0
        else:
            candidate = None
            
        # 4. Update Consecutive Agreement Tracker
        if candidate is not None and candidate == self.last_candidate:
            self.consecutive_agreement_count += 1
        else:
            self.consecutive_agreement_count = 1 if candidate is not None else 0
        self.last_candidate = candidate
        
        # 5. Determine Uncertainty
        is_uncertain = (0.5 - self.config['uncertainty_threshold']) <= confidence <= (0.5 + self.config['uncertainty_threshold'])
        
        # 6. State Machine Transitions
        prev_state = self.state
        action = Action.WAIT
        reason = ""
        
        if self.state == State.INITIALIZING:
            if self.window_index >= self.config['minimum_consecutive_windows']:
                self.state = State.WAITING
                action = Action.WAIT
                reason = "Initialization period complete."
            else:
                action = Action.WAIT
                reason = "Gathering initial evidence."
                
        elif self.state == State.WAITING:
            if candidate is not None and self.consecutive_agreement_count >= self.config['minimum_consecutive_windows']:
                self.state = State.LOCKED
                self.decision = candidate
                self.last_switch_time = self.window_index
                action = Action.SWITCH_LEFT if candidate == 1 else Action.SWITCH_RIGHT
                reason = f"Confidence exceeded threshold for {self.consecutive_agreement_count} consecutive windows."
                if self.metrics['first_decision_time'] is None:
                    self.metrics['first_decision_time'] = self.window_index
            elif self.time_in_state >= self.config['maximum_wait_time']:
                self.state = State.UNCERTAIN
                action = Action.REJECT
                self.metrics['rejects'] += 1
                reason = "Exceeded maximum wait time without reaching confidence."
            else:
                action = Action.WAIT
                reason = "Accumulating evidence."
                
        elif self.state == State.LOCKED:
            action = Action.HOLD
            
            if self.time_in_state < self.config['minimum_lock_duration']:
                reason = f"Within minimum lock duration ({self.time_in_state}/{self.config['minimum_lock_duration']}). Ignoring fluctuations."
            else:
                if is_uncertain:
                    self.state = State.UNCERTAIN
                    self.decision = None
                    action = Action.REJECT
                    self.metrics['rejects'] += 1
                    reason = "Confidence collapsed into uncertainty zone."
                elif candidate is not None and candidate != self.decision:
                    if self.consecutive_agreement_count >= self.config['minimum_consecutive_windows']:
                        if (self.window_index - self.last_switch_time) >= self.config['minimum_switch_gap']:
                            self.state = State.SWITCHING
                            action = Action.HOLD
                            reason = "Opposing confidence built up and switch gap satisfied. Entering switching."
                        else:
                            reason = "Opposing confidence built up, but minimum switch gap not met."
                            self.metrics['forced_decisions'] += 1
                    else:
                        reason = "Opposing confidence forming, awaiting consecutive confirmations."
                else:
                    reason = "Confidence remains stable."
                    
        elif self.state == State.SWITCHING:
            # Automatically transition to LOCKED on the new candidate
            self.state = State.LOCKED
            self.decision = candidate
            self.last_switch_time = self.window_index
            self.metrics['switches'] += 1
            action = Action.SWITCH_LEFT if candidate == 1 else Action.SWITCH_RIGHT
            reason = "Completed switch to new target."
            
        elif self.state == State.UNCERTAIN:
            action = Action.REJECT
            if candidate is not None and self.consecutive_agreement_count >= self.config['minimum_consecutive_windows']:
                self.state = State.LOCKED
                self.decision = candidate
                self.last_switch_time = self.window_index
                action = Action.SWITCH_LEFT if candidate == 1 else Action.SWITCH_RIGHT
                reason = "Recovered confidence from uncertain state."
            else:
                reason = "Remaining uncertain."
                
        # 7. Metrics Updates
        if self.state == prev_state:
            self.time_in_state += 1
        else:
            if prev_state == State.LOCKED:
                self.metrics['lock_durations'].append(self.time_in_state)
            elif prev_state == State.UNCERTAIN:
                self.metrics['uncertainty_durations'].append(self.time_in_state)
            
            # Detect fast oscillations
            if prev_state == State.LOCKED and self.state == State.LOCKED and self.time_in_state < self.config['minimum_switch_gap'] * 2:
                self.metrics['oscillations'] += 1
                
            self.time_in_state = 1
            
        self.metrics['state_occupancy'][self.state] += 1
        
        return {
            'state': self.state,
            'decision': self.decision,
            'action': action,
            'confidence': confidence,
            'evidence': self.evidence,
            'reason': reason
        }

    def statistics(self):
        # Finalize metrics
        if self.state == State.LOCKED:
            self.metrics['lock_durations'].append(self.time_in_state)
        elif self.state == State.UNCERTAIN:
            self.metrics['uncertainty_durations'].append(self.time_in_state)
            
        avg_lock = np.mean(self.metrics['lock_durations']) if self.metrics['lock_durations'] else 0
        avg_uncertain = np.mean(self.metrics['uncertainty_durations']) if self.metrics['uncertainty_durations'] else 0
        
        # Normalize state occupancy
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
