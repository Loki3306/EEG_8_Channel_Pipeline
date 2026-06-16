# Research Integrity Audit Report

**Status:** 173 violations found.

The following files contain forbidden words that violate `RESEARCH_RULES.md`:

| File | Line | Forbidden Keyword | Context |
|---|---|---|---|
| `DATA_ANALYSIS.md` | 364 | `\brandom\b` | `1. **`random`** (default):` |
| `DATA_ANALYSIS.md` | 369 | `\brandom\b` | `- Sample opposite stream with random temporal shift` |
| `DATA_ANALYSIS.md` | 382 | `\brandom\b` | `--negative-mode {random\|nearby\|same-trial\|mixed}` |
| `DATA_ANALYSIS.md` | 410 | `\brandom\b` | `python training/train_temporal_cnn_loso.py --objective contrastive --negative-mode random` |
| `analysis\audio_normalization_smoke_test.py` | 145 | `\brandom\b` | `y_shuf = np.random.permutation(y)` |
| `analysis\audio_separability.py` | 162 | `\brandom\b` | `y_shuffled = np.random.permutation(y)` |
| `analysis\audit_audio_pipeline.py` | 7 | `\brandom\b` | `import random` |
| `analysis\audit_audio_pipeline.py` | 45 | `\brandom\b` | `# 3. Select 20 random trials` |
| `analysis\audit_audio_pipeline.py` | 46 | `\brandom\b` | `random.seed(42)` |
| `analysis\audit_audio_pipeline.py` | 47 | `\brandom\b` | `sample_indices = random.sample(range(len(trials)), min(20, len(trials)))` |
| `analysis\check_model_collapse.py` | 63 | `\brandom\b` | `print("\n[Diagnostic] Extracting predictions for 10 random test trials...")` |
| `analysis\check_model_collapse.py` | 66 | `\brandom\b` | `np.random.seed(42)` |
| `analysis\check_model_collapse.py` | 67 | `\brandom\b` | `indices = np.random.choice(len(X_te), size=10, replace=False)` |
| `analysis\eegnet_channel_discovery.py` | 159 | `\brandom\b` | `np.random.seed(42)` |
| `analysis\eegnet_channel_discovery.py` | 160 | `\brandom\b` | `np.random.shuffle(train_exs)` |
| `analysis\eegnet_channel_discovery.py` | 298 | `\brandom\b` | `np.random.seed(42)` |
| `analysis\eegnet_channel_discovery.py` | 299 | `\brandom\b` | `np.random.shuffle(train_exs)` |
| `analysis\evaluate_mapping.py` | 204 | `\brandom\b` | `trial_index = int(np.random.default_rng(restart).integers(0, trial_count))` |
| `analysis\gammatone_audit.py` | 7 | `\brandom\b` | `import random` |
| `analysis\gammatone_audit.py` | 88 | `\brandom\b` | `random.seed(42)` |
| `analysis\gammatone_audit.py` | 89 | `\brandom\b` | `sample_indices = random.sample(range(len(trials)), min(20, len(trials)))` |
| `analysis\recover_speaker_identity.py` | 233 | `\brandom\b` | `trial_index = int(np.random.default_rng(restart).integers(0, trial_count))` |
| `analysis\subband_screening.py` | 161 | `\brandom\b` | `np.random.shuffle(x_shuf)` |
| `baselines\ridge_within_screening.py` | 85 | `\brandom\b` | `np.random.seed(42)` |
| `baselines\ridge_within_screening.py` | 86 | `\brandom\b` | `np.random.shuffle(exs)` |
| `evaluation\clean_8ch_ridge_test.py` | 119 | `\brandom\b` | `np.random.seed(42)` |
| `evaluation\clean_8ch_ridge_test.py` | 120 | `\brandom\b` | `np.random.shuffle(all_eegs)` |
| `evaluation\dtu_64ch_sanity.py` | 158 | `\brandom\b` | `np.random.shuffle(x_shuf) # Shuffles across time` |
| `evaluation\dtu_8ch_reproduction.py` | 173 | `\brandom\b` | `np.random.seed(42)` |
| `evaluation\dtu_8ch_reproduction.py` | 174 | `\brandom\b` | `shuffle_indices = np.random.permutation(num_trials)` |
| `evaluation\dtu_8ch_reproduction.py` | 176 | `\brandom\b` | `shuffle_indices = np.random.permutation(num_trials)` |
| `evaluation\dtu_8ch_sanity.py` | 158 | `\brandom\b` | `np.random.shuffle(x_shuf) # Shuffles across time` |
| `evaluation\dtu_paper_reproduction.py` | 174 | `\brandom\b` | `np.random.seed(42)` |
| `evaluation\dtu_paper_reproduction.py` | 175 | `\brandom\b` | `shuffle_indices = np.random.permutation(num_trials)` |
| `evaluation\dtu_paper_reproduction.py` | 177 | `\brandom\b` | `shuffle_indices = np.random.permutation(num_trials)` |
| `evaluation\dtu_validate_channels.py` | 161 | `\brandom\b` | `np.random.shuffle(x_shuf) # Shuffles across time` |
| `evaluation\eeg_sanity_suite.py` | 17 | `\brandom\b` | `Test D - Random Noise EEG (replace EEG with Gaussian noise matching mean/std)` |
| `evaluation\eeg_sanity_suite.py` | 187 | `\brandom\b` | `np.random.shuffle(all_eegs)` |
| `evaluation\eeg_sanity_suite.py` | 206 | `\brandom\b` | `new_exs.append(replace(ex, eeg=np.random.normal(loc=mean, scale=std, size=ex.eeg.shape)))` |
| `evaluation\eeg_sanity_suite.py` | 267 | `\brandom\b` | `results.append(run_sanity_test("Test D: Random Noise EEG", subject_examples, transform_noise))` |
| `evaluation\permutation_test.py` | 121 | `\brandom\b` | `# Pre-generate 1000 random label assignments.` |
| `evaluation\permutation_test.py` | 122 | `\brandom\b` | `# Each subject has ~60 trials. We'll generate a massive block of random labels per subject.` |
| `evaluation\permutation_test.py` | 123 | `\brandom\b` | `# We use random coin flips (0 or 1) where 0=A, 1=B.` |
| `evaluation\permutation_test.py` | 125 | `\brandom\b` | `np.random.seed(42)` |
| `evaluation\permutation_test.py` | 188 | `\brandom\b` | `# Generate random labels for this subject for all permutations` |
| `evaluation\permutation_test.py` | 190 | `\brandom\b` | `perm_labels = np.random.randint(0, 2, size=(NUM_PERMUTATIONS, num_trials))` |
| `evaluation\permutation_test.py` | 266 | `\brandom\b` | `print(" => NOT SIGNIFICANT: The accuracy is indistinguishable from random noise.")` |
| `scratch\check_epsilon_leak.py` | 8 | `\brandom\b` | `np.random.seed(42)` |
| `scratch\check_epsilon_leak.py` | 10 | `\brandom\b` | `a = np.random.randn(100, 28) * 0.1` |
| `scratch\check_epsilon_leak.py` | 12 | `\brandom\b` | `b = np.random.randn(100, 28) * 1.0` |
| `scratch\profile_matchnet.py` | 15 | `\bmock\b` | `# Mock Data parameters` |
| `scratch\profile_matchnet.py` | 24 | `\brandom\b` | `X = np.random.randn(N_SAMPLES, EEG_CHANNELS, SEQ_LEN).astype(np.float32)` |
| `scratch\profile_matchnet.py` | 25 | `\brandom\b` | `YA = np.random.randn(N_SAMPLES, AUDIO_CHANNELS, SEQ_LEN).astype(np.float32)` |
| `scratch\profile_matchnet.py` | 26 | `\brandom\b` | `YB = np.random.randn(N_SAMPLES, AUDIO_CHANNELS, SEQ_LEN).astype(np.float32)` |
| `scratch\test_gammatone_stability.py` | 5 | `\brandom\b` | `audio = np.random.randn(fs * 10) * 10000` |
| `training\loso_ridge_runner.py` | 143 | `\brandom\b` | `rng = np.random.default_rng(seed)` |
| `training\train_atcnet_gammatone.py` | 176 | `\brandom\b` | `np.random.seed(42)` |
| `training\train_atcnet_gammatone.py` | 177 | `\brandom\b` | `shuffle_indices = np.random.permutation(len(X))` |
| `training\train_atcnet_gammatone.py` | 179 | `\brandom\b` | `shuffle_indices = np.random.permutation(len(X))` |
| `training\train_atcnet_gammatone.py` | 241 | `\brandom\b` | `np.random.seed(42)` |
| `training\train_atcnet_gammatone.py` | 242 | `\brandom\b` | `np.random.shuffle(train_exs)` |
| `training\train_atcnet_loso.py` | 126 | `\brandom\b` | `np.random.seed(42)` |
| `training\train_atcnet_loso.py` | 127 | `\brandom\b` | `shuffle_indices = np.random.permutation(len(X))` |
| `training\train_atcnet_loso.py` | 129 | `\brandom\b` | `shuffle_indices = np.random.permutation(len(X))` |
| `training\train_atcnet_loso.py` | 190 | `\brandom\b` | `np.random.seed(42)` |
| `training\train_atcnet_loso.py` | 191 | `\brandom\b` | `np.random.shuffle(train_exs)` |
| `training\train_atcnet_loso.py` | 252 | `\brandom\b` | `print(f"  -> Accuracy Random : {rand_acc*100:.2f}%")` |
| `training\train_atcnet_loso.py` | 267 | `\brandom\b` | `print(f" Random EEG  : {final_random*100:.2f}%")` |
| `training\train_atcnet_screening.py` | 128 | `\brandom\b` | `np.random.seed(42)` |
| `training\train_atcnet_screening.py` | 129 | `\brandom\b` | `shuffle_indices = np.random.permutation(len(X))` |
| `training\train_atcnet_screening.py` | 131 | `\brandom\b` | `shuffle_indices = np.random.permutation(len(X))` |
| `training\train_atcnet_screening.py` | 194 | `\brandom\b` | `np.random.seed(42)` |
| `training\train_atcnet_screening.py` | 195 | `\brandom\b` | `np.random.shuffle(train_exs)` |
| `training\train_atcnet_screening.py` | 256 | `\brandom\b` | `print(f"  -> Accuracy Random : {rand_acc*100:.2f}%")` |
| `training\train_atcnet_screening.py` | 271 | `\brandom\b` | `print(f" Random EEG  : {final_random*100:.2f}%")` |
| `training\train_eegnet_adaptation_screening.py` | 220 | `\brandom\b` | `np.random.seed(42)` |
| `training\train_eegnet_adaptation_screening.py` | 221 | `\brandom\b` | `np.random.shuffle(train_exs)` |
| `training\train_eegnet_adaptation_screening.py` | 280 | `\brandom\b` | `np.random.seed(123)  # Fixed seed for reproducibility` |
| `training\train_eegnet_adaptation_screening.py` | 282 | `\brandom\b` | `perm = np.random.permutation(n_test)` |
| `training\train_eegnet_gammatone.py` | 175 | `\brandom\b` | `np.random.seed(42)` |
| `training\train_eegnet_gammatone.py` | 176 | `\brandom\b` | `shuffle_indices = np.random.permutation(len(X))` |
| `training\train_eegnet_gammatone.py` | 178 | `\brandom\b` | `shuffle_indices = np.random.permutation(len(X))` |
| `training\train_eegnet_gammatone.py` | 240 | `\brandom\b` | `np.random.seed(42)` |
| `training\train_eegnet_gammatone.py` | 241 | `\brandom\b` | `np.random.shuffle(train_exs)` |
| `training\train_eegnet_loso.py` | 126 | `\brandom\b` | `np.random.seed(42)` |
| `training\train_eegnet_loso.py` | 127 | `\brandom\b` | `shuffle_indices = np.random.permutation(len(X))` |
| `training\train_eegnet_loso.py` | 129 | `\brandom\b` | `shuffle_indices = np.random.permutation(len(X))` |
| `training\train_eegnet_loso.py` | 190 | `\brandom\b` | `np.random.seed(42)` |
| `training\train_eegnet_loso.py` | 191 | `\brandom\b` | `np.random.shuffle(train_exs)` |
| `training\train_eegnet_loso.py` | 252 | `\brandom\b` | `print(f"  -> Accuracy Random : {rand_acc*100:.2f}%")` |
| `training\train_eegnet_loso.py` | 267 | `\brandom\b` | `print(f" Random EEG  : {final_random*100:.2f}%")` |
| `training\train_eegnet_screening.py` | 124 | `\brandom\b` | `np.random.seed(42)` |
| `training\train_eegnet_screening.py` | 125 | `\brandom\b` | `shuffle_indices = np.random.permutation(len(X))` |
| `training\train_eegnet_screening.py` | 127 | `\brandom\b` | `shuffle_indices = np.random.permutation(len(X))` |
| `training\train_eegnet_screening.py` | 190 | `\brandom\b` | `np.random.seed(42)` |
| `training\train_eegnet_screening.py` | 191 | `\brandom\b` | `np.random.shuffle(train_exs)` |
| `training\train_eegnet_screening.py` | 252 | `\brandom\b` | `print(f"  -> Accuracy Random : {rand_acc*100:.2f}%")` |
| `training\train_eegnet_screening.py` | 267 | `\brandom\b` | `print(f" Random EEG  : {final_random*100:.2f}%")` |
| `training\train_eegnet_subband.py` | 175 | `\brandom\b` | `np.random.seed(42)` |
| `training\train_eegnet_subband.py` | 176 | `\brandom\b` | `shuffle_indices = np.random.permutation(len(X))` |
| `training\train_eegnet_subband.py` | 178 | `\brandom\b` | `shuffle_indices = np.random.permutation(len(X))` |
| `training\train_eegnet_subband.py` | 240 | `\brandom\b` | `np.random.seed(42)` |
| `training\train_eegnet_subband.py` | 241 | `\brandom\b` | `np.random.shuffle(train_exs)` |
| `training\train_eegnet_tcn_screening.py` | 133 | `\brandom\b` | `np.random.seed(42)` |
| `training\train_eegnet_tcn_screening.py` | 134 | `\brandom\b` | `shuffle_indices = np.random.permutation(len(X))` |
| `training\train_eegnet_tcn_screening.py` | 136 | `\brandom\b` | `shuffle_indices = np.random.permutation(len(X))` |
| `training\train_eegnet_tcn_screening.py` | 193 | `\brandom\b` | `np.random.seed(42)` |
| `training\train_eegnet_tcn_screening.py` | 194 | `\brandom\b` | `np.random.shuffle(train_exs)` |
| `training\train_matchnet_diagnostic.py` | 126 | `\brandom\b` | `np.random.seed(42)` |
| `training\train_matchnet_diagnostic.py` | 127 | `\brandom\b` | `shuffle_indices = np.random.permutation(len(X))` |
| `training\train_matchnet_diagnostic.py` | 129 | `\brandom\b` | `shuffle_indices = np.random.permutation(len(X))` |
| `training\train_matchnet_diagnostic.py` | 206 | `\brandom\b` | `np.random.seed(42)` |
| `training\train_matchnet_diagnostic.py` | 207 | `\brandom\b` | `np.random.shuffle(train_exs)` |
| `training\train_matchnet_diagnostic.py` | 261 | `\brandom\b` | `perm = np.random.permutation(len(X_tr))` |
| `training\train_matchnet_loso.py` | 128 | `\brandom\b` | `np.random.seed(42)` |
| `training\train_matchnet_loso.py` | 129 | `\brandom\b` | `shuffle_indices = np.random.permutation(len(X))` |
| `training\train_matchnet_loso.py` | 131 | `\brandom\b` | `shuffle_indices = np.random.permutation(len(X))` |
| `training\train_matchnet_loso.py` | 206 | `\brandom\b` | `np.random.seed(42)` |
| `training\train_matchnet_loso.py` | 207 | `\brandom\b` | `np.random.shuffle(train_exs)` |
| `training\train_matchnet_screening.py` | 118 | `\brandom\b` | `np.random.seed(42)` |
| `training\train_matchnet_screening.py` | 119 | `\brandom\b` | `shuffle_indices = np.random.permutation(len(X))` |
| `training\train_matchnet_screening.py` | 121 | `\brandom\b` | `shuffle_indices = np.random.permutation(len(X))` |
| `training\train_matchnet_screening.py` | 192 | `\brandom\b` | `np.random.seed(42)` |
| `training\train_matchnet_screening.py` | 193 | `\brandom\b` | `np.random.shuffle(train_exs)` |
| `training\train_matchnet_screening.py` | 246 | `\brandom\b` | `perm = np.random.permutation(len(X_tr))` |
| `training\train_matchnet_within_screening.py` | 118 | `\brandom\b` | `np.random.seed(42)` |
| `training\train_matchnet_within_screening.py` | 119 | `\brandom\b` | `shuffle_indices = np.random.permutation(len(X))` |
| `training\train_matchnet_within_screening.py` | 121 | `\brandom\b` | `shuffle_indices = np.random.permutation(len(X))` |
| `training\train_matchnet_within_screening.py` | 183 | `\brandom\b` | `np.random.seed(42)` |
| `training\train_matchnet_within_screening.py` | 184 | `\brandom\b` | `np.random.shuffle(exs)` |
| `training\train_matchnet_within_screening.py` | 204 | `\bdummy\b` | `# Use dummy prepare_dataset wrapper to reuse code` |
| `training\train_matchnet_within_screening.py` | 226 | `\brandom\b` | `perm = np.random.permutation(len(X_tr))` |
| `training\train_temporal_cnn_loso.py` | 5 | `\brandom\b` | `import random` |
| `training\train_temporal_cnn_loso.py` | 57 | `\brandom\b` | `random.seed(seed)` |
| `training\train_temporal_cnn_loso.py` | 58 | `\brandom\b` | `np.random.seed(seed)` |
| `training\train_temporal_cnn_loso.py` | 148 | `\brandom\b` | `np.random.shuffle(indices)` |
| `training\train_temporal_cnn_loso.py` | 166 | `\brandom\b` | `negative_mode: str = "random",` |
| `training\train_temporal_cnn_loso.py` | 172 | `\brandom\b` | `- random: opposite stream at same time (current default)` |
| `training\train_temporal_cnn_loso.py` | 177 | `\brandom\b` | `if negative_mode == "random":` |
| `training\train_temporal_cnn_loso.py` | 195 | `\brandom\b` | `shift = np.random.randint(min_shift, max_shift + 1)` |
| `training\train_temporal_cnn_loso.py` | 212 | `\brandom\b` | `shift = np.random.randint(min_shift, max_shift + 1)` |
| `training\train_temporal_cnn_loso.py` | 223 | `\brandom\b` | `strategies = ["random", "nearby", "same_trial"]` |
| `training\train_temporal_cnn_loso.py` | 224 | `\brandom\b` | `chosen = np.random.choice(strategies)` |
| `training\train_temporal_cnn_loso.py` | 377 | `\brandom\b` | `negative_mode: str = "random",` |
| `training\train_temporal_cnn_loso.py` | 577 | `\brandom\b` | `eeg = np.random.randn(*eeg.shape).astype(np.float32)` |
| `training\train_temporal_cnn_loso.py` | 671 | `\brandom\b` | `negative_mode: str = "random",` |
| `training\train_temporal_cnn_loso.py` | 816 | `\brandom\b` | `negative_mode: str = "random",` |
| `training\train_temporal_cnn_loso.py` | 1175 | `\brandom\b` | `parser.add_argument("--negative-mode", choices=["random", "nearby", "same-trial", "mixed"], defau...` |
| `training\train_temporal_cnn_loso.py` | 1184 | `\brandom\b` | `parser.add_argument("--random-eeg", action="store_true", help="Replace EEG with random noise dyna...` |
| `training\train_temporal_cnn_loso.py` | 1187 | `\brandom\b` | `parser.add_argument("--random-audio-pairs", action="store_true", help="Break EEG/audio correspond...` |
| `training\train_temporal_cnn_loso.py` | 1210 | `\brandom\b` | `if args.random_eeg: notify("-> Random EEG")` |
| `training\train_temporal_cnn_loso.py` | 1213 | `\brandom\b` | `if args.random_audio_pairs: notify("-> Random Audio Pairing")` |
| `training\train_temporal_cnn_loso.py` | 1218 | `\brandom\b` | `rng = np.random.default_rng(args.seed)` |
| `training\train_vlaai_channel_ablation.py` | 129 | `\brandom\b` | `np.random.seed(42)` |
| `training\train_vlaai_channel_ablation.py` | 130 | `\brandom\b` | `shuffle_indices = np.random.permutation(len(X))` |
| `training\train_vlaai_channel_ablation.py` | 132 | `\brandom\b` | `shuffle_indices = np.random.permutation(len(X))` |
| `training\train_vlaai_channel_ablation.py` | 196 | `\brandom\b` | `np.random.seed(42)` |
| `training\train_vlaai_channel_ablation.py` | 197 | `\brandom\b` | `np.random.shuffle(train_exs)` |
| `training\train_vlaai_channel_ablation.py` | 258 | `\brandom\b` | `print(f"  -> Accuracy Random : {rand_acc*100:.2f}%")` |
| `training\train_vlaai_channel_ablation.py` | 273 | `\brandom\b` | `print(f" Random EEG  : {final_random*100:.2f}%")` |
| `training\train_vlaai_lite_loso.py` | 124 | `\brandom\b` | `np.random.seed(42)` |
| `training\train_vlaai_lite_loso.py` | 125 | `\brandom\b` | `shuffle_indices = np.random.permutation(len(X))` |
| `training\train_vlaai_lite_loso.py` | 127 | `\brandom\b` | `shuffle_indices = np.random.permutation(len(X))` |
| `training\train_vlaai_lite_loso.py` | 188 | `\brandom\b` | `np.random.seed(42)` |
| `training\train_vlaai_lite_loso.py` | 189 | `\brandom\b` | `np.random.shuffle(train_exs)` |
| `training\train_vlaai_lite_loso.py` | 252 | `\brandom\b` | `print(f"  -> Accuracy Random : {rand_acc*100:.2f}%")` |
| `training\train_vlaai_lite_loso.py` | 267 | `\brandom\b` | `print(f" Random EEG  : {final_random*100:.2f}%")` |
| `training\train_vlaai_sweep.py` | 108 | `\brandom\b` | `np.random.seed(42)` |
| `training\train_vlaai_sweep.py` | 109 | `\brandom\b` | `shuffle_indices = np.random.permutation(len(X))` |
| `training\train_vlaai_sweep.py` | 111 | `\brandom\b` | `shuffle_indices = np.random.permutation(len(X))` |
| `training\train_vlaai_sweep.py` | 229 | `\brandom\b` | `np.random.seed(42)` |
| `training\train_vlaai_sweep.py` | 230 | `\brandom\b` | `np.random.shuffle(train_exs)` |
| `training\train_vlaai_sweep.py` | 260 | `\brandom\b` | `print(f" -> Random : {res['random_acc']*100:.2f}%")` |
