Your previous implementation failed review.

Task: Validate EEG script

EEG Validator Issues:
- scratch/flawed_eeg.py: Potential LOSO leakage - using random train_test_split instead of Leave-One-Subject-Out.

ChatGPT Blocking Issues:
- Methodological violation: prepare_eeg_data() claims to prepare a LOSO dataset but performs a random train_test_split, causing subject-level leakage and invalid evaluation for EEG subject-generalization experiments.
- contrastive_loss() computes a similarity tensor of shape [batch] using torch.einsum('bdt,bdt->b', ...) and then calls mean(dim=1), which raises an IndexError because dimension 1 does not exist.
- contrastive_loss() is not a valid contrastive/InfoNCE loss. It returns a reduced similarity score instead of constructing positive/negative logits and optimizing a contrastive objective.

ChatGPT Warnings:
- train_test_split() is called without a fixed random_state, reducing experiment reproducibility.
- No subject identifiers are present, making proper Leave-One-Subject-Out partitioning impossible.
- No validation split is produced for model selection or early stopping.
- No input validation is performed to ensure z_eeg and z_audio have matching shapes.
- The printed message ('Preparing LOSO dataset...') contradicts the implemented behavior, which can mislead users and reviewers.
- Synthetic random tensors are acceptable for a scratch example but are not representative of a realistic EEG preprocessing pipeline.

Apply ONLY the required fixes. Do not change unrelated code. Preserve functionality.
After editing, save the files and call autonomous_review again.
