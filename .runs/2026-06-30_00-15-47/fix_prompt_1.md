Your previous implementation failed review.

Task: Validate EEG script

EEG Validator Issues:
- scratch/flawed_eeg_2.py: Potential LOSO leakage - using random train_test_split instead of Leave-One-Subject-Out.

ChatGPT Blocking Issues:
- Methodological violation: prepare_eeg_data() claims to prepare a LOSO dataset but performs a random train_test_split, causing subject-level leakage and invalid evaluation for EEG subject-generalization experiments.
- contrastive_loss() computes a similarity tensor of shape [batch] using torch.einsum('bdt,bdt->b', ...) and then calls sim.mean(dim=1). Since sim is 1-dimensional, this raises an IndexError because dimension 1 does not exist.
- Methodological error: contrastive_loss() collapses temporal information directly into a single similarity score per sample instead of constructing a pairwise similarity matrix required for an InfoNCE-style contrastive objective, making the implementation inconsistent with contrastive learning.

ChatGPT Warnings:
- prepare_eeg_data() generates synthetic random EEG tensors without subject identities, making a true LOSO split impossible and unsuitable for validating EEG evaluation pipelines.
- train_test_split() is called without an explicit random_state, reducing experiment reproducibility.
- The printed message 'Preparing LOSO dataset...' is inconsistent with the actual implementation and may mislead users during experimentation.
- The function named contrastive_loss() returns a reduced similarity statistic rather than an actual contrastive loss value.

Apply ONLY the required fixes. Do not change unrelated code. Preserve functionality.
After editing, save the files and call autonomous_review again.
