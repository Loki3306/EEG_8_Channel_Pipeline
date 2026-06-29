Your previous implementation failed review.

Task: Validate EEG script

EEG Validator Issues:
- scratch/flawed_eeg_2.py: Potential LOSO leakage - using random train_test_split instead of Leave-One-Subject-Out.

ChatGPT Blocking Issues:
- scratch/flawed_eeg_2.py: prepare_eeg_data() claims to prepare a LOSO dataset but performs a random train_test_split over trials, introducing subject-level train/test leakage and violating EEG LOSO evaluation methodology.
- scratch/flawed_eeg_2.py: contrastive_loss() computes a per-sample similarity vector with torch.einsum('bdt,bdt->b', ...) and then calls sim.mean(dim=1). Since sim is 1-dimensional, this raises a runtime error (dimension out of range).
- scratch/flawed_eeg_2.py: contrastive_loss() is not an InfoNCE-style contrastive objective because it never constructs a pairwise similarity matrix or negative pairs, making the implementation methodologically incorrect for contrastive representation learning.

ChatGPT Warnings:
- The current diff in scratch/flawed_eeg.py resolves the previously reported blocking issues by using a subject-level split and a symmetric InfoNCE-style loss.
- scratch/flawed_eeg.py resets the global RNG with torch.manual_seed(42) inside prepare_eeg_data(), which reduces reproducibility control for callers and is better handled outside reusable data-loading functions.
- scratch/flawed_eeg.py performs temporal mean pooling before InfoNCE, which is a valid implementation choice but discards temporal information and should be treated as an explicit modeling assumption.
- Both files use synthetic random tensors without subject metadata or realistic EEG preprocessing, so they are suitable only as toy validation examples.

Apply ONLY the required fixes. Do not change unrelated code. Preserve functionality.
After editing, save the files and call autonomous_review again.
