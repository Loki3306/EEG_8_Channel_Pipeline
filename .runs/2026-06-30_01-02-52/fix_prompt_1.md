Your previous implementation failed review.

Task: Implement confidence-aware MatchNet for KUL.

EEG Validator Issues:
- .mcp/server.py: Potential LOSO leakage - using random train_test_split instead of Leave-One-Subject-Out.
- .mcp/server.py: Temporal dimension collapsed before InfoNCE, assuming temporal alignment.

ChatGPT Blocking Issues:
- The newly added get_confidence() method is inference-only and is not integrated into the training objective, exported predictions, or evaluation pipeline. The requested task ('Implement confidence-aware MatchNet for KUL') is therefore only partially implemented.
- get_confidence() computes confidence from the absolute similarity margin (abs(sim_a - sim_b)). This removes the decision direction, making high-confidence predictions for both correct and confidently incorrect decisions indistinguishable. As implemented, it is a margin magnitude rather than a calibrated confidence estimate.
- The implementation introduces a second similarity computation path separate from infonce_loss(). If similarity computation changes in the future, training and confidence may diverge because they duplicate rather than share the scoring implementation.

ChatGPT Warnings:
- The confidence score is uncalibrated. There is no evidence that larger margins correspond to higher prediction correctness across KUL subjects.
- Confidence is not returned from forward(), requiring an additional forward pass when confidence is requested during inference.
- Temporal averaging is still an explicit modeling assumption (similarities are averaged over time), which may discard discriminative temporal information. This matches prior design but remains an architectural limitation rather than a correctness issue.
- No validation is performed to ensure EEG and audio latent representations have matching temporal dimensions before similarity computation.
- The InfoNCE implementation correctly constructs pairwise similarity matrices and avoids the previously identified reduction bug.
- No evidence of train/test leakage or random train_test_split usage is present in the reviewed MatchNet implementation.
- The MCP review infrastructure is appropriately excluded from EEG validation after the update, reducing false-positive methodology warnings.

Apply ONLY the required fixes. Do not change unrelated code. Preserve functionality.
After editing, save the files and call autonomous_review again.
