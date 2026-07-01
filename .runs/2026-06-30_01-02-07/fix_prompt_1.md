Your previous implementation failed review.

Task: Implement confidence-aware MatchNet for KUL.

EEG Validator Issues:
- .mcp/server.py: Potential LOSO leakage - using random train_test_split instead of Leave-One-Subject-Out.
- .mcp/server.py: Temporal dimension collapsed before InfoNCE, assuming temporal alignment.

ChatGPT Blocking Issues:
- The submitted changes do not implement the requested task ('Implement confidence-aware MatchNet for KUL'). The only reviewed source is an MCP server implementation and contains no MatchNet model, KUL training pipeline, confidence estimation, or EEG model changes.
- review_code() assumes reviewer.review() returns valid JSON but performs no contract enforcement. If the reviewer returns prose (as MockReviewer does) or malformed output, downstream parsing fails and autonomous_review degrades into a synthetic blocking issue rather than a reliable review result.

ChatGPT Warnings:
- review_experiment() relies on fragile substring matching (e.g. searching for 'train_test_split', 'loso', 'mean(dim=1)') and can both miss real methodological violations and report false positives.
- is_binary() classifies binary files solely via UTF-8 decode failure, which may misclassify some text encodings and some binary formats.
- Many helper functions silently swallow all exceptions during AST parsing or filesystem traversal, making repository inspection failures difficult to diagnose.
- search_experiment_memory() uses simple keyword matching and may retrieve unrelated experiments because any sufficiently long query token can trigger a match.
- create_checkpoint() performs 'git stash apply' immediately after stashing without checking command results or merge conflicts.
- restore_checkpoint() performs destructive 'git reset --hard' and 'git clean -fd' before confirming stash application succeeds, risking data loss if restoration fails.
- review_code() and verify_fix() both trust LLM output structure without schema validation beyond JSON parsing.
- The binary .pyc file being included in review input adds no review value and should normally be excluded from AI review.

Apply ONLY the required fixes. Do not change unrelated code. Preserve functionality.
After editing, save the files and call autonomous_review again.
