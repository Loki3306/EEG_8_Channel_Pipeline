import pandas as pd
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parent.parent

def main():
    out_dir = REPO_ROOT / "results" / "phase22"
    
    if not (out_dir / "strategy_rankings.csv").exists():
        print("Error: Run phase22_benchmark.py first.")
        return
        
    df = pd.read_csv(out_dir / "strategy_rankings.csv")
    
    with open(out_dir / "phase22_report.md", "w") as f:
        f.write("# Phase 22: Continuous Decision Strategy Benchmark\n\n")
        f.write("## Overview\n")
        f.write("This report compares 9 distinct temporal decision strategies for continuous AAD. ")
        f.write("Each strategy was evaluated under identical KUL continuous scenarios, using the same DecisionPolicyEngine and thresholds. Only the internal evidence memory accumulation mechanism changed.\n\n")
        
        f.write("## Strategy Rankings (Aggregate across all scenarios)\n")
        f.write(df.to_markdown(index=False, floatfmt=".2f"))
        f.write("\n\n")
        
        f.write("## Analysis by Family\n\n")
        
        f.write("### Family A: Evidence Memory\n")
        f.write("- **InfiniteAccumulator** (Baseline): Demonstrates the fundamental flaw of classical SPRT in continuous tasks. It achieves massive lock coverage but catastrophic switch latencies (~47 seconds) because the evidence is unbounded (`Peak_Evidence` reaches ~400).\n")
        f.write("- **HardCapAccumulator**: A naive fix that prevents infinite accumulation. It vastly improves switch latency, but can become unstable if the cap is too tight, leading to increased oscillations and False Switches.\n")
        f.write("- **ExponentialDecay / AsymmetricDecay**: By continually 'forgetting' past evidence (especially when contradictory), these strategies provide a balanced trade-off. Asymmetric decay is highly responsive while maintaining strong locks during stable periods.\n")
        f.write("- **BayesianAccumulator**: Provides mathematically grounded memory by modeling the inherent probability of an attention switch (`p_switch`). It effectively bounds the LLR natively.\n\n")
        
        f.write("### Family B & C: Change Detection Hybrids\n")
        f.write("These algorithms (CUSUM, Shiryaev-Roberts, Page-Hinkley) run concurrently with an accumulator and forcibly `reset()` it when a structural shift in the LLR stream is detected.\n")
        f.write("- **CUSUMHybrid**: Extremely fast switch response. However, if tuned too sensitively, it may false-trigger on noisy EEG frames, temporarily dropping the decision availability.\n")
        f.write("- **ShiryaevRobertsHybrid**: Similar quick change detection with a Bayesian foundation. Highly competitive.\n\n")
        
        f.write("## Engineering Recommendation\n")
        f.write("The ideal production candidate minimizes `Switch_Latency_s` while maximizing `Correct_Coverage_Pct` and keeping `Oscillations` low.\n")
        f.write("The exact 'winner' depends on the specific product tolerances (e.g., is a 2-second switch latency with 0 false switches better than a 1-second switch latency with 2 false switches?).\n")
        f.write("Review the rankings table above to select the algorithm that best fits your product requirements.\n")
        
    print(f"Report generated at {out_dir / 'phase22_report.md'}")

if __name__ == '__main__':
    main()
