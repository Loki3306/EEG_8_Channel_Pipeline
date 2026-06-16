# Current System Flow Diagram

## High-Level Repository Pipeline

```mermaid
flowchart TD
    %% Dataset and Subject Loading
    RAW[Raw KUL Dataset\nS*.mat files] --> |subject_files()| SL[Subject Loader\nload_subject_examples()]
    SL --> |Trial Construction| TR[Trial Examples\nSubject, Trial, EEG, WavA, WavB, Label]
    TR --> PRE[prepare_dataset()]
    
    subgraph Preprocessing Pipeline
        PRE --> CH[Channel Selection\n[13, 46, 43, 23, 50, 0, 52, 14]]
        CH --> BP[Bandpass Filter\n1.0Hz - 6.0Hz]
        BP --> ZN[Z-score Normalization\nPer Channel]
        
        %% Audio Pipeline
        MAP[audio_mapping.json] --> |get_mapping_data()| A_LOAD[Load Gammatone Envelopes\n28 Subbands]
        A_LOAD --> TRUNC[Truncate to Minimum Length\nmin(EEG_len, Audio_len)]
        TRUNC --> A_NORM[Z-score Normalization\nPer Subband]
    end
    
    ZN --> TRUNC
    
    TRUNC --> CHNK[chunk_trial()\nSlice into T-second overlapping windows]
    A_NORM --> CHNK
    
    subgraph Training / MatchNet
        CHNK --> MN[ContrastiveMatchNet]
        
        MN --> |x_chunk| EEG_ENC[EEG Encoder\nEEGNet/MultiScale/etc.]
        MN --> |ya_chunk, yb_chunk| A_LAG[Audio Lag Modeling\nlags=[3,6,10,13,16]]
        A_LAG --> A_ENC[Audio Encoder\nStandard/Inception]
        
        EEG_ENC --> |[B, 64, T]| Z_EEG[z_eeg]
        A_ENC --> |[B, 64, T]| Z_A[z_a]
        A_ENC --> |[B, 64, T]| Z_B[z_b]
        
        Z_EEG --> LOSS{Contrastive Loss}
        Z_A --> LOSS
        Z_B --> LOSS
    end
    
    LOSS --> OPT[Optimizer Update\nAdamW, CosineAnnealing]
    
    subgraph Validation
        Z_EEG -.-> VAL_SIM[F.cosine_similarity\nMean over time]
        Z_A -.-> VAL_SIM
        Z_B -.-> VAL_SIM
        
        VAL_SIM --> VAL_DECISION[sim_a > sim_b ?]
        VAL_DECISION --> VAL_ACC[Validation Accuracy]
        VAL_ACC --> CHK[Checkpoint Selection\nSave Best Model]
    end
    
    subgraph LOSO Evaluation
        CHK --> TE[Test Subject\nHeld Out]
        TE --> TE_CHUNK[Evaluate Chunk by Chunk]
        TE_CHUNK --> TE_MN[Pre-trained MatchNet]
        TE_MN --> TE_SIM[sim_a, sim_b]
        TE_SIM --> WIN[Window Aggregation\nevidence_accumulation_study.py]
        WIN --> FINAL[Final Accuracy vs Latency]
    end
```
