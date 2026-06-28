import torch
import torch.nn as nn
from models.eegnet import EEGNet

class EEGClassifier(nn.Module):
    """
    EEG-only representation learning network.
    Uses EEGNet backbone, pools over time, and projects to a fixed embedding dimension.
    Provides a binary classification head for Track 1 vs Track 2 prediction.
    """
    def __init__(self, in_channels=8, embedding_dim=128, eegnet_f1=8, eegnet_d=2, eegnet_f2=16):
        super().__init__()
        
        # Instantiate base EEGNet without the output projection
        self.backbone = EEGNet(in_channels=in_channels, F1=eegnet_f1, D=eegnet_d, F2=eegnet_f2)
        
        # The backbone produces [Batch, F2, Time]
        # We will temporal mean-pool this to [Batch, F2]
        
        # Project pooled features to the desired embedding dim
        self.embedding_proj = nn.Sequential(
            nn.Linear(eegnet_f2, embedding_dim),
            nn.GELU(),
            nn.Dropout(0.25)
        )
        
        # Classification head for binary classification (Track 1 or Track 2)
        self.classifier = nn.Linear(embedding_dim, 2)
        
    def extract_embedding(self, x):
        """
        Extract the 128-D embedding from the EEG chunk.
        Input: [Batch, Channels, Time]
        Output: [Batch, embedding_dim]
        """
        # Run backbone manually to bypass its own output_proj
        orig_len = x.shape[-1]
        x = x.unsqueeze(1) # [Batch, 1, Channels, Time]
        x = self.backbone.block1(x) # [Batch, F1*D, 1, Time+1]
        x = self.backbone.block2(x) # [Batch, F2, 1, Time+2]
        x = x.squeeze(2)   # [Batch, F2, Time+2]
        
        # Temporal Mean Pooling
        x = x.mean(dim=-1) # [Batch, F2]
        
        # Project to embedding
        emb = self.embedding_proj(x) # [Batch, embedding_dim]
        return emb

    def forward(self, x, return_embedding=False):
        """
        Input: [Batch, Channels, Time]
        Output: Logits [Batch, 2] (or tuple with embedding if requested)
        """
        emb = self.extract_embedding(x)
        logits = self.classifier(emb)
        
        if return_embedding:
            return logits, emb
        return logits

def print_summary():
    model = EEGClassifier()
    x = torch.randn(2, 8, 320)
    logits, emb = model(x, return_embedding=True)
    print(f"Input: {x.shape}")
    print(f"Embedding: {emb.shape}")
    print(f"Logits: {logits.shape}")
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"EEGClassifier Parameter Count: {params:,}")
    
if __name__ == "__main__":
    print_summary()
