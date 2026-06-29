import torch
import torch.nn.functional as F

def prepare_eeg_data():
    print("Preparing dataset with proper subject splits...")
    torch.manual_seed(42)
    # 5 subjects, 20 trials each
    data = torch.randn(5, 20, 64, 256)
    labels = torch.randint(0, 2, (5, 20))
    
    # Proper Leave-One-Subject-Out split
    X_train = data[:4].reshape(-1, 64, 256)
    y_train = labels[:4].reshape(-1)
    
    X_test = data[4].reshape(-1, 64, 256)
    y_test = labels[4].reshape(-1)
    
    return X_train, X_test, y_train, y_test

def contrastive_loss(z_eeg, z_audio):
    if z_eeg.shape != z_audio.shape:
        raise ValueError("Shapes must match")
        
    # Pool representations before InfoNCE to avoid temporal alignment assumptions
    z_eeg_pooled = z_eeg.mean(dim=2) 
    z_audio_pooled = z_audio.mean(dim=2)
    
    z_eeg_norm = F.normalize(z_eeg_pooled, dim=-1)
    z_audio_norm = F.normalize(z_audio_pooled, dim=-1)
    
    logits = torch.matmul(z_eeg_norm, z_audio_norm.T) / 0.1
    
    labels = torch.arange(logits.size(0), device=logits.device)
    loss = (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)) / 2
    
    return loss
