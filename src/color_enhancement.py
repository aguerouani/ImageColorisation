"""
Module pour améliorer la colorisation des images en résolvant le problème des couleurs ternes.

Ce module contient:
1. Loss functions améliorées pour l'entraînement
2. Fonctions de post-processing pour amplifier les couleurs
3. Utilities pour la visualisation et la comparaison

Auteur: Enhanced Colorization Utils
Date: 2025
"""

import torch
import torch.nn as nn
import numpy as np
from skimage.color import lab2rgb
import matplotlib.pyplot as plt


# ===================================================================
# LOSS FUNCTIONS AMÉLIORÉES
# ===================================================================

class ColorfulnessLoss(nn.Module):
    """
    Loss qui pénalise les prédictions trop proches de 0 (images grises/ternes).
    Encourage le modèle à prédire des couleurs plus saturées.
    """
    def __init__(self, alpha=0.1):
        super().__init__()
        self.alpha = alpha
    
    def forward(self, pred_ab, target_ab):
        # Loss de reconstruction standard (L1 ou MSE)
        reconstruction_loss = torch.nn.functional.smooth_l1_loss(pred_ab, target_ab)
        
        # Pénalité pour prédictions trop proches de 0 (couleurs ternes)
        color_magnitude = torch.sqrt(pred_ab[:, 0]**2 + pred_ab[:, 1]**2)
        colorfulness_penalty = -torch.mean(color_magnitude)
        
        # Loss totale
        total_loss = reconstruction_loss + self.alpha * colorfulness_penalty
        
        return total_loss


class PerceptualColorLoss(nn.Module):
    """
    Loss combinée qui inclut:
    1. Reconstruction Loss (L1 smooth)
    2. Colorfulness Loss (encourage saturation)
    3. Color Distribution Loss (encourage diversité des couleurs)
    """
    def __init__(self, alpha_colorfulness=0.05, alpha_distribution=0.02):
        super().__init__()
        self.alpha_colorfulness = alpha_colorfulness
        self.alpha_distribution = alpha_distribution
    
    def forward(self, pred_ab, target_ab):
        # 1. Reconstruction loss
        reconstruction_loss = torch.nn.functional.smooth_l1_loss(pred_ab, target_ab)
        
        # 2. Colorfulness loss
        pred_magnitude = torch.sqrt(pred_ab[:, 0]**2 + pred_ab[:, 1]**2)
        target_magnitude = torch.sqrt(target_ab[:, 0]**2 + target_ab[:, 1]**2)
        colorfulness_loss = torch.nn.functional.relu(target_magnitude - pred_magnitude).mean()
        
        # 3. Distribution loss
        pred_std = torch.std(pred_ab.view(pred_ab.size(0), pred_ab.size(1), -1), dim=2).mean()
        distribution_loss = -pred_std
        
        # Loss totale
        total_loss = (reconstruction_loss + 
                     self.alpha_colorfulness * colorfulness_loss + 
                     self.alpha_distribution * distribution_loss)
        
        return total_loss, {
            'recon': reconstruction_loss.item(),
            'colorfulness': colorfulness_loss.item(),
            'distribution': distribution_loss.item()
        }


class QuantileHuberLoss(nn.Module):
    """
    Loss asymétrique qui pénalise plus fortement les sous-estimations
    de saturation que les sur-estimations.
    """
    def __init__(self, quantile=0.7, delta=1.0):
        super().__init__()
        self.quantile = quantile
        self.delta = delta
    
    def forward(self, pred_ab, target_ab):
        error = target_ab - pred_ab
        
        loss = torch.where(
            error >= 0,
            self.quantile * torch.abs(error),
            (1 - self.quantile) * torch.abs(error)
        )
        
        loss = torch.where(
            torch.abs(error) <= self.delta,
            0.5 * error**2 / self.delta,
            loss - 0.5 * self.delta
        )
        
        return loss.mean()


class SaturationWeightedLoss(nn.Module):
    """
    Loss qui donne plus d'importance aux pixels très colorés (haute saturation).
    """
    def __init__(self, base_weight=1.0, saturation_weight=3.0):
        super().__init__()
        self.base_weight = base_weight
        self.saturation_weight = saturation_weight
    
    def forward(self, pred_ab, target_ab):
        # Calculer la saturation des couleurs cibles
        target_saturation = torch.sqrt(target_ab[:, 0]**2 + target_ab[:, 1]**2)
        target_saturation_norm = (target_saturation + 1) / 2
        
        # Créer des poids
        weights = self.base_weight + self.saturation_weight * target_saturation_norm
        
        # Calculer la loss L1 pondérée
        loss_per_pixel = torch.abs(pred_ab - target_ab)
        weighted_loss = (loss_per_pixel * weights.unsqueeze(1)).mean()
        
        return weighted_loss


# ===================================================================
# POST-PROCESSING: AMPLIFICATION DES COULEURS
# ===================================================================

def enhance_colors(ab_pred, method='amplify', factor=1.5, temperature=1.0):
    """
    Améliore les couleurs prédites pour les rendre plus vives.
    
    Args:
        ab_pred: Tensor (2, H, W) des canaux ab prédits (normalisés [-1, 1])
        method: 'amplify', 'boost', ou 'temperature'
        factor: Facteur d'amplification (> 1 pour plus de couleurs)
        temperature: Temperature sampling (< 1 = conservatif, > 1 = risqué)
    
    Returns:
        Tensor (2, H, W) avec couleurs amplifiées
    """
    ab_enhanced = ab_pred.clone()
    
    if method == 'amplify':
        ab_enhanced = torch.clamp(ab_pred * factor, -1, 1)
        
    elif method == 'boost':
        sign = torch.sign(ab_pred)
        magnitude = torch.abs(ab_pred)
        boosted = torch.pow(magnitude, 1.0/factor)
        ab_enhanced = torch.clamp(sign * boosted, -1, 1)
        
    elif method == 'temperature':
        ab_enhanced = torch.clamp(ab_pred / temperature, -1, 1)
    
    return ab_enhanced


def lab_to_rgb_enhanced(L, ab, enhance_method='amplify', enhance_factor=1.5):
    """
    Convertit LAB vers RGB avec amplification des couleurs.
    
    Args:
        L: Tensor (1, H, W) canal L normalisé [0, 1]
        ab: Tensor (2, H, W) canaux ab normalisés [-1, 1]
        enhance_method: Méthode d'amplification ('amplify', 'boost', 'temperature')
        enhance_factor: Facteur d'amplification
    
    Returns:
        np.array (H, W, 3) image RGB [0, 1]
    """
    # Améliorer les couleurs
    ab_enhanced = enhance_colors(ab, method=enhance_method, factor=enhance_factor)
    
    L = L[0].cpu().numpy()
    ab_enh = ab_enhanced.cpu().numpy().transpose(1, 2, 0)

    # Dénormaliser
    L = L * 100
    ab_enh = ab_enh * 128

    lab = np.concatenate([L[..., np.newaxis], ab_enh], axis=2)
    rgb = np.clip(lab2rgb(lab.astype(np.float64)), 0, 1)
    
    return rgb


# ===================================================================
# FONCTIONS DE VISUALISATION
# ===================================================================

def visualize_enhancement_comparison(model, loader, device, num_samples=4, 
                                    factors=[1.0, 1.5, 2.0, 2.5]):
    """
    Visualise les résultats avec différents facteurs d'amplification.
    """
    model.eval()
    
    with torch.no_grad():
        L_batch, ab_batch = next(iter(loader))
        L_batch, ab_batch = L_batch.to(device), ab_batch.to(device)
        pred_ab_batch = model(L_batch)
        
        n = min(num_samples, L_batch.size(0))
        
        fig, axes = plt.subplots(n, len(factors) + 2, 
                                figsize=(3*(len(factors)+2), 3*n))
        if n == 1:
            axes = axes.reshape(1, -1)
        
        for i in range(n):
            L = L_batch[i]
            ab_gt = ab_batch[i]
            ab_pred = pred_ab_batch[i]
            
            # Grayscale
            gray = L[0].cpu().numpy()
            axes[i, 0].imshow(gray, cmap='gray')
            axes[i, 0].set_title('Input (L)')
            axes[i, 0].axis('off')
            
            # Ground Truth
            from skimage.color import lab2rgb as skimage_lab2rgb
            L_np = L[0].cpu().numpy() * 100
            ab_np = ab_gt.cpu().numpy().transpose(1, 2, 0) * 128
            lab = np.concatenate([L_np[..., np.newaxis], ab_np], axis=2)
            rgb_gt = np.clip(skimage_lab2rgb(lab.astype(np.float64)), 0, 1)
            
            axes[i, 1].imshow(rgb_gt)
            axes[i, 1].set_title('Ground Truth')
            axes[i, 1].axis('off')
            
            # Différents facteurs
            for j, factor in enumerate(factors):
                rgb_pred = lab_to_rgb_enhanced(L, ab_pred, 
                                              enhance_method='amplify', 
                                              enhance_factor=factor)
                axes[i, j+2].imshow(rgb_pred)
                axes[i, j+2].set_title(f'Pred (×{factor})')
                axes[i, j+2].axis('off')
        
        plt.tight_layout()
        plt.show()


def compare_before_after(model, loader, device, num_samples=4, best_factor=1.8):
    """
    Compare côte à côte les prédictions originales vs améliorées.
    """
    model.eval()
    
    with torch.no_grad():
        L_batch, ab_batch = next(iter(loader))
        L_batch, ab_batch = L_batch.to(device), ab_batch.to(device)
        pred_ab_batch = model(L_batch)
        
        n = min(num_samples, L_batch.size(0))
        
        fig, axes = plt.subplots(n, 4, figsize=(16, 4*n))
        if n == 1:
            axes = axes.reshape(1, -1)
        
        for i in range(n):
            L = L_batch[i]
            ab_gt = ab_batch[i]
            ab_pred = pred_ab_batch[i]
            
            # Grayscale
            gray = L[0].cpu().numpy()
            axes[i, 0].imshow(gray, cmap='gray')
            axes[i, 0].set_title('Input', fontweight='bold')
            axes[i, 0].axis('off')
            
            # Ground Truth
            from skimage.color import lab2rgb as skimage_lab2rgb
            L_np = L[0].cpu().numpy() * 100
            ab_np = ab_gt.cpu().numpy().transpose(1, 2, 0) * 128
            lab = np.concatenate([L_np[..., np.newaxis], ab_np], axis=2)
            rgb_gt = np.clip(skimage_lab2rgb(lab.astype(np.float64)), 0, 1)
            
            axes[i, 1].imshow(rgb_gt)
            axes[i, 1].set_title('Ground Truth', fontweight='bold', color='green')
            axes[i, 1].axis('off')
            
            # Original (terne)
            L_np = L[0].cpu().numpy() * 100
            ab_np = ab_pred.cpu().numpy().transpose(1, 2, 0) * 128
            lab = np.concatenate([L_np[..., np.newaxis], ab_np], axis=2)
            rgb_pred_original = np.clip(skimage_lab2rgb(lab.astype(np.float64)), 0, 1)
            
            axes[i, 2].imshow(rgb_pred_original)
            axes[i, 2].set_title('Original (Terne)', fontweight='bold', color='orange')
            axes[i, 2].axis('off')
            
            # Amélioré
            rgb_pred_enhanced = lab_to_rgb_enhanced(L, ab_pred, 
                                                   enhance_method='amplify', 
                                                   enhance_factor=best_factor)
            axes[i, 3].imshow(rgb_pred_enhanced)
            axes[i, 3].set_title(f'Amélioré (×{best_factor})', fontweight='bold', color='blue')
            axes[i, 3].axis('off')
        
        plt.suptitle('Comparaison: Avant vs Après Amplification', 
                    fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.show()


# ===================================================================
# UTILITIES
# ===================================================================

def get_recommended_loss(approach='balanced'):
    """
    Retourne une loss function recommandée selon l'approche.
    
    Args:
        approach: 'conservative', 'balanced', ou 'aggressive'
    
    Returns:
        Loss function configurée
    """
    if approach == 'conservative':
        return PerceptualColorLoss(
            alpha_colorfulness=0.03,
            alpha_distribution=0.01
        )
    elif approach == 'balanced':
        return PerceptualColorLoss(
            alpha_colorfulness=0.08,
            alpha_distribution=0.03
        )
    elif approach == 'aggressive':
        return PerceptualColorLoss(
            alpha_colorfulness=0.15,
            alpha_distribution=0.05
        )
    else:
        raise ValueError(f"Unknown approach: {approach}")


def print_usage_guide():
    """Affiche un guide d'utilisation rapide."""
    print("="*70)
    print("🎨 COLOR ENHANCEMENT UTILS - Guide d'Utilisation")
    print("="*70)
    print("\n📚 LOSS FUNCTIONS DISPONIBLES:")
    print("  • PerceptualColorLoss (RECOMMANDÉ)")
    print("  • SaturationWeightedLoss")
    print("  • ColorfulnessLoss")
    print("  • QuantileHuberLoss")
    
    print("\n🔧 POST-PROCESSING:")
    print("  • enhance_colors() - Amplifie les couleurs")
    print("  • lab_to_rgb_enhanced() - Conversion LAB→RGB améliorée")
    
    print("\n📊 VISUALISATION:")
    print("  • visualize_enhancement_comparison() - Teste plusieurs facteurs")
    print("  • compare_before_after() - Compare avant/après")
    
    print("\n💡 EXEMPLE D'UTILISATION:")
    print("```python")
    print("# Option 1: Réentraînement")
    print("criterion = get_recommended_loss('balanced')")
    print("# ... puis entraînez votre modèle ...")
    print()
    print("# Option 2: Post-processing (sans réentraînement)")
    print("rgb = lab_to_rgb_enhanced(L, ab_pred, enhance_factor=1.8)")
    print("```")
    print("="*70)


if __name__ == "__main__":
    print_usage_guide()
