#!/usr/bin/env python3
"""
Script de test rapide pour visualiser l'amélioration des couleurs.

Usage:
    python test_color_enhancement.py --factor 1.8
    python test_color_enhancement.py --factors 1.0 1.5 2.0 2.5
"""

import argparse
import sys
import torch
import matplotlib.pyplot as plt

# Importer les modules locaux
try:
    from color_enhancement import (
        compare_before_after,
        visualize_enhancement_comparison,
        print_usage_guide
    )
except ImportError:
    print("❌ Erreur: Impossible d'importer color_enhancement.py")
    print("   Assurez-vous d'être dans le répertoire src/")
    sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="Test rapide de l'amélioration des couleurs"
    )
    parser.add_argument(
        '--model-path',
        type=str,
        help='Chemin vers le modèle PyTorch (.pth)'
    )
    parser.add_argument(
        '--data-dir',
        type=str,
        default='../Data/DIV2K_train_LR_bicubic/X2_cropped/',
        help='Répertoire des images'
    )
    parser.add_argument(
        '--factor',
        type=float,
        default=1.8,
        help='Facteur d\'amplification unique (défaut: 1.8)'
    )
    parser.add_argument(
        '--factors',
        type=float,
        nargs='+',
        help='Plusieurs facteurs à comparer (ex: 1.0 1.5 2.0)'
    )
    parser.add_argument(
        '--num-samples',
        type=int,
        default=4,
        help='Nombre d\'échantillons à visualiser (défaut: 4)'
    )
    parser.add_argument(
        '--guide',
        action='store_true',
        help='Afficher le guide d\'utilisation'
    )
    
    args = parser.parse_args()
    
    # Afficher le guide si demandé
    if args.guide:
        print_usage_guide()
        return
    
    # Vérifier la disponibilité de CUDA
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️  Utilisation de: {device}")
    
    # Message si pas de modèle fourni
    if not args.model_path:
        print("\n" + "="*70)
        print("⚠️  INFORMATION")
        print("="*70)
        print("Aucun modèle spécifié (--model-path)")
        print("\nPour tester rapidement:")
        print("  1. Ouvrez le notebook ResNet_UNET.ipynb")
        print("  2. Exécutez les cellules jusqu'à l'entraînement du modèle")
        print("  3. Utilisez la visualisation directement dans le notebook")
        print("\nOu spécifiez un modèle:")
        print("  python test_color_enhancement.py --model-path ../models/model.pth")
        print("="*70)
        print("\n💡 En attendant, voici le guide d'utilisation:\n")
        print_usage_guide()
        return
    
    # Charger le modèle (si disponible)
    try:
        from ResNet_UNET import ResNet_UNet_AE
        model = ResNet_UNet_AE()
        model.load_state_dict(torch.load(args.model_path, map_location=device))
        model = model.to(device)
        model.eval()
        print(f"✅ Modèle chargé: {args.model_path}")
    except Exception as e:
        print(f"❌ Erreur lors du chargement du modèle: {e}")
        return
    
    # Charger les données
    try:
        from ResNet_UNET import RGB2LabDataset
        from torch.utils.data import DataLoader
        
        dataset = RGB2LabDataset(args.data_dir, image_size=256)
        loader = DataLoader(dataset, batch_size=8, shuffle=True)
        print(f"✅ Dataset chargé: {len(dataset)} images")
    except Exception as e:
        print(f"❌ Erreur lors du chargement des données: {e}")
        return
    
    # Visualisation
    print("\n" + "="*70)
    print("🎨 GÉNÉRATION DES VISUALISATIONS")
    print("="*70)
    
    if args.factors:
        # Comparer plusieurs facteurs
        print(f"Comparaison de plusieurs facteurs: {args.factors}")
        visualize_enhancement_comparison(
            model, loader, device,
            num_samples=args.num_samples,
            factors=args.factors
        )
    else:
        # Comparaison avant/après avec un seul facteur
        print(f"Comparaison avec facteur: {args.factor}")
        compare_before_after(
            model, loader, device,
            num_samples=args.num_samples,
            best_factor=args.factor
        )
    
    print("\n✅ Visualisations générées!")
    print("\n💡 Conseils:")
    print(f"  • Couleurs trop pâles ? → Augmentez --factor (essayez {args.factor + 0.3:.1f})")
    print(f"  • Couleurs trop saturées ? → Réduisez --factor (essayez {args.factor - 0.3:.1f})")
    print("  • Testez plusieurs facteurs avec --factors 1.0 1.5 2.0 2.5")


if __name__ == "__main__":
    main()
