"""
Script pour augmenter les données d'entraînement en créant des crops de 256x256 pixels.
Pour chaque image, on génère 5 crops aléatoires.
"""

import os
import random
from PIL import Image
from pathlib import Path
import argparse


def crop_image(image, crop_size=256):
    """
    Crée un crop aléatoire de taille crop_size x crop_size depuis l'image.
    
    Args:
        image: PIL Image object
        crop_size: Taille du crop (défaut: 256)
    
    Returns:
        PIL Image object cropée
    """
    width, height = image.size
    
    # Vérifier que l'image est assez grande
    if width < crop_size or height < crop_size:
        # Si l'image est trop petite, on la redimensionne
        scale = max(crop_size / width, crop_size / height)
        new_width = int(width * scale) + 1
        new_height = int(height * scale) + 1
        image = image.resize((new_width, new_height), Image.LANCZOS)
        width, height = image.size
    
    # Position aléatoire pour le crop
    left = random.randint(0, width - crop_size)
    top = random.randint(0, height - crop_size)
    right = left + crop_size
    bottom = top + crop_size
    
    return image.crop((left, top, right, bottom))


def process_directory(input_dir, output_dir, num_crops=5, crop_size=256):
    """
    Traite toutes les images d'un répertoire et crée des crops.
    
    Args:
        input_dir: Répertoire contenant les images d'origine
        output_dir: Répertoire où sauvegarder les crops
        num_crops: Nombre de crops par image (défaut: 5)
        crop_size: Taille des crops (défaut: 256)
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    
    # Créer le répertoire de sortie s'il n'existe pas
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Extensions d'images supportées
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
    
    # Parcourir toutes les images
    image_files = [f for f in input_path.iterdir() 
                   if f.is_file() and f.suffix.lower() in image_extensions]
    
    print(f"Trouvé {len(image_files)} images à traiter")
    print(f"Création de {num_crops} crops de {crop_size}x{crop_size} par image")
    
    total_crops = 0
    for idx, image_file in enumerate(image_files, 1):
        try:
            # Charger l'image
            img = Image.open(image_file)
            
            # Convertir en RGB si nécessaire (pour les images en niveaux de gris ou RGBA)
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            # Créer les crops
            base_name = image_file.stem
            for crop_idx in range(num_crops):
                cropped_img = crop_image(img, crop_size)
                
                # Nom du fichier de sortie
                output_filename = f"{base_name}_crop{crop_idx:02d}.png"
                output_file = output_path / output_filename
                
                # Sauvegarder le crop
                cropped_img.save(output_file, 'PNG')
                total_crops += 1
            
            if idx % 10 == 0:
                print(f"Traité {idx}/{len(image_files)} images ({total_crops} crops créés)")
                
        except Exception as e:
            print(f"Erreur lors du traitement de {image_file.name}: {e}")
            continue
    
    print(f"\nTerminé! {total_crops} crops créés dans {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Augmentation de données par cropping aléatoire"
    )
    parser.add_argument(
        '--input', '-i',
        type=str,
        required=True,
        help='Répertoire contenant les images d\'origine'
    )
    parser.add_argument(
        '--output', '-o',
        type=str,
        required=True,
        help='Répertoire où sauvegarder les crops'
    )
    parser.add_argument(
        '--num-crops', '-n',
        type=int,
        default=5,
        help='Nombre de crops par image (défaut: 5)'
    )
    parser.add_argument(
        '--crop-size', '-s',
        type=int,
        default=256,
        help='Taille des crops en pixels (défaut: 256)'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=None,
        help='Seed pour la reproductibilité (optionnel)'
    )
    
    args = parser.parse_args()
    
    # Définir le seed si fourni
    if args.seed is not None:
        random.seed(args.seed)
        print(f"Utilisation du seed: {args.seed}")
    
    # Traiter les images
    process_directory(
        args.input,
        args.output,
        num_crops=args.num_crops,
        crop_size=args.crop_size
    )


if __name__ == "__main__":
    main()
