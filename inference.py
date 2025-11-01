#!/usr/bin/env python3
"""
Script de inferencia para clasificar imágenes nuevas usando el modelo entrenado.

Uso:
    # Clasificar una imagen
    python inference.py --image path/to/image.jpg

    # Clasificar múltiples imágenes
    python inference.py --image img1.jpg --image img2.jpg --image img3.jpg

    # Clasificar todas las imágenes en una carpeta
    python inference.py --folder path/to/images/

    # Usar modelo de un directorio específico
    python inference.py --image img.jpg --model-dir export_v4/

    # Mostrar top-k predicciones
    python inference.py --image img.jpg --top-k 3
"""

import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import json
import argparse
import os
from pathlib import Path
import numpy as np


class ImprovedAutoencoder(nn.Module):
    """
    Arquitectura del Autoencoder (debe coincidir con el notebook).
    """
    def __init__(self, latent_dim=128):
        super(ImprovedAutoencoder, self).__init__()
        self.latent_dim = latent_dim

        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )

        self.to_latent = nn.Sequential(
            nn.Linear(64 * 4 * 4, latent_dim),
            nn.ReLU()
        )

        self.from_latent = nn.Sequential(
            nn.Linear(latent_dim, 64 * 4 * 4),
            nn.ReLU()
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),

            nn.ConvTranspose2d(16, 3, kernel_size=2, stride=2),
            nn.Tanh()
        )

    def forward(self, x):
        encoded = self.encoder(x)
        encoded = encoded.view(encoded.size(0), -1)
        latent_code = self.to_latent(encoded)

        decoded = self.from_latent(latent_code)
        decoded = decoded.view(decoded.size(0), 64, 4, 4)
        reconstructed = self.decoder(decoded)

        return reconstructed, latent_code

    def encode(self, x):
        encoded = self.encoder(x)
        encoded = encoded.view(encoded.size(0), -1)
        latent_code = self.to_latent(encoded)
        return latent_code


class ImprovedClassifier(nn.Module):
    """
    Arquitectura del Clasificador (debe coincidir con el notebook).
    """
    def __init__(self, autoencoder, num_classes, dropout=0.5):
        super(ImprovedClassifier, self).__init__()
        self.autoencoder = autoencoder
        self.latent_dim = autoencoder.latent_dim

        for param in self.autoencoder.parameters():
            param.requires_grad = False

        self.classifier = nn.Sequential(
            nn.Linear(self.latent_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        with torch.no_grad():
            latent_code = self.autoencoder.encode(x)

        output = self.classifier(latent_code)
        return output


class ImageClassifierPredictor:
    """
    Clase wrapper para facilitar la carga y uso del modelo en producción.
    """

    def __init__(self, model_dir='export_v4', device=None):
        """
        Inicializa el predictor cargando el modelo y metadata.

        Args:
            model_dir: Directorio donde están guardados los modelos
            device: 'cuda', 'cpu' o None (auto-detect)
        """
        self.model_dir = Path(model_dir)

        # Detectar dispositivo
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        print(f"Usando dispositivo: {self.device}")

        # Cargar metadata
        self._load_metadata()

        # Configurar transformaciones
        self._setup_transforms()

        # Cargar modelos
        self._load_models()

        print(f"✓ Modelo cargado exitosamente")
        print(f"✓ Clases: {self.classes}")
        print(f"✓ Listo para clasificar imágenes de tamaño {self.img_size}x{self.img_size}")

    def _load_metadata(self):
        """Carga los metadatos del modelo."""
        metadata_path = self.model_dir / 'metadata.json'

        if not metadata_path.exists():
            raise FileNotFoundError(
                f"No se encontró metadata.json en {self.model_dir}. "
                "Asegúrate de entrenar el modelo primero."
            )

        with open(metadata_path, 'r') as f:
            self.metadata = json.load(f)

        # Extraer información importante
        self.classes = self.metadata['classes']
        self.num_classes = self.metadata['num_classes']
        self.latent_dim = self.metadata['architecture']['latent_dim']
        self.img_size = self.metadata['architecture']['img_size']
        self.dropout = self.metadata['architecture']['dropout']

        # Transformaciones
        transform_info = self.metadata['transforms']
        self.normalize_mean = transform_info['normalize_mean']
        self.normalize_std = transform_info['normalize_std']

    def _setup_transforms(self):
        """Configura las transformaciones para las imágenes de entrada."""
        self.transform = transforms.Compose([
            transforms.Resize((self.img_size, self.img_size)),
            transforms.ToTensor(),
            transforms.Normalize(self.normalize_mean, self.normalize_std)
        ])

    def _load_models(self):
        """Carga los modelos (autoencoder y clasificador)."""
        # Crear arquitecturas
        autoencoder = ImprovedAutoencoder(latent_dim=self.latent_dim)

        # Cargar pesos del autoencoder
        ae_path = self.model_dir / 'autoencoder_state.pt'
        if not ae_path.exists():
            raise FileNotFoundError(f"No se encontró {ae_path}")

        autoencoder.load_state_dict(
            torch.load(ae_path, map_location=self.device)
        )
        autoencoder.eval()

        # Crear clasificador
        self.model = ImprovedClassifier(
            autoencoder=autoencoder,
            num_classes=self.num_classes,
            dropout=self.dropout
        )

        # Cargar pesos del clasificador
        clf_path = self.model_dir / 'classifier_state.pt'
        if not clf_path.exists():
            raise FileNotFoundError(f"No se encontró {clf_path}")

        self.model.load_state_dict(
            torch.load(clf_path, map_location=self.device)
        )
        self.model.to(self.device)
        self.model.eval()

    def load_image(self, image_path):
        """
        Carga y preprocesa una imagen.

        Args:
            image_path: Ruta a la imagen

        Returns:
            Tensor de imagen preprocesado
        """
        try:
            image = Image.open(image_path).convert('RGB')
            image_tensor = self.transform(image)
            return image_tensor
        except Exception as e:
            raise ValueError(f"Error al cargar imagen {image_path}: {e}")

    def predict(self, image_path, top_k=1):
        """
        Predice la clase de una imagen.

        Args:
            image_path: Ruta a la imagen
            top_k: Número de predicciones top a retornar

        Returns:
            Lista de tuplas (clase, probabilidad) ordenadas por probabilidad
        """
        # Cargar y preprocesar imagen
        image_tensor = self.load_image(image_path)
        image_tensor = image_tensor.unsqueeze(0).to(self.device)

        # Predicción
        with torch.no_grad():
            outputs = self.model(image_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            probs, indices = torch.topk(probabilities, k=min(top_k, self.num_classes))

        # Formatear resultados
        results = []
        for prob, idx in zip(probs[0].cpu().numpy(), indices[0].cpu().numpy()):
            results.append({
                'class': self.classes[idx],
                'probability': float(prob),
                'confidence_pct': float(prob * 100)
            })

        return results

    def predict_batch(self, image_paths, top_k=1):
        """
        Predice las clases de múltiples imágenes.

        Args:
            image_paths: Lista de rutas a imágenes
            top_k: Número de predicciones top a retornar por imagen

        Returns:
            Diccionario con resultados para cada imagen
        """
        results = {}

        for image_path in image_paths:
            try:
                predictions = self.predict(image_path, top_k=top_k)
                results[image_path] = {
                    'success': True,
                    'predictions': predictions
                }
            except Exception as e:
                results[image_path] = {
                    'success': False,
                    'error': str(e)
                }

        return results

    def predict_folder(self, folder_path, top_k=1, extensions=None):
        """
        Predice las clases de todas las imágenes en una carpeta.

        Args:
            folder_path: Ruta a la carpeta con imágenes
            top_k: Número de predicciones top a retornar
            extensions: Lista de extensiones válidas (default: jpg, jpeg, png)

        Returns:
            Diccionario con resultados para cada imagen
        """
        if extensions is None:
            extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.gif']

        folder = Path(folder_path)
        image_paths = []

        for ext in extensions:
            image_paths.extend(folder.glob(f'*{ext}'))
            image_paths.extend(folder.glob(f'*{ext.upper()}'))

        if not image_paths:
            raise ValueError(f"No se encontraron imágenes en {folder_path}")

        print(f"Encontradas {len(image_paths)} imágenes")
        return self.predict_batch([str(p) for p in image_paths], top_k=top_k)


def main():
    """Función principal para uso desde línea de comandos."""
    parser = argparse.ArgumentParser(
        description='Clasificador de imágenes con Autoencoder CNN',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    parser.add_argument(
        '--image', '-i',
        type=str,
        action='append',
        help='Ruta a una imagen para clasificar (puede usarse múltiples veces)'
    )

    parser.add_argument(
        '--folder', '-f',
        type=str,
        help='Ruta a una carpeta con imágenes para clasificar'
    )

    parser.add_argument(
        '--model-dir', '-m',
        type=str,
        default='export_v4',
        help='Directorio donde está guardado el modelo (default: export_v4)'
    )

    parser.add_argument(
        '--top-k', '-k',
        type=int,
        default=3,
        help='Número de predicciones top a mostrar (default: 3)'
    )

    parser.add_argument(
        '--device', '-d',
        type=str,
        choices=['cuda', 'cpu', 'auto'],
        default='auto',
        help='Dispositivo a usar (default: auto)'
    )

    parser.add_argument(
        '--output', '-o',
        type=str,
        help='Archivo JSON para guardar resultados (opcional)'
    )

    args = parser.parse_args()

    # Validar argumentos
    if not args.image and not args.folder:
        parser.error("Debes especificar --image o --folder")

    # Cargar modelo
    device = None if args.device == 'auto' else args.device
    try:
        predictor = ImageClassifierPredictor(
            model_dir=args.model_dir,
            device=device
        )
    except Exception as e:
        print(f"❌ Error al cargar modelo: {e}")
        return 1

    print("\n" + "="*60)
    print("CLASIFICANDO IMÁGENES")
    print("="*60 + "\n")

    # Realizar predicciones
    results = {}

    if args.folder:
        # Clasificar carpeta
        try:
            results = predictor.predict_folder(args.folder, top_k=args.top_k)
        except Exception as e:
            print(f"❌ Error al procesar carpeta: {e}")
            return 1

    elif args.image:
        # Clasificar imágenes individuales
        results = predictor.predict_batch(args.image, top_k=args.top_k)

    # Mostrar resultados
    for image_path, result in results.items():
        print(f"\n📷 Imagen: {image_path}")
        print("-" * 60)

        if result['success']:
            for i, pred in enumerate(result['predictions'], 1):
                print(f"  {i}. {pred['class']}: {pred['confidence_pct']:.2f}%")
        else:
            print(f"  ❌ Error: {result['error']}")

    # Guardar resultados si se especificó output
    if args.output:
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\n✓ Resultados guardados en: {args.output}")

    print("\n" + "="*60)
    print("CLASIFICACIÓN COMPLETADA")
    print("="*60)

    return 0


if __name__ == '__main__':
    exit(main())
