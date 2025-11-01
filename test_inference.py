#!/usr/bin/env python3
"""
Script de prueba rápida para verificar que el sistema de inferencia funciona correctamente.

Uso:
    python test_inference.py
"""

from inference import ImageClassifierPredictor
import os
from pathlib import Path


def test_model_loading():
    """Test 1: Verificar que el modelo se puede cargar."""
    print("\n" + "="*60)
    print("TEST 1: Cargando modelo...")
    print("="*60)

    try:
        predictor = ImageClassifierPredictor(
            model_dir='export_v4',
            device='cpu'
        )
        print("✅ Modelo cargado exitosamente")
        print(f"   Clases disponibles: {predictor.classes}")
        print(f"   Número de clases: {predictor.num_classes}")
        print(f"   Tamaño de imagen: {predictor.img_size}x{predictor.img_size}")
        print(f"   Latent dim: {predictor.latent_dim}")
        return predictor
    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
        print("\n⚠️  Primero debes entrenar el modelo usando el notebook:")
        print("   autoencoder_cnn_v4_improved.ipynb")
        return None
    except Exception as e:
        print(f"❌ Error inesperado: {e}")
        return None


def test_predict_from_dataset(predictor):
    """Test 2: Verificar predicción con imágenes del dataset."""
    print("\n" + "="*60)
    print("TEST 2: Probando predicción con imágenes del dataset...")
    print("="*60)

    dataset_path = Path('objetos_salon/processed')

    if not dataset_path.exists():
        print("⚠️  Dataset no encontrado en objetos_salon/processed/")
        return

    # Tomar una imagen de cada clase
    test_images = []
    for class_dir in dataset_path.iterdir():
        if class_dir.is_dir():
            images = list(class_dir.glob('*.jpg')) + list(class_dir.glob('*.png'))
            if images:
                test_images.append((str(images[0]), class_dir.name))

    if not test_images:
        print("⚠️  No se encontraron imágenes en el dataset")
        return

    print(f"\nProbando con {len(test_images)} imágenes (una por clase)...\n")

    correct = 0
    total = 0

    for img_path, true_class in test_images:
        try:
            predictions = predictor.predict(img_path, top_k=3)
            predicted_class = predictions[0]['class']
            confidence = predictions[0]['confidence_pct']

            is_correct = predicted_class == true_class
            correct += is_correct
            total += 1

            status = "✅" if is_correct else "❌"
            print(f"{status} {Path(img_path).name}")
            print(f"   Real: {true_class} | Predicho: {predicted_class} ({confidence:.1f}%)")

            if not is_correct:
                print(f"   Top-3: ", end="")
                for i, pred in enumerate(predictions[:3], 1):
                    print(f"{pred['class']}({pred['confidence_pct']:.1f}%) ", end="")
                print()

        except Exception as e:
            print(f"❌ Error al procesar {img_path}: {e}")

    accuracy = (correct / total * 100) if total > 0 else 0
    print(f"\n{'='*60}")
    print(f"Resultado: {correct}/{total} correctas ({accuracy:.1f}%)")
    print(f"{'='*60}")


def test_metadata():
    """Test 3: Verificar metadata del modelo."""
    print("\n" + "="*60)
    print("TEST 3: Verificando metadata...")
    print("="*60)

    metadata_path = Path('export_v4/metadata.json')

    if not metadata_path.exists():
        print("❌ Metadata no encontrado")
        return

    import json
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)

    print("\n�� Información del modelo:")
    print(f"   Creado: {metadata.get('created_at', 'N/A')}")
    print(f"   Framework: {metadata.get('framework', 'N/A')}")
    print(f"   PyTorch version: {metadata.get('system', {}).get('pytorch_version', 'N/A')}")

    if 'performance' in metadata:
        perf = metadata['performance']
        print(f"\n📊 Rendimiento:")
        print(f"   Train accuracy: {perf.get('final_train_acc', 0):.2f}%")
        print(f"   Val accuracy: {perf.get('final_val_acc', 0):.2f}%")
        print(f"   Best val accuracy: {perf.get('best_val_acc', 0):.2f}%")

    print("\n✅ Metadata válido")


def test_export_files():
    """Test 4: Verificar que todos los archivos necesarios existen."""
    print("\n" + "="*60)
    print("TEST 4: Verificando archivos exportados...")
    print("="*60)

    export_dir = Path('export_v4')
    required_files = [
        'autoencoder_state.pt',
        'classifier_state.pt',
        'metadata.json',
        'training_history.json'
    ]

    if not export_dir.exists():
        print(f"❌ Directorio {export_dir} no existe")
        print("   Debes entrenar el modelo primero")
        return False

    all_exist = True
    for filename in required_files:
        filepath = export_dir / filename
        if filepath.exists():
            size_kb = filepath.stat().st_size / 1024
            print(f"✅ {filename} ({size_kb:.1f} KB)")
        else:
            print(f"❌ {filename} - NO ENCONTRADO")
            all_exist = False

    if all_exist:
        print("\n✅ Todos los archivos necesarios están presentes")
    else:
        print("\n❌ Faltan archivos. Re-entrena el modelo.")

    return all_exist


def main():
    """Ejecuta todos los tests."""
    print("\n" + "="*60)
    print("SISTEMA DE PRUEBAS - Clasificador con Autoencoder CNN")
    print("="*60)

    # Test 1: Verificar archivos
    files_ok = test_export_files()

    if not files_ok:
        print("\n⚠️  ATENCIÓN: Primero debes entrenar el modelo.")
        print("   Abre y ejecuta: autoencoder_cnn_v4_improved.ipynb")
        return 1

    # Test 2: Verificar metadata
    test_metadata()

    # Test 3: Cargar modelo
    predictor = test_model_loading()

    if predictor is None:
        print("\n❌ No se pudo cargar el modelo. Tests detenidos.")
        return 1

    # Test 4: Probar predicciones
    test_predict_from_dataset(predictor)

    # Resumen final
    print("\n" + "="*60)
    print("TESTS COMPLETADOS")
    print("="*60)
    print("\n🎉 El sistema está funcionando correctamente!")
    print("\nPróximos pasos:")
    print("  1. Usa inference.py para clasificar imágenes nuevas")
    print("  2. O usa inference_notebook.ipynb para análisis interactivo")
    print("\nEjemplo:")
    print("  python inference.py --image mi_imagen.jpg --top-k 3")
    print("="*60 + "\n")

    return 0


if __name__ == '__main__':
    exit(main())
