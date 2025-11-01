# Clasificador de Imágenes con Autoencoder CNN - Versión Mejorada

Sistema completo de clasificación de imágenes usando un autoencoder convolucional pre-entrenado seguido de un clasificador. Listo para entrenar, evaluar y clasificar imágenes nuevas en producción.

## Características

✅ **Arquitectura robusta**: Autoencoder CNN con Batch Normalization
✅ **Entrenamiento optimizado**: Early stopping, learning rate scheduler, validación
✅ **Data augmentation**: Mejor generalización con transformaciones aleatorias
✅ **Sistema de inferencia completo**: Script CLI + notebook para clasificar imágenes nuevas
✅ **Guardado robusto**: Metadatos completos, transformaciones, historial de entrenamiento
✅ **Bugs corregidos**: Flujo dimensional correcto del encoder al clasificador

---

## Estructura del Proyecto

```
computer_vision/
├── autoencoder_cnn_v4_improved.ipynb  # Notebook de entrenamiento mejorado
├── inference.py                        # Script CLI para inferencia
├── inference_notebook.ipynb            # Notebook de inferencia
├── README_CLASSIFIER.md                # Este archivo
├── objetos_salon/processed/            # Dataset de entrenamiento
│   ├── mesa/
│   ├── mouse/
│   ├── nada/
│   ├── teclado/
│   ├── pantalla/
│   ├── cpu/
│   └── silla/
└── export_v4/                          # Modelos entrenados (se crea al entrenar)
    ├── autoencoder_state.pt
    ├── classifier_state.pt
    ├── complete_models.pt
    ├── metadata.json
    └── training_history.json
```

---

## Instalación

### Requisitos

```bash
pip install torch torchvision
pip install numpy matplotlib seaborn
pip install scikit-learn pillow
pip install pandas  # opcional, para exportar a CSV
```

---

## Uso

### 1️⃣ Entrenamiento (Primera vez)

Abre el notebook [autoencoder_cnn_v4_improved.ipynb](autoencoder_cnn_v4_improved.ipynb) y ejecuta todas las celdas. Esto:

1. Cargará tu dataset desde `objetos_salon/processed/`
2. Entrenará el autoencoder
3. Entrenará el clasificador
4. Evaluará el modelo
5. Guardará todo en `export_v4/`

**Tiempo estimado**: 10-30 minutos (dependiendo de tu hardware)

#### Configuración personalizada

En el notebook, puedes ajustar los hiperparámetros en la sección de `CONFIG`:

```python
CONFIG = {
    'data_dir': 'objetos_salon/processed',
    'batch_size': 64,
    'latent_dim': 128,
    'ae_epochs': 30,
    'classifier_epochs': 25,
    'img_size': 32,
    # ... más opciones
}
```

---

### 2️⃣ Clasificar Imágenes Nuevas

Una vez entrenado el modelo, tienes **dos opciones** para clasificar imágenes nuevas:

#### Opción A: Script de línea de comandos

```bash
# Hacer el script ejecutable (solo una vez)
chmod +x inference.py

# Clasificar una imagen
python inference.py --image mi_imagen.jpg

# Clasificar múltiples imágenes
python inference.py --image img1.jpg --image img2.jpg --image img3.jpg

# Clasificar todas las imágenes en una carpeta
python inference.py --folder imagenes_nuevas/

# Ver top-5 predicciones
python inference.py --image img.jpg --top-k 5

# Guardar resultados en JSON
python inference.py --folder imagenes/ --output resultados.json

# Usar modelo de otro directorio
python inference.py --image img.jpg --model-dir export_v3/
```

#### Opción B: Notebook interactivo

Abre [inference_notebook.ipynb](inference_notebook.ipynb) y sigue las instrucciones. Es ideal para:
- Visualizar las predicciones
- Experimentar interactivamente
- Exportar resultados a CSV
- Análisis exploratorio

---

## Ejemplos de Uso

### Ejemplo 1: Clasificar una imagen con Python

```python
from inference import ImageClassifierPredictor

# Cargar modelo
predictor = ImageClassifierPredictor(model_dir='export_v4')

# Predecir
predictions = predictor.predict('nueva_imagen.jpg', top_k=3)

# Ver resultados
for i, pred in enumerate(predictions, 1):
    print(f"{i}. {pred['class']}: {pred['confidence_pct']:.2f}%")
```

### Ejemplo 2: Procesar carpeta completa

```python
from inference import ImageClassifierPredictor

predictor = ImageClassifierPredictor(model_dir='export_v4')
results = predictor.predict_folder('carpeta_imagenes/', top_k=1)

# Resumen
for img_path, result in results.items():
    if result['success']:
        pred = result['predictions'][0]
        print(f"{img_path}: {pred['class']} ({pred['confidence_pct']:.1f}%)")
```

### Ejemplo 3: Clasificación con umbral de confianza

```python
predictions = predictor.predict('imagen_dudosa.jpg', top_k=1)
confidence = predictions[0]['confidence_pct']

if confidence >= 80:
    print(f"✓ Alta confianza: {predictions[0]['class']}")
elif confidence >= 50:
    print(f"⚠ Confianza media: {predictions[0]['class']}")
else:
    print(f"❌ Confianza baja, revisar manualmente")
```

---

## Arquitectura del Modelo

### Autoencoder

```
Encoder:
  3x32x32 → Conv(16) → MaxPool → 16x16x16
         → Conv(32) → MaxPool → 32x8x8
         → Conv(64) → MaxPool → 64x4x4
         → Flatten → Linear(128) → Latent Code

Decoder:
  Latent Code → Linear(1024) → Reshape(64x4x4)
              → ConvTranspose → 32x8x8
              → ConvTranspose → 16x16x16
              → ConvTranspose → 3x32x32
```

### Clasificador

```
Latent Code (128) → Linear(256) → ReLU → Dropout(0.5)
                  → Linear(128) → ReLU → Dropout(0.5)
                  → Linear(num_classes)
```

**Parámetros totales**: ~500K (autoencoder) + ~50K (clasificador)

---

## Mejoras Implementadas vs v3

| Característica | v3 (Original) | v4 (Mejorada) |
|---|---|---|
| Bug dimensional | ❌ Incorrecto | ✅ Corregido |
| Data augmentation | ❌ No | ✅ Sí |
| Validación | ❌ Solo train/test | ✅ Train/val/test |
| Early stopping | ❌ No | ✅ Sí |
| Batch normalization | ❌ No | ✅ Sí |
| LR scheduler | ❌ No | ✅ Sí |
| Metadatos | ⚠️ Incompletos | ✅ Completos |
| Inferencia | ❌ No disponible | ✅ CLI + Notebook |
| Transformaciones guardadas | ❌ No | ✅ Sí |
| Visualizaciones | ⚠️ Básicas | ✅ Completas |

---

## Flujo de Trabajo Recomendado

### Para agregar nuevas imágenes en el futuro:

1. **Opción A: Re-entrenar con datos nuevos**
   ```bash
   # 1. Agregar imágenes al dataset
   # objetos_salon/processed/nueva_clase/*.jpg

   # 2. Ejecutar notebook de entrenamiento
   # Esto creará un nuevo modelo en export_v4/
   ```

2. **Opción B: Usar modelo actual (solo inferencia)**
   ```bash
   # Si las nuevas imágenes son de clases ya conocidas
   python inference.py --folder nuevas_imagenes/
   ```

### Para agregar una nueva clase:

1. Crear carpeta para la nueva clase:
   ```bash
   mkdir objetos_salon/processed/nueva_clase
   # Agregar imágenes de entrenamiento
   ```

2. Re-entrenar el modelo completo usando el notebook

3. El nuevo modelo clasificará todas las clases (antiguas + nueva)

---

## Resolución de Problemas

### Error: "No se encontró metadata.json"

**Causa**: El modelo no ha sido entrenado todavía.
**Solución**: Ejecuta el notebook de entrenamiento primero.

### Error: "CUDA out of memory"

**Causa**: GPU sin suficiente memoria.
**Solución**: Reduce `batch_size` en CONFIG o usa CPU:
```python
predictor = ImageClassifierPredictor(device='cpu')
```

### Predicciones con confianza muy baja

**Causa**: La imagen es muy diferente al dataset de entrenamiento.
**Solución**:
1. Verifica que la imagen sea de una clase conocida
2. Agrega más datos de entrenamiento similares
3. Re-entrena el modelo

### Las imágenes no se clasifican correctamente

**Posibles causas**:
1. **Imagen muy diferente al training set**: Agregar más datos similares
2. **Clase desbalanceada**: Verificar que cada clase tenga suficientes imágenes
3. **Overfitting**: Usar más data augmentation o aumentar dropout

---

## Personalización Avanzada

### Cambiar el tamaño de las imágenes

```python
CONFIG = {
    'img_size': 64,  # Cambiar de 32 a 64
    # ... otros parámetros
}
```

**Nota**: Deberás ajustar la arquitectura del autoencoder si cambias el tamaño.

### Descongelar el encoder para fine-tuning

En el notebook, modifica:

```python
classifier = ImprovedClassifier(
    autoencoder=autoencoder,
    num_classes=num_classes,
    dropout=CONFIG['dropout'],
    freeze_encoder=False  # Cambiar a False
)
```

### Exportar a ONNX para producción

```python
import torch.onnx

dummy_input = torch.randn(1, 3, 32, 32).to(device)
torch.onnx.export(
    classifier,
    dummy_input,
    "model.onnx",
    input_names=['input'],
    output_names=['output']
)
```

---

## Métricas de Rendimiento

El modelo reporta:
- **Accuracy** en train/val/test
- **Precision, Recall, F1-score** por clase
- **Matriz de confusión**
- **Curvas de aprendizaje** (loss y accuracy)

Ejemplo de output:

```
Test Accuracy: 92.34%

Classification Report:
              precision    recall  f1-score   support
        cpu       0.91      0.89      0.90        45
       mesa       0.94      0.93      0.94        52
      mouse       0.88      0.91      0.89        38
       nada       0.96      0.97      0.97        61
   pantalla       0.92      0.90      0.91        48
      silla       0.93      0.95      0.94        43
    teclado       0.90      0.88      0.89        50
```

---

## Licencia y Créditos

Proyecto creado con PyTorch. Para uso educativo y de investigación.

---

## FAQ

**P: ¿Puedo usar este modelo para otras tareas?**
R: Sí, solo necesitas cambiar el dataset en `CONFIG['data_dir']` y re-entrenar.

**P: ¿Cuántas imágenes necesito por clase?**
R: Mínimo 50-100 por clase para buenos resultados. Más es mejor.

**P: ¿Puedo usar el autoencoder para otras tareas?**
R: Sí, el autoencoder aprende representaciones genéricas que puedes usar para clustering, detección de anomalías, etc.

**P: ¿Cómo sé si mi modelo está en overfitting?**
R: Si train_acc >> val_acc (ej: 95% vs 70%), hay overfitting. Aumenta data augmentation o regularización.

**P: ¿Puedo usar el modelo sin GPU?**
R: Sí, funcionará en CPU pero será más lento. Para inferencia es aceptable.

---

## Próximos Pasos

1. ✅ **Ya tienes**: Sistema completo de entrenamiento e inferencia
2. 📊 **Recomendado**: Agregar más datos de entrenamiento si accuracy < 90%
3. 🚀 **Avanzado**: Implementar API REST con Flask/FastAPI
4. 📱 **Producción**: Exportar a ONNX o TorchScript para deployment

---

## Soporte

Si encuentras problemas:
1. Revisa la sección "Resolución de Problemas"
2. Verifica que todos los archivos estén en su lugar
3. Confirma que el modelo fue entrenado correctamente (existe `export_v4/`)

**¡Listo para clasificar! 🚀**
