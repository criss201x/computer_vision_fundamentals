# computer_vision_fundamentals

#### codigo del encoder-cnn + clasificador -> [autoencoder_cnn_v5_improved.ipynb](https://github.com/criss201x/computer_vision_fundamentals/blob/main/autoencoder_cnn_v5_improved.ipynb) 
#### Codigo para clasificar imagenes externas con el modelo entrenado anteriormente  [inference_notebook.ipynb](https://nbviewer.org/github/criss201x/computer_vision_fundamentals/blob/main/inference_notebook.ipynb.ipynb) 
#### Codigo para clasificar varios objetos en una misma imagen con el modelo entrenado anteriormente  [yolo_integration.ipynb](https://github.com/criss201x/computer_vision_fundamentals/blob/main/yolo_integration.ipynb) 
#### Codigo para clasificar varios objetos en en un video o en tiempo real con el modelo entrenado anteriormente  [yolo_video_integration.ipynb](https://github.com/criss201x/computer_vision_fundamentals/blob/main/yolo_video_integration.ipynb) 
#### video de clasificación multiobjeto en tiempo real con el modelo entrenado anteriormente [video_procesado.mp4](https://github.com/criss201x/computer_vision_fundamentals/blob/main/video_procesado.mp4) 


# Computer Vision Fundamentals

Sistema completo de clasificación de imágenes usando **Autoencoder CNN** + **Clasificador** en PyTorch.

---

## 📋 Tabla de Contenidos

- [Características](#-características)
- [Estructura del Proyecto](#-estructura-del-proyecto)
- [Instalación](#-instalación)
- [Uso Rápido](#-uso-rápido)
- [Notebooks](#-notebooks)
- [Scripts de Inferencia](#-scripts-de-inferencia)
- [Workflow Completo](#-workflow-completo)
- [Configuración](#%EF%B8%8F-configuración)
- [Resultados](#-resultados)

---

## ✨ Características

- ✅ **Autoencoder Convolucional** con representación latente comprimida
- ✅ **Clasificador multicategoria** basado en features aprendidas
- ✅ **Data Augmentation** para mejor generalización
- ✅ **Early Stopping** y validación automática
- ✅ **Batch Normalization** y Dropout
- ✅ **Learning Rate Scheduler** adaptativo
- ✅ **Sistema completo de inferencia** para imágenes nuevas
- ✅ **Visualización de métricas** en tiempo real
- ✅ **Exportación robusta** de modelos

---

## 📁 Estructura del Proyecto

```
computer_vision/
├── autoencoder_cnn_v4_improved.ipynb  # 🎓 Entrenamiento completo
├── inference_notebook.ipynb           # 🔮 Predicción interactiva
├── inference.py                       # 🚀 CLI para inferencia
├── test_inference.py                  # ✔️ Verificación del sistema
├── export_v4/                         # 💾 Modelos entrenados
│   ├── autoencoder_state.pt
│   ├── classifier_state.pt
│   ├── metadata.json
│   └── training_history.json
├── objetos_salon/processed/           # 📸 Dataset (ignorado en git)
│   ├── clase1/
│   ├── clase2/
│   └── ...
└── README.md
```

---

## 🔧 Instalación

### Requisitos
- Python 3.8+
- PyTorch 1.10+
- CUDA (opcional, para GPU)

### Instalar dependencias

```bash
pip install torch torchvision matplotlib numpy scikit-learn seaborn pillow
```

### Verificar instalación

```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import torch; print(f'CUDA disponible: {torch.cuda.is_available()}')"
```

---

## 🚀 Uso Rápido

### 1️⃣ Entrenar el Modelo (una sola vez)

```bash
jupyter notebook autoencoder_cnn_v4_improved.ipynb
```

**Ejecuta todas las celdas** → El modelo se guardará en `export_v4/`

---

### 2️⃣ Verificar que Funciona

```bash
python test_inference.py
```

Salida esperada:
```
✓ Modelos cargados correctamente
✓ Imagen de prueba clasificada
✓ Sistema listo para usar
```

---

### 3️⃣ Clasificar Imágenes Nuevas

#### **Opción A: Línea de Comandos** 🖥️

```bash
# Clasificar una imagen
python inference.py --image nueva_imagen.jpg --top-k 3

# Clasificar carpeta completa
python inference.py --folder imagenes_nuevas/

# Guardar resultados en JSON
python inference.py --folder imgs/ --output resultados.json
```

---

#### **Opción B: Script Python** 🐍

```python
from inference import ImageClassifierPredictor

# Cargar predictor
predictor = ImageClassifierPredictor('export_v4')

# Predecir una imagen
predictions = predictor.predict('nueva_imagen.jpg', top_k=3)

# Mostrar resultados
for pred in predictions:
    print(f"{pred['class']}: {pred['confidence_pct']:.2f}%")
```

---

#### **Opción C: Notebook Interactivo** 📓

```bash
jupyter notebook inference_notebook.ipynb
```

Ejecuta las celdas para:
- Cargar el modelo
- Subir/seleccionar imágenes
- Ver predicciones con visualizaciones

---

## 📚 Notebooks

### 🎓 [`autoencoder_cnn_v4_improved.ipynb`](autoencoder_cnn_v4_improved.ipynb)

**Entrenamiento completo del sistema**

**Secciones:**
1. Configuración de hiperparámetros
2. Carga de datos con augmentation
3. Arquitectura del Autoencoder
4. Entrenamiento con validación
5. Visualización de reconstrucciones
6. Clasificador mejorado (bug corregido)
7. Evaluación en test set
8. Guardado robusto de modelos

**Output:** Modelos entrenados en `export_v4/`

---

### 🔮 [`inference_notebook.ipynb`](https://nbviewer.org/github/criss201x/computer_vision_fundamentals/blob/main/inference_notebook.ipynb)

**Clasificación interactiva de imágenes**

**Características:**
- Carga automática de modelos
- Upload de imágenes desde el navegador
- Visualización de predicciones con confianza
- Top-K predicciones configurables

---

## 🛠️ Scripts de Inferencia

### `inference.py`

**CLI completo para clasificación**

**Uso:**

```bash
# Ayuda
python inference.py --help

# Clasificar una imagen
python inference.py --image foto.jpg

# Top-3 predicciones
python inference.py --image foto.jpg --top-k 3

# Batch de imágenes
python inference.py --folder carpeta_imagenes/

# Guardar resultados
python inference.py --folder imgs/ --output predicciones.json

# Usar modelo custom
python inference.py --image test.jpg --model-dir mi_modelo/
```

---

### `test_inference.py`

**Verificación del sistema de inferencia**

```bash
python test_inference.py
```

**Comprueba:**
- ✓ Carga de modelos
- ✓ Metadata correcta
- ✓ Predicción funcional
- ✓ Formato de salida

---

## 🔄 Workflow Completo

### Primera Vez (Setup Inicial)

```mermaid
graph LR
    A[Preparar Dataset] --> B[Entrenar Modelo]
    B --> C[Verificar Sistema]
    C --> D[Listo para Usar]
```

1. **Organiza tu dataset:**
   ```
   objetos_salon/processed/
   ├── clase1/
   │   ├── img1.jpg
   │   └── img2.jpg
   ├── clase2/
   │   └── ...
   ```

2. **Entrena el modelo:**
   ```bash
   jupyter notebook autoencoder_cnn_v4_improved.ipynb
   # Ejecutar todas las celdas
   ```

3. **Verifica:**
   ```bash
   python test_inference.py
   ```

---

### Uso Diario (Clasificar Imágenes)

```bash
# Solo necesitas esto ⬇️
python inference.py --image nueva_foto.jpg
```

**No necesitas re-entrenar**, el modelo ya está guardado en `export_v4/`

---

### Actualizar el Modelo (Opcional)

Si agregas más datos o quieres mejorar:

1. Agrega imágenes a `objetos_salon/processed/`
2. Re-ejecuta `autoencoder_cnn_v4_improved.ipynb`
3. El nuevo modelo sobrescribirá `export_v4/`

---

## ⚙️ Configuración

### Hiperparámetros (en el notebook)

```python
CONFIG = {
    'img_size': 32,           # Tamaño de entrada
    'latent_dim': 128,        # Dimensión del espacio latente
    'batch_size': 64,         # Tamaño del batch
    'ae_epochs': 30,          # Épocas del autoencoder
    'classifier_epochs': 25,  # Épocas del clasificador
    'dropout': 0.5,           # Dropout del clasificador
    'ae_lr': 1e-3,           # Learning rate autoencoder
    'classifier_lr': 1e-3,   # Learning rate clasificador
    'ae_patience': 5,        # Early stopping autoencoder
    'classifier_patience': 7 # Early stopping clasificador
}
```

---

## 📊 Resultados

El sistema genera:

### Durante el Entrenamiento
- 📈 Curvas de aprendizaje (loss & accuracy)
- 🖼️ Visualizaciones de reconstrucciones
- 📋 Reporte de clasificación (precision, recall, F1)
- 🔥 Matriz de confusión

### Archivos Exportados
```
export_v4/
├── autoencoder_state.pt      # Pesos del autoencoder
├── classifier_state.pt       # Pesos del clasificador
├── complete_models.pt        # Backup completo
├── metadata.json             # Info de clases y config
└── training_history.json     # Métricas de entrenamiento
```

### Ejemplo de Metadata
```json
{
  "classes": ["clase1", "clase2", "clase3"],
  "num_classes": 3,
  "architecture": {
    "latent_dim": 128,
    "img_size": 32
  },
  "performance": {
    "best_val_acc": 95.5
  }
}
```

---

## 🎯 Ejemplos de Uso

### Caso 1: Clasificar imagen individual

```python
from inference import ImageClassifierPredictor

predictor = ImageClassifierPredictor('export_v4')
result = predictor.predict('test.jpg', top_k=1)

print(f"Clase: {result[0]['class']}")
print(f"Confianza: {result[0]['confidence_pct']:.1f}%")
```

**Output:**
```
Clase: laptop
Confianza: 98.3%
```

---

### Caso 2: Batch processing

```python
import os
from inference import ImageClassifierPredictor

predictor = ImageClassifierPredictor('export_v4')

for img in os.listdir('nueva_carpeta/'):
    if img.endswith(('.jpg', '.png')):
        preds = predictor.predict(f'nueva_carpeta/{img}', top_k=1)
        print(f"{img}: {preds[0]['class']} ({preds[0]['confidence_pct']:.1f}%)")
```

---

### Caso 3: Integración en aplicación

```python
from inference import ImageClassifierPredictor
from flask import Flask, request, jsonify

app = Flask(__name__)
predictor = ImageClassifierPredictor('export_v4')

@app.route('/predict', methods=['POST'])
def predict():
    file = request.files['image']
    file.save('temp.jpg')
    results = predictor.predict('temp.jpg', top_k=3)
    return jsonify(results)

if __name__ == '__main__':
    app.run(debug=True)
```

---

## 📝 Notas

- ⚠️ El dataset `objetos_salon/` está excluido de git (ver `.gitignore`)
- 💾 Los modelos entrenados ocupan ~10-50 MB
- 🖥️ GPU recomendada para entrenamiento, opcional para inferencia
- 🔄 El sistema soporta cualquier número de clases (2+)

---

## 🤝 Contribuciones

¿Encontraste un bug o tienes una mejora? ¡Abre un issue o pull request!

---

## 📄 Licencia

MIT License - Usa libremente para proyectos personales o comerciales.

---

## 🔗 Enlaces Útiles

- [Notebook de Entrenamiento](autoencoder_cnn_v4_improved.ipynb)
- [Notebook de Inferencia](https://nbviewer.org/github/criss201x/computer_vision_fundamentals/blob/main/inference_notebook.ipynb)
- [PyTorch Documentation](https://pytorch.org/docs/)
- [Torchvision Transforms](https://pytorch.org/vision/stable/transforms.html)

---

**¿Preguntas?** Abre un issue en el repositorio.
