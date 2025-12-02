# Autoencoder y clasificador convolucional de Fashion-MNIST con pre-entrenamiento 
### Trabajo Final Integrador - Brunello, Florencia Luciana (2025)

Este repositorio contiene la implementación de un autoencoder convolucional profundo y un clasificador convolucional profundo entrenados sobre el dataset Fashion-MNIST, incluyendo experimentos de evaluación de hiperparámetros, pre-entrenamiento, fine-tuning y análisis de resultados.

El objetivo principal es estudiar cómo diferentes arquitecturas, tasas de aprendizaje, niveles de dropout y estrategias de entrenamiento impactan la reconstrucción, extracción de características y capacidad de generalización del modelo.

## Estructura del repositorio

- `autoencoder/` - contiene las tres arquitecturas evaluadas, con sus módulos de codificador y decodificador
- `clasificadora/` - incluye las redes de dos y tres capas utilizadas para la clasificación
- `experimentos/` — scripts de entrenamiento (autoencoder, pre-entrenamiento, fine-tuning, scratch).  

## Dataset: Fashion-MNIST

- 60.000 imágenes de entrenamiento  
- 10.000 imágenes de validación/test  
- 28 × 28 pixeles, escala de grises  
- 10 categorías de prendas de vestir  

## Arquitectura del Autoencoder

Se evaluaron tres configuraciones variando número de filtros, tamaño de kernels y cantidad de capas.
**Mejor configuración:** Experimento 1  
**Mejor learning rate:** `1e-3

## Arquitectura del Clasificador

El codificador pre-entrenado se reutiliza como extractor de características.  
Se evalúan redes de **2** y **3 capas**:
**Mejor arquitectura:**  
Clasificador de **2 capas**, dropout **0.2**.

## Estrategias de entrenamiento evaluadas

1. **Entrenamiento completo sin pre-entrenamiento**  
2. **Fine-tuning completo** con el codificador pre-entrenado  
3. **Entrenamiento solo del clasificador**, codificador congelado  
4. **Pre-entrenamiento del autoencoder + clasificador desde cero**

## Resultados principales

### Autoencoder
- Mejor pérdida: **0.0025**
- Mejor LR: `1e-3`
- La arquitectura más simple obtuvo el mejor rendimiento

### Clasificador
- Arquitecturas más profundas presentan mayor **sobreajuste**
- Mejor precisión:
  - Sin pre-entrenamiento: **92.66%**
  - Fine-tuning: **92.7%** (pero con fuerte overfitting)
- Clasificador con codificador congelado:
  - **86.99%**, sin overfitting
