# 🍃 Actividad #3: Servicio Web de Detección de Enfermedades en Hojas de Café

**Autor:** Ninrrol Garcia  
**Asignatura:** Computación en la Nube  
**Despliegue Web:** https://ninrrolgarcia-actividad3.streamlit.app/
**Documentación Técnica Completa (PDF):** [Descargar Manual Técnico PDF](./Documentacion_Tecnica_Servicio_Nube_Cafe.pdf)

---

## Descripción del Proyecto

Este proyecto consiste en el desarrollo e implementación de un **Servicio Web basado en Computación en la Nube** que permite detectar enfermedades y plagas en hojas de café (*Coffea arabica*) a partir de fotografías digitales. 

El sistema utiliza una **Red Neuronal Convolucional (CNN)** basada en `MobileNetV2` para la clasificación visual de patologías y se integra con la **API de Groq (Llama 3.3)** para generar diagnósticos agronómicos personalizados y recomendaciones técnicas preventivas en tiempo real.

---

## Tecnologías y Servicios Cloud Utilizados

- **Google Colab (GPU Cloud):** Extracción, preprocesamiento de imágenes, Data Augmentation y entrenamiento del modelo con Transfer Learning.
- **Python 3.11:** Lenguaje principal de desarrollo.
- **TensorFlow / Keras:** Framework de Aprendizaje Profundo utilizado para el diseño e inferencia de la red neuronal.
- **Streamlit Community Cloud:** Plataforma de hosting y despliegue continuo de la aplicación web.
- **API de Groq (`llama-3.3-70b-versatile`):** Servicio de Inteligencia Artificial Generativa para la generación de reportes técnicos.
- **GitHub:** Control de versiones, repositorio de código y orquestación del despliegue.

---

## Estructura del Repositorio

```text
├── app.py                  # Aplicación principal de Streamlit
├── modelo_cafe.keras       # Modelo preentrenado de Inteligencia Artificial
├── class_names.json        # Mapeo de índices a clases/enfermedades
├── requirements.txt        # Dependencias de Python para el entorno Cloud
└── README.md               # Documentación del repositorio
