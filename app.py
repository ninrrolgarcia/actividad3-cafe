import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import json
import os
from groq import Groq

# Configuración de página con diseño ancho
st.set_page_config(
    page_title="AgroDetect - Diagnóstico de Café",
    page_icon="🍃",
    layout="wide"
)

# Estilo personalizado para emular la interfaz
st.markdown("""
    <style>
    .main-title { font-size: 2.2rem; font-weight: 700; color: #2C3E50; }
    .subtitle { font-size: 1.1rem; color: #7F8C8D; margin-bottom: 20px; }
    .card { background-color: #F8F9FA; padding: 20px; border-radius: 10px; border: 1px solid #E9ECEF; }
    .metric-box { font-size: 2.5rem; font-weight: bold; color: #27AE60; text-align: right; }
    </style>
""", unsafe_allow_html=True)

# 1. Cargar artefactos del modelo
@st.cache_resource
def load_model_and_classes():
    model = tf.keras.models.load_model('modelo_cafe.keras')
    with open('class_names.json', 'r') as f:
        class_dict = json.load(f)
    return model, class_dict

model, class_dict = load_model_and_classes()

# 2. Configuración de API de Groq
GROQ_API_KEY = st.secrets.get("GROQ_API_KEY", os.getenv("GROQ_API_KEY", ""))

def consultar_groq(enfermedad):
    if not GROQ_API_KEY:
        return "⚠️ Clave de API de Groq no configurada. Por favor define 'GROQ_API_KEY' en los Secrets de Streamlit."
    
    client = Groq(api_key=GROQ_API_KEY)
    
    prompt = f"""
    Actúa como un Ingeniero Agrónomo experto en el cultivo de café.
    Se ha detectado la siguiente condición o enfermedad en una hoja de café: '{enfermedad}'.

    Proporciona un reporte técnico estructurado en formato Markdown con las siguientes secciones:
    1. **Diferenciación y Síntomas:** Explicación breve de los signos visuales.
    2. **Manejo Agronómico Preventivo y Correctivo:** Tratamientos, fungicidas o prácticas culturales sugeridas.
    3. **Consulta Técnica:** Cuándo acudir a un especialista o laboratorio.
    4. **Monitoreo y Seguimiento:** Frecuencia de revisión del cultivo y condiciones de riesgo.
    5. **Registro y Trazabilidad:** Qué variables registrar en finca.
    
    Responde con tono profesional, conciso y directo para el agricultor.
    """
    
    try:
        response = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="llama-3.3-70b-versatile"
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error al conectar con la API de Groq: {str(e)}"

# 3. Interfaz Principal
st.markdown('<p class="main-title">Captura de Imagen Foliar</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Posicione la hoja de café. El sistema detectará automáticamente signos de Roya, Cercospora o Plagas.</p>', unsafe_allow_html=True)

col_left, col_right = st.columns([1, 1], gap="large")

with col_left:
    st.subheader("Subir Imagen")
    uploaded_file = st.file_uploader("Seleccione una fotografía de la hoja de café...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert('RGB')
        st.image(image, caption="Imagen cargada", use_container_width=True)

with col_right:
    if uploaded_file is not None:
        # Preprocesar imagen
        img_resized = image.resize((224, 224))
        img_array = np.array(img_resized) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # Predicción
        predictions = model.predict(img_array)
        class_idx = np.argmax(predictions[0])
        confidence = float(predictions[0][class_idx]) * 100
        detected_class = class_dict.get(str(class_idx), f"Clase {class_idx}")

        # Mostrar métricas principales
        res_col1, res_col2 = st.columns([2, 1])
        with res_col1:
            st.subheader(f"Diagnóstico: **{detected_class.capitalize()}**")
            st.caption("Estatus: Detectado recientemente")
        with res_col2:
            st.markdown(f'<div class="metric-box">{confidence:.1f}%</div>', unsafe_allow_html=True)
            st.caption("<div style='text-align: right;'>Confianza</div>", unsafe_allow_html=True)

        st.divider()

        # Consulta a Groq
        st.subheader("💡 Orientación y Manejo Preventivo (Generado por IA Groq)")
        with st.spinner("Generando recomendaciones agronómicas especializadas..."):
            recomendaciones = consultar_groq(detected_class)
            st.markdown(recomendaciones)
    else:
        st.info("Cargue una imagen en el panel izquierdo para obtener el diagnóstico y las recomendaciones técnicas.")
