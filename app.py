import streamlit as st
import pickle
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
import os
import pandas as pd
import altair as alt
import numpy as np
from PIL import Image

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DOCS_DIR = BASE_DIR
IMG_DIR = os.path.join(DOCS_DIR, 'imagenes_notebook')

st.set_page_config(page_title='Proyecto Especializacion', layout='wide', initial_sidebar_state='expanded')

st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;800&display=swap');
    
    html, body, [class*="css"] { font-family: 'Inter', sans-serif !important; }
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    .hero {
        background: linear-gradient(135deg, #0B192C 0%, #1A365D 50%, #2A4365 100%);
        padding: 60px 30px; border-radius: 16px; color: white; text-align: center;
        box-shadow: 0 10px 30px rgba(0,0,0,0.15); margin-bottom: 40px; margin-top: 20px;
    }
    .hero h1 { color: white !important; font-size: 3.5rem; font-weight: 800; letter-spacing: -1.5px; margin-bottom: 10px; line-height: 1.2; }
    .hero p { color: #A0AEC0 !important; font-size: 1.2rem; font-weight: 300; margin-top: 0; }
    .hero-badge { background: rgba(255,255,255,0.1); padding: 8px 15px; border-radius: 20px; font-size: 0.9rem; display: inline-block; margin-bottom: 15px; letter-spacing: 1px; }

    .section-title {
        font-size: 2.2rem; font-weight: 800; color: var(--text-color);
        margin-bottom: 30px; border-bottom: 4px solid #3182CE; display: inline-block; padding-bottom: 5px;
    }
    
    .info-card {
        background-color: var(--background-color); padding: 30px; border-radius: 12px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.2); border-top: 5px solid #3182CE; height: 100%;
        margin-bottom: 20px; border: 1px solid rgba(128,128,128,0.2);
    }
    .info-card h3 { font-weight: 800; margin-top: 0; font-size: 1.5rem; color: var(--text-color); }
    .info-card p { font-size: 1.05rem; line-height: 1.7; color: var(--text-color); }
    
    div.stTextArea > div > div > textarea { border: 2px solid #3182CE; border-radius: 12px; font-size: 1.15rem; padding: 20px; }
    
    div.stButton > button { 
        background: linear-gradient(135deg, #3182CE 0%, #2B6CB0 100%); color: white; border-radius: 50px;
        font-weight: 800; font-size: 1.2rem; padding: 12px 0; border: none; box-shadow: 0 4px 15px rgba(49, 130, 206, 0.4); transition: all 0.3s ease;
    }
    div.stButton > button:hover { transform: translateY(-2px); box-shadow: 0 6px 20px rgba(49, 130, 206, 0.6); }

    .pred-card { padding: 30px; border-radius: 16px; text-align: center; box-shadow: 0 10px 25px rgba(0,0,0,0.1); margin: 30px 0; animation: fadeIn 0.5s ease-in; }
    .pred-pos { background: linear-gradient(135deg, #1B5E20 0%, #2E7D32 100%); border-bottom: 8px solid #81C784; color: white !important;}
    .pred-neg { background: linear-gradient(135deg, #B71C1C 0%, #C62828 100%); border-bottom: 8px solid #E57373; color: white !important;}
    .pred-title { font-size: 1.2rem; text-transform: uppercase; letter-spacing: 2px; font-weight: 800; margin-bottom: 5px; color: white !important;}
    .pred-score { font-size: 4rem; font-weight: 900; margin: 0; line-height: 1; color: white !important;}
    
    .expl-box { background-color: var(--secondary-background-color); padding: 20px; border-radius: 12px; font-size: 0.95rem; margin-top: 15px; border-left: 4px solid #3182CE; line-height: 1.6; border: 1px solid rgba(128,128,128,0.2);}
    
    @keyframes fadeIn { from { opacity: 0; transform: translateY(10px); } to { opacity: 1; transform: translateY(0); } }
    </style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_nltk_resources():
    try:
        nltk.data.find('tokenizers/punkt_tab')
    except LookupError:
        nltk.download('punkt')
        nltk.download('punkt_tab')
        nltk.download('stopwords')

load_nltk_resources()

@st.cache_resource
def load_models():
    model_path = os.path.join(BASE_DIR, 'modelo_lr.pkl')
    vec_path = os.path.join(BASE_DIR, 'vectorizador.pkl')
    with open(model_path, 'rb') as f:
        modelo = pickle.load(f)
    with open(vec_path, 'rb') as f:
        vectorizador = pickle.load(f)
    return modelo, vectorizador

try:
    modelo_lr, vectorizador = load_models()
    
    palabras_vacias = set(stopwords.words('english'))
    negaciones = {'no', 'not', 'nor', 'neither', 'none', 'never', 'cannot', 'isn\'t', 'aren\'t', 'wasn\'t', 'weren\'t', 'haven\'t', 'hasn\'t', 'hadn\'t', 'won\'t', 'wouldn\'t', 'don\'t', 'doesn\'t', 'didn\'t', 'can\'t', 'couldn\'t', 'shouldn\'t', 'mightn\'t', 'mustn\'t'}
    palabras_vacias = palabras_vacias - negaciones
    stemmer = PorterStemmer()

    def limpiar_texto(texto):
        texto = str(texto).lower()
        texto = re.sub(r'https?://\S+|www\.\S+', '', texto)
        texto = re.sub(r'@\w+', '', texto)
        texto = re.sub(r'[^a-zA-Z\s]', ' ', texto)
        tokens = word_tokenize(texto)
        tokens_limpios = [stemmer.stem(palabra) for palabra in tokens if palabra not in palabras_vacias]
        return ' '.join(tokens_limpios)

    with st.sidebar:
        st.markdown("<div style='text-align: center; margin-bottom: 30px;'><img src='https://cdn-icons-png.flaticon.com/512/2040/2040946.png' width='90' style='filter: drop-shadow(0px 4px 6px rgba(0,0,0,0.3));'></div>", unsafe_allow_html=True)
        
        st.markdown("<h3 style='text-align: center; color: white; font-weight: 800; font-size: 1.2rem; letter-spacing: 1px;'>PANEL DE CONTROL</h3><hr style='margin-top: 5px; margin-bottom: 20px; border-color: #4A5568;'>", unsafe_allow_html=True)
        
        opcion = st.radio("", ["1. Portada General", "2. Analisis de Datos (EDA)", "3. Rendimiento y Metricas", "4. Interfaz Predictiva (Demo)"])
        
        st.markdown("<br><br>", unsafe_allow_html=True)
        
        # Tarjeta lateral de perfil premium (En una sola linea para evitar bugs de formato Markdown)
        st.markdown("""<div style="background: linear-gradient(145deg, #1A202C 0%, #2D3748 100%); padding: 25px; border-radius: 15px; color: white; box-shadow: 0 10px 20px rgba(0,0,0,0.2); border: 1px solid #4A5568;"><p style="color: #63B3ED; margin-bottom: 5px; font-size: 12px; text-transform: uppercase; letter-spacing: 1.5px; font-weight: bold;">MetodologÃ­a</p><p style="margin-top: 0; font-size: 18px; font-weight: 800; color: #F7FAFC;">CRISP-DM</p><p style="color: #63B3ED; margin-bottom: 10px; font-size: 12px; text-transform: uppercase; letter-spacing: 1.5px; font-weight: bold; margin-top: 25px;">Stack TecnolÃ³gico</p><div style="display: flex; gap: 8px; flex-wrap: wrap; margin-bottom: 5px;"><span style="background: rgba(49, 130, 206, 0.2); border: 1px solid #3182CE; padding: 4px 10px; border-radius: 20px; font-size: 12px; font-weight: 600; color: #90CDF4;">Python</span><span style="background: rgba(49, 130, 206, 0.2); border: 1px solid #3182CE; padding: 4px 10px; border-radius: 20px; font-size: 12px; font-weight: 600; color: #90CDF4;">Scikit-Learn</span><span style="background: rgba(49, 130, 206, 0.2); border: 1px solid #3182CE; padding: 4px 10px; border-radius: 20px; font-size: 12px; font-weight: 600; color: #90CDF4;">NLTK</span><span style="background: rgba(49, 130, 206, 0.2); border: 1px solid #3182CE; padding: 4px 10px; border-radius: 20px; font-size: 12px; font-weight: 600; color: #90CDF4;">Streamlit</span></div><p style="color: #63B3ED; margin-bottom: 5px; font-size: 12px; text-transform: uppercase; letter-spacing: 1.5px; font-weight: bold; margin-top: 25px;">Autor</p><p style="margin-top: 0; font-size: 18px; font-weight: 800; color: #F7FAFC;">SebastiÃ¡n Isaza</p></div>""", unsafe_allow_html=True)

    if opcion == "1. Portada General":
        st.markdown("""
            <div class="hero">
                <div class="hero-badge">TEXT & WEB ANALYTICS</div>
                <h1>Analisis de Sentimiento<br>en Redes Sociales</h1>
                <p>Clasificacion algoritmica avanzada mediante Procesamiento de Lenguaje Natural</p>
            </div>
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("""
                <div class="info-card">
                    <h3>El Contexto (Problema)</h3>
                    <p>Las marcas modernas reciben un flujo masivo de interacciones diarias en Twitter/X. Intentar clasificar manualmente si la opinion del publico es buena o mala es <b>logisticamente imposible</b>. Ademas, los diccionarios tradicionales fallan constantemente ante el sarcasmo, las abreviaturas y el lenguaje coloquial del internet.</p>
                </div>
            """, unsafe_allow_html=True)
        with col2:
            st.markdown("""
                <div class="info-card">
                    <h3>La Solucion (Objetivo)</h3>
                    <p>Desarrollamos un <b>modelo de Inteligencia Artificial (Machine Learning)</b> capaz de comprender el texto estructurado, identificar sus raices semanticas mediante NLP (TF-IDF) y clasificar la opinion del usuario en tiempo real con una precision superior al 80%, brindando asi una herramienta critica para el <b>Marketing Digital</b>.</p>
                </div>
            """, unsafe_allow_html=True)
            
    elif opcion == "2. Analisis de Datos (EDA)":
        st.markdown('<div class="section-title">Analisis Exploratorio (EDA)</div>', unsafe_allow_html=True)
        st.markdown("<p style='font-size:1.1rem;'>Visualizacion descriptiva del comportamiento de los datos extraidos (Dataset: Sentiment140, Stanford Univ).</p><br>", unsafe_allow_html=True)
        
        img_path1 = os.path.join(IMG_DIR, 'img_1.png')
        if os.path.exists(img_path1): 
            st.image(Image.open(img_path1), use_container_width=True)
            st.markdown("<div class='expl-box'><b>Balance de Clases:</b> El corpus de 200,000 tweets fue extraido manteniendo un balance perfecto (50/50) para evitar que el algoritmo desarrolle sesgos estadisticos hacia la clase mayoritaria.</div><br><br>", unsafe_allow_html=True)
            
        img_path2 = os.path.join(IMG_DIR, 'img_2.png')
        if os.path.exists(img_path2): 
            st.image(Image.open(img_path2), use_container_width=True)
            st.markdown("<div class='expl-box'><b>Longitud de los Textos:</b> La mayoria de los tweets analizados concentran su longitud entre las 5 y 20 palabras, confirmando el uso de lenguaje directo y rapido caracteristico de la plataforma.</div><br><br>", unsafe_allow_html=True)
            
        st.markdown("<h3 style='font-weight:800; margin-top:20px;'>Top Lexico: Palabras mas Frecuentes</h3>", unsafe_allow_html=True)
        
        img_path4 = os.path.join(IMG_DIR, 'img_4.png')
        if os.path.exists(img_path4): 
            st.image(Image.open(img_path4), use_container_width=True)
            st.markdown("<div class='expl-box'><b>Frecuencia en Sentimiento POSITIVO:</b><br>Al observar los terminos mas repetidos en los tweets etiquetados como positivos, destacan palabras como <i>'good', 'love', 'day', 'thanks'</i>. Esto confirma matematicamente que el dataset es coherente y las personas utilizan estas raices lexicas para expresar alegria o satisfaccion.</div><br><br>", unsafe_allow_html=True)
            
        img_path3 = os.path.join(IMG_DIR, 'img_3.png')
        if os.path.exists(img_path3): 
            st.image(Image.open(img_path3), use_container_width=True)
            st.markdown("<div class='expl-box'><b>Frecuencia en Sentimiento NEGATIVO:</b><br>Por el contrario, en el bloque negativo predominan palabras vinculadas a quejas, cansancio o tristeza, como <i>'work', 'today', 'miss', 'bad'</i>. El algoritmo usa estas diferencias de frecuencia para aprender a clasificar textos nuevos.</div><br><br>", unsafe_allow_html=True)
            
    elif opcion == "3. Rendimiento y Metricas":
        st.markdown('<div class="section-title">Evaluacion y Desempeno</div>', unsafe_allow_html=True)
        
        img_path9 = os.path.join(IMG_DIR, 'img_9.png')
        if os.path.exists(img_path9): 
            st.image(Image.open(img_path9), use_container_width=True)
            st.markdown("<div class='expl-box'><b>Precision Global:</b> De los algoritmos evaluados (Regresion Logistica, Naive Bayes y SVC), la Regresion Logistica evidencio el mejor balance predictivo superando el objetivo de negocio del 80% (F1-Score).</div><br><br>", unsafe_allow_html=True)
            
        img_path8 = os.path.join(IMG_DIR, 'img_8.png')
        if os.path.exists(img_path8): 
            st.image(Image.open(img_path8), use_container_width=True)
            st.markdown("<div class='expl-box'><b>Capacidad Discriminativa:</b> La Curva ROC (AUC ~0.87) demuestra la robusta capacidad matematica del modelo para separar un comentario positivo de uno negativo sin adivinar al azar.</div><br><br>", unsafe_allow_html=True)
            
        st.markdown("<h3 style='font-weight:800; margin-top:20px;'>Analisis del Error (Matriz de Confusion)</h3>", unsafe_allow_html=True)
        img_path5 = os.path.join(IMG_DIR, 'img_5.png')
        if os.path.exists(img_path5): 
            st.image(Image.open(img_path5), use_container_width=True)
            
        st.markdown("""
        <div class="info-card" style="margin-top:20px;">
            <h3>Interpretacion del Mapa de Calor</h3>
            <p>La densa concentracion en la diagonal principal (azul oscuro) valida la alta tasa de Verdaderos Positivos y Verdaderos Negativos.<br><br>
            <b>El margen de error (20% residual):</b> Se debe intrinsecamente a la ironia y el sarcasmo humano. Por ejemplo, usuarios escribiendo "This is exactly what I wanted... NOT!", donde el vocabulario es matematicamente positivo pero la intencion es la opuesta.</p>
        </div>
        """, unsafe_allow_html=True)
            
    elif opcion == "4. Interfaz Predictiva (Demo)":
        st.markdown("""
            <div class="hero" style="padding: 30px; margin-bottom: 20px;">
                <h1 style="font-size:2.5rem; color: white !important;">Demo Interactiva</h1>
                <p style="color: #A0AEC0 !important;">Clasificador en vivo de Regresion Logistica</p>
            </div>
        """, unsafe_allow_html=True)
        
        col_izq, col_centro, col_der = st.columns([1, 8, 1])
        with col_centro:
            user_input = st.text_area("Analiza una frase en ingles (Ej: Comentarios de clientes):", "The new design is absolutely fantastic, but the customer service is terrible.", height=110)
            
            if st.button("PROCESAR TEXTO MEDIANTE IA", use_container_width=True):
                if user_input.strip() != "":
                    texto_limpio = limpiar_texto(user_input)
                    texto_vec = vectorizador.transform([texto_limpio])
                    
                    prediccion = modelo_lr.predict(texto_vec)[0]
                    probabilidad = modelo_lr.predict_proba(texto_vec)[0]
                    
                    if prediccion == 1:
                        st.markdown(f'''
                            <div class="pred-card pred-pos">
                                <div class="pred-title">Veredicto de la Inteligencia Artificial</div>
                                <div class="pred-score score-pos">POSITIVO ({probabilidad[1]:.1%})</div>
                            </div>
                        ''', unsafe_allow_html=True)
                    else:
                        st.markdown(f'''
                            <div class="pred-card pred-neg">
                                <div class="pred-title">Veredicto de la Inteligencia Artificial</div>
                                <div class="pred-score score-neg">NEGATIVO ({probabilidad[0]:.1%})</div>
                            </div>
                        ''', unsafe_allow_html=True)
                        
                    with st.expander("Ver Tokens Estructurados (Proceso Interno)"):
                        st.code(texto_limpio)
                    
                    st.markdown("<br><h3 style='text-align:center; font-weight:800; border-bottom:2px solid #3182CE; padding-bottom:10px; display:inline-block;'>Radiografia de Explicabilidad</h3><br><br>", unsafe_allow_html=True)
                    col_graf1, col_graf2 = st.columns([1, 1.5])
                    
                    with col_graf1:
                        df_probs = pd.DataFrame({'Sentimiento': ['Negativo', 'Positivo'], 'Probabilidad': [probabilidad[0], probabilidad[1]]})
                        donut = alt.Chart(df_probs).mark_arc(innerRadius=70, cornerRadius=8).encode(
                            theta=alt.Theta(field="Probabilidad", type="quantitative"),
                            color=alt.Color(field="Sentimiento", type="nominal", scale=alt.Scale(domain=['Negativo', 'Positivo'], range=['#E53E3E', '#38A169']), legend=alt.Legend(orient="bottom", title=None, labelFontSize=14, labelColor='white')),
                            tooltip=['Sentimiento', alt.Tooltip('Probabilidad', format='.1%')]
                        ).properties(height=350, title=alt.TitleParams(text="Distribucion Probabilistica", font="Inter", fontSize=18, anchor='middle', color='white'))
                        text = alt.Chart(pd.DataFrame({'text': [f"{max(probabilidad):.1%}"]})).mark_text(size=40, fontWeight='bold', color='white').encode(text='text')
                        st.altair_chart(donut + text, use_container_width=True)
                        st.markdown("<div class='expl-box'><b>Certeza del Algoritmo:</b> Muestra matematicamente que tan seguro esta el modelo en base a la frontera de decision (0 a 100%).</div>", unsafe_allow_html=True)
                        
                    with col_graf2:
                        feature_names = vectorizador.get_feature_names_out()
                        coefs = modelo_lr.coef_[0]
                        nonzero_indices = texto_vec.nonzero()[1]
                        
                        if len(nonzero_indices) > 0:
                            words, weights = [], []
                            for idx in nonzero_indices:
                                words.append(feature_names[idx])
                                weights.append(coefs[idx])
                                
                            df_words = pd.DataFrame({'Palabra': words, 'Peso': weights, 'Impacto': ['Positivo' if w > 0 else 'Negativo' for w in weights]})
                            bars = alt.Chart(df_words).mark_bar(cornerRadiusEnd=6).encode(
                                y=alt.Y('Palabra:N', sort=alt.EncodingSortField(field="Peso", order="descending"), title="", axis=alt.Axis(labelFontSize=14, labelFontWeight='bold', labelColor='white')),
                                x=alt.X('Peso:Q', title="Impacto Matematico de la Palabra", axis=alt.Axis(grid=True, labelFontSize=12, titleColor='white', labelColor='white')),
                                color=alt.Color('Impacto:N', scale=alt.Scale(domain=['Negativo', 'Positivo'], range=['#E53E3E', '#38A169']), legend=None),
                                tooltip=['Palabra', alt.Tooltip('Peso:Q', format='.3f')]
                            ).properties(height=alt.Step(45), title=alt.TitleParams(text="Impacto Semantico de tu Frase", font="Inter", fontSize=18, anchor='middle', color='white'))
                            
                            rule = alt.Chart(pd.DataFrame({'x': [0]})).mark_rule(color='white', strokeDash=[4, 4], strokeWidth=2).encode(x='x')
                            st.altair_chart(bars + rule, use_container_width=True)
                            st.markdown("<div class='expl-box'><b>Abriendo la 'Caja Negra':</b> El modelo separo tu oracion en tokens. Las barras verdes indican que esa palabra suma puntos hacia lo positivo; las rojas indican que la palabra resta puntos hacia lo negativo. Asi es exactamente como la IA llega al veredicto final.</div>", unsafe_allow_html=True)
                        else:
                            st.info("El modelo no identifico palabras de peso en su vocabulario pre-entrenado.")
                else:
                    st.warning("Por favor ingresa un texto para analizar.")

except FileNotFoundError as e:
    st.error(f"No se encontraron los archivos del modelo: {e}")

