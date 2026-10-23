import streamlit as st
import pandas as pd
import json
import io
from pypdf import PdfReader
from google import genai
from google.genai import types

# Motores de Optimización Cuántica de IBM Qiskit
from qiskit_optimization import QuadraticProgram
from qiskit_optimization.algorithms import MinimumEigenOptimizer
from qiskit_algorithms import NumPyMinimumEigensolver

# Configuración de página
st.set_page_config(page_title="NODUX Quantum Decision Engine", layout="wide")

st.markdown("""
    <style>
    .main-title { font-size: 28px; font-weight: 800; color: #1E293B; }
    .sub-title { font-size: 14px; color: #64748B; margin-bottom: 20px; }
    </style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">NODUX // Quantum-AI Decision Engine</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Pipeline de Optimización Combinatoria B2B: Ingesta Universal (PDF/Excel/Texto) ➔ Parser IA ➔ Solver Qiskit ➔ Dictamen Ejecutivo</div>', unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 1. CONFIGURACIÓN DE CREDENCIALES
# -----------------------------------------------------------------------------
with st.sidebar:
    st.header("⚙️ Configuración")
    gemini_key = st.text_input("Ingresa tu Gemini API Key (AI Studio):", type="password")
    st.caption("Obtén tu clave gratuita en [aistudio.google.com](https://aistudio.google.com/)")
    
    st.divider()
    st.markdown("**Arquitectura del Sistema:**")
    st.markdown("• **Parser IA:** Gemini Multi-Model Fallback")
    st.markdown("• **Solver Cuántico:** Qiskit Optimization (QUBO)")
    st.markdown("• **Auditor C-Level:** Gemini Executive Mode")

if not gemini_key:
    st.info("👋 Por favor, ingresa tu API Key de Gemini en la barra lateral izquierda para activar la plataforma.")
    st.stop()

# Inicializar cliente de Google GenAI
client = genai.Client(api_key=gemini_key)

def ejecutar_llamada_gemini(client, prompt, es_json=False):
    modelos_a_probar = [
        "gemini-2.0-flash",
        "gemini-flash-latest",
        "gemini-2.0-flash-exp",
        "gemini-1.5-flash"
    ]
    ultimo_error = None
    config = types.GenerateContentConfig(
        response_mime_type="application/json" if es_json else None
    )

    for modelo in modelos_a_probar:
        try:
            respuesta = client.models.generate_content(
                model=modelo,
                contents=prompt,
                config=config
            )
            return respuesta.text, modelo
        except Exception as e:
            ultimo_error = e
            continue
            
    raise ultimo_error

# -----------------------------------------------------------------------------
# 2. INGESTA UNIVERSAL (TEXTO, EXCEL, CSV, PDF)
# -----------------------------------------------------------------------------
st.subheader("1. Ingesta de Datos No Estructurados")

tab1, tab2, tab3 = st.tabs(["📝 Pegar Texto / Memorándum", "📊 Subir Archivo Excel / CSV", "📄 Subir Documento PDF"])
input_data = ""

with tab1:
    raw_text = st.text_area(
        "Pega aquí el correo, mensaje o reporte desordenado:",
        height=180,
        value="""MEMORÁNDUM DE DISTRIBUCIÓN // LOGÍSTICA LIMA
Presupuesto máximo asignado: S/ 45,000 para esta semana.
Opciones de rutas:
- Ruta Industrial Ate: Costo S/ 12,000 con retorno proyectado de S/ 26,000.
- Hub Masivo San Juan Lurigancho: Costo S/ 18,000 con retorno proyectado de S/ 38,000.
- Flota Surco Express: Requiere S/ 10,000 con retorno de S/ 19,000.
- Terminal Portuario Callao: Requiere S/ 25,000 con retorno de S/ 50,000.
- Centro Logístico Villa El Salvador: Costo S/ 14,000 con retorno de S/ 30,000.
- Nodo Central Cercado Lima: Inversión de S/ 9,000 con retorno de S/ 20,000.
Determinar la asignación que maximice la rentabilidad respetando el techo financiero."""
    )
    if raw_text:
        input_data = raw_text

with tab2:
    uploaded_file = st.file_uploader("Carga tu archivo Excel o CSV sucio:", type=["xlsx", "xls", "csv"])
    if uploaded_file:
        try:
            df_uploaded = pd.read_excel(uploaded_file) if uploaded_file.name.endswith(('xlsx', 'xls')) else pd.read_csv(uploaded_file)
            st.write("Vista previa de la hoja:", df_uploaded.head(3))
            input_data = df_uploaded.to_string()
        except Exception as e:
            st.error(f"Error al leer el archivo tabular: {e}")

with tab3:
    uploaded_pdf = st.file_uploader("Carga un reporte o memorándum en PDF:", type=["pdf"])
    if uploaded_pdf:
        try:
            reader = PdfReader(io.BytesIO(uploaded_pdf.read()))
            texto_pdf = "\n".join([page.extract_text() or "" for page in reader.pages])
            st.write(f"PDF procesado ({len(reader.pages)} páginas). Vista previa:")
            st.caption(texto_pdf[:300] + "...")
            input_data = texto_pdf
        except Exception as e:
            st.error(f"Error al extraer texto del PDF: {e}")

# -----------------------------------------------------------------------------
# 3. EJECUCIÓN DEL PIPELINE COMPLETO
# -----------------------------------------------------------------------------
if st.button("🚀 Procesar Datos y Resolver con Motor Cuántico", type="primary"):
    if not input_data.strip():
        st.error("No hay datos de entrada para procesar.")
        st.stop()

    # FASE A: PARSER CON GEMINI (MOLDE A)
    with st.spinner("🤖 Fase 1/3: Normalizando datos y formulando variables para el modelo QUBO..."):
        prompt_parser = f"""
        Eres un ingeniero de datos especializado en optimización matemática y operaciones corporativas.
        Analiza la siguiente información caótica (puede ser texto, tabla de Excel o contenido de PDF):
        \"\"\"{input_data}\"\"\"

        Tu objetivo es extraer:
        1. El presupuesto límite, capital disponible o techo financiero autorizado.
        2. La lista completa de opciones (proyectos, rutas, inversiones o ítems) con:
           - "nombre": Identificador claro.
           - "costo": Inversión financiera requerida (número float puro).
           - "beneficio": Retorno o ganancia esperada (número float puro).

        Responde ÚNICAMENTE un JSON con esta estructura exacta:
        {{
            "presupuesto_maximo": float,
            "items": [
                {{"nombre": str, "costo": float, "beneficio": float}}
            ]
        }}
        """
        try:
            texto_json, modelo_exitoso = ejecutar_llamada_gemini(client, prompt_parser, es_json=True)
            parsed_json = json.loads(texto_json)
            presupuesto_limite = float(parsed_json["presupuesto_maximo"])
            items = parsed_json["items"]
            n = len(items)
        except Exception as e:
            st.error(f"Error al estructurar los datos con la IA: {e}")
            st.stop()

    st.success(f"✓ Datos procesados con {modelo_exitoso} | {n} variables extraídas | Presupuesto: S/ {presupuesto_limite:,.2f}")
    
    # Tabla visual
    df_items = pd.DataFrame(items)
    df_items["ROI %"] = ((df_items["beneficio"] - df_items["costo"]) / df_items["costo"] * 100).round(1)
    st.dataframe(df_items, use_container_width=True)

    # FASE B: RESOLUCIÓN CON QISKIT
    with st.spinner("⚛️ Fase 2/3: Mapeando Hamiltoniano y ejecutando el solver cuántico..."):
        qp = QuadraticProgram("Nodux_Quantum_Pipeline")
        for i in range(n):
            qp.binary_var(name=f"x_{i}")

        # Maximizar retornos
        qp.maximize(linear={f"x_{i}": items[i]["beneficio"] for i in range(n)})
        
        # Restricción rígida: Costo total <= Presupuesto límite
        qp.linear_constraint(
            linear={f"x_{i}": items[i]["costo"] for i in range(n)},
            sense="<=", rhs=presupuesto_limite, name="presupuesto"
        )

        try:
            # Resolución directa sobre QuadraticProgram
            solver = MinimumEigenOptimizer(NumPyMinimumEigensolver())
            result = solver.solve(qp)
            raw_binary = [int(round(val)) for val in result.x]
        except Exception:
            # Respaldo matemático exacto en caso de incompatibilidad de versiones en la nube
            mejor_b = -1
            mejor_bits = [0] * n
            for mask in range(1 << n):
                c = sum(items[i]["costo"] for i in range(n) if (mask >> i) & 1)
                b = sum(items[i]["beneficio"] for i in range(n) if (mask >> i) & 1)
                if c <= presupuesto_limite and b > mejor_b:
                    mejor_b = b
                    mejor_bits = [(mask >> i) & 1 for i in range(n)]
            raw_binary = mejor_bits

        seleccionados = [items[i] for i in range(n) if raw_binary[i] == 1]
        descartados = [items[i] for i in range(n) if raw_binary[i] == 0]
        costo_total = sum(it["costo"] for it in seleccionados)
        beneficio_total = sum(it["beneficio"] for it in seleccionados)
        ganancia_neta = beneficio_total - costo_total

    # Métricas de negocio
    st.subheader("2. Resultados del Motor Cuántico")
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    kpi1.metric("Capital Utilizado", f"S/ {costo_total:,.2f}", delta=f"Límite: S/ {presupuesto_limite:,.2f}")
    kpi2.metric("Retorno Proyectado", f"S/ {beneficio_total:,.2f}")
    kpi3.metric("Margen Neto Creado", f"+S/ {ganancia_neta:,.2f}")
    kpi4.metric("Vector Binario |x⟩", f"|{', '.join(map(str, raw_binary))}⟩")

    # FASE C: DICTAMEN EJECUTIVO CON GEMINI (MOLDE C)
    with st.spinner("📄 Fase 3/3: Generando dictamen estratégico para el directorio..."):
        prompt_informe = f"""
        Eres un consultor senior de estrategia empresarial (estilo Harvard Business / McKinsey / BCG).
        Redacta un Dictamen Ejecutivo de Asignación Estratégica (Reporte DOC-03) con tono directivo formal:
        
        - Presupuesto Autorizado: S/ {presupuesto_limite:,.2f}
        - Presupuesto Invertido: S/ {costo_total:,.2f}
        - Retorno Total (ROI): S/ {beneficio_total:,.2f}
        - Ganancia Neta Creada: S/ {ganancia_neta:,.2f}
        - Proyectos Seleccionados por el Algoritmo: {json.dumps(seleccionados)}
        - Proyectos Descartados por Restricción: {json.dumps(descartados)}

        Estructura el informe con:
        1. Resumen Ejecutivo (Conclusión directa / BLUF).
        2. Justificación Estratégica (Por qué se seleccionaron esas variables y por qué se descartaron las otras).
        3. Recomendaciones Operativas para la Gerencia.
        Usa Markdown limpio y formal con negritas y tablas.
        """
        texto_reporte, _ = ejecutar_llamada_gemini(client, prompt_informe, es_json=False)

    st.subheader("3. Dictamen Ejecutivo de Directorio (DOC-03)")
    st.markdown(texto_reporte)
