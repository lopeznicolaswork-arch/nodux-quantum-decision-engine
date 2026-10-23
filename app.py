import streamlit as st
import pandas as pd
import json
from google import genai
from google.genai import types

# Motores de Optimización Cuántica de IBM Qiskit
from qiskit_optimization import QuadraticProgram
from qiskit_optimization.converters import QuadraticProgramToQubo
from qiskit_optimization.algorithms import MinimumEigenOptimizer
from qiskit_algorithms import NumPyMinimumEigensolver

st.set_page_layout(page_title="NODUX Quantum Decision Engine", layout="wide")

# Estilos visuales limpios
st.markdown("""
    <style>
    .main-title { font-size: 28px; font-weight: 800; color: #1E293B; }
    .sub-title { font-size: 14px; color: #64748B; margin-bottom: 20px; }
    </style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">NODUX // Quantum-AI Decision Engine</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Pipeline de Optimización Combinatoria B2B: Ingesta Caótica ➔ Parser IA ➔ Solver Qiskit ➔ Dictamen Ejecutivo</div>', unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 1. CONFIGURACIÓN DE CREDENCIALES (SIDEBAR)
# -----------------------------------------------------------------------------
with st.sidebar:
    st.header("⚙️ Configuración")
    gemini_key = st.text_input("Ingresa tu Gemini API Key (AI Studio):", type="password")
    st.caption("Obtén tu clave gratuita en [aistudio.google.com](https://aistudio.google.com/)")
    
    st.divider()
    st.markdown("**Arquitectura:**")
    st.markdown("• **Parser:** Gemini 2.5 Flash")
    st.markdown("• **Solver:** Qiskit Optimization (QUBO)")
    st.markdown("• **Auditor:** Gemini Executive Mode")

if not gemini_key:
    st.info("👋 Por favor, ingresa tu API Key de Gemini en la barra lateral izquierda para activar la plataforma.")
    st.stop()

# Inicializar cliente de Google AI Studio
client = genai.Client(api_key=gemini_key)

# -----------------------------------------------------------------------------
# 2. INGESTA DE DATOS (EXCEL, CSV O TEXTO)
# -----------------------------------------------------------------------------
st.subheader("1. Ingesta de Datos No Estructurados")

tab1, tab2 = st.tabs(["📝 Pegar Texto / Memorándum", "📁 Subir Archivo Excel / CSV"])
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
        df_uploaded = pd.read_excel(uploaded_file) if uploaded_file.name.endswith(('xlsx', 'xls')) else pd.read_csv(uploaded_file)
        st.write("Vista previa del archivo cargado:", df_uploaded.head(3))
        input_data = df_uploaded.to_string()

# -----------------------------------------------------------------------------
# 3. EJECUCIÓN DEL PIPELINE COMPLETO
# -----------------------------------------------------------------------------
if st.button("🚀 Procesar Datos y Resolver con Motor Cuántico", type="primary"):
    if not input_data.strip():
        st.error("No hay datos de entrada para procesar.")
        st.stop()

    # FASE A: PARSER CON GEMINI
    with st.spinner("🤖 Fase 1/3: La IA está analizando el texto y estructurando las variables..."):
        prompt_parser = f"""
        Eres un ingeniero de datos especializado en optimización matemática.
        Analiza el siguiente texto de un negocio y extrae:
        1. El presupuesto límite o restricción presupuestaria máxima.
        2. La lista de proyectos, rutas o ítems con su nombre, su costo financiero y su beneficio/retorno (ROI).

        TEXTO DEL NEGOCIO:
        \"\"\"{input_data}\"\"\"

        Devuelve ESTRICTAMENTE un JSON con esta estructura exacta, sin código markdown extra:
        {{
            "presupuesto_maximo": float,
            "items": [
                {{"nombre": str, "costo": float, "beneficio": float}}
            ]
        }}
        """
        try:
            resp_parser = client.models.generate_content(
                model='gemini-2.5-flash',
                contents=prompt_parser,
                config=types.GenerateContentConfig(response_mime_type="application/json")
            )
            parsed_json = json.loads(resp_parser.text)
            presupuesto_limite = float(parsed_json["presupuesto_maximo"])
            items = parsed_json["items"]
            n = len(items)
        except Exception as e:
            st.error(f"Error extrayendo datos con la IA: {e}")
            st.stop()

    st.success(f"✓ Estructuración exitosa: {n} variables detectadas. Presupuesto límite: S/ {presupuesto_limite:,.2f}")
    
    # Mostrar tabla limpia
    df_items = pd.DataFrame(items)
    df_items["ROI %"] = ((df_items["beneficio"] - df_items["costo"]) / df_items["costo"] * 100).round(1)
    st.dataframe(df_items, use_container_width=True)

    # FASE B: RESOLUCIÓN CON QISKIT (QUBO)
    with st.spinner("⚛️ Fase 2/3: Mapeando restricciones a QUBO y resolviendo con Qiskit..."):
        qp = QuadraticProgram("Nodux_Quantum_Pipeline")
        for i in range(n):
            qp.binary_var(name=f"x_{i}")

        # Función objetivo: Maximizar beneficios
        qp.maximize(linear={f"x_{i}": items[i]["beneficio"] for i in range(n)})
        
        # Restricción: Costo total <= Presupuesto límite
        qp.linear_constraint(
            linear={f"x_{i}": items[i]["costo"] for i in range(n)},
            sense="<=", rhs=presupuesto_limite, name="presupuesto"
        )

        # Conversión matemática a QUBO
        conv = QuadraticProgramToQubo()
        qubo = conv.convert(qp)

        # Resolución algorítmica
        solver = MinimumEigenOptimizer(NumPyMinimumEigensolver())
        result = solver.solve(qubo)
        raw_binary = [int(round(val)) for val in conv.interpret(result)]

        seleccionados = [items[i] for i in range(n) if raw_binary[i] == 1]
        descartados = [items[i] for i in range(n) if raw_binary[i] == 0]
        costo_total = sum(it["costo"] for it in seleccionados)
        beneficio_total = sum(it["beneficio"] for it in seleccionados)
        ganancia_neta = beneficio_total - costo_total

    # Mostrar métricas del motor
    st.subheader("2. Resultados del Motor Cuántico")
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    kpi1.metric("Capital Utilizado", f"S/ {costo_total:,.2f}", delta=f"Límite: S/ {presupuesto_limite:,.2f}")
    kpi2.metric("Retorno Proyectado", f"S/ {beneficio_total:,.2f}")
    kpi3.metric("Margen Neto Creado", f"+S/ {ganancia_neta:,.2f}")
    kpi4.metric("Vector Binario |x⟩", f"|{', '.join(map(str, raw_binary))}⟩")

    # FASE C: DICTAMEN EJECUTIVO CON GEMINI
    with st.spinner("📄 Fase 3/3: Gemini redactando informe de auditoría para el directorio..."):
        prompt_informe = f"""
        Eres un consultor senior de estrategia empresarial (estilo Harvard Business / McKinsey).
        Redacta un Dictamen Ejecutivo de Asignación Estratégica (Reporte DOC-03) formal basado en la optimización:
        
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
        resp_informe = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=prompt_informe
        )

    st.subheader("3. Dictamen Ejecutivo de Directorio (DOC-03)")
    st.markdown(resp_informe.text)