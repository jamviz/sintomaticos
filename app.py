import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler

# Configuración de la página
st.set_page_config(page_title="Predicción de Enfermedades Respiratorias en Ilo", layout="wide")

# Carga de datos y modelos
@st.cache_resource
def load_model():
    return joblib.load('linear_regression_model.joblib')

@st.cache_resource
def load_scaler():
    return joblib.load('scaler.joblib')

@st.cache_data
def load_data():
    return pd.read_csv("datos_casosRespiratorios.csv")

model = load_model()
scaler = load_scaler()
data = load_data()

# Función para crear predicciones
def make_prediction(input_data):
    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)
    return prediction[0]

# Función para mostrar gráficos
def show_graphs():
    st.subheader("Visualización de Datos")

    # Mapa de calor de correlación
    st.write("Mapa de Calor de Correlación")
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(data.corr(), annot=True, cmap="BuPu", ax=ax)
    st.pyplot(fig)

    # Gráficos de dispersión
    st.write("Gráficos de Dispersión")
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    sns.scatterplot(x=data["Cantidad_Sintomaticos"], y=data["H2S_24h"], ax=axes[0])
    axes[0].set_title('H2S vs Casos Sintomáticos')
    sns.scatterplot(x=data["Cantidad_Sintomaticos"], y=data["SO2_24h"], ax=axes[1])
    axes[1].set_title('SO2 vs Casos Sintomáticos')
    sns.scatterplot(x=data["Cantidad_Sintomaticos"], y=data["CO"], ax=axes[2])
    axes[2].set_title('CO vs Casos Sintomáticos')
    st.pyplot(fig)

    # Explicación de los gráficos
    st.write("""
    **Interpretación de los Gráficos:**
    
    1. **Mapa de Calor de Correlación:** Este gráfico muestra la fuerza de la relación entre las diferentes variables. 
       Colores más oscuros indican una correlación más fuerte.
    
    2. **Gráficos de Dispersión:** Estos gráficos muestran la relación entre cada contaminante y la cantidad de casos sintomáticos.
    """)

    st.markdown("""
        ### 1. H₂S vs Casos Sintomáticos

        El gráfico de dispersión para H₂S muestra una tendencia positiva moderada:

        - **Correlación**: Se observa una correlación positiva, indicando que un aumento en los niveles de H₂S tiende a asociarse con un incremento en los casos sintomáticos.
        - **Dispersión**: La nube de puntos muestra una dispersión considerable, lo que sugiere la presencia de variabilidad no explicada solo por H₂S.
        - **No linealidad**: Hay indicios de una ligera no linealidad, especialmente en concentraciones más altas, lo que justifica la exploración de características polinomiales en nuestro modelo.
        
        **Implicaciones para el modelo**: La relación moderadamente fuerte sugiere que H₂S es un predictor importante. La no linealidad observada respalda la decisión de utilizar características polinomiales para capturar esta complejidad.

        ### 2. SO₂ vs Casos Sintomáticos

        El gráfico de SO₂ exhibe la tendencia positiva más pronunciada:

        - **Correlación**: Muestra una fuerte correlación positiva, indicando que SO₂ podría ser el predictor más influyente en nuestro modelo.
        - **Heterocedasticidad**: Se observa un aumento en la variabilidad de los casos sintomáticos a medida que aumentan los niveles de SO₂, lo que sugiere heterocedasticidad.
        - **Cluster**: Hay una concentración de puntos en niveles bajos de SO₂, lo que podría afectar la estimación de coeficientes en el modelo lineal.
        - **Efecto umbral**: Parece haber un "efecto umbral" donde los casos sintomáticos aumentan más rápidamente después de cierto nivel de SO₂.

        **Implicaciones para el modelo**: La fuerte correlación justifica la inclusión de SO₂ como predictor clave. La heterocedasticidad observada podría requerir técnicas de regularización como LASSO (que vemos implementado en el código) para mejorar la robustez del modelo.

        ### 3. CO vs Casos Sintomáticos

        El gráfico de CO muestra la relación menos clara de los tres contaminantes:

        - **Correlación débil**: La tendencia positiva es menos pronunciada, sugiriendo una correlación más débil con los casos sintomáticos.
        - **Alta dispersión**: La nube de puntos es más dispersa, indicando una relación más compleja o la influencia de otros factores no capturados.
        - **Clusters**: Se observan algunos agrupamientos de puntos, lo que podría indicar la presencia de subpoblaciones o efectos estacionales.

        **Implicaciones para el modelo**: Aunque la relación es más débil, la inclusión de CO en el modelo multivariable podría capturar interacciones importantes con otros contaminantes.
        """)

# Interfaz de usuario
def main():
    st.sidebar.title("Navegación")
    page = st.sidebar.radio("Ir a", ["Introducción", "Predicción", "Análisis de Datos"])

    if page == "Introducción":
        st.image("caratula.png", 
                caption="Sistema de predicción de casos sintomáticos de enfermedades respiratorias",
                use_column_width=True)
        
        st.write("""
        ## Bienvenido a nuestra aplicación
        
        Esta herramienta innovadora utiliza datos históricos sobre la contaminación del aire 
        para estimar el número de casos sintomáticos de enfermedades respiratorias en Ilo.
        """)
        
        st.markdown("""
        ### Navegación
        
        Utilice la barra lateral para explorar las diferentes secciones de la aplicación:
        
        - **Predicción**: Ingrese los niveles actuales de contaminantes para obtener una predicción.
        - **Análisis de Datos**: Explore visualizaciones de los datos históricos y su interpretación.
        """)
        
        st.markdown("""
        ---
        ### Créditos
        
        Este proyecto fue desarrollado por:
        
        - Jamir Balcona
        - Carlos Mamani
        - Ivan Ccaso
        - Gabriela
        
        © 2024 Todos los derechos reservados
        """)

        st.warning("""
        **Nota Importante**: Esta aplicación es solo para fines educativos y de demostración. 
        No debe utilizarse como único recurso para tomar decisiones médicas o de salud pública.
        """)
    elif page == "Predicción":
        st.title("Predicción de Casos Sintomáticos")
        st.write("Ingrese los niveles actuales de contaminantes para obtener una predicción.")

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Formulario de Entrada")
            h2s = st.number_input('H2S (24h)', min_value=0.0, max_value=100.0, value=10.0)
            so2 = st.number_input('SO2 (24h)', min_value=0.0, max_value=100.0, value=20.0)
            co = st.number_input('CO', min_value=0.0, max_value=1000.0, value=200.0)

        with col2:
            st.subheader("Ajuste Fino")
            h2s = st.slider('H2S (24h)', 0.0, 100.0, h2s)
            so2 = st.slider('SO2 (24h)', 0.0, 100.0, so2)
            co = st.slider('CO', 0.0, 1000.0, co)

        if st.button('Realizar Predicción'):
            input_df = pd.DataFrame({'H2S_24h': [h2s], 'SO2_24h': [so2], 'CO': [co]})
            prediction = make_prediction(input_df)
            st.success(f'El número estimado de casos sintomáticos es: {prediction:.2f}')

            # Mostrar gráficos después de la predicción
            show_graphs()

    elif page == "Análisis de Datos":
        st.title("Análisis de Datos Históricos")
        show_graphs()

if __name__ == "__main__":
    main()