import streamlit as st
import pandas as pd
import plotly.express as px
from prophet import Prophet
import io

# Configuración
st.set_page_config(page_title="BeautyFlow AI", layout="centered")
st.markdown("<h1 style='text-align: center; color: #ff69b4;'>BeautyFlow AI</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>El ERP ligero que predice tu próximo best-seller</p>", unsafe_allow_html=True)

# Subir archivo
uploaded_file = st.file_uploader("Sube tu CSV de ventas (fecha, producto, unidades)", type="csv")

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        st.success("¡Datos cargados!")
        
        # Mostrar datos
        st.write("### Tus ventas")
        st.dataframe(df.head(10))
        
        # Gráfico
        if 'fecha' in df.columns and 'unidades' in df.columns:
            fig = px.line(df, x='fecha', y='unidades', color='producto', title="Ventas por producto")
            st.plotly_chart(fig)
        
        # Predicción
        if st.button("Predecir 2026"):
            with st.spinner("Calculando tu futuro..."):
                pred_df = df[['fecha', 'unidades']].copy()
                pred_df['fecha'] = pd.to_datetime(pred_df['fecha'])
                pred_df = pred_df.rename(columns={'fecha': 'ds', 'unidades': 'y'})
                m = Prophet(yearly_seasonality=True, weekly_seasonality=True)
                m.fit(pred_df)
                future = m.make_future_dataframe(periods=365)
                forecast = m.predict(future)
                st.success("¡Predicción 2026 lista!")
                st.line_chart(forecast[['ds', 'yhat']].set_index('ds'))
                top_pred = forecast['yhat'].iloc[-30:].mean()
                st.markdown(f"### Predicción promedio 2026: **{int(top_pred):,} unidades/mes**")
                
    except Exception as e:
        st.error("Error en el CSV. Usa: fecha,producto,unidades")
else:
    st.info("Sube un CSV para empezar. Ejemplo:")
    example = pd.DataFrame({
        'fecha': ['2025-01-01', '2025-01-02', '2025-01-03'],
        'producto': ['Labial Berry', 'Labial Nude', 'Labial Berry'],
        'unidades': [120, 80, 140]
    })
    st.dataframe(example)
    st.download_button("Descargar ejemplo CSV", example.to_csv(index=False), "ejemplo_ventas.csv")
