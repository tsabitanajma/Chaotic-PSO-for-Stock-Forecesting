# ============================================
# IMPORTS
# ============================================

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import warnings
import sys
import traceback

# SET PAGE CONFIG HARUS DULUAN SETELAH IMPORT STREAMLIT
st.set_page_config(
    page_title="Prediksi Saham BRIS",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# INISIALISASI VARIABLE DI LUAR TRY-EXCEPT
PLOTLY_AVAILABLE = False

# IMPORT PLOTLY DENGAN FALLBACK KE MATPLOTLIB
try:
    import plotly.graph_objects as go
    import plotly.express as px
    PLOTLY_AVAILABLE = True
    PLOTLY_LOADED = True
except ImportError as e:
    PLOTLY_LOADED = False
    st.warning(f"⚠️ Plotly tidak tersedia: {e}. Menggunakan matplotlib sebagai fallback.")
    # Fallback ke matplotlib
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use('Agg')

warnings.filterwarnings('ignore')

# ============================================
# CUSTOM CSS - BSI THEME
# ============================================
st.markdown("""
<style>
    /* Warna Theme BSI */
    :root {
        --bsi-green: #00A651;
        --bsi-orange: #F37021;
        --bsi-dark: #1A3A2A;
        --bsi-light: #F5F9F7;
    }
    
    /* Main container */
    .main {
        background-color: var(--bsi-light);
    }
    
    /* Headers - FIXED */
    .main-header {
        color: var(--bsi-green) !important;
        font-weight: 700 !important;
        border-bottom: 3px solid var(--bsi-orange);
        padding-bottom: 10px;
        margin-bottom: 20px !important;
        font-size: 2.5rem !important;
    }
    
    .sub-header {
        color: var(--bsi-dark) !important;
        font-weight: 600 !important;
        margin-top: 10px !important;
        margin-bottom: 20px !important;
    }
    
    /* Sidebar */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, var(--bsi-green), var(--bsi-dark));
        color: white;
    }
    
    [data-testid="stSidebar"] .sidebar-content {
        color: white !important;
    }
    
    /* Buttons */
    .stButton > button {
        background: linear-gradient(90deg, var(--bsi-orange), #FF8C42);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 10px 24px;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(243, 112, 33, 0.3);
    }
    
    /* Metric cards */
    [data-testid="stMetric"] {
        background: linear-gradient(135deg, #ffffff, #f8fdf9);
        border-radius: 12px;
        padding: 20px;
        border-left: 5px solid var(--bsi-orange);
        box-shadow: 0 4px 12px rgba(0,0,0,0.05);
        margin: 15px 0 !important;
    }
    
    [data-testid="stMetricValue"] {
        color: var(--bsi-green) !important;
        font-size: 28px !important;
        font-weight: 700 !important;
    }
    
    [data-testid="stMetricLabel"] {
        color: var(--bsi-dark) !important;
        font-weight: 600 !important;
    }
    
    /* Badges */
    .badge-container {
        display: flex;
        gap: 10px;
        margin: 10px 0;
        flex-wrap: wrap;
    }
    
    .badge {
        display: inline-block;
        padding: 8px 16px;
        border-radius: 20px;
        font-size: 14px;
        font-weight: 600;
        margin: 5px;
    }
    
    .badge-green { 
        background-color: #D4F4E5; 
        color: var(--bsi-green);
        border: 1px solid var(--bsi-green);
    }
    
    .badge-orange { 
        background-color: #FFE8D9; 
        color: var(--bsi-orange);
        border: 1px solid var(--bsi-orange);
    }
    
    /* Custom cards */
    .input-card {
        background: white;
        border-radius: 12px;
        padding: 25px;
        border: 2px solid var(--bsi-green);
        box-shadow: 0 4px 12px rgba(0,0,0,0.05);
        margin-bottom: 20px;
    }
    
    .result-card {
        background: linear-gradient(135deg, #ffffff, #f8fdf9);
        border-radius: 12px;
        padding: 25px;
        border-left: 5px solid var(--bsi-orange);
        box-shadow: 0 6px 15px rgba(0,0,0,0.08);
        margin: 20px 0;
    }
    
    /* Footer Styling */
    .footer-container {
        text-align: center;
        padding: 25px 20px;
        color: var(--bsi-dark);
        font-size: 14px;
        border-top: 2px solid var(--bsi-green);
        margin-top: 40px;
        background: linear-gradient(90deg, #f8fdf9, white);
        border-radius: 0 0 10px 10px;
    }
    
    .footer-grid {
        display: flex;
        justify-content: center;
        gap: 40px;
        margin-bottom: 20px;
        flex-wrap: wrap;
    }
    
    .footer-item {
        text-align: center;
    }
    
    .footer-label {
        font-weight: 700;
        color: #00A651;
        font-size: 16px;
        margin-bottom: 5px;
    }
    
    .footer-value {
        color: #1A3A2A;
        font-size: 14px;
    }
    
    .separator-line {
        height: 3px;
        background: linear-gradient(90deg, #00A651, #F37021);
        width: 120px;
        margin: 15px auto;
        border-radius: 3px;
    }
    
    /* Spacing fix */
    .st-emotion-cache-16txtl3 {
        padding-top: 2rem !important;
    }
    
    /* Fix for header spacing */
    h1, h2, h3 {
        margin-top: 0.5rem !important;
        margin-bottom: 1rem !important;
    }
    
    /* Label styling */
    .input-label {
        font-weight: 600;
        color: var(--bsi-dark);
        margin-bottom: 8px;
        display: block;
    }

    /* Fix untuk expander */
    .st-expander {
        margin-top: 20px !important;
        margin-bottom: 20px !important;
    }
    
    .st-expander > div {
        padding: 15px !important;
    }
    
    /* Fix untuk list dalam markdown */
    ul {
        margin-top: 8px !important;
        margin-bottom: 8px !important;
        padding-left: 20px !important;
    }
    
    li {
        margin-bottom: 5px !important;
    }
    
    /* Fix untuk container */
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }

</style>
""", unsafe_allow_html=True)

# ============================================
# LOAD MODEL FUNGSI
# ============================================
@st.cache_resource
def load_model():
    try:
        with open('model_cpso.pkl', 'rb') as f:
            model = pickle.load(f)
        with open('scaler_x.pkl', 'rb') as f:
            scaler_x = pickle.load(f)
        with open('scaler_y.pkl', 'rb') as f:
            scaler_y = pickle.load(f)
        return model, scaler_x, scaler_y
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None, None

def predict_price(model, scaler_x, scaler_y, features):
    """Prediksi harga dengan model"""
    try:
        features_scaled = scaler_x.transform(features)
        pred_scaled = model.predict(features_scaled)
        pred_price = float(scaler_y.inverse_transform(pred_scaled.reshape(-1, 1)))
        return pred_price
    except Exception as e:
        st.error(f"Error prediksi: {e}")
        return None

def create_bar_chart_matplotlib(current_price, predicted_price):
    """Create bar chart using matplotlib as fallback"""
    # Pastikan matplotlib sudah diimport
    if not PLOTLY_LOADED:
        import matplotlib.pyplot as plt
        import matplotlib
        matplotlib.use('Agg')
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    categories = ['Hari Ini', 'Prediksi Besok']
    values = [current_price, predicted_price]
    colors = ['#00A651', '#F37021']
    
    bars = ax.bar(categories, values, color=colors, width=0.6, edgecolor='white', linewidth=2)
    
    # Add value labels on top of bars
    for bar, value in zip(bars, values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'Rp {value:,.0f}', ha='center', va='bottom', 
                fontsize=12, fontweight='bold')
    
    # Customize chart
    ax.set_ylabel('Harga (IDR)', fontsize=12, fontweight='bold')
    ax.set_title('Perbandingan Harga', fontsize=16, fontweight='bold', pad=20)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'Rp {x:,.0f}'))
    ax.grid(axis='y', alpha=0.3)
    ax.set_axisbelow(True)
    
    plt.tight_layout()
    return fig

# ============================================
# SIDEBAR
# ============================================
with st.sidebar:
    # Logo dan Judul Sidebar
    st.markdown("""
    <div style='text-align: center; padding: 15px 0;'>
        <div style='
            background: linear-gradient(135deg, #00A651, #F37021);
            width: 70px;
            height: 70px;
            border-radius: 15px;
            display: flex;
            align-items: center;
            justify-content: center;
            margin: 0 auto 15px auto;
        '>
            <span style='font-size: 35px; color: white;'>📈</span>
        </div>
        <h3 style='color: white; margin: 0;'>BSI Stock Predictor</h3>
        <p style='color: rgba(255,255,255,0.8); font-size: 14px; margin-top: 5px;'>
        Prediksi Cerdas Saham BRIS
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Cara Penggunaan
    with st.expander("📖 Cara Penggunaan", expanded=True):
        st.markdown("""
        **Langkah-langkah:**
        1. **Input Data** - Masukkan data OHLCV hari ini
        2. **Klik Prediksi** - Tekan tombol prediksi
        3. **Lihat Hasil** - Sistem akan menampilkan prediksi
        
        **Keterangan:**
        - **O** = Open (Harga pembukaan)
        - **H** = High (Harga tertinggi)
        - **L** = Low (Harga terendah)
        - **C** = Close (Harga penutupan)
        - **V** = Volume (Jumlah transaksi)
        """)
    
    st.markdown("---")
    
    # Informasi Model
    st.markdown("**⚙️ Model Details**")
    st.markdown("""
    - **Algorithm**: XGBoost-CPSO
    - **Features**: 5 parameters
    - **Training**: 2021-2025 data
    """)
    
    st.markdown("---")
    
    # Disclaimer
    st.warning("""
    **⚠️ Disclaimer:**
    Prediksi ini untuk tujuan edukasi.
    Keputusan investasi adalah tanggung jawab investor.
    """)

# ============================================
# HEADER UTAMA
# ============================================

# Logo dan judul dalam satu baris
col_logo, col_title = st.columns([1, 4])

with col_logo:
    st.markdown("""
    <div style='text-align: center;'>
        <div style='
            background: linear-gradient(135deg, #00A651, #F37021);
            width: 80px;
            height: 80px;
            border-radius: 20px;
            display: flex;
            align-items: center;
            justify-content: center;
            margin-top: 10px;
        '>
            <span style='font-size: 40px; color: white;'>📈</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

with col_title:
    st.markdown("<h1 class='main-header'>PREDIKSI HARGA SAHAM BRIS</h1>", unsafe_allow_html=True)

# Badges dan deskripsi di bawah judul
st.markdown("""
<div class='badge-container'>
    <div class='badge badge-green'>XGBoost</div>
    <div class='badge badge-orange'>Chaotic PSO</div>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<p style='font-size: 18px; color: var(--bsi-dark); margin-bottom: 30px;'>
Aplikasi prediksi harga saham BRIS berbasis model machine learning XGBoost-Chaotic PSO
</p>
""", unsafe_allow_html=True)

st.markdown("---")

# ============================================
# LOAD MODEL
# ============================================
model, scaler_x, scaler_y = load_model()

if model is None:
    st.error("❌ Model tidak dapat dimuat. Pastikan file model (.pkl) tersedia.")
    st.stop()

# ============================================
# FORM INPUT MANUAL
# ============================================
st.markdown("### 📊 Input Data Saham Hari Ini")

with st.container():
    st.markdown("<div class='input-card'>", unsafe_allow_html=True)
    
    st.markdown("**Masukkan data OHLCV saham BRIS untuk hari ini:**")
    
    # Grid untuk input fields
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.markdown("<span class='input-label'>💰 Open Price</span>", unsafe_allow_html=True)
        open_price = st.number_input(
            "Harga Pembukaan (IDR)",
            min_value=0.0,
            value=1000.0,
            step=10.0,
            label_visibility="collapsed",
            key="open_manual"
        )
        st.markdown(f"**Rp {open_price:,.0f}**")
    
    with col2:
        st.markdown("<span class='input-label'>📈 High Price</span>", unsafe_allow_html=True)
        high_price = st.number_input(
            "Harga Tertinggi (IDR)",
            min_value=0.0,
            value=1050.0,
            step=10.0,
            label_visibility="collapsed",
            key="high_manual"
        )
        st.markdown(f"**Rp {high_price:,.0f}**")
    
    with col3:
        st.markdown("<span class='input-label'>📉 Low Price</span>", unsafe_allow_html=True)
        low_price = st.number_input(
            "Harga Terendah (IDR)",
            min_value=0.0,
            value=980.0,
            step=10.0,
            label_visibility="collapsed",
            key="low_manual"
        )
        st.markdown(f"**Rp {low_price:,.0f}**")
    
    with col4:
        st.markdown("<span class='input-label'>💵 Close Price</span>", unsafe_allow_html=True)
        close_price = st.number_input(
            "Harga Penutupan (IDR)",
            min_value=0.0,
            value=1025.0,
            step=10.0,
            label_visibility="collapsed",
            key="close_manual"
        )
        st.markdown(f"**Rp {close_price:,.0f}**")
    
    with col5:
        st.markdown("<span class='input-label'>📊 Volume</span>", unsafe_allow_html=True)
        volume = st.number_input(
            "Volume (lot)",
            min_value=0,
            value=1500000,
            step=10000,
            label_visibility="collapsed",
            key="volume_manual"
        )
        st.markdown(f"**{volume:,.0f} lot**")
    
    st.markdown("</div>", unsafe_allow_html=True)

# Tombol prediksi
st.markdown("<br>", unsafe_allow_html=True)
col_btn1, col_btn2, col_btn3 = st.columns([1, 2, 1])
with col_btn2:
    predict_btn = st.button(
        "🚀 **LAKUKAN PREDIKSI**",
        type="primary",
        use_container_width=True,
        help="Klik untuk memprediksi harga saham besok"
    )

# ============================================
# HASIL PREDIKSI
# ============================================
if predict_btn:
    with st.spinner("🧠 Memproses prediksi..."):
        features = np.array([[open_price, high_price, low_price, close_price, volume]])
        predicted_price = predict_price(model, scaler_x, scaler_y, features)
        
        if predicted_price:
            # Hitung perubahan
            change = predicted_price - close_price
            pct_change = (change / close_price) * 100
            
            # Tampilkan hasil prediksi
            st.markdown("---")
            st.markdown("## 📊 Hasil Prediksi")
            
            # Metrics cards
            st.markdown("<div class='result-card'>", unsafe_allow_html=True)
            
            col_result1, col_result2, col_result3 = st.columns(3)
            
            with col_result1:
                st.metric(
                    label="Harga Hari Ini",
                    value=f"Rp {close_price:,.0f}",
                    help="Harga penutupan yang diinput"
                )
            
            with col_result2:
                st.metric(
                    label="Prediksi Besok",
                    value=f"Rp {predicted_price:,.0f}",
                    delta=f"Rp {change:+,.0f}",
                    help="Prediksi harga penutupan untuk besok"
                )
            
            with col_result3:
                # Tampilkan dengan icon yang sesuai
                if pct_change > 0:
                    icon = "📈"
                    color = "normal"
                else:
                    icon = "📉"
                    color = "inverse"
                
                st.metric(
                    label="Perubahan",
                    value=f"{pct_change:+.2f}%",
                    delta=icon,
                    delta_color=color,
                    help="Persentase perubahan dari harga hari ini"
                )
            
            st.markdown("</div>", unsafe_allow_html=True)
            
            # Visualisasi perbandingan
            st.markdown("### 📈 Visualisasi Perbandingan")
            
            # Gunakan PLOTLY_LOADED untuk cek apakah plotly berhasil diimport
            if PLOTLY_LOADED:
                # Bar chart dengan Plotly
                fig_bar = go.Figure(data=[
                    go.Bar(
                        name='Hari Ini',
                        x=['Harga'],
                        y=[close_price],
                        marker_color='#00A651',
                        width=0.4,
                        text=[f'Rp {close_price:,.0f}'],
                        textposition='outside'
                    ),
                    go.Bar(
                        name='Prediksi Besok',
                        x=['Prediksi'],
                        y=[predicted_price],
                        marker_color='#F37021',
                        width=0.4,
                        text=[f'Rp {predicted_price:,.0f}'],
                        textposition='outside'
                    )
                ])
                
                fig_bar.update_layout(
                    title='Perbandingan Harga',
                    yaxis_title='Harga (IDR)',
                    template='plotly_white',
                    height=400,
                    showlegend=True,
                    bargap=0.5
                )
                fig_bar.update_yaxes(tickprefix='Rp ')
                
                st.plotly_chart(fig_bar, use_container_width=True)
            else:
                # Fallback ke matplotlib
                fig = create_bar_chart_matplotlib(close_price, predicted_price)
                st.pyplot(fig)
            
            # Detail hasil
            with st.expander("📋 Detail Hasil Prediksi", expanded=True):
                col_detail1, col_detail2 = st.columns(2)
                
                with col_detail1:
                    st.markdown("**📥 Data Input:**")
                    st.markdown(f"""
                    - **Open**: Rp {open_price:,.0f}
                    - **High**: Rp {high_price:,.0f}
                    - **Low**: Rp {low_price:,.0f}
                    - **Close**: Rp {close_price:,.0f}
                    - **Volume**: {volume:,.0f} lot
                    """)
                
                with col_detail2:
                    st.markdown("**📤 Hasil Prediksi:**")
                    st.markdown(f"""
                    - **Prediksi**: Rp {predicted_price:,.0f}
                    - **Perubahan**: Rp {change:+,.0f}
                    - **% Perubahan**: {pct_change:+.2f}%
                    - **Model**: XGBoost-CPSO
                    """)

# ============================================
# FOOTER - VERSI FIXED
# ============================================
import streamlit.components.v1 as components

components.html("""
<style>
.footer {
    font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI",
                 Roboto, "Helvetica Neue", Arial, sans-serif;
    color: #6b7280;
    font-size: 13px;
    text-align: center;
    padding: 16px 8px;
    border-top: 1px solid #e5e7eb;
    margin-top: 32px;
}

.footer strong {
    color: #374151;
}

.footer small {
    font-size: 12px;
}
</style>

<div class="footer">
    <strong>BSI Stock Predictor</strong> — XGBoost + Chaotic PSO<br>
    <small>Educational Purpose • © 2024</small>
</div>
""", height=120)


# ============================================
# REFRESH BUTTON
# ============================================
with st.sidebar:
    st.markdown("---")
    if st.button("🔄 Reset Form", type="secondary", use_container_width=True):
        st.rerun()

# ============================================
# DEBUG INFO (Optional - bisa dihapus setelah deploy sukses)
# ============================================
with st.sidebar:
    with st.expander("ℹ️ Debug Info", expanded=False):
        st.write(f"Python: {sys.version.split()[0]}")
        st.write(f"Streamlit: {st.__version__}")
        st.write(f"Plotly Loaded: {PLOTLY_LOADED}")
