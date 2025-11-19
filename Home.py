from sklearn.neighbors import KNeighborsClassifier
import streamlit as st
import pandas as pd
import numpy as np
import altair as alt

# -------------------- CSS Minimal Pastel --------------------
st.markdown("""
<style>

/* Body */
body {
    background: #fff0f5 !important;
    font-family: 'Segoe UI', sans-serif;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #ffe6f0, #fff0f5);
    color: #333;
    padding: 15px;
    border-radius: 15px;
}

/* Title Card */
.title-card {
    background: #ffccdd;
    padding: 20px;
    border-radius: 20px;
    text-align: center;
    color: #222;
    font-size: 32px;
    font-weight: 600;
    margin-bottom: 25px;
    box-shadow: 0 4px 15px rgba(0,0,0,0.05);
}

/* Section Card */
.section-card {
    background: #ffffffcc;
    padding: 20px;
    border-radius: 20px;
    box-shadow: 0 4px 15px rgba(0,0,0,0.05);
    margin-top: 25px;
    transition: transform 0.2s ease;
}
.section-card:hover {
    transform: scale(1.01);
}

/* Sub Header */
.sub-header {
    font-size: 22px;
    font-weight: 600;
    text-align: center;
    color: #ff6f91;
    margin-bottom: 15px;
}

/* Minimal Button */
.minimal-btn {
    background: #ff9ebc;
    color: white !important;
    padding: 12px 28px;
    border-radius: 25px;
    border: none;
    font-size: 16px;
    box-shadow: 0 2px 6px rgba(255, 120, 160, 0.3);
    transition: 0.3s ease;
}
.minimal-btn:hover {
    box-shadow: 0 2px 12px rgba(255, 120, 160, 0.5);
    cursor: pointer;
    transform: scale(1.05);
}

/* Input Fields */
.stSlider>div>div>div>div { color: #ff6f91 !important; }
.stNumberInput>div>input {
    border-radius: 12px !important;
    border: 1.5px solid #ffb6c1 !important;
}

</style>
""", unsafe_allow_html=True)

# -------------------- Sidebar --------------------
st.sidebar.markdown("## 🌸 Pink Minimal Dashboard")
st.sidebar.image("./img/pin.jpg", width=160)
st.sidebar.markdown("""
**Menu**
- 📌 Home
- 🌼 Flower Samples
- 📊 Statistics
- 🔮 Prediction
""")

# -------------------- Header --------------------
st.markdown('<div class="title-card">🌸 Iris Classification Dashboard 🌸</div>', unsafe_allow_html=True)
st.write("### ยินดีต้อนรับสู่ระบบทำนายสายพันธุ์ดอกไม้ 🩷")

# -------------------- Flower Sample --------------------
st.markdown('<div class="section-card">', unsafe_allow_html=True)
st.markdown('<div class="sub-header">🌼 ตัวอย่างภาพสายพันธุ์ดอกไม้</div>', unsafe_allow_html=True)
col1, col2, col3 = st.columns(3)
with col1: st.image("./img/iris1.jpg", caption="Versicolor")
with col2: st.image("./img/iris2.jpg", caption="Virginica")
with col3: st.image("./img/iris3.jpg", caption="Setosa")
st.markdown('</div>', unsafe_allow_html=True)

# -------------------- Statistics --------------------
st.markdown('<div class="section-card">', unsafe_allow_html=True)
st.markdown('<div class="sub-header">📊 สถิติข้อมูลดอกไม้</div>', unsafe_allow_html=True)
dt = pd.read_csv("./data/iris.csv")
st.write(dt.head(10))

dt_sum = dt.sum().reset_index()
dt_sum.columns = ["Feature", "Total"]

if st.button("📈 แสดงกราฟ", key="chart_minimal"):
    chart = alt.Chart(dt_sum).mark_bar(cornerRadiusTopLeft=5, cornerRadiusTopRight=5).encode(
        x=alt.X('Feature', sort=None),
        y='Total',
        color=alt.Color('Feature', scale=alt.Scale(range=['#ffb6c1','#ff9eac','#ffc0cb','#ff7f9e']))
    ).properties(width=600, height=350)
    st.altair_chart(chart, use_container_width=True)
else:
    st.info("กดปุ่มเพื่อดูกราฟสถิติ")

st.markdown('</div>', unsafe_allow_html=True)

# -------------------- Prediction --------------------
st.markdown('<div class="section-card">', unsafe_allow_html=True)
st.markdown('<div class="sub-header">🔮 ระบบทำนายสายพันธุ์ดอกไม้</div>', unsafe_allow_html=True)
pt_len = st.slider("Petal Length", 0.0, 10.0, 1.0)
pt_wd = st.slider("Petal Width", 0.0, 5.0, 1.0)
sp_len = st.number_input("Sepal Length")
sp_wd = st.number_input("Sepal Width")

if st.button("🔮 ทำนาย", key="predict_minimal"):
    X = dt.drop('variety', axis=1)
    y = dt["variety"]
    model = KNeighborsClassifier(n_neighbors=3)
    model.fit(X, y)

    x_input = np.array([[pt_len, pt_wd, sp_len, sp_wd]])
    result = model.predict(x_input)

    st.success(f"ผลการทำนาย: **{result[0]}** 🌸")
    if result[0] == 'Setosa': st.image("./img/iris3.jpg", caption="Setosa")
    elif result[0] == 'Versicolor': st.image("./img/iris1.jpg", caption="Versicolor")
    else: st.image("./img/iris2.jpg", caption="Virginica")
else:
    st.info("กรอกข้อมูลแล้วกดปุ่มเพื่อทำนาย")

st.markdown('</div>', unsafe_allow_html=True)
