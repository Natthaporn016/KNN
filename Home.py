from sklearn.neighbors import KNeighborsClassifier 
import streamlit as st
import pandas as pd
import numpy as np
import altair as alt

# ---------------------------------------------------------
# 🌸 Minimal Pastel Theme CSS
# ---------------------------------------------------------
st.markdown("""
<style>
/* Body */
body {
    background: #fff0f5 !important;
    font-family: 'Segoe UI', sans-serif;
    color: #333;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: #ffe4f0;
    color: #333;
    border-radius: 15px;
    padding: 15px;
}

/* Title Card */
.title-card {
    background: #ffb6c1;
    padding: 25px;
    border-radius: 20px;
    text-align: center;
    color: white;
    font-size: 32px;
    font-weight: 600;
    margin-bottom: 20px;
}

/* Section Card */
.section-card {
    background: #ffffffdd;
    padding: 20px;
    border-radius: 20px;
    box-shadow: 0 4px 15px rgba(255,182,193,0.2);
    margin-top: 20px;
    transition: all 0.2s ease;
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
.stButton>button {
    background-color: #ff9aa2;
    color: white;
    padding: 10px 28px;
    border-radius: 20px;
    border: none;
    font-size: 16px;
    font-weight: 500;
    transition: all 0.2s ease;
}
.stButton>button:hover {
    background-color: #ff7f9e;
    transform: scale(1.05);
    cursor: pointer;
}

/* Slider & Input Fields */
.stSlider>div>div>div>div { color: #ff6f91 !important; }
.stNumberInput>div>input {
    border-radius: 10px !important;
    border: 1.5px solid #ffb6c1 !important;
    padding: 5px;
}
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# Sidebar
# ---------------------------------------------------------
st.sidebar.markdown("## 🌸 Pink Dashboard")
st.sidebar.image("./img/pin.jpg", width=150)
st.sidebar.markdown("""
**เมนู**
- 📌 หน้าแรก
- 🌼 ข้อมูลสายพันธุ์ดอกไม้
- 📊 Visualization
- 🔮 ระบบทำนาย
- 💖 ติดต่อปิ่น
""")

# ---------------------------------------------------------
# Header
# ---------------------------------------------------------
st.markdown('<div class="title-card">🌸 Pin - Iris Classification 🌸</div>', unsafe_allow_html=True)
st.write("ยินดีต้อนรับสู่ Pink Dashboard สำหรับทำนายสายพันธุ์ดอกไม้ไอริส 💗")

# ---------------------------------------------------------
# ตัวอย่างภาพสายพันธุ์ดอกไม้
# ---------------------------------------------------------
st.markdown('<div class="section-card">', unsafe_allow_html=True)
st.markdown('<div class="sub-header">🌼 ตัวอย่างสายพันธุ์ดอกไม้</div>', unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)
with col1: st.image("./img/iris1.jpg", caption="Versicolor")
with col2: st.image("./img/iris2.jpg", caption="Virginica")
with col3: st.image("./img/iris3.jpg", caption="Setosa")

st.markdown('</div>', unsafe_allow_html=True)

# ---------------------------------------------------------
# สถิติข้อมูลดอกไม้ + Visualization
# ---------------------------------------------------------
st.markdown('<div class="section-card">', unsafe_allow_html=True)
st.markdown('<div class="sub-header">📊 สถิติข้อมูลดอกไม้</div>', unsafe_allow_html=True)

dt = pd.read_csv("./data/iris.csv")
st.dataframe(dt.head(10), use_container_width=True)

dt_sum = dt.sum().reset_index()
dt_sum.columns = ["Feature", "Total"]

if st.button("✨ แสดงกราฟข้อมูล ✨"):
    chart = alt.Chart(dt_sum).mark_bar(cornerRadiusTopLeft=5, cornerRadiusTopRight=5).encode(
        x=alt.X('Feature', sort=None),
        y='Total',
        color=alt.Color('Feature', scale=alt.Scale(range=['#ffb6c1','#ffc0cb','#ff9aa2','#ff7f9e']))
    ).properties(width=600, height=350)
    st.altair_chart(chart, use_container_width=True)
else:
    st.info("กดปุ่มเพื่อแสดงกราฟ")

st.markdown('</div>', unsafe_allow_html=True)

# ---------------------------------------------------------
# ระบบทำนายข้อมูลดอกไม้
# ---------------------------------------------------------
st.markdown('<div class="section-card">', unsafe_allow_html=True)
st.markdown('<div class="sub-header">🔮 ระบบทำนายสายพันธุ์ดอกไม้</div>', unsafe_allow_html=True)

pt_len = st.slider("เลือกค่า Petal Length", 0.0, 10.0, 1.0)
pt_wd = st.slider("เลือกค่า Petal Width", 0.0, 5.0, 1.0)
sp_len = st.number_input("กรอกค่า Sepal Length")
sp_wd = st.number_input("กรอกค่า Sepal Width")

if st.button("✨ ทำนายสายพันธุ์ดอกไม้ ✨"):
    X = dt.drop('variety', axis=1)
    y = dt["variety"]
    model = KNeighborsClassifier(n_neighbors=3)
    model.fit(X, y)

    x_input = np.array([[pt_len, pt_wd, sp_len, sp_wd]])
    result = model.predict(x_input)

    st.success(f"ผลการทำนายคือ ➜ 🌸 **{result[0]}** 🌸")

    if result[0] == 'Setosa': st.image("./img/iris3.jpg", caption="Setosa")
    elif result[0] == 'Versicolor': st.image("./img/iris1.jpg", caption="Versicolor")
    else: st.image("./img/iris2.jpg", caption="Virginica")
else:
    st.info("กรอกข้อมูลแล้วกดปุ่มเพื่อทำนาย")

st.markdown('</div>', unsafe_allow_html=True)
