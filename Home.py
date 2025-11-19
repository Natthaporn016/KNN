from sklearn.neighbors import KNeighborsClassifier
import streamlit as st
import pandas as pd
import numpy as np
import altair as alt

# ---------------------------------------------------------
# 🌸 Pastel Theme + Glow Button + Card UI CSS
# ---------------------------------------------------------
st.markdown("""
<style>

/* Body */
body {
    background: linear-gradient(180deg, #ffeef5, #fff0f8) !important;
    font-family: 'Segoe UI', sans-serif;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #ffd6e8, #ffeaf6);
    color: #333;
    border-radius: 20px;
    padding: 15px;
}

/* Title Card */
.title-card {
    background: linear-gradient(135deg, #ff9a9e, #fad0c4, #fbc2eb);
    padding: 30px;
    border-radius: 30px;
    text-align: center;
    color: #fff;
    font-size: 36px;
    font-weight: bold;
    margin-bottom: 30px;
    box-shadow: 0 8px 25px rgba(255,150,200,0.4);
}

/* Section Card */
.section-card {
    background: #ffffffdd;
    padding: 25px;
    border-radius: 25px;
    box-shadow: 0 6px 25px rgba(255,150,200,0.25);
    margin-top: 25px;
    transition: all 0.3s ease;
}
.section-card:hover {
    transform: scale(1.02);
}

/* Sub Header */
.sub-header {
    font-size: 26px;
    font-weight: bold;
    text-align: center;
    color: #ff6f91;
    margin-bottom: 15px;
}

/* Glow Button */
.glow-btn {
    background: linear-gradient(135deg, #ffb6c1, #ff9a9e);
    color: white !important;
    padding: 14px 35px;
    border-radius: 35px;
    border: none;
    font-size: 18px;
    box-shadow: 0 0 15px #ff8ab5, 0 0 25px #ff8ab5 inset;
    transition: all 0.3s ease;
    font-weight: bold;
}
.glow-btn:hover {
    box-shadow: 0 0 30px #ff5f9e, 0 0 40px #ff8ab5 inset;
    cursor: pointer;
    transform: scale(1.1);
}

/* Slider & Input Fields */
.stSlider>div>div>div>div { color: #ff6f91 !important; }
.stNumberInput>div>input {
    border-radius: 15px !important;
    border: 2px solid #ffb6c1 !important;
}

</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# Sidebar
# ---------------------------------------------------------
st.sidebar.markdown("## 🌸 Pink Dashboard")
st.sidebar.image("./img/pin.jpg", width=180)
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
st.markdown('<div class="title-card">🌸 Pin - Pink Iris Classification Dashboard 🌸</div>', unsafe_allow_html=True)
st.image("./img/pin.jpg", width=250)
st.write("### ยินดีต้อนรับเข้าสู่ Pink Dashboard สำหรับทำนายสายพันธุ์ดอกไม้ไอริส 💗")

# ---------------------------------------------------------
# ตัวอย่างภาพสายพันธุ์ดอกไม้
# ---------------------------------------------------------
st.markdown('<div class="section-card">', unsafe_allow_html=True)
st.markdown('<div class="sub-header">🌼 ตัวอย่างภาพสายพันธุ์ดอกไม้</div>', unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)
with col1: st.image("./img/iris1.jpg", caption="Versicolor")
with col2: st.image("./img/iris2.jpg", caption="Virginica")
with col3: st.image("./img/iris3.jpg", caption="Setosa")

st.markdown('</div>', unsafe_allow_html=True)

# ---------------------------------------------------------
# สถิติข้อมูลดอกไม้ + Visualization pastel
# ---------------------------------------------------------
st.markdown('<div class="section-card">', unsafe_allow_html=True)
st.markdown('<div class="sub-header">📊 สถิติข้อมูลดอกไม้</div>', unsafe_allow_html=True)

dt = pd.read_csv("./data/iris.csv")
st.write(dt.head(10))

dt_sum = dt.sum().reset_index()
dt_sum.columns = ["Feature", "Total"]

if st.button("✨ แสดงกราฟข้อมูล (Visualization) ✨", key="chart3"):
    st.markdown("### ✨ กราฟข้อมูล✨")
    chart = alt.Chart(dt_sum).mark_bar(cornerRadiusTopLeft=8, cornerRadiusTopRight=8).encode(
        x=alt.X('Feature', sort=None),
        y='Total',
        color=alt.Color('Feature', scale=alt.Scale(range=['#ffb6c1','#ff9a9e','#ffc0cb','#ff7f9e']))
    ).properties(width=600, height=400)
    st.altair_chart(chart, use_container_width=True)
else:
    st.info("กดปุ่มเพื่อแสดงกราฟภาพรวม")

st.markdown('</div>', unsafe_allow_html=True)

# ---------------------------------------------------------
# ระบบทำนายข้อมูลดอกไม้
# ---------------------------------------------------------
st.markdown('<div class="section-card">', unsafe_allow_html=True)
st.markdown('<div class="sub-header">🔮 ระบบทำนายข้อมูลดอกไม้</div>', unsafe_allow_html=True)

pt_len = st.slider("เลือกค่า Petal Length", 0.0, 10.0, 1.0)
pt_wd = st.slider("เลือกค่า Petal Width", 0.0, 5.0, 1.0)
sp_len = st.number_input("กรอกค่า Sepal Length")
sp_wd = st.number_input("กรอกค่า Sepal Width")

if st.button("✨ ทำนายสายพันธุ์ดอกไม้ ✨", key="predict2"):
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
