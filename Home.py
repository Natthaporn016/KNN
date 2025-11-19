from sklearn.neighbors import KNeighborsClassifier
import streamlit as st
import pandas as pd
import numpy as np

# ---------------------------------------------------------
# 🌸 Pastel Theme + Glow Button + Cute Card UI CSS
# ---------------------------------------------------------
st.markdown("""
<style>

body {
    background: #ffeef5 !important;
    font-family: 'Segoe UI', sans-serif;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #ffd6e8, #ffeaf6);
    color: #333;
}
[data-testid="stSidebar"] .css-1d391kg {
    color: #333;
}

/* Title Card */
.title-card {
    background: linear-gradient(135deg, #ff9a9e, #fad0c4, #fbc2eb);
    padding: 25px;
    border-radius: 25px;
    text-align: center;
    color: #fff;
    font-size: 35px;
    font-weight: bold;
    margin-bottom: 25px;
    box-shadow: 0px 4px 20px rgba(255,150,200,0.5);
}

/* Section Card */
.section-card {
    background-color: #ffffffcc;
    padding: 25px;
    border-radius: 25px;
    box-shadow: 0 4px 25px rgba(255,150,200,0.3);
    margin-top: 25px;
    transition: transform 0.3s ease;
}
.section-card:hover {
    transform: scale(1.02);
}

/* Sub Header */
.sub-header {
    font-size: 24px;
    font-weight: bold;
    text-align: center;
    color: #ff6f91;
    padding-bottom: 10px;
}

/* Glow Button */
.glow-btn {
    background: linear-gradient(135deg, #ffb6c1, #ff9a9e);
    color: white !important;
    padding: 14px 30px;
    border-radius: 35px;
    border: none;
    font-size: 18px;
    box-shadow: 0 0 15px #ff8ab5;
    transition: 0.3s;
}
.glow-btn:hover {
    box-shadow: 0 0 30px #ff5f9e;
    cursor: pointer;
    transform: scale(1.1);
}

/* Input Fields Styling */
.stSlider>div>div>div>div {
    color: #ff6f91 !important;
}
.stNumberInput>div>input {
    border-radius: 15px !important;
    border: 2px solid #ffb6c1 !important;
}

</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# 🌸 Sidebar UI
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
# 🌼 ตัวอย่างภาพสายพันธุ์ดอกไม้
# ---------------------------------------------------------
st.markdown('<div class="section-card">', unsafe_allow_html=True)
st.markdown('<div class="sub-header">🌼 ตัวอย่างภาพสายพันธุ์ดอกไม้</div>', unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)
with col1: st.image("./img/iris1.jpg", caption="Versicolor")
with col2: st.image("./img/iris2.jpg", caption="Virginica")
with col3: st.image("./img/iris3.jpg", caption="Setosa")

st.markdown('</div>', unsafe_allow_html=True)

# ---------------------------------------------------------
# 📊 สถิติข้อมูลดอกไม้ + Visualization
# ---------------------------------------------------------
st.markdown('<div class="section-card">', unsafe_allow_html=True)
st.markdown('<div class="sub-header">📊 สถิติข้อมูลดอกไม้</div>', unsafe_allow_html=True)

dt = pd.read_csv("./data/iris.csv")
st.write(dt.head(10))

dt_sum = dt.sum()
dx = dt_sum.values
dx2 = pd.DataFrame(dx, index=dt_sum.index)

if st.button("✨ แสดงการจินตทัศน์ข้อมูล (Visualization) ✨", key="chart2"):
    st.markdown("### ✨ กราฟแบบพาสเทลฟุ้ง ๆ ✨")
    st.bar_chart(dx2)
else:
    st.info("กดปุ่มเพื่อแสดงข้อมูลภาพรวม")

st.markdown('</div>', unsafe_allow_html=True)

# ---------------------------------------------------------
# 🔮 ระบบทำนายข้อมูลดอกไม้
# ---------------------------------------------------------
st.markdown('<div class="section-card">', unsafe_allow_html=True)
st.markdown('<div class="sub-header">🔮 ระบบทำนายข้อมูลดอกไม้</div>', unsafe_allow_html=True)

pt_len = st.slider("เลือกค่า Petal Length", 0.0, 10.0, 1.0)
pt_wd = st.slider("เลือกค่า Petal Width", 0.0, 5.0, 1.0)
sp_len = st.number_input("กรอกค่า Sepal Length")
sp_wd = st.number_input("กรอกค่า Sepal Width")

predict_btn = st.button("✨ ทำนายสายพันธุ์ดอกไม้ ✨", key="predict_btn")

if predict_btn:
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
