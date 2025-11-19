from sklearn.neighbors import KNeighborsClassifier
import streamlit as st
import pandas as pd
import numpy as np

# --------------------------------------------
# Custom CSS สำหรับตกแต่งหน้าเว็บ
# --------------------------------------------
st.markdown("""
<style>
    .title-card {
        background: linear-gradient(135deg, #ff9a9e, #fad0c4);
        padding: 20px;
        border-radius: 20px;
        text-align: center;
        color: #222;
        font-size: 32px;
        font-weight: bold;
        margin-bottom: 20px;
        box-shadow: 0px 4px 12px rgba(0,0,0,0.1);
    }
    .section-card {
        background-color: #ffffff;
        padding: 18px;
        border-radius: 15px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
        margin-top: 20px;
    }
    .sub-header {
        font-size: 22px;
        font-weight: bold;
        text-align: center;
        color: #333;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# Header
# ---------------------------------------------------------
st.markdown('<div class="title-card">🌸 Pin - Iris Classification App 🌸</div>', unsafe_allow_html=True)

st.image("./img/pin.jpg", width=250)
st.write("## ยินดีต้อนรับเข้าสู่ระบบทำนายสายพันธุ์ดอกไม้ไอริส")

# ---------------------------------------------------------
# รูปภาพสายพันธุ์
# ---------------------------------------------------------
st.markdown('<div class="section-card">', unsafe_allow_html=True)
st.markdown('<div class="sub-header">ตัวอย่างภาพสายพันธุ์ดอกไม้</div>', unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)

with col1:
    st.image("./img/iris1.jpg", caption="Versicolor")

with col2:
    st.image("./img/iris2.jpg", caption="Virginica")

with col3:
    st.image("./img/iris3.jpg", caption="Setosa")

st.markdown('</div>', unsafe_allow_html=True)

# ---------------------------------------------------------
# แสดงตารางข้อมูล
# ---------------------------------------------------------
st.markdown('<div class="section-card">', unsafe_allow_html=True)
st.markdown('<div class="sub-header">📊 สถิติข้อมูลดอกไม้</div>', unsafe_allow_html=True)

dt = pd.read_csv("./data/iris.csv")
st.write(dt.head(10))

dt_sum = dt.sum()
dx = dt_sum.values
dx2 = pd.DataFrame(dx, index=dt_sum.index)

if st.button("แสดงการจินตทัศน์ข้อมูล (Visualization)"):
    st.bar_chart(dx2)
else:
    st.info("กดปุ่มเพื่อแสดงข้อมูลภาพรวม")

st.markdown('</div>', unsafe_allow_html=True)

# ---------------------------------------------------------
# ทำนายข้อมูล
# ---------------------------------------------------------
st.markdown('<div class="section-card">', unsafe_allow_html=True)
st.markdown('<div class="sub-header">🔮 ระบบทำนายข้อมูลดอกไม้</div>', unsafe_allow_html=True)

pt_len = st.slider("เลือกค่า Petal Length", 0.0, 10.0, 1.0)
pt_wd = st.slider("เลือกค่า Petal Width", 0.0, 5.0, 1.0)

sp_len = st.number_input("กรอกค่า Sepal Length")
sp_wd = st.number_input("กรอกค่า Sepal Width")

if st.button("ทำนายสายพันธุ์ดอกไม้"):
    X = dt.drop('variety', axis=1)
    y = dt["variety"]

    model = KNeighborsClassifier(n_neighbors=3)
    model.fit(X, y)

    x_input = np.array([[pt_len, pt_wd, sp_len, sp_wd]])
    result = model.predict(x_input)

    st.success(f"ผลการทำนายคือ ➜ **{result[0]}**")

    if result[0] == 'Setosa':
        st.image("./img/iris3.jpg", caption="Setosa")
    elif result[0] == 'Versicolor':
        st.image("./img/iris1.jpg", caption="Versicolor")
    else:
        st.image("./img/iris2.jpg", caption="Virginica")

else:
    st.info("กรอกข้อมูลแล้วกดปุ่มเพื่อทำนาย")

st.markdown('</div>', unsafe_allow_html=True)
