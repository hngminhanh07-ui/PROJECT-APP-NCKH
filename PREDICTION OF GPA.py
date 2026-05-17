import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler

st.set_page_config(page_title="Dự Đoán GPA", page_icon="🎓", layout="centered")

st.markdown("""
<style>
    .stApp { background-color: #f0f7ff; }
    .title { text-align: center; font-size: 70px; font-weight: bold; color: #1565C0; }
    .result-box { background-color: #E3F2FD; padding: 30px; border-radius: 20px; text-align: center; }
    .gpa-text { color: #0D47A1; font-size: 55px; font-weight: bold; }
    div.stButton > button { background-color: #1E88E5; color: white; font-size: 20px; font-weight: bold; width: 100%; }
</style>
""", unsafe_allow_html=True)

st.markdown('<p class="title">🎓 GPA PREDICTOR</p>', unsafe_allow_html=True)

@st.cache_resource
def load_and_train():
    df = pd.read_csv('student_data.csv', encoding='utf-8')
    
    mapping = {
        'Dưới 5 giờ':2.5, '5–10 giờ':7.5, '11–15 giờ':13, '16–20 giờ':18, 'Trên 20 giờ':22,
        '3–4 môn':3.5, '5–6 môn':5.5, '7–8 môn':7.5, 'Trên 8 môn':9,
        'Dưới 5 giờ':4, '5–6 giờ':5.5, '6–7 giờ':6.5, '7–8 giờ':7.5, 'Trên 8 giờ':8.5,
        'Dưới 1 giờ':0.5, '1–2 giờ':1.5, '3–4 giờ':3.5, '5–6 giờ':5.5, 'Trên 6 giờ':7,
        'Không':0, 'Có':1,
        'Tự học cá nhân':0, 'Học nhóm':1, 'Kết hợp cả hai':2,
        'Dưới 50%':25, '50–70%':60, '70–85%':77.5, '85–95%':90, 'Trên 95%':97.5,
        'Dưới 2.0':1.5, '2.0–2.5':2.25, '2.5–3.0':2.75, '3.0–3.5':3.25, 'Trên 3.5':3.75
    }
    
    for col in df.columns[1:10]:
        df[col] = df[col].map(mapping)
    
    df = df.dropna()
    X = df.iloc[:, 1:9]
    y = df.iloc[:, 9]
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_scaled, y)
    
    return model, scaler

model, scaler = load_and_train()

c1, c2 = st.columns(2)

with c1:
    hoc = st.selectbox("⏰ Giờ học/tuần", ['Dưới 5 giờ','5–10 giờ','11–15 giờ','16–20 giờ','Trên 20 giờ'])
    mon = st.selectbox("📚 Số môn", ['3–4 môn','5–6 môn','7–8 môn','Trên 8 môn'])
    ngu = st.selectbox("😴 Giờ ngủ/ngày", ['Dưới 5 giờ','5–6 giờ','6–7 giờ','7–8 giờ','Trên 8 giờ'])
    mxh = st.selectbox("📱 MXH/ngày", ['Dưới 1 giờ','1–2 giờ','3–4 giờ','5–6 giờ','Trên 6 giờ'])

with c2:
    parttime = st.radio("💼 Part-time?", ['Không','Có'])
    ngoaikhoa = st.radio("🎯 Ngoại khóa?", ['Không','Có'])
    hinhthuc = st.selectbox("📖 Hình thức học", ['Tự học cá nhân','Học nhóm','Kết hợp cả hai'])
    chuyencan = st.selectbox("📊 Chuyên cần", ['Dưới 50%','50–70%','70–85%','85–95%','Trên 95%'])

map_input = {
    'Dưới 5 giờ':2.5, '5–10 giờ':7.5, '11–15 giờ':13, '16–20 giờ':18, 'Trên 20 giờ':22,
    '3–4 môn':3.5, '5–6 môn':5.5, '7–8 môn':7.5, 'Trên 8 môn':9,
    'Dưới 5 giờ':4, '5–6 giờ':5.5, '6–7 giờ':6.5, '7–8 giờ':7.5, 'Trên 8 giờ':8.5,
    'Dưới 1 giờ':0.5, '1–2 giờ':1.5, '3–4 giờ':3.5, '5–6 giờ':5.5, 'Trên 6 giờ':7,
    'Không':0, 'Có':1, 'Tự học cá nhân':0, 'Học nhóm':1, 'Kết hợp cả hai':2,
    'Dưới 50%':25, '50–70%':60, '70–85%':77.5, '85–95%':90, 'Trên 95%':97.5
}

if st.button("🚀 DỰ ĐOÁN GPA"):
    input_data = [[map_input[hoc], map_input[mon], map_input[parttime], map_input[ngu], 
                   map_input[ngoaikhoa], map_input[mxh], map_input[hinhthuc], map_input[chuyencan]]]
    gpa = model.predict(scaler.transform(input_data))[0]
    
    rank = "Yếu" if gpa < 2 else "Trung bình" if gpa < 2.5 else "Khá" if gpa < 3 else "Giỏi" if gpa < 3.5 else "Xuất sắc 🎉"
    
    st.markdown(f"""
    <div class="result-box">
        <div class="gpa-text">{gpa:.2f} / 4.0</div>
        <h2>Xếp loại: {rank}</h2>
    </div>
    """, unsafe_allow_html=True)
    st.balloons()