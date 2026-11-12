import streamlit as st
import os
import math
import random
import hashlib
import dlib
import numpy as np
from PIL import Image, ImageDraw
import plotly.graph_objects as go
import pandas as pd

# 페이지 설정
st.set_page_config(
    page_title="인공지능 수학 수행평가 플랫폼",
    page_icon="📐",
    layout="wide"
)

# 랜드마크 모델 경로 및 자동 다운로드 설정
model_path = 'shape_predictor_68_face_landmarks.dat'
if not os.path.exists(model_path):
    import urllib.request
    url = "https://github.com/italojs/facial-landmarks-recognition/raw/master/shape_predictor_68_face_landmarks.dat"
    urllib.request.urlretrieve(url, model_path)

@st.cache_resource
def load_detector():
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(model_path)
    return detector, predictor

detector, predictor = load_detector()

# 데이터 풀 설정 (샘플 파일명 기준)
full_pool = {
    '행복': [{'filename': 'KA.HA1.29.tiff'}, {'filename': 'KL.HA1.158.tiff'}, {'filename': 'KM.HA3.6.tiff'}, {'filename': 'YM.HA3.54.tiff'}, {'filename': 'NM.HA1.95.tiff'}],
    '놀람': [{'filename': 'KA.SU3.38.tiff'}, {'filename': 'KM.SU3.16.tiff'}, {'filename': 'UY.SU1.143.tiff'}, {'filename': 'NM.SU2.102.tiff'}, {'filename': 'TM.SU1.187.tiff'}],
    '혐오': [{'filename': 'NA.DI1.214.tiff'}, {'filename': 'KL.DI2.171.tiff'}, {'filename': 'UY.DI2.150.tiff'}, {'filename': 'KA.DI2.43.tiff'}, {'filename': 'YM.DI3.66.tiff'}]
}

st.title("📐 인공지능 수학 수행평가 통합 플랫폼")
st.markdown("탭을 이동하여 **[1탭] 안면 랜드마크 측정** 또는 **[2탭] 3D 군집 시각화**를 이용할 수 있습니다.")

tab1, tab2 = st.tabs(["📏 1. 랜드마크 변수 측정기", "📊 2. 3D 공간 시각화 및 분석"])

# ==========================================
# [탭 1] 랜드마크 변수 측정기
# ==========================================
with tab1:
    st.subheader("🎯 안면 랜드마크 직접 측정기")
    
    if 'file_map' not in st.session_state:
        train_list, test_list, file_map = [], [], {}
        emotions = ['행복', '놀람', '혐오']
        test_pools = {}
        for emo in emotions:
            shuffled = full_pool[emo].copy()
            random.shuffle(shuffled)
            for i in range(3):
                item = shuffled[i]
                tag = f"{emo}{i+1}"
                file_map[tag] = item['filename']
                train_list.append(tag)
            test_pools[emo] = shuffled[3:]
        chosen_emotions = random.sample(emotions, 2)
        test_items = [random.choice(test_pools[emo]) for emo in chosen_emotions]
        for i, item in enumerate(test_items):
            tag = f"테스트{i+1}"
            file_map[tag] = item['filename']
            test_list.append(tag)
        st.session_state.file_map = file_map
        st.session_state.all_choices = train_list + test_list

    selected_tag = st.selectbox("📁 분석할 사진 선택", st.session_state.all_choices)
    st.info("💡 클라우드 환경에서는 이미지가 포함된 폴더 경로를 함께 세팅하거나 업로드 기능으로 대체할 수 있습니다.")

# ==========================================
# [탭 2] 3D 공간 시각화 및 분석
# ==========================================
with tab2:
    st.subheader("📊 3D 공간 시각화 및 분석")
    st.markdown("**🟢 행복(HA) | 🟠 놀람(SU) | 🟣 혐오(DI) | 🟡 테스트(X)**")
    
    default_df = pd.DataFrame([
        ["행복1 (Train)", 11, 13, 5],
        ["행복2 (Train)", 12, 12, 5],
        ["행복3 (Train)", 13, 11, 5],
        ["놀람1 (Train)", 5, 25, 15],
        ["놀람2 (Train)", 6, 26, 13],
        ["놀람3 (Train)", 7, 24, 14],
        ["혐오1 (Train)", 30, 30, 31],
        ["혐오2 (Train)", 32, 33, 32],
        ["혐오3 (Train)", 33, 32, 33],
        ["테스트1 (Test)", 10, 10, 9],
        ["테스트2 (Test)", 25, 25, 25]
    ], columns=["항목/태그", "X값 (입 가로)", "Y값 (눈 세로)", "Z값 (눈썹-눈)"])

    edited_df = st.data_editor(default_df, num_rows="dynamic", use_container_width=True)

    if st.button("🚀 3차원 공간 시각화 실행", type="primary"):
        fig = go.Figure()
        for _, row in edited_df.iterrows():
            tag, x, y, z = str(row[0]), float(row[1]), float(row[2]), float(row[3])
            color, symbol = "#F1C40F", "x"
            if "행복" in tag: color, symbol = "#2ECC71", "square"
            elif "놀람" in tag: color, symbol = "#E67E22", "diamond"
            elif "혐오" in tag: color, symbol = "#8E44AD", "circle"

            fig.add_trace(go.Scatter3d(
                x=[x], y=[y], z=[z],
                mode='markers+text', text=[tag],
                textposition='top center',
                marker=dict(size=6, color=color, symbol=symbol),
                name=tag
            ))

        fig.update_layout(
            title="<b>실제 안면 랜드마크 측정 데이터 3D 시각화</b>",
            scene=dict(xaxis_title='X (입 가로)', yaxis_title='Y (눈 세로)', zaxis_title='Z (눈썹-눈)')
        )
        st.plotly_chart(fig, use_container_width=True)
