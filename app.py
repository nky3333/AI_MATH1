import os
import math
import random
import hashlib
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
import streamlit as st
import plotly.graph_objects as go

# ── 페이지 설정 ──
st.set_page_config(
    page_title="인공지능 수학 수행평가 플랫폼",
    page_icon="📐",
    layout="wide",
)

st.title("📐 인공지능 수학 수행평가 통합 플랫폼")
st.markdown("사이드바 또는 탭을 통해 안면 랜드마크 측정 및 3D 군집 분석을 진행할 수 있습니다.")

# ── 15개 정예 이미지 풀 (images/ 폴더 기준) ──
IMAGE_DIR = "./images"
full_pool = {
    "행복": [
        "KA.HA1.29.tiff",
        "KL.HA1.158.tiff",
        "KM.HA3.6.tiff",
        "YM.HA3.54.tiff",
        "NM.HA1.95.tiff",
    ],
    "놀람": [
        "KA.SU3.38.tiff",
        "KM.SU3.16.tiff",
        "UY.SU1.143.tiff",
        "NM.SU2.102.tiff",
        "TM.SU1.187.tiff",
    ],
    "혐오": [
        "NA.DI1.214.tiff",
        "KL.DI2.171.tiff",
        "UY.DI2.150.tiff",
        "KA.DI2.43.tiff",
        "YM.DI3.66.tiff",
    ],
}

# ── 세션 상태 초기화 (데이터셋 고정용) ──
if "file_map" not in st.session_state:
  train_list = []
  test_list = []
  file_map = {}
  test_pools = {}

  for emo, filenames in full_pool.items():
    shuffled = filenames.copy()
    random.shuffle(shuffled)
    for i in range(3):
      fn = shuffled[i]
      tag = f"{emo}{i+1}"
      file_map[tag] = fn
      train_list.append(tag)
    test_pools[emo] = shuffled[3:]

  chosen_emotions = random.sample(list(full_pool.keys()), 2)
  test_items = []
  for emo in chosen_emotions:
    test_items.append(random.choice(test_pools[emo]))

  for i, fn in enumerate(test_items):
    tag = f"테스트{i+1}"
    file_map[tag] = fn
    test_list.append(tag)

  st.session_state.file_map = file_map
  st.session_state.all_choices = train_list + test_list

  combined_filenames = "".join(sorted(file_map.values()))
  st.session_state.case_id = (
      hashlib.md5(combined_filenames.encode()).hexdigest()[:5].upper()
  )

# ── 상단 탭 나누기 ──
tab1, tab2 = st.tabs(
    ["📏 1. 랜드마크 변수 측정기", "📊 2. 3D 공간 시각화 및 분석"]
)

with tab1:
  st.markdown(
      f"### 🎯 부여된 수행평가 케이스: `Case #{st.session_state.case_id}`"
  )
  st.info(
      "📌 **[HA / SU / DI 데이터 세트 배정 완료]** 수행평가 활동지에 위 **Case"
      " 번호**를 반드시 적어주세요!"
  )

  col1, col2 = st.columns([1, 2])

  with col1:
    selected_tag = st.selectbox(
        "📁 분석할 사진 선택", st.session_state.all_choices
    )
    img_filename = st.session_state.file_map.get(selected_tag)
    img_path = os.path.join(IMAGE_DIR, img_filename)

    if st.button("사진 불러오기", type="primary"):
      st.session_state.loaded_img = img_path
      st.success(f"✅ [{selected_tag}] 로드 완료!")

  with col2:
    if "loaded_img" in st.session_state and os.path.exists(
        st.session_state.loaded_img
    ):
      image = Image.open(st.session_state.loaded_img).convert("RGB")
      st.image(
          image,
          caption=f"선택된 이미지: {selected_tag} ({img_filename})",
          use_container_width=True,
      )
    else:
      st.warning(
          "⚠ 왼쪽에서 사진을 선택하고 [사진 불러오기] 버튼을 눌러주세요."
      )

with tab2:
  st.markdown(
      "### 11개 데이터(훈련 9개 + 테스트 2개)의 좌표를 확인하고 3D 공간 상의"
      " 분포를 시각화하세요."
  )
  st.markdown("**🟢 행복(HA) | 🟠 놀람(SU) | 🟣 혐오(DI) | 🟡 테스트(X)**")

  # 기본 데이터프레임 설정
  default_df = pd.DataFrame(
      [
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
          ["테스트2 (Test)", 25, 25, 25],
      ],
      columns=["항목/태그", "X값 (입 가로)", "Y값 (눈 세로)", "Z값 (눈썹-눈)"],
  )

  edited_df = st.data_editor(default_df, num_rows="dynamic", use_container_width=True)

  if st.button("🚀 3차원 공간 시각화 실행", type="primary"):
    try:
      fig = go.Figure()
      style_map = {
          "행복": {"color": "#2ECC71", "symbol": "square", "size": 6},
          "놀람": {"color": "#E67E22", "symbol": "diamond", "size": 6},
          "혐오": {"color": "#8E44AD", "symbol": "circle", "size": 6},
          "테스트": {"color": "#F1C40F", "symbol": "x", "size": 8},
      }

      for _, row in edited_df.iterrows():
        tag = str(row["항목/태그"])
        x_val, y_val, z_val = float(row[1]), float(row[2]), float(row[3])

        style = style_map["테스트"]
        for key in ["행복", "놀람", "혐오"]:
          if key in tag:
            style = style_map[key]
            break

        fig.add_trace(
            go.Scatter3d(
                x=[x_val],
                y=[y_val],
                z=[z_val],
                mode="markers+text",
                text=[tag],
                textposition="top center",
                marker=dict(
                    size=style["size"],
                    color=style["color"],
                    symbol=style["symbol"],
                ),
                name=tag,
            )
        )

      fig.update_layout(
          title="<b>🎓 안면 랜드마크 측정 데이터 3D 시각화</b>",
          scene=dict(
              xaxis_title="X (입 가로 폭)",
              yaxis_title="Y (눈 세로 높이)",
              zaxis_title="Z (눈썹-눈 거리)",
          ),
          margin=dict(l=0, r=0, b=0, t=40),
      )
      st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
      st.error(f"시각화 중 오류 발생: {e}")
