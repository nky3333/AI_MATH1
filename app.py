import bz2
import hashlib
import math
import os
import random
import urllib.request
import dlib
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
import plotly.graph_objects as go
import streamlit as st
from streamlit_image_coordinates import streamlit_image_coordinates

# ── 페이지 설정 ──
st.set_page_config(
    page_title="인공지능 수학 수행평가 플랫폼",
    page_icon="📐",
    layout="wide",
)

st.title("📐 인공지능 수학 수행평가 통합 플랫폼")
st.markdown("안면 랜드마크 직접 측정 및 3D 군집 분석 통합 플랫폼입니다.")

# ── dlib 랜드마크 모델 자동 다운로드 및 초기화 ──
MODEL_FILE = "shape_predictor_68_face_landmarks.dat"
if not os.path.exists(MODEL_FILE):
  with st.spinner("📥 랜드마크 모델 파일을 다운로드 중입니다 (약 99MB)..."):
    url = "http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2"
    urllib.request.urlretrieve(url, MODEL_FILE + ".bz2")
    with bz2.BZ2File(MODEL_FILE + ".bz2") as fr, open(MODEL_FILE, "wb") as fw:
      fw.write(fr.read())
    os.remove(MODEL_FILE + ".bz2")

@st.cache_resource
def load_dlib_tools():
  detector = dlib.get_frontal_face_detector()
  predictor = dlib.shape_predictor(MODEL_FILE)
  return detector, predictor


detector, predictor = load_dlib_tools()

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

# ── 세션 상태 초기화 ──
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

if "points" not in st.session_state:
  st.session_state.points = []  # 저장 형태: (landmark_idx, x, y)
if "current_image_tag" not in st.session_state:
  st.session_state.current_image_tag = None

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
        "📁 분석할 사진 선택",
        st.session_state.all_choices,
        key="selected_tag_box",
    )

    if st.session_state.current_image_tag != selected_tag:
      st.session_state.current_image_tag = selected_tag
      st.session_state.points = []

    img_filename = st.session_state.file_map.get(selected_tag)
    img_path = os.path.join(IMAGE_DIR, img_filename)

    if st.button("🔄 선택 초기화", type="secondary"):
      st.session_state.points = []
      st.rerun()

    st.markdown("---")
    st.markdown("### 📊 측정 안내")
    pts = st.session_state.points
    n = len(pts)

    msg = ""
    if n == 0:
      msg = "👉 **1단계:** **변수 x (입 가로 폭)** 첫 번째 점을 클릭하세요."
    elif n == 1:
      msg = "👉 **1단계:** **변수 x (입 가로 폭)** 두 번째 점을 클릭하세요."
    elif n == 2:
      dx = abs(pts[1][1] - pts[0][1])
      dy = abs(pts[1][2] - pts[0][2])
      dist_x = math.sqrt(dx**2 + dy**2)
      msg = f"🔴 **변수 x 완료!** ({dist_x:.1f}px)\n👉 **2단계:** **변수 y (눈 세로 폭)** 첫 번째 점을 클릭하세요."
    elif n == 3:
      msg = "👉 **2단계:** **변수 y (눈 세로 폭)** 두 번째 점을 클릭하세요."
    elif n == 4:
      dx = abs(pts[3][1] - pts[2][1])
      dy = abs(pts[3][2] - pts[2][2])
      dist_y = math.sqrt(dx**2 + dy**2)
      msg = f"🟡 **변수 y 완료!** ({dist_y:.1f}px)\n👉 **3단계:** **변수 z (눈썹-눈 거리)** 첫 번째 점을 클릭하세요."
    elif n == 5:
      msg = "👉 **3단계:** **변수 z (눈썹-눈 거리)** 두 번째 점을 클릭하세요."
    elif n >= 6:
      x_val = math.sqrt(
          (pts[1][1] - pts[0][1]) ** 2 + (pts[1][2] - pts[0][2]) ** 2
      )
      y_val = math.sqrt(
          (pts[3][1] - pts[2][1]) ** 2 + (pts[3][2] - pts[2][2]) ** 2
      )
      z_val = math.sqrt(
          (pts[5][1] - pts[4][1]) ** 2 + (pts[5][2] - pts[4][2]) ** 2
      )
      msg = (
          f"🎉 **3가지 변수 측정 완료!**\n"
          f"🔴 **x (입 가로):** {x_val:.1f} px\n"
          f"🟡 **y (눈 세로):** {y_val:.1f} px\n"
          f"🔵 **z (눈썹-눈):** {z_val:.1f} px\n\n"
          f"✨ **최종 결과값:** x:{x_val:.1f} y:{y_val:.1f} z:{z_val:.1f}"
      )

    st.text_area("측정 상태", value=msg, height=160)

  with col2:
    if os.path.exists(img_path):
      orig_pil = Image.open(img_path).convert("RGB")
      orig_w, orig_h = orig_pil.size

      # 화면 표시용 스케일 적용 (너비 450px 기준)
      base_width = 450
      scale = base_width / float(orig_w)
      h_size = int(float(orig_h) * float(scale))
      img_resized = orig_pil.resize(
          (base_width, h_size), Image.Resampling.LANCZOS
      )

      # dlib으로 68개 랜드마크 검출
      faces = detector(np.array(orig_pil), 1)
      lm_coords = []
      if len(faces) > 0:
        shape = predictor(np.array(orig_pil), faces[0])
        # 리사이즈된 화면 크기에 맞게 좌표 스케일링
        lm_coords = [
            (int(shape.part(i).x * scale), int(shape.part(i).y * scale))
            for i in range(68]
        )

      # 이미지 위에 하얀색 점(랜드마크) 및 선택된 측정 점/선 그리기
      draw_img = img_resized.copy()
      draw = ImageDraw.Draw(draw_img)

      # 1. 68개 전체 하얀색 점 표시
      for lx, ly in lm_coords:
        r = 3.0
        draw.ellipse(
            [lx - r, ly - r, lx + r, ly + r],
            fill="white",
            outline="black",
            width=1,
        )

      # 2. 사용자가 선택한 점들 색상 입히기 (빨강, 노랑, 파랑)
      colors = ["red", "yellow", "blue"]
      for i, pt in enumerate(pts):
        px, py = pt[1], pt[2]
        color_idx = i // 2
        c = colors[color_idx] if color_idx < 3 else "white"
        r = 6
        draw.ellipse(
            [px - r, py - r, px + r, py + r],
            fill=c,
            outline="white",
            width=2,
        )

      # 3. 측정 선 긋기
      for pair_idx in range(len(pts) // 2):
        p1 = pts[pair_idx * 2]
        p2 = pts[pair_idx * 2 + 1]
        c = colors[pair_idx]
        draw.line([p1[1], p1[2], p2[1], p2[2]], fill=c, width=3)

      st.markdown(
          "👇 **얼굴에 표시된 하얀색 랜드마크 점 위를 마우스로 클릭하세요**"
          " (가장 가까운 랜드마크에 자동으로 스냅됩니다)"
      )

      # 클릭 감지 컴포넌트
      value = streamlit_image_coordinates(
          draw_img, key=f"img_click_{selected_tag}_{len(pts)}"
      )

      if value is not None and len(lm_coords) > 0:
        click_x, click_y = value["x"], value["y"]

        # 가장 가까운 dlib 랜드마크 찾기 (스냅 기능)
        closest_idx = -1
        min_dist = float("inf")
        for idx, (lx, ly) in enumerate(lm_coords):
          d = math.sqrt((click_x - lx) ** 2 + (click_y - ly) ** 2)
          if d < min_dist:
            min_dist = d
            closest_idx = idx

        # 40픽셀 이내의 점을 클릭했을 때만 인정
        if min_dist <= 40 and closest_idx != -1:
          selected_pt = lm_coords[closest_idx]
          # 중복 클릭 방지
          if not pts or pts[-1][0] != closest_idx:
            if len(pts) < 6:
              pts.append((closest_idx, selected_pt[0], selected_pt[1]))
              st.session_state.points = pts
              st.rerun()
    else:
      st.error(f"이미지 파일을 찾을 수 없습니다: {img_path}")

with tab2:
  st.markdown(
      "### 11개 데이터(훈련 9개 + 테스트 2개)의 좌표를 확인하고 3D 공간 상의"
      " 분포를 시각화하세요."
  )
  st.markdown("**🟢 행복(HA) | 🟠 놀람(SU) | 🟣 혐오(DI) | 🟡 테스트(X)**")

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

  edited_df = st.data_editor(
      default_df, num_rows="dynamic", use_container_width=True, key="coord_table"
  )

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
