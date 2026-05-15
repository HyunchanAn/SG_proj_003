import streamlit as st
from PIL import Image
import numpy as np
import pandas as pd
from pathlib import Path
import sys

# 프로젝트 루트 추가
sys.path.append(str(Path(__file__).parent))

from vsams.analysis.surface_evaluator import SurfaceEvaluator
from vsams.utils.substrate_db import SubstrateDB
from streamlit_drawable_canvas import st_canvas

# --- Config ---
st.set_page_config(
    page_title="V-SAMS Core Hub",
    page_icon="🛡️",
    layout="wide"
)

# --- Resources ---
def load_resources():
    evaluator = SurfaceEvaluator()
    db = SubstrateDB()
    return evaluator, db

evaluator, db = load_resources()

# --- UI Styles ---
st.markdown("""
<style>
    .main { background-color: #0e1117; }
    .metric-card {
        background-color: #1e2130;
        padding: 20px;
        border-radius: 10px;
        border: 1px solid #3e44fe;
        text-align: center;
    }
    .metric-value { font-size: 2.5rem; font-weight: bold; color: #00ff00; }
    .metric-label { font-size: 1rem; color: #888; }
</style>
""", unsafe_allow_html=True)

# --- Header ---
st.title("🛡️ V-SAMS Core Hub")
st.markdown("동전-반사광 분석 기반 고정밀 표면 분석 시스템")

# --- Sidebar ---
with st.sidebar:
    st.header("⚙️ 분석 설정")
    uploaded_file = st.file_uploader("분석할 사진 업로드", type=['jpg', 'jpeg', 'png'])
    st.divider()
    st.info("💡 팁: 실제 동전과 그 반사된 이미지가 모두 잘 보이도록 촬영해 주세요.")

# --- Main Logic ---
if uploaded_file:
    img = Image.open(uploaded_file).convert("RGB")
    w_orig, h_orig = img.size
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("1. 분석 영역 지정 (ROI Selection)")
        st.info("파란색 박스로 실제 동전 을, 빨간색 박스로 반사된 이미지 를 드래그하여 지정하세요.")
        
        # 캔버스 크기 조정
        canvas_width = 800
        canvas_height = int(h_orig * (canvas_width / w_orig))
        
        canvas_result = st_canvas(
            fill_color="rgba(255, 255, 255, 0.2)",
            stroke_width=3,
            stroke_color="#00ff00",
            background_image=img,
            height=canvas_height,
            width=canvas_width,
            drawing_mode="rect",
            key="analysis_canvas",
        )
        
        if st.button("영역 초기화 🧹"):
            st.rerun()

    with col2:
        st.subheader("2. 분석 결과 (Analysis)")
        
        # 상태 관리: 수동 박스가 있는지 확인
        manual_boxes = None
        if canvas_result.json_data is not None:
            objects = canvas_result.json_data["objects"]
            if len(objects) >= 2:
                manual_boxes = []
                for obj in objects[:2]:
                    x1 = int(obj["left"] * (w_orig / canvas_width))
                    y1 = int(obj["top"] * (h_orig / canvas_height))
                    x2 = int((obj["left"] + obj["width"]) * (w_orig / canvas_width))
                    y2 = int((obj["top"] + obj["height"]) * (h_orig / canvas_height))
                    manual_boxes.append([x1, y1, x2, y2])

        # 분석 실행 (수동 박스가 있으면 수동, 없으면 자동)
        with st.spinner("표면 상태 분석 중..."):
            result = evaluator.analyze(img, custom_boxes=manual_boxes)
        
        if result and "error" not in result:
            # 결과 표시 카드
            c1, c2 = st.columns(2)
            with c1:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">추정 조도 (Ra)</div>
                    <div class="metric-value">{result['roughness']:.4f}</div>
                </div>
                """, unsafe_allow_html=True)
            with c2:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">추정 광택도 (%)</div>
                    <div class="metric-value">{result['gloss']:.1f}</div>
                </div>
                """, unsafe_allow_html=True)
            
            st.divider()
            st.success(f"🎯 예측 분류: {result['predicted_label']}")
            
            # DB 매칭
            closest = db.find_closest(result['roughness'], result['gloss'])
            if closest:
                st.info(f"📚 유사 제품 (DB): {closest['product_name']} (오차: {closest['distance']:.4f})")
            
            # 감지 영역 시각화 (동전/반사광 위치 확인용)
            st.image(evaluator.get_overlay_image(img.copy(), result), caption="감지된 분석 영역 (Blue: Coin, Red: Reflection)")
            
            # 크롭된 이미지 직접 확인
            with st.expander("🔍 추출된 영역 상세보기"):
                c1, c2 = st.columns(2)
                coin_img = img.crop(result["coin_box"])
                ref_img = img.crop(result["ref_box"])
                with c1:
                    st.image(coin_img, caption="추출된 동전 (Source)")
                with c2:
                    st.image(ref_img, caption="추출된 반사광 (Target)")
            
            # 정답지 대조
            with st.expander("📊 기준 물성치 표 보기"):
                st.dataframe(db.df[["product_name", "roughness_avg", "gloss_avg"]].sort_values("product_name"))
        elif result and "error" in result:
            st.error(f"⚠️ {result['error']}")
            st.info("사이드바의 캔버스에서 직접 동전과 반사광 영역을 사각형으로 그려주세요.")

else:
    st.info("사이드바에서 분석할 이미지를 업로드해 주세요.")
