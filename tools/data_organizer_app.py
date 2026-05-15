import streamlit as st
import os
import sys
import json
import csv
from pathlib import Path
from PIL import Image
import pandas as pd
import numpy as np
import datetime
import torch
from streamlit_drawable_canvas import st_canvas

# 프로젝트 루트 경로 추가
sys.path.append(str(Path(__file__).parent))

from vsams.utils.substrate_db import SubstrateDB
from vsams.analysis.surface_evaluator import SurfaceEvaluator

# --- 페이지 설정 ---
st.set_page_config(
    page_title="V-SAMS Data Organizer",
    page_icon="📂",
    layout="wide"
)

# --- 스타일 정의 ---
st.markdown("""
<style>
    .main { background-color: #0e1117; }
    .stButton>button { width: 100%; border-radius: 5px; height: 3em; font-weight: bold; }
    .status-box { padding: 10px; border-radius: 5px; border: 1px solid #3e44fe; background-color: #1e2130; margin-bottom: 10px; }
    .instruction-box { padding: 15px; border-radius: 5px; border-left: 5px solid #ffaa00; background-color: #262730; margin-bottom: 20px; }
</style>
""", unsafe_allow_html=True)

# --- 리소스 로딩 ---
@st.cache_resource
def load_resources():
    db = SubstrateDB()
    evaluator = SurfaceEvaluator()
    return db, evaluator

# --- 세션 상태 초기화 ---
if 'file_index' not in st.session_state:
    st.session_state.file_index = 0
if 'files_to_process' not in st.session_state:
    st.session_state.files_to_process = []
if 'current_analysis' not in st.session_state:
    st.session_state.current_analysis = None

# --- 메인 로직 ---
st.title("📂 V-SAMS Data Organizer")
st.markdown("학습용 표면 분석 데이터를 검증하고 정제하는 도구입니다.")

db, evaluator = load_resources()

# --- 장치 정보 표시 ---
if evaluator.device.type == 'cpu':
    st.warning("⚠️ GPU 메모리 부족으로 인해 CPU 모드로 작동 중입니다. 분석 속도가 느려질 수 있습니다.")
else:
    st.success(f"✅ GPU 가속 활성화됨 ({torch.cuda.get_device_name(0)})")

# --- 사이드바: 작업 경로 설정 ---
with st.sidebar:
    st.header("⚙️ 작업 설정")
    source_dir = st.text_input("원본 이미지 폴더 경로", value="test_260420_surface")
    target_dir = st.text_input("저장될 데이터셋 경로", value="dataset/verified")
    
    if st.button("목록 갱신 🔄"):
        p = Path(source_dir)
        if p.exists():
            extensions = ['*.jpg', '*.jpeg', '*.png']
            files = []
            for ext in extensions:
                files.extend(list(p.rglob(ext)))
            st.session_state.files_to_process = sorted([str(f) for f in files])
            st.session_state.file_index = 0
            st.session_state.current_analysis = None
            st.success(f"{len(files)}개의 이미지를 찾았습니다.")
        else:
            st.error("경로가 존재하지 않습니다.")

    st.divider()
    
    if st.session_state.files_to_process:
        progress = (st.session_state.file_index + 1) / len(st.session_state.files_to_process)
        st.progress(progress)
        st.write(f"진행도: {st.session_state.file_index + 1} / {len(st.session_state.files_to_process)}")
        
        new_idx = st.number_input("이동할 파일 인덱스", min_value=0, 
                                  max_value=len(st.session_state.files_to_process)-1, 
                                  value=st.session_state.file_index)
        if new_idx != st.session_state.file_index:
            st.session_state.file_index = new_idx
            st.session_state.current_analysis = None

# --- 분석 및 라벨링 화면 ---
if st.session_state.files_to_process:
    current_file_path = st.session_state.files_to_process[st.session_state.file_index]
    st.info(f"📄 현재 파일: `{Path(current_file_path).name}`")
    
    img = Image.open(current_file_path).convert("RGB")
    w_orig, h_orig = img.size
    
    canvas_width = 800
    canvas_height = int(h_orig * (canvas_width / w_orig))
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        mode = st.radio("마스킹 모드", ["자동 (Auto)", "수동 박스 (Manual Box)"], horizontal=True)
        
        if mode == "수동 박스 (Manual Box)":
            st.markdown("""
            <div class='instruction-box'>
            <b>수동 가이드:</b> 1. <b>실제 동전</b> 영역 드래그(박스) → 2. <b>반사광</b> 영역 드래그(박스)
            </div>
            """, unsafe_allow_html=True)
            
            # 파일 인덱스를 key에 포함하여 파일 변경 시 캔버스가 자동 초기화되도록 함
            canvas_key = f"canvas_{st.session_state.file_index}"
            
            canvas_result = st_canvas(
                fill_color="rgba(0, 0, 255, 0.2)",
                stroke_width=2,
                stroke_color="#0000ff",
                background_image=img,
                update_streamlit=True,
                height=canvas_height,
                width=canvas_width,
                drawing_mode="rect",
                key=canvas_key,
            )
            
            if st.button("캔버스 초기화 🧹"):
                st.rerun()

            if canvas_result.json_data is not None:
                boxes = []
                for obj in canvas_result.json_data["objects"]:
                    if obj["type"] == "rect":
                        # 캔버스 좌표를 원본 이미지 좌표로 변환 [x1, y1, x2, y2]
                        x1 = int(obj["left"] * (w_orig / canvas_width))
                        y1 = int(obj["top"] * (h_orig / canvas_height))
                        x2 = int((obj["left"] + obj["width"]) * (w_orig / canvas_width))
                        y2 = int((obj["top"] + obj["height"]) * (h_orig / canvas_height))
                        boxes.append([x1, y1, x2, y2])
                
                if len(boxes) >= 2:
                    if st.button("수동 박스로 분석 실행 🎯"):
                        with st.spinner("수동 박스 기반 분석 중..."):
                            # 첫 번째 박스는 동전, 두 번째 박스는 반사광
                            result = evaluator.analyze(img, custom_boxes=boxes[:2])
                            result['path'] = current_file_path
                            st.session_state.current_analysis = result
                elif len(boxes) == 1:
                    st.warning("반사광 영역도 박스로 지정해 주세요.")
        
        else: # 자동 모드
            if st.session_state.current_analysis is None or st.session_state.current_analysis.get('path') != current_file_path:
                with st.spinner("자동 분석 중..."):
                    result = evaluator.analyze(img)
                    result['path'] = current_file_path
                    st.session_state.current_analysis = result
            
            st.image(img, width='stretch')

        # 분석 결과 시각화
        if st.session_state.current_analysis:
            st.divider()
            st.subheader("🔬 분석 결과 (ROI 마스킹)")
            overlay_img = evaluator.get_overlay_image(img, st.session_state.current_analysis)
            st.image(overlay_img, width='stretch', caption="Blue: Real Coin | Red: Reflection")

    with col2:
        st.subheader("📝 라벨링 및 검증")
        
        if st.session_state.current_analysis:
            analysis = st.session_state.current_analysis
            with st.container():
                st.markdown("<div class='status-box'>", unsafe_allow_html=True)
                st.write("**추정 물성치**")
                st.write(f"조도(Ra): `{analysis['roughness']:.4f}` um")
                st.write(f"광택도(Gloss): `{analysis['gloss']:.2f}` deg")
                st.write(f"반사광 감지: `{'성공' if analysis['has_reflection'] else '실패'}`")
                
                st.divider()
                st.markdown(f"**🤖 AI 예측 결과: <span style='color:#00ff00; font-size:1.2em;'>{analysis.get('predicted_label', 'Unknown')}</span>**", unsafe_allow_html=True)
                st.markdown("</div>", unsafe_allow_html=True)
            
            surface_types = ["BA", "#4", "HL", "SM", "2B", "Other"]
            default_label_idx = 0
            for i, t in enumerate(surface_types):
                if t.lower() in current_file_path.lower():
                    default_label_idx = i
                    break
            
            label = st.selectbox("표면 종류 선택", surface_types, index=default_label_idx)
            mask_quality = st.radio("마스킹 품질", ["Good ✅", "Bad ❌", "Uncertain ❓"], horizontal=True)
            notes = st.text_area("메모 (선택사항)", placeholder="특이사항 입력...")
            
            st.divider()
            
            c1, c2 = st.columns(2)
            
            if c1.button("확정 및 저장 💾", type="primary"):
                save_path = Path(target_dir)
                save_path.mkdir(parents=True, exist_ok=True)
                
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                file_id = f"{timestamp}_{Path(current_file_path).stem}"
                
                # 결과 이미지 저장
                overlay_img = evaluator.get_overlay_image(img, analysis)
                overlay_img.save(save_path / f"{file_id}_mask.png")
                
                # CSV 기록
                csv_path = save_path / "metadata.csv"
                file_exists = csv_path.exists()
                
                with open(csv_path, mode='a', newline='', encoding='utf-8') as f:
                    writer = csv.writer(f)
                    if not file_exists:
                        writer.writerow(["id", "original_path", "label", "roughness", "gloss", "mask_quality", "notes", "timestamp", "mode"])
                    
                    writer.writerow([
                        file_id, 
                        current_file_path, 
                        label, 
                        f"{analysis['roughness']:.6f}", 
                        f"{analysis['gloss']:.2f}", 
                        mask_quality, 
                        notes, 
                        timestamp,
                        f"Box_{mode}"
                    ])
                
                st.success("저장 완료!")
                if st.session_state.file_index < len(st.session_state.files_to_process) - 1:
                    st.session_state.file_index += 1
                    st.session_state.current_analysis = None
                    st.rerun()
                else:
                    st.balloons()
                    st.info("모든 파일 처리가 완료되었습니다!")

            if c2.button("건너뛰기 ⏩"):
                if st.session_state.file_index < len(st.session_state.files_to_process) - 1:
                    st.session_state.file_index += 1
                    st.session_state.current_analysis = None
                    st.rerun()
                else:
                    st.info("마지막 파일입니다.")
        else:
            st.warning("분석 결과가 없습니다. 자동 또는 수동으로 분석을 실행해 주세요.")

    with st.expander("ℹ️ 표면 종류별 기준 물성 (DB 참조)"):
        if db.df is not None:
            st.dataframe(db.df[["product_name", "roughness_avg", "gloss_avg"]].sort_values("product_name"))

else:
    st.info("사이드바에서 원본 이미지 폴더를 지정하고 '목록 갱신'을 눌러주세요.")
