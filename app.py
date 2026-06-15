import sys
from pathlib import Path

import pandas as pd
import streamlit as st
from PIL import Image
from streamlit_drawable_canvas import st_canvas

# 프로젝트 루트 경로 추가 (패키지 임포트 지원)
sys.path.append(str(Path(__file__).parent))

from vsams.analysis.surface_evaluator import SurfaceEvaluator
from vsams.utils.substrate_db import SubstrateDB

# --- 설정 및 스타일 ---
st.set_page_config(
    page_title="V-SAMS Surface Analysis Lab", page_icon="🔍", layout="wide"
)

st.markdown(
    """
<style>
    .main {
        background-color: #0e1117;
    }
    .stMetric {
        background-color: #1e2130;
        padding: 15px;
        border-radius: 10px;
        border: 1px solid #3e44fe;
    }
    .css-1r6slb0 { /* Sidebar padding */
        padding-top: 2rem;
    }
</style>
""",
    unsafe_allow_html=True,
)


# --- 리소스 로딩 (캐싱) ---
@st.cache_resource
def load_resources():
    db = SubstrateDB()
    evaluator = SurfaceEvaluator()
    return db, evaluator


# --- 앱 로직 시작 ---
st.title("🛡️ V-SAMS Surface Analysis Lab")
st.markdown("동전 반사광 기반 정밀 표면 분석 및 피착재 식별 시스템")

try:
    db, evaluator = load_resources()
except Exception as e:
    st.error(f"모델 또는 DB 로드 중 오류 발생: {e}")
    st.stop()

# --- 사이드바: 데이터 선택 ---
with st.sidebar:
    st.header("⚙️ 분석 설정")

    masking_mode = st.radio(
        "마스킹 모드", ["자동 (Auto)", "수동 박스 (Manual Box)"], horizontal=True
    )
    input_mode = st.radio("이미지 입력 방식", ["테스트 데이터셋", "직접 업로드"])

    target_image = None

    if input_mode == "테스트 데이터셋":
        test_root = Path("test_260420_surface")
        if test_root.exists():
            folders = [f.name for f in test_root.iterdir() if f.is_dir()]
            selected_folder = st.selectbox("표면 폴더 선택", folders)

            folder_path = test_root / selected_folder
            images = list(folder_path.glob("*.jpg"))
            selected_img_path = st.selectbox(
                "이미지 선택", [img.name for img in images]
            )

            if selected_img_path:
                target_image = Image.open(folder_path / selected_img_path)
        else:
            st.warning("테스트 데이터셋 경로를 찾을 수 없습니다.")
    else:
        uploaded_file = st.file_uploader("이미지 업로드", type=["jpg", "jpeg", "png"])
        if uploaded_file:
            target_image = Image.open(uploaded_file).convert("RGB")

    st.divider()
    if st.button("분석 실행 🚀", use_container_width=True, type="primary"):
        st.session_state.do_analysis = True
    else:
        if "do_analysis" not in st.session_state:
            st.session_state.do_analysis = False

# --- 메인 화면 대시보드 ---
if target_image:
    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("📸 입력 이미지")
        boxes = None
        if masking_mode == "자동 (Auto)":
            st.image(target_image, use_container_width=True)
        else:
            st.markdown(
                "**수동 가이드:** 1. **실제 동전** 영역 드래그 → 2. **반사광** 영역 드래그"
            )
            w_orig, h_orig = target_image.size
            canvas_width = 600
            canvas_height = int(h_orig * (canvas_width / w_orig))

            canvas_result = st_canvas(
                fill_color="rgba(0, 0, 255, 0.2)",
                stroke_width=2,
                stroke_color="#0000ff",
                background_image=target_image,
                update_streamlit=True,
                height=canvas_height,
                width=canvas_width,
                drawing_mode="rect",
                key="manual_canvas",
            )

            if st.button("캔버스 초기화 🧹"):
                st.rerun()

            if canvas_result.json_data is not None:
                boxes = []
                for obj in canvas_result.json_data["objects"]:
                    if obj["type"] == "rect":
                        x1 = int(obj["left"] * (w_orig / canvas_width))
                        y1 = int(obj["top"] * (h_orig / canvas_height))
                        x2 = int((obj["left"] + obj["width"]) * (w_orig / canvas_width))
                        y2 = int(
                            (obj["top"] + obj["height"]) * (h_orig / canvas_height)
                        )
                        boxes.append([x1, y1, x2, y2])

    with col2:
        if st.session_state.do_analysis:
            if masking_mode == "수동 박스 (Manual Box)" and (
                boxes is None or len(boxes) < 2
            ):
                st.warning("동전과 반사광 영역을 모두 지정해주세요. (최소 2개 박스)")
                st.session_state.do_analysis = False
            else:
                with st.spinner("SAM 기반 정밀 분석 중..."):
                    # 분석 실행
                    custom_boxes = boxes[:2] if boxes else None
                    result = evaluator.analyze(target_image, custom_boxes=custom_boxes)
                    overlay_img = evaluator.get_overlay_image(target_image, result)

                # DB 매칭
                match = db.find_closest(result["roughness"], result["gloss"])

                st.subheader("🔬 분석 결과 (ROI)")
                st.image(
                    overlay_img,
                    use_container_width=True,
                    caption="Green: Real Coin | Blue: Reflection",
                )

                # 결과 메트릭
                m1, m2 = st.columns(2)
                m1.metric("추정 조도 (Ra)", f"{result['roughness']:.4f} μm")
                m2.metric("추정 광택도 (Gloss)", f"{result['gloss']:.1f} GU")

                if match:
                    st.success(f"🏆 **최적 매칭 피착재**: {match['product_name']}")
                    st.info(
                        f"물성 정보: Ra {match['roughness_avg']:.4f} | Gloss {match['gloss_avg']:.2f}"
                    )
                else:
                    st.warning("DB에서 적절한 매칭 결과를 찾지 못했습니다.")
        else:
            st.info("분석 실행 버튼을 눌러주세요.")

    # 하단 분석 이력 (Optional)
    if st.session_state.do_analysis and "history" not in st.session_state:
        st.session_state.history = []

    if st.session_state.do_analysis:
        new_entry = {
            "파일명": (
                selected_img_path if input_mode == "테스트 데이터셋" else "Uploaded"
            ),
            "Ra": f"{result['roughness']:.4f}",
            "Gloss": f"{result['gloss']:.1f}",
            "판정결과": match["product_name"] if match else "N/A",
        }
        # 중복 추가 방지 (간단하게 마지막 항목과 비교)
        if not st.session_state.history or st.session_state.history[-1] != new_entry:
            st.session_state.history.append(new_entry)

    if "history" in st.session_state and st.session_state.history:
        st.divider()
        st.subheader("📊 최근 분석 이력")
        st.table(pd.DataFrame(st.session_state.history).tail(5))

else:
    st.info("좌측 사이드바에서 이미지를 선택하거나 업로드해 주세요.")

# --- 제품 정보 팁 ---
with st.expander("ℹ️ 표면별 물성 기준표 (Target Only)"):
    if db.df is not None:
        st.dataframe(
            db.df[["product_name", "roughness_avg", "gloss_avg"]].sort_values(
                "product_name"
            )
        )
    else:
        st.write("DB를 로드할 수 없습니다.")
