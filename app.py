import streamlit as st
import torch
import os
from PIL import Image
import time
from vsams.models.classifier import SurfaceClassifier
from vsams.utils.db_handler import query_recommendation, load_db, save_db

# ... (Existing Language Dict - omitted for brevity in replacement, but I need to make sure I don't delete it.
# Actually, I should probably append the Admin strings to the dictionary first or handle it inline if it's easier.
# Let's verify where line 7 is first.


# --- Config & Setup ---
st.set_page_config(
    page_title="V-SAMS Prototype",
    page_icon="🛡️",
    layout="wide"
)

# Load Model (Cached)
# Load Model (Cached)
@st.cache_resource
def load_model():
    checkpoint_path = 'checkpoints/v_sams_model.pth'
    model = SurfaceClassifier(num_materials=6, num_finishes=7)
    
    msg = ""
    status = "mock"
    
    if os.path.exists(checkpoint_path):
        try:
            # MacBook Pro M2 Pro (Apple Silicon) MPS 가동
            if torch.backends.mps.is_available():
                device = torch.device("mps")
            else:
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            state_dict = torch.load(checkpoint_path, map_location=device)
            model.load_state_dict(state_dict)
            model.to(device)
            msg = f"실제 AI 모델 가동 중 (분석 장치: {device})"
            status = "real"
        except Exception as e:
            msg = f"모델 가중치 로드 실패: {e}"
            status = "error"
    else:
        msg = "모델 파일 없음 (MOCK 시뮬레이션 모드)"
        status = "mock"
    
    model.eval()
    return model, msg, status

model_obj, load_msg, load_status = load_model()

# UI 상단에 로드 상태 표시
if load_status == "real":
    st.toast(load_msg)
elif load_status == "mock":
    st.toast(load_msg)
else:
    st.error(load_msg)

# --- Prediction Logic ---
def predict_multiple(images, image_names):
    """
    여러 장의 이미지를 분석하여 종합적인 결과를 도출합니다.
    """
    if load_status == "real":
        from torchvision import transforms
        preprocess = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ])
        
        device = next(model_obj.parameters()).device
        
        all_mat_probs = []
        all_fin_probs = []
        
        for img in images:
            input_tensor = preprocess(img).unsqueeze(0).to(device)
            with torch.no_grad():
                mat_logits, fin_logits = model_obj(input_tensor)
                all_mat_probs.append(torch.softmax(mat_logits, dim=1)[0])
                all_fin_probs.append(torch.softmax(fin_logits, dim=1)[0])
        
        # 확률 평균 계산 (Ensemble)
        avg_mat_probs = torch.stack(all_mat_probs).mean(dim=0)
        avg_fin_probs = torch.stack(all_fin_probs).mean(dim=0)
        
        MATERIALS = ["Metal", "Plastic", "Glass", "Painted", "Wood", "Other"]
        FINISHES = ["Mirror", "Rough", "Hairline", "Matte", "Glossy", "Pattern", "Other"]
        
        mat_idx = torch.argmax(avg_mat_probs).item()
        fin_idx = torch.argmax(avg_fin_probs).item()
        
        return {
            "Material": MATERIALS[mat_idx],
            "Finish": FINISHES[fin_idx],
            "Scores": {
                MATERIALS[mat_idx]: avg_mat_probs[mat_idx].item(),
                FINISHES[fin_idx]: avg_fin_probs[fin_idx].item()
            }
        }
    
    # Simulation logic (Mock)
    time.sleep(1.0)
    # 갯수에 상관없이 첫 번째 파일명 기반으로 간단히 시뮬레이션
    name = image_names[0].lower()
    if "mirror" in name or "ba" in name or "sm" in name:
        return {"Material": "Metal", "Finish": "Mirror", "Scores": {"Metal": 0.95, "Mirror": 0.98}}
    elif "hl" in name or "hairline" in name:
        return {"Material": "Metal", "Finish": "Hairline", "Scores": {"Metal": 0.92, "Hairline": 0.94}}
    elif "rough" in name or "4" in name:
        return {"Material": "Metal", "Finish": "Pattern", "Scores": {"Metal": 0.88, "Pattern": 0.85}}
    else:
        return {"Material": "Other", "Finish": "Other", "Scores": {"Other": 0.50, "Other": 0.50}}

def get_substrate_type(material, finish):
    """
    AI 분류 결과를 현장 용어로 맵핑합니다 (Based on 260410 memo.txt).
    """
    mapping = {
        ("Metal", "Mirror"): "Sus (BA/SM) / High-gloss Steel",
        ("Metal", "Hairline"): "Sus (HL / Hairline)",
        ("Metal", "Pattern"): "Sus (#4) / Patterned Metal",
        ("Metal", "Rough"): "Rough Metal / Sandblast",
        ("Metal", "Matte"): "Matte Color Steel",
        ("Painted", "Glossy"): "Glossy Color Steel",
        ("Painted", "Matte"): "Matte/Ultra-matte Color Steel",
    }
    return mapping.get((material, finish), f"{material} ({finish})")

# --- Language Config ---
LANG_DICT = {
    "English": {
        "title": "🛡️ V-SAMS Analysis Hub",
        "subtitle": "**Multi-View Surface Analysis System** (Prototype)",
        "sidebar_header": "Setup",
        "upload_label": "Upload Surface Images (Max 5)",
        "upload_tip": "💡 Tip: Upload multiple angles for better accuracy.",
        "debug_checkbox": "Show Debug Info",
        "img_acq": "1. Multi-View Acquisition",
        "img_caption": "Input Photo",
        "ai_analysis": "2. Integrated AI Analysis",
        "analyzing": "Synthesizing multi-view data...",
        "success": "Analysis Complete",
        "det_material": "Material Group",
        "det_finish": "Surface Finish",
        "det_substrate": "Identified Substrate (Field Term)",
        "mat_conf": "Material Confidence",
        "finish_conf": "Texture Confidence",
        "recommendation": "3. Decision Output",
        "best_match": "### 🔍 Surface Identification Result",
        "welcome_title": "### Welcome to V-SAMS Multi-View Demo",
        "welcome_msg": """
        This system analyzes surface properties from multiple photos to identify specific industrial substrates.
        
        **Workflow:**
        1.  **Upload** 1 to 5 photos of the material (different angles recommended).
        2.  **AI Engine** synthesizes all views to identify Material and Finish.
        3.  **Result** outputs the professional substrate name (e.g., Sus BA, HL).
        """,
        "mode_select": "Select Mode",
        "mode_user": "Analysis Demo",
        "mode_admin": "Developer Info"
    },
    "Korean": {
        "title": "🛡️ V-SAMS 분석 허브",
        "subtitle": "**다각도 표면 분석 시스템** (Prototype)",
        "sidebar_header": "환경 설정",
        "upload_label": "이미지 업로드 (최대 5장)",
        "upload_tip": "💡 팁: 여러 각도의 사진을 올리면 정확도가 높아집니다.",
        "debug_checkbox": "디버그 정보 표시",
        "img_acq": "1. 다각도 이미지 획득",
        "img_caption": "입력 이미지",
        "ai_analysis": "2. 통합 AI 분석 결과",
        "analyzing": "모든 각도의 데이터를 종합 분석 중...",
        "success": "분석 완료",
        "det_material": "재질 그룹",
        "det_finish": "표면 마감",
        "det_substrate": "식별된 표면 종류 (현장 용어)",
        "mat_conf": "재질 신뢰도",
        "finish_conf": "텍스처 신뢰도",
        "recommendation": "3. 최종 판단 결과",
        "best_match": "### 🔍 표면 종류 식별 결과",
        "welcome_title": "### V-SAMS 다각도 데모에 오신 것을 환영합니다",
        "welcome_msg": """
        이 시스템은 여러 장의 사진을 종합 분석하여 실제 현장에서 사용하는 표면 종류를 식별합니다.
        
        **워크플로우:**
        1.  **업로드**: 피착제 사진을 1~5장까지 업로드합니다 (다양한 각도 권장).
        2.  **AI 통합 분석**: 업로드된 모든 사진을 종합하여 재질과 마감을 판단합니다.
        3.  **결과 출력**: 식별된 표면의 현장 용어(예: Sus BA, HL 등)를 출력합니다.
        """,
        "mode_select": "모드 선택",
        "mode_user": "분석 데모",
        "mode_admin": "개발 정보"
    }
}

# --- UI Layout ---
with st.sidebar:
    lang_code = st.radio("Language / 언어", ["English", "Korean"], index=1)
    txt = LANG_DICT[lang_code]
    st.divider()
    mode = st.radio(txt["mode_select"], [txt["mode_user"], txt["mode_admin"]])
    st.divider()
    
    if mode == txt["mode_user"]:
        st.header(txt["sidebar_header"])
        uploaded_files = st.file_uploader(txt["upload_label"], type=['jpg', 'png', 'jpeg'], accept_multiple_files=True)
        if uploaded_files and len(uploaded_files) > 5:
            st.warning("Max 5 files allowed. Only the first 5 will be processed.")
            uploaded_files = uploaded_files[:5]
        st.info(txt["upload_tip"])
        
        if st.checkbox(txt["debug_checkbox"]):
            st.write(f"Loaded Files: {len(uploaded_files) if uploaded_files else 0}")
            st.write("Device: MPS" if torch.backends.mps.is_available() else "Device: CPU")

# User Mode UI
if mode == txt["mode_user"]:
    st.title(txt["title"])
    st.markdown(txt["subtitle"])

    if uploaded_files:
        # 1. Display Images in Carousel-like Grid
        st.subheader(txt["img_acq"])
        cols = st.columns(len(uploaded_files))
        processed_images = []
        file_names = []
        
        for i, file in enumerate(uploaded_files):
            img = Image.open(file).convert("RGB")
            processed_images.append(img)
            file_names.append(file.name)
            with cols[i]:
                st.image(img, caption=f"{txt['img_caption']} {i+1}", use_container_width=True)
            
        st.divider()

        # 2. Integrated AI Analysis
        st.subheader(txt["ai_analysis"])
        with st.spinner(txt["analyzing"]):
            result = predict_multiple(processed_images, file_names)
        
        st.success(txt["success"])
        
        # Mapping to Field Term
        substrate_name = get_substrate_type(result['Material'], result['Finish'])
        
        col1, col2 = st.columns([1, 1])
        with col1:
            st.metric(txt["det_material"], result['Material'])
            st.progress(result['Scores'][result['Material']], text=f"{txt['mat_conf']}")
            
        with col2:
            st.metric(txt["det_finish"], result['Finish'])
            st.progress(result['Scores'][result['Finish']], text=f"{txt['finish_conf']}")

        st.divider()

        # 3. Final Decision Result
        st.header(txt["recommendation"])
        st.markdown(f"{txt['best_match']}")
        st.info(f"✨ **{substrate_name}**")
        
        # Additional Field Context based on Substrate
        if "Sus" in substrate_name:
            st.write("💡 *Sus surfaces require careful adhesive selection based on gloss levels (BA/SM).*")
        elif "Color Steel" in substrate_name:
            st.write("💡 *Coated surfaces may have variable surface energy; check matte/gloss levels.*")
            
    else:
        st.markdown(txt["welcome_title"])
        st.markdown(txt["welcome_msg"])

# Admin/Dev Info
else:
    st.title(txt["mode_admin"])
    st.subheader("System Architecture")
    st.write("- Model: ResNet50 Dual-Head Classifier")
    st.write("- Feature Extraction: 2048-dim vectors")
    st.write("- Ensemble Method: Logit/Probability Averaging")
    st.write("- Optimized for: Apple M2 Pro (MPS)")
    
    st.divider()
    st.subheader("Raw Prediction Mapping Targets")
    st.json({
        "Materials": ["Metal", "Plastic", "Glass", "Painted", "Wood", "Other"],
        "Finishes": ["Mirror", "Rough", "Hairline", "Matte", "Glossy", "Pattern", "Other"]
    })

