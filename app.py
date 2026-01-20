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
    # Initialize with current label counts
    model = SurfaceClassifier(num_materials=6, num_finishes=7)
    
    msg = ""
    status = "mock"
    
    if os.path.exists(checkpoint_path):
        try:
            model.load_state_dict(torch.load(checkpoint_path, map_location='cpu'))
            msg = "✅ Real AI model weights loaded."
            status = "real"
        except Exception as e:
            msg = f"Error loading weights: {e}"
            status = "error"
    else:
        msg = "⚠️ Weight file not found. Running in MOCK/Simulation mode."
        status = "mock"
    
    model.eval()
    return model, msg, status

model, load_msg, load_status = load_model()

if load_status == "real":
    st.toast(load_msg)
elif load_status == "mock":
    st.toast(load_msg)
else:
    st.error(load_msg)

# --- Prediction Logic ---
def predict(image, image_name):
    """
    Real inference if model is trained, else simulation.
    """
    checkpoint_path = 'checkpoints/v_sams_model.pth'
    
    if os.path.exists(checkpoint_path):
        # Real Inference
        from torchvision import transforms
        preprocess = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ])
        input_tensor = preprocess(image).unsqueeze(0)
        
        with torch.no_grad():
            mat_logits, fin_logits = model(input_tensor)
            mat_probs = torch.softmax(mat_logits, dim=1)[0]
            fin_probs = torch.softmax(fin_logits, dim=1)[0]
            
        MATERIALS = ["Metal", "Plastic", "Glass", "Painted", "Wood", "Other"]
        FINISHES = ["Mirror", "Rough", "Hairline", "Matte", "Glossy", "Pattern", "Other"]
        
        mat_idx = torch.argmax(mat_probs).item()
        fin_idx = torch.argmax(fin_probs).item()
        
        return {
            "Material": MATERIALS[mat_idx],
            "Finish": FINISHES[fin_idx],
            "Scores": {
                MATERIALS[mat_idx]: mat_probs[mat_idx].item(),
                FINISHES[fin_idx]: fin_probs[fin_idx].item()
            }
        }
    
    # Simulation logic (Mock)
    time.sleep(1.0) 
    image_name = image_name.lower()
    
    if "mirror" in image_name or "shiny" in image_name or "101" in image_name:
        return {"Material": "Metal", "Finish": "Mirror", "Scores": {"Metal": 0.92, "Mirror": 0.95}}
    elif "rough" in image_name or "sandblast" in image_name or "305" in image_name:
        return {"Material": "Metal", "Finish": "Rough", "Scores": {"Metal": 0.88, "Rough": 0.91}}
    elif "paint" in image_name or "glossy" in image_name or "500" in image_name:
        return {"Material": "Painted", "Finish": "Glossy", "Scores": {"Painted": 0.94, "Glossy": 0.89}}
    else:
        return {"Material": "Metal", "Finish": "Mirror", "Scores": {"Metal": 0.60, "Mirror": 0.55}}

# --- Language Config ---
LANG_DICT = {
    "English": {
        "title": "🛡️ V-SAMS",
        "subtitle": "**Visual-based Surface Analysis & Matching System** (Proprietary Demo)",
        "sidebar_header": "Upload Environment",
        "upload_label": "Upload Product Image",
        "upload_tip": "💡 Tip: Try uploading images of metal or plastic surfaces.",
        "debug_checkbox": "Show Debug Info",
        "img_acq": "1. Image Acquisition",
        "img_caption": "Preprocessed Input",
        "ai_analysis": "2. AI Analysis Result",
        "analyzing": "Analyzing Texture & Material...",
        "success": "Analysis Complete",
        "det_material": "Detected Material",
        "det_finish": "Detected Finish",
        "mat_conf": "Material Confidence",
        "finish_conf": "Texture Confidence",
        "recommendation": "3. Intelligent Recommendation",
        "best_match": "### ✨ Best Match",
        "desc_label": "**Description:**",
        "specs_label": "#### Specs",
        "report_btn": "📄 Generate Report",
        "no_match": "No perfect match found in current database.",
        "welcome_title": "### Welcome to V-SAMS Demo",
        "welcome_msg": """
        This system analyzes surface properties to recommend protective films.
        
        **Workflow:**
        1.  **Upload** a photo of the implementation material.
        2.  **AI** identifies Material Type and Surface Finish.
        3.  **System** matches the best Protective Film from the database.
        """,
        "mode_select": "Select Mode",
        "mode_user": "User Demo",
        "mode_admin": "DB Management",
        "admin_title": "🔧 Database Management",
        "add_section": "Add New Product",
        "p_id": "Product ID (e.g., PF-900)",
        "p_name": "Product Name",
        "p_desc": "Description",
        "p_base": "Base Material (e.g., PET, PE)",
        "p_adh": "Adhesive Type (e.g., Silicone)",
        "p_tack": "Tack Force (e.g., 50 gf/25mm)",
        "tgt_mat": "Target Materials",
        "tgt_fin": "Target Finishes",
        "save_btn": "💾 Save Product",
        "saved_msg": "✅ Product saved successfully to database!"
    },
    "Korean": {
        "title": "🛡️ V-SAMS",
        "subtitle": "**시각 기반 표면 분석 및 보호필름 매칭 시스템** (Proprietary Demo)",
        "sidebar_header": "환경 설정",
        "upload_label": "제품 이미지 업로드",
        "upload_tip": "💡 팁: 금속이나 플라스틱 표면 사진을 업로드해보세요.",
        "debug_checkbox": "디버그 정보 표시",
        "img_acq": "1. 이미지 획득 (Image Acquisition)",
        "img_caption": "전처리된 입력 이미지",
        "ai_analysis": "2. AI 분석 결과 (AI Analysis)",
        "analyzing": "텍스처 및 재질 분석 중...",
        "success": "분석 완료",
        "det_material": "감지된 재질",
        "det_finish": "감지된 마감",
        "mat_conf": "재질 신뢰도",
        "finish_conf": "텍스처 신뢰도",
        "recommendation": "3. 지능형 제품 추천 (Recommendation)",
        "best_match": "### ✨ 최적 매칭 제품",
        "desc_label": "**제품 설명:**",
        "specs_label": "#### 상세 스펙",
        "report_btn": "📄 리포트 생성",
        "no_match": "현재 데이터베이스에서 완벽하게 일치하는 제품을 찾을 수 없습니다.",
        "welcome_title": "### V-SAMS 데모에 오신 것을 환영합니다",
        "welcome_msg": """
        이 시스템은 표면 특성을 분석하여 최적의 보호 필름을 추천합니다.
        
        **워크플로우:**
        1.  **업로드**: 피착제(제품)의 사진을 업로드합니다.
        2.  **AI 분석**: 인공지능이 재질 종류와 표면 마감 상태를 식별합니다.
        3.  **추천**: 데이터베이스에서 가장 적합한 보호 필름을 매칭합니다.
        """,
        "mode_select": "모드 선택",
        "mode_user": "사용자 데모",
        "mode_admin": "DB 관리 도구",
        "admin_title": "🔧 데이터베이스 관리",
        "add_section": "신규 제품 등록",
        "p_id": "제품 ID (예: PF-900)",
        "p_name": "제품명",
        "p_desc": "설명",
        "p_base": "기재 (Base Material)",
        "p_adh": "점착제 (Adhesive)",
        "p_tack": "점착력 (Tack Force)",
        "tgt_mat": "타겟 재질 (복수 선택)",
        "tgt_fin": "타겟 마감 (복수 선택)",
        "save_btn": "💾 제품 저장",
        "saved_msg": "✅ 제품이 데이터베이스에 저장되었습니다!"
    }
}

# --- UI Layout ---
with st.sidebar:
    # Language Toggle
    lang_code = st.radio("Language / 언어", ["English", "Korean"], index=1)
    txt = LANG_DICT[lang_code]
    
    st.divider()
    
    # Mode Toggle
    mode = st.radio(txt["mode_select"], [txt["mode_user"], txt["mode_admin"]])
    
    st.divider()
    
    if mode == txt["mode_user"]:
        st.header(txt["sidebar_header"])
        uploaded_file = st.file_uploader(txt["upload_label"], type=['jpg', 'png', 'jpeg'])
        st.info(txt["upload_tip"])
        
        if st.checkbox(txt["debug_checkbox"]):
            st.write("System Status: Online")
            st.write("Model: ResNet50-DualHead")
            st.write("Database: v1.0 (JSON)")

# User Mode UI
if mode == txt["mode_user"]:
    st.title(txt["title"])
    st.markdown(txt["subtitle"])

    col1, col2 = st.columns([1, 1])

    if uploaded_file is not None:
        # 1. Display Image
        image = Image.open(uploaded_file)
        with col1:
            st.subheader(txt["img_acq"])
            st.image(image, caption=txt["img_caption"], use_container_width=True)
            
        # 2. AI Analysis
        with col2:
            st.subheader(txt["ai_analysis"])
            
            with st.spinner(txt["analyzing"]):
                result = predict(image, uploaded_file.name)
            
            # Visualize Confidence
            st.success(txt["success"])
            
            m_col, f_col = st.columns(2)
            with m_col:
                st.metric(txt["det_material"], result['Material'], f"{result['Scores'][result['Material']]*100:.1f}%")
            with f_col:
                st.metric(txt["det_finish"], result['Finish'], f"{result['Scores'][result['Finish']]*100:.1f}%")
                
            st.progress(result['Scores'][result['Material']], text=f"{txt['mat_conf']}: {result['Material']}")
            st.progress(result['Scores'][result['Finish']], text=f"{txt['finish_conf']}: {result['Finish']}")

        st.divider()

        # 3. Recommendation
        st.header(txt["recommendation"])
        
        recommendations = query_recommendation(result['Material'], result['Finish'])
        
        if recommendations:
            best_match = recommendations[0]
            st.markdown(f"{txt['best_match']}: {best_match['name']}")
            
            rec_col1, rec_col2 = st.columns([1, 2])
            
            # Product Image (Mock)
            with rec_col1:
                # Try to load mock product image if exists
                img_path = best_match.get('image_url', '')
                if os.path.exists(img_path):
                    st.image(img_path, width=200)
                else:
                    st.markdown("Easy-to-peel Protection")
            
            with rec_col2:
                st.markdown(f"{txt['desc_label']} {best_match['description']}")
                st.markdown(txt["specs_label"])
                st.json(best_match['specs'])
                
                st.button(txt["report_btn"])
        else:
            st.warning(txt["no_match"])

    else:
        # Welcome Screen
        st.markdown(txt["welcome_title"])
        st.markdown(txt["welcome_msg"])

# Admin Mode UI
else:
    st.title(txt["admin_title"])
    
    # 1. Product List (Read-only view)
    st.subheader("Current Database")
    current_db = load_db()
    if current_db:
        st.dataframe(current_db) # Simple Table View
    
    st.divider()
    
    # 2. Add New Product Form
    with st.form("product_form"):
        st.subheader(txt["add_section"])
        
        col1, col2 = st.columns(2)
        with col1:
            p_id = st.text_input(txt["p_id"])
            p_name = st.text_input(txt["p_name"])
            p_base = st.text_input(txt["p_base"])
        with col2:
            p_desc = st.text_input(txt["p_desc"])
            p_adh = st.text_input(txt["p_adh"])
            p_tack = st.text_input(txt["p_tack"])
            
        params_mat = st.multiselect(txt["tgt_mat"], ["Metal", "Plastic", "Glass", "Painted"])
        params_fin = st.multiselect(txt["tgt_fin"], ["Mirror", "Rough", "Glossy", "Matte", "Hairline", "Sandblast"])
        
        submitted = st.form_submit_button(txt["save_btn"])
        
        if submitted:
            if not p_id or not p_name:
                st.error("ID and Name are required!")
            else:
                new_product = {
                    "id": p_id,
                    "name": p_name,
                    "description": p_desc,
                    "specs": {
                        "base_material": p_base,
                        "adhesive": p_adh,
                        "tack_force": p_tack
                    },
                    "target_condition": {
                        "material_category": params_mat,
                        "finish_type": params_fin,
                        "risk_residue": "Medium"
                    },
                    "image_url": "images/placeholder.png"
                }
                
                current_db.append(new_product)
                save_db(current_db)
                st.success(txt["saved_msg"])
                time.sleep(1)
                st.rerun()

