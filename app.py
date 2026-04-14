import streamlit as st
import torch
import os
from PIL import Image
import time
import cv2
import numpy as np
import pandas as pd
from segment_anything import sam_model_registry, SamPredictor
from vsams.models.classifier import SurfaceClassifier
from vsams.utils.db_handler import query_recommendation
from vsams.utils.substrate_db import SubstrateDB

# --- Config & Setup ---
st.set_page_config(
    page_title="V-SAMS Prototype",
    page_icon="🛡️",
    layout="wide"
)

# Load Model (Cached)
@st.cache_resource
def load_model():
    checkpoint_path = 'checkpoints/v_sams_model.pth'
    model = SurfaceClassifier(num_materials=6, num_finishes=7)
    
    msg = ""
    status = "mock"
    
    if os.path.exists(checkpoint_path):
        try:
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
substrate_db = SubstrateDB()

# Load SAM Model (Cached)
@st.cache_resource
def load_sam():
    checkpoint = "checkpoints/sam_vit_l_0b3195.pth"
    model_type = "vit_l"
    if not os.path.exists(checkpoint):
        return None, "SAM checkpoint not found"
    
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        sam = sam_model_registry[model_type](checkpoint=checkpoint)
        sam.to(device=device)
        predictor = SamPredictor(sam)
        return predictor, "SAM Loaded"
    except Exception as e:
        return None, f"Error loading SAM: {e}"

sam_predictor, sam_msg = load_sam()

# --- Property Estimation Engine ---
def estimate_properties(images):
    roughness_scores = []
    gloss_scores = []
    
    for img in images:
        open_cv_image = np.array(img)
        open_cv_image = open_cv_image[:, :, ::-1].copy()
        gray = cv2.cvtColor(open_cv_image, cv2.COLOR_BGR2GRAY)
        
        # Roughness (Ra)
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        mag = np.sqrt(sobelx**2 + sobely**2)
        edge_density = np.mean(mag)
        ra_est = 0.3 + (edge_density / 50.0) * 0.5 
        ra_est = float(np.clip(ra_est, 0.3, 1.0))
        roughness_scores.append(ra_est)
        
        # Glossiness (%)
        contrast = gray.std()
        avg_bright = gray.mean()
        gloss_est = 10.0 + (contrast / 80.0) * 30.0
        if avg_bright < 40 or avg_bright > 220:
             gloss_est *= 0.8
        gloss_est = float(np.clip(gloss_est, 5.0, 50.0))
        gloss_scores.append(gloss_est)
        
    return np.mean(roughness_scores), np.mean(gloss_scores)

# --- Prediction Logic ---
def run_sam_masking(images):
    if sam_predictor is None:
        return images
    
    masked_images = []
    for img in images:
        cv_img = np.array(img)
        cv_img = cv_img[:, :, ::-1].copy()
        h, w, _ = cv_img.shape
        sam_predictor.set_image(cv_img)
        input_point = np.array([[w // 2, h // 2]])
        input_label = np.array([1])
        masks, _, _ = sam_predictor.predict(input_point, input_label, multimask_output=False)
        mask = masks[0]
        masked_img = cv_img.copy()
        masked_img[~mask] = 0
        y_indices, x_indices = np.where(mask)
        if len(y_indices) > 0:
            y1, y2, x1, x2 = y_indices.min(), y_indices.max(), x_indices.min(), x_indices.max()
            masked_img = masked_img[y1:y2, x1:x2]
        masked_img = masked_img[:, :, ::-1].copy()
        masked_images.append(Image.fromarray(masked_img))
    return masked_images

def predict_multiple(images, image_names, use_sam=False):
    processed_for_ai = images
    if use_sam and sam_predictor:
        processed_for_ai = run_sam_masking(images)

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
        all_features = []
        
        for img in images:
            input_tensor = preprocess(img).unsqueeze(0).to(device)
            with torch.no_grad():
                mat_logits, fin_logits = model_obj(input_tensor)
                all_mat_probs.append(torch.softmax(mat_logits, dim=1)[0])
                all_fin_probs.append(torch.softmax(fin_logits, dim=1)[0])
        
        for img in processed_for_ai:
            input_tensor = preprocess(img).unsqueeze(0).to(device)
            with torch.no_grad():
                feat = model_obj.extract_features(input_tensor)
                all_features.append(feat.cpu().numpy().flatten())
        
        avg_mat_probs = torch.stack(all_mat_probs).mean(dim=0)
        avg_fin_probs = torch.stack(all_fin_probs).mean(dim=0)
        avg_features = np.mean(all_features, axis=0)
        
        MATERIALS = ["Metal", "Plastic", "Glass", "Painted", "Wood", "Other"]
        FINISHES = ["Mirror", "Rough", "Hairline", "Matte", "Glossy", "Pattern", "Other"]
        mat_idx = torch.argmax(avg_mat_probs).item()
        fin_idx = torch.argmax(avg_fin_probs).item()
        
        est_ra, est_gloss = estimate_properties(images)
        visual_matches = substrate_db.find_visual_match(avg_features, k=10)
        property_matches = substrate_db.find_closest_top_k(est_ra, est_gloss, k=10)
        
        # Hybrid Scoring
        scored_results = []
        vis_dict = {m['product_name']: m['similarity'] for m in visual_matches} if visual_matches else {}
        prop_dict = {m['product_name']: m['distance'] for m in property_matches} if property_matches else {}
        all_products = set(vis_dict.keys()) | set(prop_dict.keys())
        
        for p in all_products:
            v_score = vis_dict.get(p, 0.0)
            p_dist = prop_dict.get(p, 10.0)
            p_score = 1.0 / (1.0 + p_dist)
            hybrid_score = (v_score * 0.6) + (p_score * 0.4)
            scored_results.append({
                'product_name': p,
                'hybrid_score': hybrid_score,
                'visual_score': v_score,
                'property_score': p_score,
                'property_dist': p_dist
            })
        
        scored_results = sorted(scored_results, key=lambda x: x['hybrid_score'], reverse=True)
        final_winner = scored_results[0] if scored_results else None
        
        return {
            "Material": MATERIALS[mat_idx],
            "Finish": FINISHES[fin_idx],
            "Est_Roughness": est_ra,
            "Est_Gloss": est_gloss,
            "Final_Winner": final_winner,
            "Visual_Matches": visual_matches,
            "Property_Matches": property_matches,
            "Hybrid_Scores": scored_results[:5],
            "Processed_Images": processed_for_ai,
            "Scores": {MATERIALS[mat_idx]: avg_mat_probs[mat_idx].item(), FINISHES[fin_idx]: avg_fin_probs[fin_idx].item()}
        }
    
    return {"Material": "Other", "Finish": "Other", "Est_Roughness": 0.5, "Est_Gloss": 20.0}

# --- Language Config ---
LANG_DICT = {
    "Korean": {
        "title": "🛡️ V-SAMS 통합 분석 허브",
        "subtitle": "다각도 표면 분석 및 지능형 제품 식별 시스템",
        "sidebar_header": "분석 설정",
        "upload_label": "이미지 업로드 (최대 5장)",
        "upload_tip": "💡 팁: 여러 각도의 표면 사진을 업로드하면 정확도가 향상됩니다.",
        "debug_checkbox": "상세 분석 로그 표시",
        "img_acq": "1. 다각도 이미지 수집",
        "ai_analysis": "2. AI 통합 분석 및 물성 추정",
        "analyzing": "데이터 종합 분석 및 식별 중...",
        "success": "분석이 완료되었습니다.",
        "recommendation": "3. 최종 제품 식별 및 정밀 분석 결과",
    }
}

# --- UI Layout ---
with st.sidebar:
    txt = LANG_DICT["Korean"]
    st.header(txt["sidebar_header"])
    uploaded_files = st.file_uploader(txt["upload_label"], type=['jpg', 'png', 'jpeg'], accept_multiple_files=True)
    if uploaded_files and len(uploaded_files) > 5:
        st.warning("최대 5장까지만 처리가 가능합니다.")
        uploaded_files = uploaded_files[:5]
    st.info(txt["upload_tip"])
    use_sam = st.toggle("SAM 정밀 마스킹 활성화", value=True)
    if use_sam and not sam_predictor:
        st.error("SAM 모델을 로드할 수 없습니다.")
    
    # Library Load Status
    if substrate_db.visual_library is not None:
        st.sidebar.success(f"📚 Visual Library Loaded ({len(substrate_db.visual_library)} items)")
    else:
        st.sidebar.error("❌ Visual Library NOT Loaded")
        
    show_debug = st.checkbox(txt["debug_checkbox"])

st.title(txt["title"])
st.markdown(txt["subtitle"])

if uploaded_files:
    st.subheader(txt["img_acq"])
    cols = st.columns(len(uploaded_files))
    processed_images = []
    file_names = []
    for i, file in enumerate(uploaded_files):
        img = Image.open(file).convert("RGB")
        processed_images.append(img)
        file_names.append(file.name)
        with cols[i]:
            st.image(img, caption=f"입력 {i+1}", use_container_width=True)
    
    st.divider()
    st.subheader(txt["ai_analysis"])
    with st.spinner(txt["analyzing"]):
        result = predict_multiple(processed_images, file_names, use_sam=use_sam)
    
    st.success(txt["success"])
    
    # 3. Final Product Identification
    st.header(txt["recommendation"])
    winner = result.get('Final_Winner')
    if winner:
        w_col1, w_col2 = st.columns([1, 1])
        with w_col1:
            st.success(f"🏆 **최종 판정 제품**: {winner['product_name']}")
            st.write(f"**종합 신뢰도**: {winner['hybrid_score']:.4f}")
            st.write(f"(시각 일치율: {winner['visual_score']:.2f}, 물성 일치율: {winner['property_score']:.2f})")
            
            # Additional recommendation
            recs = query_recommendation(result['Material'], result['Finish'])
            if recs:
                st.warning(f"📝 **추천 보호필름**: {recs[0]['name']} (및 {len(recs)-1}개 더)")
        
        with w_col2:
            vis_matches = result.get('Visual_Matches', [])
            ref_path = next((m['ref_image'] for m in vis_matches if m['product_name'] == winner['product_name']), None)
            if ref_path and os.path.exists(ref_path):
                st.image(Image.open(ref_path), caption=f"DB 참조 이미지: {winner['product_name']}", width=400)
    
    st.divider()
    
    # --- 🛠️ ADVANCED DEBUG DASHBOARD ---
    st.header("🛠️ 고급 분석 대시보드 (Advanced Debug)")
    
    # ROI display
    st.subheader("A. SAM 마스킹 결과 (ROI 분석)")
    masked_imgs = result.get("Processed_Images", [])
    if use_sam and len(masked_imgs) > 0:
        r_cols = st.columns(len(masked_imgs))
        for i, m_img in enumerate(masked_imgs):
            with r_cols[i]:
                st.image(m_img, caption=f"ROI {i+1}", use_container_width=True)
    else:
        st.info("SAM 마스킹이 비활성 상태이거나 실패했습니다.")

    # Top-K tables
    dcol1, dcol2 = st.columns(2)
    with dcol1:
        st.subheader("B. 시각적 유사도 TOP 5")
        v_matches = result.get('Visual_Matches', [])
        if v_matches:
            st.table(pd.DataFrame(v_matches)[['product_name', 'similarity']].head(5))
            
    with dcol2:
        st.subheader("C. 물성치 유사도 TOP 5")
        p_matches = result.get('Property_Matches', [])
        if p_matches:
            st.table(pd.DataFrame(p_matches)[['product_name', 'distance', 'roughness', 'gloss']].head(5))

    # Raw metrics
    st.subheader("D. 엔진 원천 지표")
    mcol1, mcol2, mcol3 = st.columns(3)
    mcol1.metric("추정 조도 (Ra)", f"{result['Est_Roughness']:.4f}")
    mcol2.metric("추정 광택도 (%)", f"{result['Est_Gloss']:.2f}")
    mcol3.metric("AI 분류 (Material/Finish)", f"{result['Material']} / {result['Finish']}")

else:
    st.info("상단 설정창에서 이미지를 업로드하여 분석을 시작하세요.")
