import cv2
import numpy as np
from PIL import Image, ImageDraw
import torch
from pathlib import Path

class SurfaceEvaluator:
    """
    V-SAMS 핵심 분석 엔진:
    동전(Source)과 반사광(Target)의 물리적 상관관계를 분석하여 
    표면 조도(Ra)와 광택도(Gloss)를 추정합니다.
    """
    
    def __init__(self, device=None):
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.classifier = None 
        
    def analyze(self, image, custom_boxes=None):
        """이미지를 분석하여 물성치를 반환합니다."""
        if isinstance(image, Image.Image):
            img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        else:
            img_cv = image
            
        # 1. 박스 결정 (수동 또는 자동)
        boxes = custom_boxes
        if boxes is None or len(boxes) < 2:
            auto_boxes = self._auto_detect_boxes(img_cv)
            if auto_boxes:
                boxes = auto_boxes
            else:
                return {"roughness": 0.5, "gloss": 20.0, "has_reflection": False, "error": "동전을 찾을 수 없습니다."}

        # 2. 영역 추출
        coin_box, ref_box = boxes[0], boxes[1]
        coin_img = self._crop_box(img_cv, coin_box)
        ref_img = self._crop_box(img_cv, ref_box)
        
        # 3. 안전 검사 (영역이 비어있는지 확인)
        if coin_img is None or coin_img.size == 0 or ref_img is None or ref_img.size == 0:
            return {"roughness": 0.5, "gloss": 20.0, "has_reflection": False, "error": "분석 영역이 유효하지 않습니다 (너무 작거나 경계 밖)."}
        
        # 4. 물리 분석 수행
        try:
            ra_val = self._estimate_roughness(coin_img, ref_img)
            ra_val = max(0.0, min(1.0, float(ra_val)))
            
            gloss_val = self._estimate_gloss(coin_img, ref_img)
            gloss_val = max(0.0, float(gloss_val))
            
            return {
                "roughness": ra_val,
                "gloss": gloss_val,
                "has_reflection": True,
                "coin_box": coin_box,
                "ref_box": ref_box,
                "predicted_label": self._map_to_label(ra_val, gloss_val)
            }
        except Exception as e:
            return {"roughness": 0.5, "gloss": 20.0, "has_reflection": False, "error": f"분석 도중 오류 발생: {str(e)}"}

    def _auto_detect_boxes(self, img):
        """이미지 중앙 영역과 질감 분석을 결합하여 진짜 동전을 찾습니다."""
        h, w = img.shape[:2]
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # 전처리: 대비를 극대화하여 동전 윤곽 추출
        clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8,8))
        gray_pre = clahe.apply(gray)
        gray_pre = cv2.medianBlur(gray_pre, 7)
        
        circles = cv2.HoughCircles(
            gray_pre, cv2.HOUGH_GRADIENT, dp=1.1, minDist=w//5,
            param1=50, param2=30, minRadius=int(h*0.06), maxRadius=int(h*0.18)
        )
        
        if circles is not None:
            circles = np.around(circles[0, :]).astype(np.int32)
            
            best_candidate = None
            max_score = -1
            
            for c in circles:
                cx, cy, cr = c
                if cx-cr < 0 or cx+cr >= w or cy-cr < 0 or cy+cr >= h:
                    continue
                
                # 1. 위치 점수 (중앙에 가까울수록 높음)
                dist_to_center = np.sqrt((cx - w/2)**2 + (cy - h/2)**2)
                pos_score = 1.0 - (dist_to_center / (np.sqrt((w/2)**2 + (h/2)**2)))
                
                # 2. 질감 점수 (동전 내부의 복잡도)
                roi = gray[cy-cr:cy+cr, cx-cr:cx+cr]
                texture_score = np.std(roi) / 128.0
                
                # 최종 점수 (위치 + 질감)
                score = pos_score * 0.4 + texture_score * 0.6
                
                if score > max_score:
                    max_score = score
                    best_candidate = c
            
            if best_candidate is None:
                best_candidate = circles[0]
                
            x, y, r = best_candidate
            pad = int(r * 0.1)
            
            coin_box = [
                max(0, int(x - r - pad)),
                max(0, int(y - r - pad)),
                min(w, int(x + r + pad)),
                min(h, int(y + r + pad))
            ]
            
            # 반사광 영역 자동 생성
            ref_y1 = coin_box[3] + 15
            ref_height = coin_box[3] - coin_box[1]
            
            ref_box = [
                coin_box[0],
                min(h - 10, ref_y1),
                coin_box[2],
                min(h, ref_y1 + ref_height)
            ]
            
            return [coin_box, ref_box]
        
        return None

    def _crop_box(self, img, box):
        """주어진 좌표로 이미지를 자릅니다."""
        try:
            x1, y1, x2, y2 = map(int, box)
            if x2 <= x1 or y2 <= y1:
                return np.array([])
            return img[y1:y2, x1:x2]
        except:
            return np.array([])

    def _estimate_roughness(self, coin_img, ref_img):
        """광택도에 따른 가변 블러링으로 조도(Ra) 계산"""
        gray_coin = cv2.cvtColor(coin_img, cv2.COLOR_BGR2GRAY)
        gray_ref = cv2.cvtColor(ref_img, cv2.COLOR_BGR2GRAY)
        
        # 1. 먼저 광택도(Contrast)를 간이 측정
        std_coin = np.std(gray_coin)
        std_ref = np.std(gray_ref)
        temp_gloss = (std_ref / (std_coin + 1e-6)) * 600.0
        
        # 2. 광택도에 따른 블러링 전략 수립
        if temp_gloss > 350: # Mirror/Glossy (BA, SM)
            gray_coin = cv2.GaussianBlur(gray_coin, (3, 3), 0)
            gray_ref = cv2.GaussianBlur(gray_ref, (3, 3), 0)
            weight = 0.7 # Mirror는 조금 더 너그럽게
        else: # Hairline/Rough (HL, #4)
            gray_coin = cv2.medianBlur(gray_coin, 3)
            gray_ref = cv2.medianBlur(gray_ref, 5)
            weight = 0.9 # Rough는 더 엄격하게
        
        sharpness_coin = cv2.Laplacian(gray_coin, cv2.CV_64F).var()
        sharpness_ref = cv2.Laplacian(gray_ref, cv2.CV_64F).var()
        
        ratio = sharpness_ref / (sharpness_coin + 1e-6)
        ratio = np.clip(ratio, 0.0, 1.0)
        
        return weight * (1.0 - ratio) + 0.02

    def _estimate_gloss(self, coin_img, ref_img):
        """광택도(Gloss) 계산"""
        gray_coin = cv2.cvtColor(coin_img, cv2.COLOR_BGR2GRAY)
        gray_ref = cv2.cvtColor(ref_img, cv2.COLOR_BGR2GRAY)
        
        std_coin = np.std(gray_coin)
        std_ref = np.std(gray_ref)
        
        contrast_ratio = std_ref / (std_coin + 1e-6)
        contrast_ratio = np.clip(contrast_ratio, 0.0, 1.0)
        
        return contrast_ratio * 600.0

    def _map_to_label(self, ra, gloss):
        """물성치 기반 산업 현장 용어 맵핑 (SM < BA < HL < #4)"""
        # 초고광택Mirror 특성 반영
        if gloss > 400:
            if ra < 0.15: return "SM (Super Mirror)"
            if ra < 0.35: return "BA (Bright Annealed)"
            
        # 일반 마감 판정
        if ra < 0.05:
            return "SM (Super Mirror)"
        elif ra < 0.15:
            return "BA (Bright Annealed)"
        elif ra < 0.85: # HL 범위를 넓혀서 거친 HL도 수용
            return "HL (Hairline)"
        elif ra >= 0.85:
            return "#4 (Rough)"
        else:
            return "Other"

    def get_overlay_image(self, image, result):
        """분석 영역 시각화"""
        if not isinstance(image, Image.Image):
            image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            
        draw = ImageDraw.Draw(image)
        if "coin_box" in result:
            draw.rectangle(result["coin_box"], outline="blue", width=5)
        if "ref_box" in result:
            draw.rectangle(result["ref_box"], outline="red", width=5)
            
        return image
