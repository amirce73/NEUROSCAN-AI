"""
NeuroScan AI - Clinical Brain CT Benchmark & Real Patient Case Suite
Generates both calibrated synthetic CT scans and processes real uploaded
patient case studies into isolated single-brain diagnostic slices.
"""

import cv2
import numpy as np
import os
from typing import Dict, Any, List, Tuple
from PIL import Image


def create_base_brain_ct(size: int = 512) -> Tuple[np.ndarray, np.ndarray, Tuple[int, int]]:
    """Synthesizes realistic brain anatomical architecture."""
    img = np.zeros((size, size), dtype=np.uint8)
    center = (size // 2, size // 2)

    # 1. Outer scalp & subcutaneous fat
    cv2.ellipse(img, center, (190, 220), 0, 0, 360, 60, -1)

    # 2. Dense skull bone (calvarium)
    cv2.ellipse(img, center, (180, 210), 0, 0, 360, 240, -1)

    # 3. Inner table / Intracranial brain parenchyma
    brain_mask = np.zeros((size, size), dtype=np.uint8)
    cv2.ellipse(img, center, (168, 198), 0, 0, 360, 110, -1)
    cv2.ellipse(brain_mask, center, (168, 198), 0, 0, 360, 255, -1)

    # Add realistic soft-tissue micro-texture
    noise = np.random.normal(0, 3.5, (size, size)).astype(np.float32)
    brain_floats = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    img = np.where(brain_mask > 0, brain_floats, img)

    # 4. Symmetrical Lateral Ventricles
    cv2.ellipse(img, (center[0] - 22, center[1] - 25), (10, 32), -15, 0, 360, 32, -1)
    cv2.ellipse(img, (center[0] + 22, center[1] - 25), (10, 32), 15, 0, 360, 32, -1)
    cv2.rectangle(img, (center[0] - 3, center[1] - 10), (center[0] + 3, center[1] + 25), 34, -1)

    # 5. Interhemispheric fissure & cortical sulci
    cv2.line(img, (center[0], center[1] - 195), (center[0], center[1] + 195), 45, 1)
    for rad, angle in [(155, 30), (160, 60), (160, 120), (155, 150),
                       (155, 210), (160, 240), (160, 300), (155, 330)]:
        x1 = int(center[0] + (rad - 15) * np.cos(np.radians(angle)))
        y1 = int(center[1] + (rad - 15) * np.sin(np.radians(angle)))
        x2 = int(center[0] + rad * np.cos(np.radians(angle)))
        y2 = int(center[1] + rad * np.sin(np.radians(angle)))
        cv2.line(img, (x1, y1), (x2, y2), 48, 1)

    return img, brain_mask, center


def generate_glioblastoma_scan(size: int = 512) -> np.ndarray:
    """Case 1: Glioblastoma Multiforme (GBM) - Grade IV Malignant Cancer."""
    img, brain_mask, center = create_base_brain_ct(size)
    lx, ly = center[0] + 68, center[1] + 15

    # Vasogenic Edema around tumor
    cv2.ellipse(img, (lx, ly), (60, 68), 25, 0, 360, 75, -1)
    for dx, dy in [(30, 25), (-25, 35), (35, -20), (-30, -30), (0, 45)]:
        cv2.circle(img, (lx + dx, ly + dy), 24, 72, -1)

    # Infiltrative tumor core with irregular lobulated spicules
    pts = []
    num_pts = 16
    for i in range(num_pts):
        angle = i * (2 * np.pi / num_pts)
        r = 38 + 12 * np.sin(3 * angle) + 6 * np.cos(5 * angle)
        px = int(lx + r * np.cos(angle))
        py = int(ly + r * 1.15 * np.sin(angle))
        pts.append([px, py])
    cv2.fillPoly(img, [np.array(pts, dtype=np.int32)], 205)

    # Invasive peripheral nodules
    cv2.circle(img, (lx + 26, ly - 20), 16, 215, -1)
    cv2.circle(img, (lx - 24, ly + 26), 18, 210, -1)
    cv2.circle(img, (lx + 32, ly + 18), 17, 220, -1)
    cv2.circle(img, (lx - 12, ly - 28), 14, 205, -1)

    # Central Necrotic / Cystic Core
    cv2.ellipse(img, (lx, ly), (18, 22), 15, 0, 360, 48, -1)

    # Compression of right lateral ventricle
    cv2.ellipse(img, (center[0] + 20, center[1] - 25), (3, 14), 15, 0, 360, 50, -1)

    blurred = cv2.GaussianBlur(img, (5, 5), 1.2)
    final_img = np.where(brain_mask > 0, blurred, img)
    return cv2.cvtColor(final_img, cv2.COLOR_GRAY2BGR)


def generate_meningioma_scan(size: int = 512) -> np.ndarray:
    """Case 2: Benign Meningioma - Grade I."""
    img, brain_mask, center = create_base_brain_ct(size)
    lx, ly = center[0] - 105, center[1] - 55

    cv2.circle(img, (lx, ly), 36, 85, -1)
    cv2.circle(img, (lx, ly), 30, 185, -1)

    blurred = cv2.GaussianBlur(img, (3, 3), 0.8)
    final_img = np.where(brain_mask > 0, blurred, img)
    return cv2.cvtColor(final_img, cv2.COLOR_GRAY2BGR)


def generate_pituitary_adenoma_scan(size: int = 512) -> np.ndarray:
    """Case 3: Pituitary Macroadenoma / Sellar Lesion."""
    img, brain_mask, center = create_base_brain_ct(size)
    lx, ly = center[0], center[1] - 15
    cv2.ellipse(img, (lx, ly), (22, 26), 0, 0, 360, 178, -1)

    blurred = cv2.GaussianBlur(img, (3, 3), 0.8)
    final_img = np.where(brain_mask > 0, blurred, img)
    return cv2.cvtColor(final_img, cv2.COLOR_GRAY2BGR)


def generate_normal_brain_scan(size: int = 512) -> np.ndarray:
    """Case 4: Healthy Normal Brain CT Scan."""
    img, brain_mask, center = create_base_brain_ct(size)
    blurred = cv2.GaussianBlur(img, (3, 3), 0.5)
    final_img = np.where(brain_mask > 0, blurred, img)
    return cv2.cvtColor(final_img, cv2.COLOR_GRAY2BGR)


def extract_and_save_real_cases(samples_dir: str) -> List[Dict[str, str]]:
    """
    Scans for user-uploaded clinical files in samples directory,
    crops and standardizes them into clean SINGLE-BRAIN cases,
    and returns rich metadata for UI selection.
    """
    raw_mappings = [
        {
            "src": "HH_Case_Study_-_Fig_1_corrected22-1024x475.jpg",
            "dst": "real_case_1_temporal_gbm.png",
            "id": "real_1_gbm",
            "title_fa": "🔬 مورد واقعی ۱: گلیوبلاستوما گیجگاهی (Real Brain GBM)",
            "title_en": "Real Patient Case 1: Temporal Glioblastoma",
            "diagnosis": "Malignant"
        },
        {
            "src": "1842-1.avif",
            "dst": "real_case_2_meningioma.png",
            "id": "real_2_meningioma",
            "title_fa": "🔬 مورد واقعی ۲: مننژیوما اکسیپیتال (Real Meningioma)",
            "title_en": "Real Patient Case 2: Occipital Meningioma",
            "diagnosis": "Malignant"
        },
        {
            "src": "images (1).jpg",
            "dst": "real_case_3_frontal_mass.png",
            "id": "real_3_frontal",
            "title_fa": "🔬 مورد واقعی ۳: توده فرونتال (Real Frontal Mass)",
            "title_en": "Real Patient Case 3: Frontal Lesion",
            "diagnosis": "Malignant"
        },
        {
            "src": "images (2).jpg",
            "dst": "real_case_4_flair_hyperintense.png",
            "id": "real_4_flair",
            "title_fa": "🔬 مورد واقعی ۴: ضایعه T2/FLAIR هایپرسیگنال (Real MRI)",
            "title_en": "Real Patient Case 4: FLAIR Hyperintensity",
            "diagnosis": "Malignant"
        },
        {
            "src": "images (3).jpg",
            "dst": "real_case_5_ring_enhancing.png",
            "id": "real_5_ring",
            "title_fa": "🔬 مورد واقعی ۵: تومور با رینگ محیطی (Real Ring Lesion)",
            "title_en": "Real Patient Case 5: Ring-Enhancing Tumor",
            "diagnosis": "Malignant"
        },
        {
            "src": "images.jpg",
            "dst": "real_case_6_glioma_axial.png",
            "id": "real_6_axial",
            "title_fa": "🔬 مورد واقعی ۶: گلیوما بدخیم اکسیال (Real Axial Glioma)",
            "title_en": "Real Patient Case 6: Axial High-Grade Glioma",
            "diagnosis": "Malignant"
        }
    ]

    real_cases = []
    for m in raw_mappings:
        src_path = os.path.join(samples_dir, m["src"])
        dst_path = os.path.join(samples_dir, m["dst"])

        if os.path.exists(src_path):
            try:
                # Load with PIL for AVIF support
                pil_im = Image.open(src_path).convert('RGB')
                arr = cv2.cvtColor(np.array(pil_im), cv2.COLOR_RGB2BGR)

                # Isolate single brain
                h, w = arr.shape[:2]
                if float(w) / h >= 1.6:
                    half_w = w // 2
                    left_h = arr[:, :half_w]
                    right_h = arr[:, half_w:]
                    l_std = float(np.std(cv2.cvtColor(left_h, cv2.COLOR_BGR2GRAY)))
                    r_std = float(np.std(cv2.cvtColor(right_h, cv2.COLOR_BGR2GRAY)))
                    arr = right_h if r_std > l_std else left_h
                    h, w = arr.shape[:2]

                gray = cv2.cvtColor(arr, cv2.COLOR_BGR2GRAY)
                _, thresh = cv2.threshold(gray, 22, 255, cv2.THRESH_BINARY)
                cv2.rectangle(thresh, (0, 0), (w, h), 0, 4)
                cnts, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                candidates = []
                min_area = (h * w) * 0.04
                for c in cnts:
                    area = cv2.contourArea(c)
                    if area > min_area:
                        x, y, bw, bh = cv2.boundingRect(c)
                        asp = float(bw) / bh
                        if 0.50 < asp < 1.50 and bw > 65 and bh > 65:
                            candidates.append((x, y, bw, bh, area))

                if candidates:
                    def score_c(c):
                        x, y, bw, bh, area = c
                        crop = gray[y:y+bh, x:x+bw]
                        return float(np.std(crop)) * float(area)
                    best = max(candidates, key=score_c)
                    x, y, bw, bh, _ = best
                    pad = int(min(bw, bh) * 0.06)
                    x1 = max(0, x - pad)
                    y1 = max(0, y - pad)
                    x2 = min(w, x + bw + pad)
                    y2 = min(h, y + bh + pad)
                    single = arr[y1:y2, x1:x2]
                else:
                    single = arr

                # Make square and resize to 512x512
                sh, sw = single.shape[:2]
                max_d = max(sh, sw)
                sq = np.zeros((max_d, max_d, 3), dtype=np.uint8)
                dx = (max_d - sw) // 2
                dy = (max_d - sh) // 2
                sq[dy:dy+sh, dx:dx+sw] = single
                single_512 = cv2.resize(sq, (512, 512), interpolation=cv2.INTER_LANCZOS4)

                cv2.imwrite(dst_path, single_512)

                real_cases.append({
                    "id": m["id"],
                    "filename": m["dst"],
                    "filepath": dst_path,
                    "title_fa": m["title_fa"],
                    "title_en": m["title_en"],
                    "diagnosis": m["diagnosis"]
                })
            except Exception as e:
                print(f"Error processing real case {m['src']}: {e}")

    return real_cases


def generate_all_samples(output_dir: str = "samples") -> List[Dict[str, str]]:
    """Generates benchmark cases and includes processed real patient cases."""
    os.makedirs(output_dir, exist_ok=True)

    # 1. Synthetic Calibration Cases
    synth_cases = [
        {
            "id": "case_1_gbm_malignant",
            "filename": "case_1_glioblastoma_malignant.png",
            "title_fa": "📐 بنچمارک ۱: گلیوبلاستوما مولتی‌فرم (بدخیم / سرطانی)",
            "title_en": "Benchmark 1: Glioblastoma Multiforme (GBM)",
            "diagnosis": "Malignant",
            "generator": generate_glioblastoma_scan
        },
        {
            "id": "case_2_meningioma_benign",
            "filename": "case_2_meningioma_benign.png",
            "title_fa": "📐 بنچمارک ۲: مننژیوما هموژن (خوش‌خیم)",
            "title_en": "Benchmark 2: Benign Meningioma",
            "diagnosis": "Benign",
            "generator": generate_meningioma_scan
        },
        {
            "id": "case_3_adenoma_benign",
            "filename": "case_3_pituitary_adenoma.png",
            "title_fa": "📐 بنچمارک ۳: آدنوم هیپوفیز (خوش‌خیم)",
            "title_en": "Benchmark 3: Pituitary Adenoma",
            "diagnosis": "Benign",
            "generator": generate_pituitary_adenoma_scan
        },
        {
            "id": "case_4_normal_healthy",
            "filename": "case_4_normal_healthy_brain.png",
            "title_fa": "📐 بنچمارک ۴: سی‌تی اسکن نرمال و بدون تومور",
            "title_en": "Benchmark 4: Normal Healthy Brain CT",
            "diagnosis": "Normal",
            "generator": generate_normal_brain_scan
        }
    ]

    all_cases = []

    # 2. Extract Real Cases first if present
    real_cases = extract_and_save_real_cases(output_dir)
    all_cases.extend(real_cases)

    # 3. Add synthetic benchmarks
    for case in synth_cases:
        filepath = os.path.join(output_dir, case["filename"])
        if not os.path.exists(filepath):
            img_bgr = case["generator"](512)
            cv2.imwrite(filepath, img_bgr)
        all_cases.append({
            "id": case["id"],
            "filename": case["filename"],
            "filepath": filepath,
            "title_fa": case["title_fa"],
            "title_en": case["title_en"],
            "diagnosis": case["diagnosis"]
        })

    return all_cases


if __name__ == "__main__":
    cases = generate_all_samples("samples")
    print(f"Loaded and generated {len(cases)} total brain cases (Real & Benchmark).")
