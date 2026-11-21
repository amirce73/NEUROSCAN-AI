"""
NeuroScan AI - Core Computer Vision Engine (OpenCV)
Advanced Brain CT Scan Image Processing, Skull Stripping,
Tumor Segmentation, and Morphometric Feature Extraction.
"""

import cv2
import numpy as np
import base64
from typing import Dict, Any, Tuple, List, Optional


class BrainCTProcessor:
    """
    High-performance OpenCV pipeline designed specifically for Brain CT / MRI scans.
    Features:
    - Bilateral edge-preserving noise filtering
    - Precise morphological Skull Stripping (Brain Extraction)
    - Contrast-Limited Adaptive Histogram Equalization (CLAHE)
    - Morphological & Watershed lesion segmentation
    - Quantitative clinical morphometry (area, diameter, location, asymmetry)
    """

    def __init__(self, target_size: Tuple[int, int] = (512, 512), pixel_spacing_mm: float = 0.43):
        self.target_size = target_size
        self.pixel_spacing_mm = pixel_spacing_mm  # Typical CT slice pixel spacing (mm)

    def isolate_single_brain(self, img: np.ndarray) -> np.ndarray:
        """
        Detects multi-slice montages, grids, or side-by-side scans and
        automatically isolates and extracts a SINGLE centered brain slice.
        """
        h, w = img.shape[:2]
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # 1. Check for extreme side-by-side aspect ratio (e.g., dual scan w/h >= 1.5)
        aspect = float(w) / h
        if aspect >= 1.6:
            half_w = w // 2
            left_half = img[:, :half_w]
            right_half = img[:, half_w:]
            left_std = float(np.std(cv2.cvtColor(left_half, cv2.COLOR_BGR2GRAY)))
            right_std = float(np.std(cv2.cvtColor(right_half, cv2.COLOR_BGR2GRAY)))
            primary = right_half if right_std > left_std else left_half
            return self.isolate_single_brain(primary)

        # 2. Find significant brain contours / cranial components
        _, thresh = cv2.threshold(gray, 22, 255, cv2.THRESH_BINARY)
        # Suppress image outer border lines to avoid picking up the whole frame
        cv2.rectangle(thresh, (0, 0), (w, h), 0, 4)

        cnts, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        candidates = []
        min_area = (h * w) * 0.04
        for c in cnts:
            area = cv2.contourArea(c)
            if area > min_area:
                x, y, bw, bh = cv2.boundingRect(c)
                comp_aspect = float(bw) / bh
                if 0.50 < comp_aspect < 1.50 and bw > 65 and bh > 65:
                    candidates.append((x, y, bw, bh, area))

        if len(candidates) > 1:
            # Multiple brains in a grid/sheet: pick the slice with highest lesion/structural variance
            def score_cand(cand):
                x, y, bw, bh, area = cand
                crop = gray[y:y+bh, x:x+bw]
                return float(np.std(crop)) * float(area)

            best = max(candidates, key=score_cand)
            x, y, bw, bh, _ = best
            pad = int(min(bw, bh) * 0.07)
            x1 = max(0, x - pad)
            y1 = max(0, y - pad)
            x2 = min(w, x + bw + pad)
            y2 = min(h, y + bh + pad)
            single = img[y1:y2, x1:x2]
        elif len(candidates) == 1:
            x, y, bw, bh, _ = candidates[0]
            # If the single candidate occupies less than 75% of the frame (e.g. lots of black border/labels)
            if float(bw * bh) / (w * h) < 0.75:
                pad = int(min(bw, bh) * 0.05)
                x1 = max(0, x - pad)
                y1 = max(0, y - pad)
                x2 = min(w, x + bw + pad)
                y2 = min(h, y + bh + pad)
                single = img[y1:y2, x1:x2]
            else:
                single = img
        else:
            single = img

        # Make perfectly square to preserve anatomical proportions
        sh, sw = single.shape[:2]
        max_dim = max(sh, sw)
        square = np.zeros((max_dim, max_dim, 3), dtype=np.uint8)
        dx = (max_dim - sw) // 2
        dy = (max_dim - sh) // 2
        square[dy:dy+sh, dx:dx+sw] = single

        return cv2.resize(square, self.target_size, interpolation=cv2.INTER_LANCZOS4)

    def load_image(self, input_data: Any) -> np.ndarray:
        """Loads an image from a file path, base64 string, or raw bytes into a BGR uint8 array."""
        if isinstance(input_data, np.ndarray):
            img = input_data.copy()
        elif isinstance(input_data, str):
            if input_data.startswith("data:image"):
                # Handle base64 data URI
                encoded_data = input_data.split(",")[1]
                img_bytes = base64.b64decode(encoded_data)
                nparr = np.frombuffer(img_bytes, np.uint8)
                img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            elif input_data.startswith("base64,"):
                encoded_data = input_data[7:]
                img_bytes = base64.b64decode(encoded_data)
                nparr = np.frombuffer(img_bytes, np.uint8)
                img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            else:
                # File path - handle AVIF or non-standard formats with PIL fallback
                img = cv2.imread(input_data, cv2.IMREAD_COLOR)
                if img is None:
                    try:
                        from PIL import Image
                        pil_img = Image.open(input_data).convert('RGB')
                        img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
                    except Exception:
                        img = None
        elif isinstance(input_data, bytes):
            nparr = np.frombuffer(input_data, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        else:
            raise ValueError("Unsupported image input format.")

        if img is None:
            raise ValueError("Failed to decode image data.")

        # Automatically isolate single brain from multi-slice/montage images
        single_brain = self.isolate_single_brain(img)
        return single_brain

    def apply_brain_window(self, gray: np.ndarray, is_raw_dicom_hu: bool = False) -> np.ndarray:
        """
        Normalizes and prepares CT scan contrast.
        If is_raw_dicom_hu is True, applies standard brain window (L=40, W=80).
        For standard 8-bit CT images (0-255), enhances dynamic range without destroying
        the distinction between bone (>220) and soft brain tissue (70-150).
        """
        if is_raw_dicom_hu:
            # 12-bit/16-bit HU windowing
            hu_min = 40 - 40
            hu_max = 40 + 40
            windowed = np.clip(gray, hu_min, hu_max)
            return cv2.normalize(windowed, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        else:
            # Standard 8-bit CT contrast optimization
            return cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

    def extract_brain_mask(self, gray: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Performs robust Skull Stripping to separate intracranial brain tissue
        from the bright calvarial bone and external artifacts.
        Returns (brain_mask, stripped_brain_gray).
        """
        # Step 1: Smooth noise while preserving anatomical cranial boundaries
        blurred = cv2.bilateralFilter(gray, d=9, sigmaColor=75, sigmaSpace=75)

        # Step 2: Head thresholding (everything that is not deep black background)
        _, head_thresh = cv2.threshold(blurred, 25, 255, cv2.THRESH_BINARY)

        # Step 3: Find head contours
        contours, _ = cv2.findContours(head_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            mask = np.ones_like(gray) * 255
            return mask, gray

        # Largest external contour is the head
        head_contour = max(contours, key=cv2.contourArea)
        head_mask = np.zeros_like(gray)
        cv2.drawContours(head_mask, [head_contour], -1, 255, thickness=cv2.FILLED)

        # Step 4: Detect skull bone (in 8-bit CT scans, calvarial bone has highest intensity > 215)
        _, bone_thresh = cv2.threshold(blurred, 215, 255, cv2.THRESH_BINARY)
        bone_thresh = cv2.bitwise_and(bone_thresh, bone_thresh, mask=head_mask)

        # If skull bone contour is detected, the intracranial cavity is the region enclosed by it
        bone_cnts, _ = cv2.findContours(bone_thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        intracranial_mask = np.zeros_like(gray)
        if bone_cnts:
            # Find the largest bone ring or cavity inside
            # Erode head mask inwards slightly to remove scalp/skin
            kernel_inward = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (19, 19))
            inner_cavity = cv2.erode(head_mask, kernel_inward, iterations=1)
            # Exclude dense bone
            intracranial_mask = cv2.bitwise_and(inner_cavity, cv2.bitwise_not(bone_thresh))
        else:
            kernel_inward = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
            intracranial_mask = cv2.erode(head_mask, kernel_inward, iterations=1)

        # Morphological opening to remove thin bridges and isolate parenchyma
        kernel_clean = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        opened = cv2.morphologyEx(intracranial_mask, cv2.MORPH_OPEN, kernel_clean, iterations=1)

        # Connected component to keep the dominant brain parenchyma
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(opened)
        if num_labels > 1:
            largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
            brain_mask = np.uint8(labels == largest_label) * 255
        else:
            brain_mask = opened

        # Fill internal holes (ventricles, necrotic cores, or lesions should remain inside mask)
        cnts, _ = cv2.findContours(brain_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if cnts:
            main_brain_cnt = max(cnts, key=cv2.contourArea)
            filled_mask = np.zeros_like(gray)
            cv2.drawContours(filled_mask, [main_brain_cnt], -1, 255, thickness=cv2.FILLED)
            brain_mask = filled_mask

        # Apply mask to original grayscale
        stripped_brain = cv2.bitwise_and(gray, gray, mask=brain_mask)

        return brain_mask, stripped_brain

    def enhance_contrast(self, stripped_gray: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """
        Applies Contrast-Limited Adaptive Histogram Equalization (CLAHE)
        specifically restricted to the brain parenchyma for optimal lesion contrast.
        """
        clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
        enhanced = clahe.apply(stripped_gray)
        enhanced = cv2.bitwise_and(enhanced, enhanced, mask=mask)
        return enhanced

    def segment_tumor_candidates(self, enhanced_gray: np.ndarray, brain_mask: np.ndarray) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
        """
        Segments abnormal lesions / tumor candidates using adaptive thresholding,
        watershed gradient, and morphological contour filtering.
        Returns (tumor_mask, list_of_detected_lesions).
        """
        # Calculate mean & std of the healthy brain parenchyma
        brain_pixels = enhanced_gray[brain_mask > 0]
        if len(brain_pixels) == 0:
            return np.zeros_like(enhanced_gray), []

        mean_val = float(np.mean(brain_pixels))
        std_val = float(np.std(brain_pixels))

        # Lesions often present as hyperdense masses or heterogeneous mixed-density regions
        # Threshold at mean + 1.15 * std to capture abnormal hyperdensity
        thresh_val = min(240, max(115, int(mean_val + 1.10 * std_val)))
        _, raw_lesion_mask = cv2.threshold(enhanced_gray, thresh_val, 255, cv2.THRESH_BINARY)
        raw_lesion_mask = cv2.bitwise_and(raw_lesion_mask, raw_lesion_mask, mask=brain_mask)

        # Morphological opening to eliminate tiny speckles, followed by closing
        kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
        cleaned_mask = cv2.morphologyEx(raw_lesion_mask, cv2.MORPH_OPEN, kernel_open, iterations=1)
        cleaned_mask = cv2.morphologyEx(cleaned_mask, cv2.MORPH_CLOSE, kernel_close, iterations=2)

        # Find candidate lesion contours
        contours, _ = cv2.findContours(cleaned_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        total_brain_area = float(np.count_nonzero(brain_mask))
        min_lesion_area = max(110.0, total_brain_area * 0.0035)  # Meaningful lesion threshold
        max_lesion_area = total_brain_area * 0.40  # Less than 40% of brain

        lesions = []
        filtered_mask = np.zeros_like(enhanced_gray)

        # Midline X for brain hemisphere localization
        h, w = enhanced_gray.shape
        midline_x = w / 2.0

        for cnt in contours:
            area_px = cv2.contourArea(cnt)
            if area_px < min_lesion_area or area_px > max_lesion_area:
                continue

            # Convex hull and perimeter
            hull = cv2.convexHull(cnt)
            hull_area = cv2.contourArea(hull)
            solidity = float(area_px / hull_area) if hull_area > 0 else 0.0
            perimeter_px = cv2.arcLength(cnt, True)
            circularity = float(4 * np.pi * area_px / (perimeter_px ** 2)) if perimeter_px > 0 else 0.0

            # Bounding box & equivalent diameter
            x, y, bw, bh = cv2.boundingRect(cnt)
            equiv_diameter_px = np.sqrt(4 * area_px / np.pi)

            # Centroid
            M = cv2.moments(cnt)
            if M["m00"] != 0:
                cx = float(M["m10"] / M["m00"])
                cy = float(M["m01"] / M["m00"])
            else:
                cx, cy = float(x + bw / 2), float(y + bh / 2)

            # Anatomical Localization
            hemisphere = "نیمکره راست (Right Hemisphere)" if cx > midline_x else "نیمکره چپ (Left Hemisphere)"

            # Lobar approximation based on vertical & horizontal coordinates
            y_ratio = cy / h
            x_dist_midline = abs(cx - midline_x) / (w / 2)

            if y_ratio < 0.38:
                lobe = "لوب پیشانی / فرونتال (Frontal Lobe)"
            elif y_ratio > 0.70:
                lobe = "لوب پس‌سری / اکسیپیتال (Occipital Lobe)"
            elif x_dist_midline > 0.40:
                lobe = "لوب گیجگاهی / تمپورال (Temporal Lobe)"
            elif 0.38 <= y_ratio <= 0.70 and x_dist_midline <= 0.40:
                lobe = "ناحیه مرکزی / پریتال و بطنی (Parietal / Periventricular)"
            else:
                lobe = "لوب آهیانه / پریتال (Parietal Lobe)"

            # Physical measurements in millimeters and cm²
            area_mm2 = area_px * (self.pixel_spacing_mm ** 2)
            area_cm2 = area_mm2 / 100.0
            diameter_mm = equiv_diameter_px * self.pixel_spacing_mm

            # Intensity metrics within lesion
            lesion_pixels = enhanced_gray[y:y+bh, x:x+bw]
            lesion_cnt_mask = np.zeros((bh, bw), dtype=np.uint8)
            cv2.drawContours(lesion_cnt_mask, [cnt - np.array([x, y])], -1, 255, -1)
            pts_vals = lesion_pixels[lesion_cnt_mask > 0]

            mean_density = float(np.mean(pts_vals)) if len(pts_vals) > 0 else 0.0
            std_density = float(np.std(pts_vals)) if len(pts_vals) > 0 else 0.0
            heterogeneity_index = float(std_density / (mean_density + 1e-5))

            # Draw onto filtered mask
            cv2.drawContours(filtered_mask, [cnt], -1, 255, thickness=cv2.FILLED)

            lesions.append({
                "contour": cnt.tolist(),
                "bbox": [int(x), int(y), int(bw), int(bh)],
                "centroid": [round(cx, 1), round(cy, 1)],
                "area_px": int(area_px),
                "area_cm2": round(area_cm2, 2),
                "area_mm2": round(area_mm2, 1),
                "diameter_mm": round(diameter_mm, 1),
                "perimeter_px": round(perimeter_px, 1),
                "circularity": round(circularity, 3),
                "solidity": round(solidity, 3),
                "hemisphere": hemisphere,
                "lobe": lobe,
                "mean_density": round(mean_density, 1),
                "std_density": round(std_density, 1),
                "heterogeneity_index": round(heterogeneity_index, 3)
            })

        # Sort lesions by size descending and keep the most significant (primary + major satellites)
        lesions.sort(key=lambda item: item["area_px"], reverse=True)
        lesions = lesions[:4]

        return filtered_mask, lesions

    def calculate_brain_asymmetry(self, brain_gray: np.ndarray, brain_mask: np.ndarray) -> float:
        """
        Calculates anatomical asymmetry between left and right brain hemispheres.
        Malignant tumors and significant mass lesions induce high structural asymmetry / midline shift.
        Returns asymmetry index (0.0 = completely symmetrical, 1.0 = highly asymmetrical).
        """
        h, w = brain_gray.shape
        mid_x = w // 2

        left_hemi = brain_gray[:, :mid_x]
        right_hemi = cv2.flip(brain_gray[:, mid_x:2 * mid_x], 1)

        diff = cv2.absdiff(left_hemi, right_hemi)
        mask_mid = brain_mask[:, :mid_x]

        valid_diff = diff[mask_mid > 0]
        if len(valid_diff) == 0:
            return 0.0

        mean_diff = float(np.mean(valid_diff))
        asymmetry_score = min(1.0, mean_diff / 50.0)
        return round(asymmetry_score, 3)

    def generate_pipeline_steps(self, bgr_image: np.ndarray) -> Dict[str, str]:
        """
        Executes and captures every step of the OpenCV pipeline as base64 images
        for display in the UI's interactive pipeline debugger.
        """
        gray = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)
        windowed = self.apply_brain_window(gray)
        brain_mask, stripped_brain = self.extract_brain_mask(windowed)
        enhanced = self.enhance_contrast(stripped_brain, brain_mask)
        tumor_mask, lesions = self.segment_tumor_candidates(enhanced, brain_mask)

        # Step 6: Contours & Bounding overlay
        annotated = bgr_image.copy()
        if lesions:
            for i, lesion in enumerate(lesions):
                cnt = np.array(lesion["contour"], dtype=np.int32)
                # Outer glow and contour
                cv2.drawContours(annotated, [cnt], -1, (0, 255, 255), 2)
                x, y, bw, bh = lesion["bbox"]
                cv2.rectangle(annotated, (x, y), (x + bw, y + bh), (0, 140, 255), 2)
                label = f"TUMOR #{i+1}: {lesion['area_cm2']} cm2"
                cv2.putText(annotated, label, (x, max(20, y - 8)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2, cv2.LINE_AA)

        # Step 7: Colorized Parenchyma Mask
        colored_brain = cv2.applyColorMap(enhanced, cv2.COLORMAP_BONE)

        def to_b64(img_array: np.ndarray) -> str:
            if len(img_array.shape) == 2:
                img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2BGR)
            _, buf = cv2.imencode('.png', img_array)
            return "data:image/png;base64," + base64.b64encode(buf).decode('utf-8')

        return {
            "step1_raw": to_b64(bgr_image),
            "step2_windowed": to_b64(windowed),
            "step3_brain_mask": to_b64(brain_mask),
            "step4_stripped_brain": to_b64(stripped_brain),
            "step5_clahe_enhanced": to_b64(enhanced),
            "step6_tumor_mask": to_b64(tumor_mask),
            "step7_annotated": to_b64(annotated),
            "step8_bone_colormap": to_b64(colored_brain)
        }

    def process_scan(self, input_data: Any) -> Dict[str, Any]:
        """
        Complete end-to-end OpenCV execution on a brain CT scan.
        Returns all extracted masks, metrics, lesions, and processed images.
        """
        bgr = self.load_image(input_data)
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

        windowed = self.apply_brain_window(gray)
        brain_mask, stripped_brain = self.extract_brain_mask(windowed)
        enhanced = self.enhance_contrast(stripped_brain, brain_mask)
        tumor_mask, lesions = self.segment_tumor_candidates(enhanced, brain_mask)
        asymmetry_score = self.calculate_brain_asymmetry(enhanced, brain_mask)

        brain_area_px = int(np.count_nonzero(brain_mask))
        brain_area_cm2 = round(brain_area_px * (self.pixel_spacing_mm ** 2) / 100.0, 2)

        has_tumor = len(lesions) > 0
        primary_lesion = lesions[0] if has_tumor else None

        return {
            "image_bgr": bgr,
            "image_gray": gray,
            "brain_mask": brain_mask,
            "stripped_brain": stripped_brain,
            "enhanced_brain": enhanced,
            "tumor_mask": tumor_mask,
            "lesions": lesions,
            "has_tumor": has_tumor,
            "primary_lesion": primary_lesion,
            "lesion_count": len(lesions),
            "brain_area_cm2": brain_area_cm2,
            "asymmetry_score": asymmetry_score
        }
