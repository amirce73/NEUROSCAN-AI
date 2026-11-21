"""
NeuroScan AI - Automated Pipeline Test Suite
Validates:
1. Benchmark CT scan generation
2. OpenCV Skull stripping and lesion segmentation
3. PyTorch Deep Feature Extraction & Grad-CAM Heatmap generation
4. Clinical Malignancy & Cancer Probability scoring
"""

import os
import sys

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core.sample_generator import generate_all_samples
from core.cv_engine import BrainCTProcessor
from core.neuro_classifier import BrainTumorClassifier


def run_tests():
    print("==================================================")
    print("🚀 Starting NeuroScan AI Validation Test...")
    print("==================================================")

    # Step 1: Generate clinical benchmark cases
    samples_dir = os.path.join(os.path.dirname(__file__), "..", "samples")
    cases = generate_all_samples(samples_dir)
    print(f"✅ Generated {len(cases)} clinical benchmark CT scans.")

    # Initialize engines
    processor = BrainCTProcessor()
    classifier = BrainTumorClassifier()
    print("✅ Initialized OpenCV Engine & PyTorch Classifier.")

    for case in cases:
        print("\n--------------------------------------------------")
        print(f"🔬 Testing: {case['title_en']} ({case['title_fa']})")
        print(f"📁 Path: {case['filepath']}")

        # 1. OpenCV Processing
        cv_res = processor.process_scan(case['filepath'])
        has_tumor = cv_res["has_tumor"]
        lesion_count = cv_res["lesion_count"]
        brain_area = cv_res["brain_area_cm2"]
        asymmetry = cv_res["asymmetry_score"]

        print(f"   [OpenCV] Brain Parenchyma Area: {brain_area} cm2")
        print(f"   [OpenCV] Asymmetry Index: {asymmetry}")
        print(f"   [OpenCV] Tumor Detected: {has_tumor} (Count: {lesion_count})")

        if has_tumor:
            p = cv_res["primary_lesion"]
            print(f"   [OpenCV] Primary Tumor Area: {p['area_cm2']} cm2 ({p['area_mm2']} mm2)")
            print(f"   [OpenCV] Approx Diameter: {p['diameter_mm']} mm")
            print(f"   [OpenCV] Location: {p['lobe']} | {p['hemisphere']}")
            print(f"   [OpenCV] Circularity: {p['circularity']} | Solidity: {p['solidity']}")

        # 2. AI Classification & Cancer Probability
        ai_res = classifier.analyze_scan(cv_res)
        print(f"   [AI] Diagnosis: {ai_res['diagnosis_en']} - {ai_res['diagnosis_fa']}")
        print(f"   [AI] Cancer / Malignancy Probability: {ai_res['cancer_probability']}%")
        print(f"   [AI] Risk Level: {ai_res['risk_level']} ({ai_res['risk_label_fa']})")
        print(f"   [AI] WHO Grade: {ai_res['who_grade']}")
        print(f"   [AI] Grad-CAM Base64 Generated: {len(ai_res['gradcam_heatmap_b64']) > 100}")
        print(f"   [AI] Diagnostic Overlay Base64: {len(ai_res['diagnostic_overlay_b64']) > 100}")

        # Verification asserts
        if case["diagnosis"] == "Malignant":
            assert has_tumor, "Malignant case must have detected tumor!"
            assert ai_res["cancer_probability"] >= 65.0, f"Expected high cancer prob, got {ai_res['cancer_probability']}%"
        elif case["diagnosis"] == "Normal":
            assert ai_res["cancer_probability"] < 20.0, f"Expected low cancer prob for normal, got {ai_res['cancer_probability']}%"
            assert not has_tumor, "Normal case should not detect tumors!"

    print("\n==================================================")
    print("🎉 ALL TESTS PASSED SUCCESSFULLY!")
    print("==================================================")


if __name__ == "__main__":
    run_tests()

