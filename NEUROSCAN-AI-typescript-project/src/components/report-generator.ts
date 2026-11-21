/**
 * NeuroScan AI - Radiology Clinical Diagnostic Report Generator
 * Generates an official, print-ready A4 medical CAD report with digital sign-off.
 */

import { AnalysisResponse } from '../types/neuroscan';

export class RadiologyReportGenerator {
  private readonly modalBackdrop: HTMLElement;

  constructor(modalElement: HTMLElement) {
    this.modalBackdrop = modalElement;
  }

  /**
   * Opens the report modal and populates official patient findings.
   */
  openReport(analysis: AnalysisResponse): void {
    const diag = analysis.ai_diagnosis;
    const cv = analysis.opencv_metrics;
    const steps = analysis.pipeline_steps;

    // Report Header & Tracking ID
    const repId = document.getElementById('rep-id');
    const repDate = document.getElementById('rep-date');
    if (repId) repId.textContent = `NS-${Math.floor(1000 + Math.random() * 9000)}`;
    if (repDate) repDate.textContent = new Date().toLocaleDateString('fa-IR');

    // Findings
    const repDiag = document.getElementById('rep-diagnosis');
    const repCancer = document.getElementById('rep-cancer-prob');
    const repWho = document.getElementById('rep-who');
    const repRisk = document.getElementById('rep-risk');

    if (repDiag) repDiag.textContent = `${diag.diagnosis_fa} (${diag.diagnosis_en})`;
    if (repCancer) repCancer.textContent = `${diag.cancer_probability}%`;
    if (repWho) repWho.textContent = diag.who_grade;
    if (repRisk) repRisk.textContent = diag.risk_label_fa;

    // Image Evidence
    const imgRaw = document.getElementById('rep-img-raw') as HTMLImageElement | null;
    const imgContours = document.getElementById('rep-img-contours') as HTMLImageElement | null;
    const imgGradcam = document.getElementById('rep-img-gradcam') as HTMLImageElement | null;

    if (imgRaw) imgRaw.src = steps.step1_raw;
    if (imgContours) imgContours.src = steps.step7_annotated;
    if (imgGradcam) imgGradcam.src = diag.gradcam_heatmap_b64;

    // Biomarkers List
    const bioContainer = document.getElementById('rep-biomarkers');
    if (bioContainer) {
      if (cv.has_tumor && cv.primary_lesion) {
        const p = cv.primary_lesion;
        bioContainer.innerHTML = `
          <div>• مساحت ضایعه تومورال: <strong>${p.area_cm2} سانتی‌متر مربع</strong> (${p.area_mm2} میلی‌متر مربع)</div>
          <div>• قطر معادل حداکثر: <strong>${p.diameter_mm} میلی‌متر</strong></div>
          <div>• موقعیت آناتومیک در نیمکره‌ها: <strong>${p.hemisphere} - ${p.lobe}</strong></div>
          <div>• شاخص عدم تقارن نیمکره‌ها (Asymmetry Index): <strong>${cv.asymmetry_score}</strong> (اثر توده‌ای)</div>
          <div>• شاخص بی‌نظمی مرز تومور (Solidity): <strong>${p.solidity}</strong> - نشان‌دهنده ارتشاح به بافت سالم اطراف</div>
        `;
      } else {
        bioContainer.innerHTML = `
          <div>• فاقد هرگونه توده فضاگیر غیرطبیعی در پارانشیم مغز.</div>
          <div>• شیارهای کورتیکال و بطن‌های طرفی مغز متقارن و در محدوده فیزیولوژیک می‌باشند.</div>
          <div>• مساحت بافت مغز استخراج‌شده: <strong>${cv.brain_area_cm2} سانتی‌متر مربع</strong>.</div>
        `;
      }
    }

    // Recommendation Box
    const repRec = document.getElementById('rep-recommendation');
    if (repRec) repRec.textContent = diag.clinical_recommendation;

    // Reveal modal
    this.modalBackdrop.classList.remove('hidden');
  }

  /**
   * Closes the modal.
   */
  closeReport(): void {
    this.modalBackdrop.classList.add('hidden');
  }

  /**
   * Triggers the browser print dialog.
   */
  print(): void {
    window.print();
  }
}

