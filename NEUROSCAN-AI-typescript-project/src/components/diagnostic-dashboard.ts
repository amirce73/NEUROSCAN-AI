/**
 * NeuroScan AI - Diagnostic Intelligence Dashboard Controller
 * Manages reactive UI state for Risk Gauge, WHO classification,
 * quantitative radiomics, and clinical oncology protocol display.
 */

import { AnalysisResponse } from '../types/neuroscan';

export interface DashboardDOMElements {
  riskPercentage: HTMLElement;
  riskPill: HTMLElement;
  gaugeFill: SVGPathElement;
  gaugeNeedle: SVGGElement;
  barMalignant: HTMLElement;
  barBenign: HTMLElement;
  barNormal: HTMLElement;
  probValMalignant: HTMLElement;
  probValBenign: HTMLElement;
  probValNormal: HTMLElement;
  diagnosisText: HTMLElement;
  whoGradeText: HTMLElement;
  subpathologyText: HTMLElement;
  recommendationText: HTMLElement;
  factorsList: HTMLElement;
  metricArea: HTMLElement;
  metricAreaMm: HTMLElement;
  metricDiameter: HTMLElement;
  metricLocation: HTMLElement;
  metricHemi: HTMLElement;
  metricAsymmetry: HTMLElement;
  metricMidline: HTMLElement;
  metricSolidity: HTMLElement;
  metricBrainArea: HTMLElement;
  hudMidline?: HTMLElement;
}

export class DiagnosticDashboard {
  private readonly dom: DashboardDOMElements;

  constructor(elements: DashboardDOMElements) {
    this.dom = elements;
  }

  /**
   * Updates all oncology panels with fresh analysis results.
   */
  update(data: AnalysisResponse): void {
    const diag = data.ai_diagnosis;
    const cv = data.opencv_metrics;
    const cancerProb = diag.cancer_probability;

    // 1. Update Gauge & Percentage
    this.dom.riskPercentage.textContent = `${cancerProb}%`;

    const maxOffset = 251.2;
    const offset = maxOffset - (cancerProb / 100) * maxOffset;
    this.dom.gaugeFill.style.strokeDashoffset = String(offset);

    const needleAngle = -90 + (cancerProb / 100) * 180;
    this.dom.gaugeNeedle.setAttribute('transform', `rotate(${needleAngle} 100 110)`);

    // 2. Risk Badge & Color Coding
    this.dom.riskPill.className = 'badge';
    if (cancerProb >= 65) {
      this.dom.riskPill.classList.add('badge-critical');
      this.dom.riskPill.textContent = diag.risk_label_fa || 'بسیار بالا / بدخیم سرطانی';
      this.dom.gaugeFill.style.stroke = '#EF4444';
      this.dom.riskPercentage.style.color = '#EF4444';
    } else if (cancerProb >= 25) {
      this.dom.riskPill.classList.add('badge-warning');
      this.dom.riskPill.textContent = diag.risk_label_fa || 'خوش‌خیم / احتمال کم';
      this.dom.gaugeFill.style.stroke = '#F59E0B';
      this.dom.riskPercentage.style.color = '#F59E0B';
    } else {
      this.dom.riskPill.classList.add('badge-success');
      this.dom.riskPill.textContent = diag.risk_label_fa || 'ریسک ناچیز / سالم';
      this.dom.gaugeFill.style.stroke = '#10B981';
      this.dom.riskPercentage.style.color = '#10B981';
    }

    // 3. Probability Breakdown Bars
    const probs = diag.probabilities;
    this.dom.barMalignant.style.width = `${probs.cancer_malignant}%`;
    this.dom.probValMalignant.textContent = `${probs.cancer_malignant}%`;
    this.dom.barBenign.style.width = `${probs.benign}%`;
    this.dom.probValBenign.textContent = `${probs.benign}%`;
    this.dom.barNormal.style.width = `${probs.normal}%`;
    this.dom.probValNormal.textContent = `${probs.normal}%`;

    // 4. WHO Classification & Subpathology
    this.dom.diagnosisText.textContent = `${diag.diagnosis_fa} (${diag.diagnosis_en})`;
    this.dom.whoGradeText.textContent = diag.who_grade;
    this.dom.subpathologyText.textContent = diag.subpathology;
    this.dom.recommendationText.textContent = diag.clinical_recommendation;

    // 5. Quantitative OpenCV Biomarkers
    this.dom.metricBrainArea.textContent = `${cv.brain_area_cm2} cm²`;
    this.dom.metricAsymmetry.textContent = String(cv.asymmetry_score);

    if (cv.asymmetry_score > 0.35) {
      this.dom.metricMidline.textContent = 'انحراف خط وسط مغز (Midline Shift)';
      this.dom.metricMidline.style.color = '#EF4444';
      if (this.dom.hudMidline) {
        this.dom.hudMidline.textContent = 'SHIFT DETECTED';
        this.dom.hudMidline.style.color = '#EF4444';
      }
    } else {
      this.dom.metricMidline.textContent = 'قرینگی نرمال بین دو نیمکره';
      this.dom.metricMidline.style.color = '#94A3B8';
      if (this.dom.hudMidline) {
        this.dom.hudMidline.textContent = 'SYMMETRIC';
        this.dom.hudMidline.style.color = '#06B6D4';
      }
    }

    if (cv.has_tumor && cv.primary_lesion) {
      const p = cv.primary_lesion;
      this.dom.metricArea.textContent = `${p.area_cm2} cm²`;
      this.dom.metricAreaMm.textContent = `${p.area_mm2} mm² (${p.area_px} px)`;
      this.dom.metricDiameter.textContent = `${p.diameter_mm} mm`;
      this.dom.metricLocation.textContent = p.lobe;
      this.dom.metricHemi.textContent = p.hemisphere;
      this.dom.metricSolidity.textContent = `${p.solidity} (گردی: ${p.circularity})`;
    } else {
      this.dom.metricArea.textContent = '0.0 cm²';
      this.dom.metricAreaMm.textContent = 'فاقد ضایعه فضاگیر';
      this.dom.metricDiameter.textContent = '0.0 mm';
      this.dom.metricLocation.textContent = 'طبیعی';
      this.dom.metricHemi.textContent = 'هر دو نیمکره سالم';
      this.dom.metricSolidity.textContent = '1.00';
    }

    // 6. Malignancy Factors
    this.dom.factorsList.innerHTML = '';
    if (diag.malignancy_factors && diag.malignancy_factors.length > 0) {
      diag.malignancy_factors.forEach((f) => {
        const li = document.createElement('li');
        li.innerHTML = `
          <span>${f.factor}</span>
          <span class="factor-impact">${f.impact}% (${f.status})</span>
        `;
        this.dom.factorsList.appendChild(li);
      });
    } else {
      const li = document.createElement('li');
      li.textContent = 'عدم مشاهده بیومارکرهای پرخطر بدخیمی در بافت اسکن‌شده.';
      this.dom.factorsList.appendChild(li);
    }
  }
}

