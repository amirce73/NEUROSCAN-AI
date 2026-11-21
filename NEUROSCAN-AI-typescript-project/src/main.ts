/**
 * NeuroScan AI - TypeScript Main Application Controller
 * Boots and binds typed services, viewer, caliper, dashboard, and report generator.
 */

import { NeuroScanApiClient } from './services/api';
import { RadiologyCanvasViewer } from './components/canvas-viewer';
import { DigitalCaliperTool } from './components/caliper-ruler';
import { DiagnosticDashboard } from './components/diagnostic-dashboard';
import { RadiologyReportGenerator } from './components/report-generator';
import { ActiveLayer, AnalysisResponse, SampleCase } from './types/neuroscan';

export class NeuroScanApp {
  private readonly api: NeuroScanApiClient;
  private readonly viewer: RadiologyCanvasViewer;
  private readonly caliper: DigitalCaliperTool;
  private readonly dashboard: DiagnosticDashboard;
  private readonly reportGenerator: RadiologyReportGenerator;

  private currentAnalysis: AnalysisResponse | null = null;
  private sampleCases: SampleCase[] = [];

  constructor() {
    this.api = new NeuroScanApiClient();

    // DOM Elements
    const canvas = document.getElementById('radiology-canvas') as HTMLCanvasElement;
    const canvasContainer = document.getElementById('canvas-container') as HTMLElement;
    const measurementSvg = document.getElementById('measurement-svg') as unknown as SVGSVGElement;
    const reportModal = document.getElementById('report-modal') as HTMLElement;

    if (!canvas || !canvasContainer || !measurementSvg || !reportModal) {
      throw new Error('Required radiology DOM elements could not be located.');
    }

    // Initialize Components
    this.viewer = new RadiologyCanvasViewer(canvas);
    this.caliper = new DigitalCaliperTool(canvasContainer, measurementSvg, canvas);
    this.reportGenerator = new RadiologyReportGenerator(reportModal);

    this.dashboard = new DiagnosticDashboard({
      riskPercentage: document.getElementById('risk-percentage')!,
      riskPill: document.getElementById('risk-pill')!,
      gaugeFill: document.getElementById('gauge-fill') as unknown as SVGPathElement,
      gaugeNeedle: document.getElementById('gauge-needle') as unknown as SVGGElement,
      barMalignant: document.getElementById('bar-malignant')!,
      barBenign: document.getElementById('bar-benign')!,
      barNormal: document.getElementById('bar-normal')!,
      probValMalignant: document.getElementById('prob-val-malignant')!,
      probValBenign: document.getElementById('prob-val-benign')!,
      probValNormal: document.getElementById('prob-val-normal')!,
      diagnosisText: document.getElementById('diagnosis-text')!,
      whoGradeText: document.getElementById('who-grade-text')!,
      subpathologyText: document.getElementById('subpathology-text')!,
      recommendationText: document.getElementById('recommendation-text')!,
      factorsList: document.getElementById('factors-list')!,
      metricArea: document.getElementById('metric-area')!,
      metricAreaMm: document.getElementById('metric-area-mm')!,
      metricDiameter: document.getElementById('metric-diameter')!,
      metricLocation: document.getElementById('metric-location')!,
      metricHemi: document.getElementById('metric-hemi')!,
      metricAsymmetry: document.getElementById('metric-asymmetry')!,
      metricMidline: document.getElementById('metric-midline')!,
      metricSolidity: document.getElementById('metric-solidity')!,
      metricBrainArea: document.getElementById('metric-brain-area')!,
      hudMidline: document.getElementById('hud-midline') || undefined,
    });

    this.bindDOMEvents();
  }

  /**
   * Initializes the application and loads the clinical cases menu.
   */
  async start(): Promise<void> {
    try {
      this.sampleCases = await this.api.fetchSamples();
      this.renderSamplesMenu();

      // Automatically analyze first sample if available
      if (this.sampleCases.length > 0) {
        await this.analyzeImage(this.sampleCases[0].thumbnail_b64);
      }
    } catch (err) {
      console.error('Failed to initialize NeuroScan application:', err);
    }
  }

  /**
   * Submits an image for analysis and updates all typed views.
   */
  async analyzeImage(imageBase64: string): Promise<void> {
    const diagText = document.getElementById('diagnosis-text');
    if (diagText) diagText.textContent = 'در حال تحلیل با OpenCV و مدل هوش مصنوعی...';

    try {
      const data = await this.api.analyzeScan(imageBase64);
      this.currentAnalysis = data;

      await this.viewer.updateLayers(data);
      this.dashboard.update(data);
      this.renderPipelineDrawer(data.pipeline_steps);
    } catch (error) {
      console.error('Analysis failed:', error);
      alert('خطا در تحلیل تصویر: ' + (error instanceof Error ? error.message : String(error)));
    }
  }

  private bindDOMEvents(): void {
    // 1. Layer Selection
    const layerBtns = document.querySelectorAll<HTMLButtonElement>('.layer-btn');
    layerBtns.forEach((btn) => {
      btn.addEventListener('click', () => {
        layerBtns.forEach((b) => b.classList.remove('active'));
        btn.classList.add('active');
        const layer = btn.dataset.layer as ActiveLayer;
        if (layer) this.viewer.setLayer(layer);
      });
    });

    // 2. Opacity Slider
    const opacitySlider = document.getElementById('opacity-slider') as HTMLInputElement | null;
    const opacityVal = document.getElementById('opacity-val');
    if (opacitySlider) {
      opacitySlider.addEventListener('input', (e) => {
        const val = parseInt((e.target as HTMLInputElement).value, 10);
        this.viewer.setOpacity(val / 100);
        if (opacityVal) opacityVal.textContent = `${val}%`;
      });
    }

    // 3. Caliper Toggle
    const btnCaliper = document.getElementById('btn-caliper');
    const hudMeasureText = document.getElementById('hud-measure-text');
    const measureMm = document.getElementById('measure-mm');

    if (btnCaliper) {
      btnCaliper.addEventListener('click', () => {
        const isActive = this.caliper.toggle();
        btnCaliper.classList.toggle('active', isActive);
        if (!isActive && hudMeasureText) {
          hudMeasureText.classList.add('hidden');
        }
      });
    }

    this.caliper.setOnMeasurementChangeListener((measurement) => {
      if (measurement) {
        if (hudMeasureText) hudMeasureText.classList.remove('hidden');
        if (measureMm) measureMm.textContent = `${measurement.distanceMm} mm`;
      }
    });

    // 4. Laser Scanner Mode
    const btnToggleScan = document.getElementById('btn-toggle-scan');
    const laserScanner = document.getElementById('laser-scanner');
    const hudScannerStatus = document.getElementById('hud-scanner-status');

    if (btnToggleScan && laserScanner) {
      btnToggleScan.addEventListener('click', () => {
        const isHidden = laserScanner.classList.toggle('hidden');
        btnToggleScan.classList.toggle('active', !isHidden);
        if (hudScannerStatus) {
          hudScannerStatus.textContent = isHidden ? 'STANDBY' : 'SCANNING [λ=532nm]';
          hudScannerStatus.className = isHidden ? '' : 'text-scanner-active';
        }
      });
    }

    // 5. Reset View
    const btnResetView = document.getElementById('btn-reset-view');
    if (btnResetView) {
      btnResetView.addEventListener('click', () => {
        this.caliper.clear();
        btnCaliper?.classList.remove('active');
        if (hudMeasureText) hudMeasureText.classList.add('hidden');
        this.viewer.reset();
      });
    }

    // 6. File Upload
    const fileUpload = document.getElementById('file-upload') as HTMLInputElement | null;
    if (fileUpload) {
      fileUpload.addEventListener('change', (e) => {
        const file = (e.target as HTMLInputElement).files?.[0];
        if (!file) return;

        const reader = new FileReader();
        reader.onload = (evt) => {
          const res = evt.target?.result as string;
          if (res) this.analyzeImage(res);
        };
        reader.readAsDataURL(file);
      });
    }

    // 7. Drag & Drop
    const canvasContainer = document.getElementById('canvas-container');
    if (canvasContainer) {
      ['dragenter', 'dragover', 'dragleave', 'drop'].forEach((eventName) => {
        canvasContainer.addEventListener(eventName, (e) => {
          e.preventDefault();
          e.stopPropagation();
        });
      });

      canvasContainer.addEventListener('drop', (e) => {
        const file = e.dataTransfer?.files?.[0];
        if (file && file.type.startsWith('image/')) {
          const reader = new FileReader();
          reader.onload = (evt) => {
            const res = evt.target?.result as string;
            if (res) this.analyzeImage(res);
          };
          reader.readAsDataURL(file);
        }
      });
    }

    // 8. Samples Menu Dropdown
    const btnSamples = document.getElementById('btn-samples');
    const samplesMenu = document.getElementById('samples-menu');
    if (btnSamples && samplesMenu) {
      btnSamples.addEventListener('click', (e) => {
        e.stopPropagation();
        samplesMenu.classList.toggle('hidden');
      });

      document.addEventListener('click', () => {
        samplesMenu.classList.add('hidden');
      });
    }

    // 9. Report Modal Controls
    const btnPrintReport = document.getElementById('btn-print-report');
    const btnCloseModal = document.getElementById('btn-close-modal');
    const btnDoPrint = document.getElementById('btn-do-print');

    if (btnPrintReport) {
      btnPrintReport.addEventListener('click', () => {
        if (!this.currentAnalysis) {
          alert('لطفاً ابتدا تصویری را جهت تحلیل انتخاب یا آپلود کنید.');
          return;
        }
        this.reportGenerator.openReport(this.currentAnalysis);
      });
    }

    if (btnCloseModal) {
      btnCloseModal.addEventListener('click', () => this.reportGenerator.closeReport());
    }

    if (btnDoPrint) {
      btnDoPrint.addEventListener('click', () => this.reportGenerator.print());
    }
  }

  private renderSamplesMenu(): void {
    const samplesMenu = document.getElementById('samples-menu');
    if (!samplesMenu) return;

    samplesMenu.innerHTML = '';
    this.sampleCases.forEach((sample) => {
      const item = document.createElement('div');
      item.className = 'sample-item';
      item.innerHTML = `
        <img src="${sample.thumbnail_b64}" alt="${sample.title_en}">
        <div class="sample-info">
          <div class="sample-title">${sample.title_fa}</div>
          <div class="sample-diag">${sample.diagnosis}</div>
        </div>
      `;
      item.addEventListener('click', () => {
        samplesMenu.classList.add('hidden');
        this.analyzeImage(sample.thumbnail_b64);
      });
      samplesMenu.appendChild(item);
    });
  }

  private renderPipelineDrawer(steps: AnalysisResponse['pipeline_steps']): void {
    const pipelineCards = document.getElementById('pipeline-cards');
    if (!pipelineCards || !steps) return;

    pipelineCards.innerHTML = '';
    const stepDefs: Array<{ key: keyof typeof steps; name: string }> = [
      { key: 'step1_raw', name: '۱. اسکن خام ورودی' },
      { key: 'step2_windowed', name: '۲. تنظیم پنجره بافتی' },
      { key: 'step3_brain_mask', name: '۳. ماسک Skull Strip' },
      { key: 'step4_stripped_brain', name: '۴. پارانشیم خالص مغز' },
      { key: 'step5_clahe_enhanced', name: '۵. بهبود کنتراست CLAHE' },
      { key: 'step6_tumor_mask', name: '۶. تقطیع ضایعه (Mask)' },
      { key: 'step7_annotated', name: '۷. کانتوریابی OpenCV' },
      { key: 'step8_bone_colormap', name: '۸. کالرمپ Bone Medical' },
    ];

    stepDefs.forEach((step) => {
      const b64 = steps[step.key];
      if (!b64) return;

      const card = document.createElement('div');
      card.className = 'step-card';
      card.innerHTML = `
        <img src="${b64}" alt="${step.name}">
        <span class="step-name">${step.name}</span>
      `;
      card.addEventListener('click', () => {
        this.viewer.previewDirectImage(b64);
      });
      pipelineCards.appendChild(card);
    });
  }
}

// Bootstrap application once DOM content is ready
document.addEventListener('DOMContentLoaded', () => {
  const app = new NeuroScanApp();
  app.start();
});

