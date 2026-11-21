/**
 * NeuroScan AI - Medical Workstation Client Application
 * Handles Canvas Layering, Interactive Caliper Measurement,
 * Real-time REST API Analysis, Pipeline Visualizer & Diagnostic Reporting.
 */

document.addEventListener('DOMContentLoaded', () => {
    // ---------------- State Management ----------------
    const state = {
        currentScanB64: null,
        analysisData: null,
        pipelineSteps: null,
        activeLayer: 'overlay', // 'overlay', 'gradcam', 'contours', 'stripped', 'raw'
        heatmapOpacity: 0.85,
        caliperActive: false,
        isMeasuring: false,
        measureStart: null,
        measureEnd: null,
        pixelSpacingMm: 0.43, // Typical axial CT pixel spacing
        zoomLevel: 1.0,
        samples: []
    };

    // ---------------- DOM Elements ----------------
    const canvas = document.getElementById('radiology-canvas');
    const ctx = canvas.getContext('2d');
    const canvasWrapper = document.getElementById('canvas-container');
    const measurementSvg = document.getElementById('measurement-svg');

    // Controls
    const opacitySlider = document.getElementById('opacity-slider');
    const opacityVal = document.getElementById('opacity-val');
    const layerBtns = document.querySelectorAll('.layer-btn');
    const btnCaliper = document.getElementById('btn-caliper');
    const btnResetView = document.getElementById('btn-reset-view');
    const btnToggleScan = document.getElementById('btn-toggle-scan');
    const laserScanner = document.getElementById('laser-scanner');
    const hudScannerStatus = document.getElementById('hud-scanner-status');
    const fileUpload = document.getElementById('file-upload');
    const btnSamples = document.getElementById('btn-samples');
    const samplesMenu = document.getElementById('samples-menu');

    // HUD Elements
    const hudMeasureText = document.getElementById('hud-measure-text');
    const measureMm = document.getElementById('measure-mm');
    const hudMidline = document.getElementById('hud-midline');

    // Diagnostic Panel Elements
    const riskPercentage = document.getElementById('risk-percentage');
    const riskPill = document.getElementById('risk-pill');
    const gaugeFill = document.getElementById('gauge-fill');
    const gaugeNeedle = document.getElementById('gauge-needle');
    const barMalignant = document.getElementById('bar-malignant');
    const barBenign = document.getElementById('bar-benign');
    const barNormal = document.getElementById('bar-normal');
    const probValMalignant = document.getElementById('prob-val-malignant');
    const probValBenign = document.getElementById('prob-val-benign');
    const probValNormal = document.getElementById('prob-val-normal');

    const diagnosisText = document.getElementById('diagnosis-text');
    const whoGradeText = document.getElementById('who-grade-text');
    const subpathologyText = document.getElementById('subpathology-text');
    const recommendationText = document.getElementById('recommendation-text');
    const factorsList = document.getElementById('factors-list');

    // Metrics
    const metricArea = document.getElementById('metric-area');
    const metricAreaMm = document.getElementById('metric-area-mm');
    const metricDiameter = document.getElementById('metric-diameter');
    const metricLocation = document.getElementById('metric-location');
    const metricHemi = document.getElementById('metric-hemi');
    const metricAsymmetry = document.getElementById('metric-asymmetry');
    const metricMidline = document.getElementById('metric-midline');
    const metricSolidity = document.getElementById('metric-solidity');
    const metricBrainArea = document.getElementById('metric-brain-area');

    // Pipeline cards
    const pipelineCards = document.getElementById('pipeline-cards');

    // Modal & Print Report
    const reportModal = document.getElementById('report-modal');
    const btnPrintReport = document.getElementById('btn-print-report');
    const btnCloseModal = document.getElementById('btn-close-modal');
    const btnDoPrint = document.getElementById('btn-do-print');

    // ---------------- Image Cache & Canvas Rendering ----------------
    const imageCache = {
        raw: null,
        stripped: null,
        contours: null,
        gradcam: null,
        overlay: null
    };

    function loadImg(src) {
        return new Promise((resolve) => {
            if (!src) return resolve(null);
            const img = new Image();
            img.onload = () => resolve(img);
            img.onerror = () => resolve(null);
            img.src = src;
        });
    }

    async function updateImageCache(data) {
        const steps = data.pipeline_steps || {};
        const diag = data.ai_diagnosis || {};

        imageCache.raw = await loadImg(steps.step1_raw);
        imageCache.stripped = await loadImg(steps.step4_stripped_brain);
        imageCache.contours = await loadImg(steps.step7_annotated);
        imageCache.gradcam = await loadImg(diag.gradcam_heatmap_b64);
        imageCache.overlay = await loadImg(diag.diagnostic_overlay_b64);
        
        renderCanvas();
    }

    function renderCanvas() {
        ctx.clearRect(0, 0, canvas.width, canvas.height);

        const rawImg = imageCache.raw;
        if (!rawImg) return;

        // Base layer: Always draw raw CT scan
        ctx.globalAlpha = 1.0;
        ctx.drawImage(rawImg, 0, 0, canvas.width, canvas.height);

        // Overlay specific layers
        if (state.activeLayer === 'stripped' && imageCache.stripped) {
            ctx.clearRect(0, 0, canvas.width, canvas.height);
            ctx.drawImage(imageCache.stripped, 0, 0, canvas.width, canvas.height);
        } else if (state.activeLayer === 'contours' && imageCache.contours) {
            ctx.globalAlpha = state.heatmapOpacity;
            ctx.drawImage(imageCache.contours, 0, 0, canvas.width, canvas.height);
            ctx.globalAlpha = 1.0;
        } else if (state.activeLayer === 'gradcam' && imageCache.gradcam) {
            ctx.globalAlpha = state.heatmapOpacity;
            ctx.drawImage(imageCache.gradcam, 0, 0, canvas.width, canvas.height);
            ctx.globalAlpha = 1.0;
        } else if (state.activeLayer === 'overlay' && imageCache.overlay) {
            ctx.globalAlpha = state.heatmapOpacity;
            ctx.drawImage(imageCache.overlay, 0, 0, canvas.width, canvas.height);
            ctx.globalAlpha = 1.0;
        }
    }

    // ---------------- REST API Integration ----------------
    async function analyzeImage(imageBase64) {
        state.currentScanB64 = imageBase64;
        diagnosisText.textContent = "در حال پردازش با OpenCV و هوش مصنوعی...";

        try {
            const response = await fetch('/api/analyze', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ image: imageBase64 })
            });

            if (!response.ok) throw new Error(`Server error: ${response.status}`);

            const data = await response.json();
            if (data.status !== 'success') throw new Error(data.message);

            state.analysisData = data;
            state.pipelineSteps = data.pipeline_steps;

            await updateImageCache(data);
            populateDiagnostics(data);
            populatePipelineDrawer(data.pipeline_steps);

        } catch (error) {
            console.error("Analysis failed:", error);
            alert("خطا در تحلیل تصویر: " + error.message);
        }
    }

    // ---------------- Diagnostic UI Updates ----------------
    function populateDiagnostics(data) {
        const diag = data.ai_diagnosis;
        const cv = data.opencv_metrics;
        const cancerProb = diag.cancer_probability;

        // 1. Update Gauge & Risk Readout
        riskPercentage.textContent = `${cancerProb}%`;
        
        // Gauge Arc (circumference = 251.2 for radius 80 over 180 degrees)
        const maxOffset = 251.2;
        const offset = maxOffset - (cancerProb / 100) * maxOffset;
        gaugeFill.style.strokeDashoffset = offset;

        // Needle rotation: from -90deg (0%) to +90deg (100%)
        const needleAngle = -90 + (cancerProb / 100) * 180;
        gaugeNeedle.setAttribute('transform', `rotate(${needleAngle} 100 110)`);

        // Risk badge and colors
        riskPill.className = 'badge';
        if (cancerProb >= 65) {
            riskPill.classList.add('badge-critical');
            riskPill.textContent = diag.risk_label_fa || 'بسیار بالا / بدخیم سرطانی';
            gaugeFill.style.stroke = '#EF4444';
            riskPercentage.style.color = '#EF4444';
        } else if (cancerProb >= 25) {
            riskPill.classList.add('badge-warning');
            riskPill.textContent = diag.risk_label_fa || 'خوش‌خیم / احتمال کم';
            gaugeFill.style.stroke = '#F59E0B';
            riskPercentage.style.color = '#F59E0B';
        } else {
            riskPill.classList.add('badge-success');
            riskPill.textContent = diag.risk_label_fa || 'ریسک ناچیز / سالم';
            gaugeFill.style.stroke = '#10B981';
            riskPercentage.style.color = '#10B981';
        }

        // Probability bars
        const probs = diag.probabilities;
        barMalignant.style.width = `${probs.cancer_malignant}%`;
        probValMalignant.textContent = `${probs.cancer_malignant}%`;
        barBenign.style.width = `${probs.benign}%`;
        probValBenign.textContent = `${probs.benign}%`;
        barNormal.style.width = `${probs.normal}%`;
        probValNormal.textContent = `${probs.normal}%`;

        // 2. WHO Grade & Pathology
        diagnosisText.textContent = `${diag.diagnosis_fa} (${diag.diagnosis_en})`;
        whoGradeText.textContent = diag.who_grade;
        subpathologyText.textContent = diag.subpathology;
        recommendationText.textContent = diag.clinical_recommendation;

        // 3. Quantitative OpenCV Radiomics
        metricBrainArea.textContent = `${cv.brain_area_cm2} cm²`;
        metricAsymmetry.textContent = `${cv.asymmetry_score}`;

        if (cv.asymmetry_score > 0.35) {
            metricMidline.textContent = "انحراف خط وسط مغز (Midline Shift)";
            metricMidline.style.color = '#EF4444';
            hudMidline.textContent = "SHIFT DETECTED";
            hudMidline.style.color = '#EF4444';
        } else {
            metricMidline.textContent = "قرینگی نرمال بین دو نیمکره";
            metricMidline.style.color = '#94A3B8';
            hudMidline.textContent = "SYMMETRIC";
            hudMidline.style.color = '#06B6D4';
        }

        if (cv.has_tumor && cv.primary_lesion) {
            const p = cv.primary_lesion;
            metricArea.textContent = `${p.area_cm2} cm²`;
            metricAreaMm.textContent = `${p.area_mm2} mm² (${p.area_px} px)`;
            metricDiameter.textContent = `${p.diameter_mm} mm`;
            metricLocation.textContent = p.lobe;
            metricHemi.textContent = p.hemisphere;
            metricSolidity.textContent = `${p.solidity} (گردی: ${p.circularity})`;
        } else {
            metricArea.textContent = "0.0 cm²";
            metricAreaMm.textContent = "فاقد ضایعه فضاگیر";
            metricDiameter.textContent = "0.0 mm";
            metricLocation.textContent = "طبیعی";
            metricHemi.textContent = "هر دو نیمکره سالم";
            metricSolidity.textContent = "1.00";
        }

        // 4. Malignancy Factors List
        factorsList.innerHTML = '';
        if (diag.malignancy_factors && diag.malignancy_factors.length > 0) {
            diag.malignancy_factors.forEach(f => {
                const li = document.createElement('li');
                li.innerHTML = `
                    <span>${f.factor}</span>
                    <span class="factor-impact">${f.impact}% (${f.status})</span>
                `;
                factorsList.appendChild(li);
            });
        } else {
            const li = document.createElement('li');
            li.textContent = "عدم مشاهده بیومارکرهای پرخطر بدخیمی در بافت اسکن‌شده.";
            factorsList.appendChild(li);
        }
    }

    // ---------------- OpenCV Pipeline Breakdown ----------------
    function populatePipelineDrawer(steps) {
        if (!steps) return;
        pipelineCards.innerHTML = '';

        const stepDefinitions = [
            { key: 'step1_raw', name: '۱. اسکن خام ورودی' },
            { key: 'step2_windowed', name: '۲. تنظیم پنجره بافتی' },
            { key: 'step3_brain_mask', name: '۳. ماسک Skull Strip' },
            { key: 'step4_stripped_brain', name: '۴. پارانشیم خالص مغز' },
            { key: 'step5_clahe_enhanced', name: '۵. بهبود کنتراست CLAHE' },
            { key: 'step6_tumor_mask', name: '۶. تقطیع ضایعه (Mask)' },
            { key: 'step7_annotated', name: '۷. کانتوریابی OpenCV' },
            { key: 'step8_bone_colormap', name: '۸. کالرمپ Bone Medical' }
        ];

        stepDefinitions.forEach(step => {
            const b64 = steps[step.key];
            if (!b64) return;

            const card = document.createElement('div');
            card.className = 'step-card';
            card.innerHTML = `
                <img src="${b64}" alt="${step.name}">
                <span class="step-name">${step.name}</span>
            `;

            // Clicking preview in main viewer
            card.addEventListener('click', async () => {
                const img = await loadImg(b64);
                ctx.clearRect(0, 0, canvas.width, canvas.height);
                ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
            });

            pipelineCards.appendChild(card);
        });
    }

    // ---------------- Interactive Caliper / Ruler Tool ----------------
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

    btnCaliper.addEventListener('click', () => {
        state.caliperActive = !state.caliperActive;
        btnCaliper.classList.toggle('active', state.caliperActive);
        canvasWrapper.classList.toggle('caliper-mode', state.caliperActive);
        
        if (!state.caliperActive) {
            measurementSvg.innerHTML = '';
            hudMeasureText.classList.add('hidden');
        }
    });

    btnResetView.addEventListener('click', () => {
        state.caliperActive = false;
        btnCaliper.classList.remove('active');
        canvasWrapper.classList.remove('caliper-mode');
        measurementSvg.innerHTML = '';
        hudMeasureText.classList.add('hidden');
        renderCanvas();
    });

    function getCanvasCoords(e) {
        const rect = canvas.getBoundingClientRect();
        const scaleX = canvas.width / rect.width;
        const scaleY = canvas.height / rect.height;
        return {
            x: (e.clientX - rect.left) * scaleX,
            y: (e.clientY - rect.top) * scaleY
        };
    }

    canvasWrapper.addEventListener('mousedown', (e) => {
        if (!state.caliperActive) return;
        state.isMeasuring = true;
        state.measureStart = getCanvasCoords(e);
        state.measureEnd = state.measureStart;
        hudMeasureText.classList.remove('hidden');
    });

    canvasWrapper.addEventListener('mousemove', (e) => {
        if (!state.caliperActive || !state.isMeasuring) return;
        state.measureEnd = getCanvasCoords(e);
        drawMeasurementLine();
    });

    window.addEventListener('mouseup', () => {
        if (state.isMeasuring) {
            state.isMeasuring = false;
        }
    });

    function drawMeasurementLine() {
        if (!state.measureStart || !state.measureEnd) return;
        const p1 = state.measureStart;
        const p2 = state.measureEnd;

        const dx = p2.x - p1.x;
        const dy = p2.y - p1.y;
        const distPx = Math.sqrt(dx * dx + dy * dy);
        const distMm = (distPx * state.pixelSpacingMm).toFixed(1);

        measureMm.textContent = `${distMm} mm`;

        // Render SVG line and tick markers
        const rect = canvas.getBoundingClientRect();
        const sx = rect.width / canvas.width;
        const sy = rect.height / canvas.height;

        const x1 = p1.x * sx;
        const y1 = p1.y * sy;
        const x2 = p2.x * sx;
        const y2 = p2.y * sy;

        measurementSvg.innerHTML = `
            <line x1="${x1}" y1="${y1}" x2="${x2}" y2="${y2}" stroke="#FACC15" stroke-width="2" stroke-dasharray="4,2"/>
            <circle cx="${x1}" cy="${y1}" r="4" fill="#FACC15"/>
            <circle cx="${x2}" cy="${y2}" r="4" fill="#FACC15"/>
            <text x="${(x1 + x2) / 2 + 8}" y="${(y1 + y2) / 2 - 8}" fill="#FACC15" font-family="JetBrains Mono" font-size="12" font-weight="bold">${distMm} mm</text>
        `;
    }

    // ---------------- Layer Selection & Opacity Slider ----------------
    layerBtns.forEach(btn => {
        btn.addEventListener('click', () => {
            layerBtns.forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            state.activeLayer = btn.dataset.layer;
            renderCanvas();
        });
    });

    opacitySlider.addEventListener('input', (e) => {
        const val = parseInt(e.target.value);
        state.heatmapOpacity = val / 100.0;
        opacityVal.textContent = `${val}%`;
        renderCanvas();
    });

    // ---------------- Sample Cases Menu ----------------
    btnSamples.addEventListener('click', (e) => {
        e.stopPropagation();
        samplesMenu.classList.toggle('hidden');
    });

    document.addEventListener('click', () => {
        samplesMenu.classList.add('hidden');
    });

    async function loadSamplesMenu() {
        try {
            const res = await fetch('/api/samples');
            const data = await res.json();
            if (data.status === 'success') {
                state.samples = data.samples;
                samplesMenu.innerHTML = '';
                data.samples.forEach(sample => {
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
                        analyzeImage(sample.thumbnail_b64);
                    });
                    samplesMenu.appendChild(item);
                });

                // Auto-load case 1 by default
                if (data.samples.length > 0) {
                    analyzeImage(data.samples[0].thumbnail_b64);
                }
            }
        } catch (e) {
            console.error("Failed to load clinical samples:", e);
        }
    }

    // ---------------- File Upload & Drag and Drop ----------------
    fileUpload.addEventListener('change', (e) => {
        const file = e.target.files[0];
        if (!file) return;

        const reader = new FileReader();
        reader.onload = (evt) => {
            analyzeImage(evt.target.result);
        };
        reader.readAsDataURL(file);
    });

    // Drag and Drop
    ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
        canvasWrapper.addEventListener(eventName, preventDefaults, false);
    });

    function preventDefaults(e) {
        e.preventDefault();
        e.stopPropagation();
    }

    canvasWrapper.addEventListener('drop', (e) => {
        const dt = e.dataTransfer;
        const file = dt.files[0];
        if (file && file.type.startsWith('image/')) {
            const reader = new FileReader();
            reader.onload = (evt) => {
                analyzeImage(evt.target.result);
            };
            reader.readAsDataURL(file);
        }
    });

    // ---------------- Official Radiology Report Modal ----------------
    btnPrintReport.addEventListener('click', () => {
        if (!state.analysisData) {
            alert("لطفاً ابتدا تصویری را جهت تحلیل انتخاب یا آپلود کنید.");
            return;
        }

        const diag = state.analysisData.ai_diagnosis;
        const cv = state.analysisData.opencv_metrics;
        const steps = state.analysisData.pipeline_steps;

        // Populate Modal Fields
        document.getElementById('rep-id').textContent = `NS-${Math.floor(1000 + Math.random() * 9000)}`;
        document.getElementById('rep-date').textContent = new Date().toLocaleDateString('fa-IR');
        document.getElementById('rep-diagnosis').textContent = `${diag.diagnosis_fa} (${diag.diagnosis_en})`;
        document.getElementById('rep-cancer-prob').textContent = `${diag.cancer_probability}%`;
        document.getElementById('rep-who').textContent = diag.who_grade;
        document.getElementById('rep-risk').textContent = diag.risk_label_fa;

        document.getElementById('rep-img-raw').src = steps.step1_raw;
        document.getElementById('rep-img-contours').src = steps.step7_annotated;
        document.getElementById('rep-img-gradcam').src = diag.gradcam_heatmap_b64;

        // Biomarkers
        const bioContainer = document.getElementById('rep-biomarkers');
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

        document.getElementById('rep-recommendation').textContent = diag.clinical_recommendation;

        reportModal.classList.remove('hidden');
    });

    btnCloseModal.addEventListener('click', () => {
        reportModal.classList.add('hidden');
    });

    btnDoPrint.addEventListener('click', () => {
        window.print();
    });

    // Initialize application
    loadSamplesMenu();
});

