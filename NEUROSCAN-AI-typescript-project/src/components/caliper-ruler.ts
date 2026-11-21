/**
 * NeuroScan AI - Digital Caliper Measurement Tool
 * Computes calibrated millimeter distances across lesions using CT pixel spacing.
 */

import { CaliperMeasurement } from '../types/neuroscan';

export class DigitalCaliperTool {
  private readonly container: HTMLElement;
  private readonly svgElement: SVGSVGElement;
  private readonly canvas: HTMLCanvasElement;
  private readonly pixelSpacingMm: number;

  private isEnabled: boolean = false;
  private isDrawing: boolean = false;
  private startPoint: { x: number; y: number } | null = null;
  private endPoint: { x: number; y: number } | null = null;

  private onMeasurementChange?: (measurement: CaliperMeasurement | null) => void;

  constructor(
    container: HTMLElement,
    svgElement: SVGSVGElement,
    canvas: HTMLCanvasElement,
    pixelSpacingMm: number = 0.43
  ) {
    this.container = container;
    this.svgElement = svgElement;
    this.canvas = canvas;
    this.pixelSpacingMm = pixelSpacingMm;

    this.bindEvents();
  }

  /**
   * Sets callback listener for measurement updates.
   */
  setOnMeasurementChangeListener(callback: (measurement: CaliperMeasurement | null) => void): void {
    this.onMeasurementChange = callback;
  }

  /**
   * Toggles the caliper tool on or off.
   */
  toggle(): boolean {
    this.isEnabled = !this.isEnabled;
    this.container.classList.toggle('caliper-mode', this.isEnabled);

    if (!this.isEnabled) {
      this.clear();
    }
    return this.isEnabled;
  }

  /**
   * Clears active measurements and hides SVG markings.
   */
  clear(): void {
    this.svgElement.innerHTML = '';
    this.startPoint = null;
    this.endPoint = null;
    this.isDrawing = false;
    this.onMeasurementChange?.(null);
  }

  private getCoordinates(e: MouseEvent): { x: number; y: number } {
    const rect = this.canvas.getBoundingClientRect();
    const scaleX = this.canvas.width / rect.width;
    const scaleY = this.canvas.height / rect.height;

    return {
      x: (e.clientX - rect.left) * scaleX,
      y: (e.clientY - rect.top) * scaleY,
    };
  }

  private bindEvents(): void {
    this.container.addEventListener('mousedown', (e) => {
      if (!this.isEnabled) return;
      this.isDrawing = true;
      this.startPoint = this.getCoordinates(e);
      this.endPoint = this.startPoint;
      this.renderMeasurement();
    });

    window.addEventListener('mousemove', (e) => {
      if (!this.isEnabled || !this.isDrawing) return;
      this.endPoint = this.getCoordinates(e);
      this.renderMeasurement();
    });

    window.addEventListener('mouseup', () => {
      if (this.isDrawing) {
        this.isDrawing = false;
      }
    });
  }

  private renderMeasurement(): void {
    if (!this.startPoint || !this.endPoint) return;

    const dx = this.endPoint.x - this.startPoint.x;
    const dy = this.endPoint.y - this.startPoint.y;
    const distPx = Math.sqrt(dx * dx + dy * dy);
    const distMm = Number((distPx * this.pixelSpacingMm).toFixed(1));

    // Map internal coordinates to container SVG space
    const rect = this.canvas.getBoundingClientRect();
    const sx = rect.width / this.canvas.width;
    const sy = rect.height / this.canvas.height;

    const x1 = this.startPoint.x * sx;
    const y1 = this.startPoint.y * sy;
    const x2 = this.endPoint.x * sx;
    const y2 = this.endPoint.y * sy;

    this.svgElement.innerHTML = `
      <line x1="${x1}" y1="${y1}" x2="${x2}" y2="${y2}" stroke="#FACC15" stroke-width="2" stroke-dasharray="4,2"/>
      <circle cx="${x1}" cy="${y1}" r="4" fill="#FACC15"/>
      <circle cx="${x2}" cy="${y2}" r="4" fill="#FACC15"/>
      <text x="${(x1 + x2) / 2 + 8}" y="${(y1 + y2) / 2 - 8}" fill="#FACC15" font-family="JetBrains Mono" font-size="12" font-weight="bold">${distMm} mm</text>
    `;

    this.onMeasurementChange?.({
      start: this.startPoint,
      end: this.endPoint,
      distancePx: distPx,
      distanceMm: distMm,
    });
  }
}

