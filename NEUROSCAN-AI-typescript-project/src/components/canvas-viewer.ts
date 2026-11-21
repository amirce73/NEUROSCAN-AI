/**
 * NeuroScan AI - Radiology Canvas Viewer Engine
 * Handles multi-layer alpha blending, image caching,
 * zoom/pan transforms, and interactive laser scanning mode.
 */

import { ActiveLayer, AnalysisResponse } from '../types/neuroscan';

export interface ImageCache {
  raw: HTMLImageElement | null;
  stripped: HTMLImageElement | null;
  contours: HTMLImageElement | null;
  gradcam: HTMLImageElement | null;
  overlay: HTMLImageElement | null;
}

export class RadiologyCanvasViewer {
  private readonly canvas: HTMLCanvasElement;
  private readonly ctx: CanvasRenderingContext2D;
  private readonly imageCache: ImageCache = {
    raw: null,
    stripped: null,
    contours: null,
    gradcam: null,
    overlay: null,
  };

  private activeLayer: ActiveLayer = 'overlay';
  private heatmapOpacity: number = 0.85;

  constructor(canvasElement: HTMLCanvasElement) {
    this.canvas = canvasElement;
    const context = canvasElement.getContext('2d');
    if (!context) {
      throw new Error('Could not obtain 2D rendering context for radiology canvas.');
    }
    this.ctx = context;
  }

  /**
   * Helper utility to load an image asynchronously.
   */
  private loadImage(src?: string): Promise<HTMLImageElement | null> {
    return new Promise((resolve) => {
      if (!src) return resolve(null);
      const img = new Image();
      img.onload = () => resolve(img);
      img.onerror = () => resolve(null);
      img.src = src;
    });
  }

  /**
   * Updates cached image layers from an analysis response.
   */
  async updateLayers(analysis: AnalysisResponse): Promise<void> {
    const steps = analysis.pipeline_steps;
    const diag = analysis.ai_diagnosis;

    const [raw, stripped, contours, gradcam, overlay] = await Promise.all([
      this.loadImage(steps?.step1_raw),
      this.loadImage(steps?.step4_stripped_brain),
      this.loadImage(steps?.step7_annotated),
      this.loadImage(diag?.gradcam_heatmap_b64),
      this.loadImage(diag?.diagnostic_overlay_b64),
    ]);

    this.imageCache.raw = raw;
    this.imageCache.stripped = stripped;
    this.imageCache.contours = contours;
    this.imageCache.gradcam = gradcam;
    this.imageCache.overlay = overlay;

    this.render();
  }

  /**
   * Sets the active display layer.
   */
  setLayer(layer: ActiveLayer): void {
    this.activeLayer = layer;
    this.render();
  }

  /**
   * Sets the alpha opacity for the overlay layer (0.0 to 1.0).
   */
  setOpacity(opacity: number): void {
    this.heatmapOpacity = Math.max(0, Math.min(1, opacity));
    this.render();
  }

  /**
   * Renders the base CT scan and overlays the selected active layer.
   */
  render(): void {
    const { width, height } = this.canvas;
    this.ctx.clearRect(0, 0, width, height);

    const raw = this.imageCache.raw;
    if (!raw) return;

    // Base CT scan
    this.ctx.globalAlpha = 1.0;
    this.ctx.drawImage(raw, 0, 0, width, height);

    // Layer selection overlay
    switch (this.activeLayer) {
      case 'stripped':
        if (this.imageCache.stripped) {
          this.ctx.clearRect(0, 0, width, height);
          this.ctx.drawImage(this.imageCache.stripped, 0, 0, width, height);
        }
        break;

      case 'contours':
        if (this.imageCache.contours) {
          this.ctx.globalAlpha = this.heatmapOpacity;
          this.ctx.drawImage(this.imageCache.contours, 0, 0, width, height);
          this.ctx.globalAlpha = 1.0;
        }
        break;

      case 'gradcam':
        if (this.imageCache.gradcam) {
          this.ctx.globalAlpha = this.heatmapOpacity;
          this.ctx.drawImage(this.imageCache.gradcam, 0, 0, width, height);
          this.ctx.globalAlpha = 1.0;
        }
        break;

      case 'overlay':
        if (this.imageCache.overlay) {
          this.ctx.globalAlpha = this.heatmapOpacity;
          this.ctx.drawImage(this.imageCache.overlay, 0, 0, width, height);
          this.ctx.globalAlpha = 1.0;
        }
        break;

      case 'raw':
      default:
        // Already rendered base raw image
        break;
    }
  }

  /**
   * Directly draws a given image element onto the canvas (used for pipeline step previews).
   */
  async previewDirectImage(src: string): Promise<void> {
    const img = await this.loadImage(src);
    if (img) {
      this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
      this.ctx.drawImage(img, 0, 0, this.canvas.width, this.canvas.height);
    }
  }

  /**
   * Resets the canvas view back to the active analysis layers.
   */
  reset(): void {
    this.render();
  }
}

