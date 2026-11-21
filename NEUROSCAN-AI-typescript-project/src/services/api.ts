/**
 * NeuroScan AI - Strongly Typed API Client Service
 * Interacts with the Python OpenCV & Deep Learning Backend.
 */

import { AnalysisResponse, SampleCase, PipelineSteps } from '../types/neuroscan';

export class NeuroScanApiClient {
  private readonly baseUrl: string;

  constructor(baseUrl: string = '') {
    this.baseUrl = baseUrl;
  }

  /**
   * Health check probe to verify backend readiness.
   */
  async checkHealth(): Promise<boolean> {
    try {
      const res = await fetch(`${this.baseUrl}/api/health`);
      if (!res.ok) return false;
      const data = await res.json();
      return data.status === 'healthy';
    } catch {
      return false;
    }
  }

  /**
   * Submits an image (base64 data URI) for full OpenCV + AI analysis.
   */
  async analyzeScan(imageBase64: string): Promise<AnalysisResponse> {
    const res = await fetch(`${this.baseUrl}/api/analyze`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ image: imageBase64 }),
    });

    if (!res.ok) {
      throw new Error(`Server returned HTTP ${res.status}: ${res.statusText}`);
    }

    const data: AnalysisResponse = await res.json();
    if (data.status !== 'success') {
      throw new Error(data.message || 'Unknown analysis error occurred.');
    }

    return data;
  }

  /**
   * Retrieves all pre-loaded clinical benchmark and real patient cases.
   */
  async fetchSamples(): Promise<SampleCase[]> {
    const res = await fetch(`${this.baseUrl}/api/samples`);
    if (!res.ok) {
      throw new Error(`Failed to load samples: ${res.statusText}`);
    }

    const data = await res.json();
    if (data.status !== 'success' || !Array.isArray(data.samples)) {
      throw new Error('Malformed samples response from server.');
    }

    return data.samples as SampleCase[];
  }

  /**
   * Retrieves intermediate OpenCV image processing steps.
   */
  async fetchPipelineSteps(imageBase64: string): Promise<PipelineSteps> {
    const res = await fetch(`${this.baseUrl}/api/pipeline-steps`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ image: imageBase64 }),
    });

    if (!res.ok) {
      throw new Error(`Failed to load pipeline steps: ${res.statusText}`);
    }

    const data = await res.json();
    return data.steps as PipelineSteps;
  }
}

