/**
 * NeuroScan AI - Node.js TypeScript Integration Example
 * Demonstrates how to consume the NeuroScan AI service from any Node.js / TypeScript backend.
 */

import * as fs from 'fs';
import * as path from 'path';
import { NeuroScanApiClient } from './services/api';
import { AnalysisResponse, SampleCase } from './types/neuroscan';

async function runDemo(): Promise<void> {
  console.log('====================================================');
  console.log('🧠 NeuroScan AI - TypeScript Client Integration Demo');
  console.log('====================================================');

  const client = new NeuroScanApiClient('http://localhost:8080');

  // 1. Health check
  const isHealthy = await client.checkHealth();
  console.log(`Backend Health Status: ${isHealthy ? '✅ ONLINE' : '❌ OFFLINE'}`);
  if (!isHealthy) {
    console.error('Please make sure Python server is running on port 8080.');
    return;
  }

  // 2. Fetch all available clinical sample cases
  const samples: SampleCase[] = await client.fetchSamples();
  console.log(`\n📋 Loaded ${samples.length} clinical patient cases from backend:`);
  samples.forEach((s, idx) => {
    console.log(`   [${idx + 1}] ${s.title_en} (${s.diagnosis})`);
  });

  // 3. Analyze the first case (Glioblastoma)
  if (samples.length > 0) {
    const targetCase = samples[0];
    console.log(`\n🔬 Sending scan for AI analysis: ${targetCase.title_en}...`);

    const result: AnalysisResponse = await client.analyzeScan(targetCase.thumbnail_b64);
    const diag = result.ai_diagnosis;
    const cv = result.opencv_metrics;

    console.log('\n📊 Oncology Diagnostic Report (Strongly Typed):');
    console.log(`   • Diagnosis: ${diag.diagnosis_en} - ${diag.diagnosis_fa}`);
    console.log(`   • Cancer Probability: ${diag.cancer_probability}%`);
    console.log(`   • Risk Level: ${diag.risk_level} (${diag.risk_label_fa})`);
    console.log(`   • WHO CNS Grade: ${diag.who_grade}`);
    console.log(`   • Brain Parenchyma Area: ${cv.brain_area_cm2} cm²`);

    if (cv.has_tumor && cv.primary_lesion) {
      const lesion = cv.primary_lesion;
      console.log(`   • Primary Lesion Area: ${lesion.area_cm2} cm² (${lesion.area_mm2} mm²)`);
      console.log(`   • Equivalent Diameter: ${lesion.diameter_mm} mm`);
      console.log(`   • Location: ${lesion.lobe} | ${lesion.hemisphere}`);
      console.log(`   • Solidity (Invasiveness): ${lesion.solidity}`);
    }

    console.log('\n🩺 Clinical Action Recommendation:');
    console.log(`   ${diag.clinical_recommendation}`);
  }

  console.log('\n🎉 TypeScript Integration Demo Finished Successfully!');
}

if (require.main === module) {
  runDemo().catch(console.error);
}

