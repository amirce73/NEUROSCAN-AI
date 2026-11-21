"""
Tests server HTTP API directly using Python urllib
"""
import urllib.request
import urllib.parse
import json
import base64
import os
import sys

def get_opener():
    proxy_handler = urllib.request.ProxyHandler({})
    return urllib.request.build_opener(proxy_handler)

def test_api(port: int = 8080):
    print(f"Testing API endpoints on port {port}...")
    base_url = f"http://127.0.0.1:{port}"
    opener = get_opener()

    # 1. Health
    res = opener.open(f"{base_url}/api/health")
    data = json.loads(res.read().decode('utf-8'))
    assert data["status"] == "healthy", f"Health failed: {data}"
    print("✅ /api/health passed.")

    # 2. Samples
    res = opener.open(f"{base_url}/api/samples")
    data = json.loads(res.read().decode('utf-8'))
    assert data["status"] == "success", f"Samples failed: {data}"
    assert len(data["samples"]) >= 4, f"Expected at least 4 samples, got {len(data['samples'])}"
    print(f"✅ /api/samples passed with {len(data['samples'])} cases (including real patient scans).")

    # 3. Analyze sample 1 (Glioblastoma)
    sample_b64 = data["samples"][0]["thumbnail_b64"]
    post_payload = json.dumps({"image": sample_b64}).encode('utf-8')
    req = urllib.request.Request(
        f"{base_url}/api/analyze",
        data=post_payload,
        headers={"Content-Type": "application/json"}
    )
    res = opener.open(req)
    analysis = json.loads(res.read().decode('utf-8'))
    assert analysis["status"] == "success", f"Analyze failed: {analysis}"
    diag = analysis["ai_diagnosis"]
    print(f"✅ /api/analyze passed: {diag['diagnosis_en']} - Cancer Prob: {diag['cancer_probability']}%")
    assert "gradcam_heatmap_b64" in diag
    assert "diagnostic_overlay_b64" in diag
    assert "pipeline_steps" in analysis

    # 4. Static frontend index.html
    res = opener.open(f"{base_url}/")
    html_content = res.read().decode('utf-8')
    assert "NEUROSCAN" in html_content
    print("✅ Static frontend index.html served successfully.")

    print("\n🎉 ALL SERVER API AND FRONTEND CHECKS PASSED!")

if __name__ == "__main__":
    test_api()
