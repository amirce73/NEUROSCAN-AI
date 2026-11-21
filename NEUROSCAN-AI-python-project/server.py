"""
NeuroScan AI - Production Web Server & REST API
Multi-threaded HTTP Server serving the Medical AI Radiology Workstation
and providing real-time OpenCV + Deep Learning diagnostic endpoints.
"""

import os
import sys
import json
import base64
import urllib.parse
from http.server import HTTPServer, SimpleHTTPRequestHandler
from socketserver import ThreadingMixIn
import traceback

# Import core modules
from core.cv_engine import BrainCTProcessor
from core.neuro_classifier import BrainTumorClassifier
from core.sample_generator import generate_all_samples


class ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
    """Handles requests in separate threads for maximum responsiveness."""
    daemon_threads = True


class NeuroScanRequestHandler(SimpleHTTPRequestHandler):
    """Custom request handler implementing REST API and static UI serving."""

    processor: BrainCTProcessor = None
    classifier: BrainTumorClassifier = None
    samples_dir: str = "samples"
    web_dir: str = "web"

    def __init__(self, *args, **kwargs):
        # Set directory to web directory for static files
        super().__init__(*args, directory=os.path.abspath(self.web_dir), **kwargs)

    def _set_headers(self, status: int = 200, content_type: str = "application/json"):
        self.send_response(status)
        self.send_header("Content-Type", f"{content_type}; charset=utf-8")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_OPTIONS(self):
        """Handles CORS pre-flight requests."""
        self._set_headers(200)

    def do_GET(self):
        """Handles REST GET requests or falls back to static files."""
        parsed_url = urllib.parse.urlparse(self.path)

        if parsed_url.path == "/api/health":
            self._set_headers(200)
            self.wfile.write(json.dumps({"status": "healthy", "service": "NeuroScan AI"}).encode('utf-8'))
            return

        elif parsed_url.path == "/api/samples":
            # Return list of pre-configured clinical benchmark cases
            try:
                cases = generate_all_samples(self.samples_dir)
                enriched_cases = []
                for case in cases:
                    with open(case["filepath"], "rb") as f:
                        b64_data = "data:image/png;base64," + base64.b64encode(f.read()).decode("utf-8")
                    enriched_cases.append({
                        "id": case["id"],
                        "filename": case["filename"],
                        "title_fa": case["title_fa"],
                        "title_en": case["title_en"],
                        "diagnosis": case["diagnosis"],
                        "thumbnail_b64": b64_data
                    })
                self._set_headers(200)
                self.wfile.write(json.dumps({"status": "success", "samples": enriched_cases}).encode('utf-8'))
            except Exception as e:
                self._set_headers(500)
                self.wfile.write(json.dumps({"status": "error", "message": str(e)}).encode('utf-8'))
            return

        # Serve static web files
        return super().do_GET()

    def do_POST(self):
        """Handles REST POST requests for analysis and pipeline inspection."""
        parsed_url = urllib.parse.urlparse(self.path)

        content_length = int(self.headers.get('Content-Length', 0))
        post_data = self.rfile.read(content_length)

        try:
            body = json.loads(post_data.decode('utf-8')) if post_data else {}
        except Exception:
            self._set_headers(400)
            self.wfile.write(json.dumps({"status": "error", "message": "Invalid JSON body"}).encode('utf-8'))
            return

        if parsed_url.path == "/api/analyze":
            # Complete OpenCV + AI Diagnostic Pipeline
            image_data = body.get("image")
            if not image_data:
                self._set_headers(400)
                self.wfile.write(json.dumps({"status": "error", "message": "Missing 'image' parameter"}).encode('utf-8'))
                return

            try:
                # 1. OpenCV Processing
                cv_res = self.processor.process_scan(image_data)
                # 2. AI Classifier & Grad-CAM Heatmap
                ai_res = self.classifier.analyze_scan(cv_res)
                # 3. OpenCV Pipeline intermediate steps
                pipeline_steps = self.processor.generate_pipeline_steps(cv_res["image_bgr"])

                # Prepare response
                response_data = {
                    "status": "success",
                    "opencv_metrics": {
                        "has_tumor": cv_res["has_tumor"],
                        "lesion_count": cv_res["lesion_count"],
                        "brain_area_cm2": cv_res["brain_area_cm2"],
                        "asymmetry_score": cv_res["asymmetry_score"],
                        "primary_lesion": cv_res["primary_lesion"],
                        "all_lesions": cv_res["lesions"]
                    },
                    "ai_diagnosis": ai_res,
                    "pipeline_steps": pipeline_steps
                }

                self._set_headers(200)
                self.wfile.write(json.dumps(response_data).encode('utf-8'))

            except Exception as e:
                traceback.print_exc()
                self._set_headers(500)
                self.wfile.write(json.dumps({"status": "error", "message": str(e)}).encode('utf-8'))
            return

        elif parsed_url.path == "/api/pipeline-steps":
            # Returns only the visual steps
            image_data = body.get("image")
            if not image_data:
                self._set_headers(400)
                self.wfile.write(json.dumps({"status": "error", "message": "Missing 'image' parameter"}).encode('utf-8'))
                return

            try:
                bgr = self.processor.load_image(image_data)
                steps = self.processor.generate_pipeline_steps(bgr)
                self._set_headers(200)
                self.wfile.write(json.dumps({"status": "success", "steps": steps}).encode('utf-8'))
            except Exception as e:
                self._set_headers(500)
                self.wfile.write(json.dumps({"status": "error", "message": str(e)}).encode('utf-8'))
            return

        self._set_headers(404)
        self.wfile.write(json.dumps({"status": "error", "message": "Endpoint not found"}).encode('utf-8'))


def run_server(port: int = 8080, host: str = "0.0.0.0"):
    """Initializes and runs the NeuroScan AI workstation server."""
    # Ensure samples exist
    os.makedirs("samples", exist_ok=True)
    generate_all_samples("samples")

    # Initialize processors once
    print("⏳ Initializing OpenCV Core & Deep Neural Network...")
    NeuroScanRequestHandler.processor = BrainCTProcessor()
    NeuroScanRequestHandler.classifier = BrainTumorClassifier()
    print("✅ AI Models and Computer Vision Pipelines Ready!")

    server_address = (host, port)
    httpd = ThreadedHTTPServer(server_address, NeuroScanRequestHandler)
    print(f"🚀 NeuroScan AI Diagnostic Server running at http://localhost:{port}")
    print("Press Ctrl+C to stop the server.")

    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n🛑 Shutting down server gracefully...")
        httpd.shutdown()


if __name__ == "__main__":
    port_arg = int(sys.argv[1]) if len(sys.argv) > 1 else 8080
    run_server(port=port_arg)

