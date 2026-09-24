"""Quick integration test for the FastAPI endpoints."""
import requests
import json
import sys

BASE = "http://127.0.0.1:8000"

def test():
    # 1. Stats
    r = requests.get(f"{BASE}/api/stats")
    assert r.status_code == 200
    print("[OK] /api/stats:", r.json())

    # 2. Model Status
    r = requests.get(f"{BASE}/api/model-status")
    assert r.status_code == 200
    tiers = r.json()["tiers"]
    for t in tiers:
        print(f"  [{t['status'].upper():>11}] {t['name']}")

    # 3. Generate
    r = requests.post(f"{BASE}/api/generate", json={
        "text": "# Test Assignment\nName: Student\n\nThis is a sample paragraph to verify the handwriting engine works correctly.",
        "ink_type": "Royal Blue Ballpoint",
        "baseline_slant": 0.4,
        "margin_drift": 12,
        "line_spacing": 95,
        "camscanner_intensity": 0.85
    })
    assert r.status_code == 200
    data = r.json()
    print(f"[OK] /api/generate: {data['page_count']} page(s), pdf_url={data['pdf_url']}")

    # 4. Preview
    r = requests.get(f"{BASE}/api/preview/0")
    assert r.status_code == 200
    print(f"[OK] /api/preview/0: {len(r.content)} bytes, type={r.headers.get('content-type')}")

    # 5. Download PDF
    r = requests.get(f"{BASE}/api/download")
    assert r.status_code == 200
    print(f"[OK] /api/download: {len(r.content)} bytes")

    # 6. Frontend
    r = requests.get(BASE)
    assert r.status_code == 200
    assert "Handwriting" in r.text
    print(f"[OK] Frontend served: {len(r.content)} bytes")

    print("\nAll tests passed!")

if __name__ == "__main__":
    test()
