import io
from fastapi.testclient import TestClient
from PIL import Image
from api import app

client = TestClient(app)


def test_analyze_endpoint():
    # Create a dummy image
    img = Image.new("RGB", (200, 200), color=(100, 100, 100))
    img_byte_arr = io.BytesIO()
    img.save(img_byte_arr, format="PNG")
    img_byte_arr.seek(0)

    # POST to /analyze
    response = client.post(
        "/analyze", files={"file": ("test_image.png", img_byte_arr, "image/png")}
    )

    # We expect a 200 OK or 422 (if evaluator rejects the dummy image)
    # The current evaluator logic might return 422 if it can't find a coin
    # Let's just check the response status and structure
    if response.status_code == 200:
        data = response.json()
        assert "roughness_ra" in data
        assert "gloss_percent" in data
        assert "detected_finish" in data
        assert "confidence" in data
        assert "matching_substrates" in data
    elif response.status_code == 422:
        # Evaluator returned error (e.g., "동전을 찾을 수 없습니다.")
        data = response.json()
        assert "detail" in data
        assert "동전을 찾을 수 없습니다" in data["detail"]
    else:
        assert False, f"Unexpected status code: {response.status_code}"


def test_analyze_invalid_file_type():
    response = client.post(
        "/analyze", files={"file": ("test.txt", b"this is text", "text/plain")}
    )
    assert response.status_code == 400
    assert "File must be an image" in response.json()["detail"]
