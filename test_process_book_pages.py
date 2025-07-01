import pytest
import numpy as np
import cv2
from process_book_pages import split_pages, deskew_image

@pytest.fixture
def sample_image() -> np.ndarray:
    """Creates a sample image for testing."""
    return np.zeros((100, 200, 3), dtype=np.uint8)

@pytest.fixture
def skewed_image() -> np.ndarray:
    """Creates a sample skewed image for testing."""
    img = np.zeros((200, 200, 3), dtype=np.uint8)
    cv2.line(img, (20, 40), (180, 45), (255, 255, 255), 2)
    cv2.line(img, (20, 100), (180, 105), (255, 255, 255), 2)
    return img

def test_split_pages(sample_image):
    """Tests the split_pages function."""
    left_page, right_page = split_pages(sample_image)
    assert left_page.shape == (100, 100, 3)
    assert right_page.shape == (100, 100, 3)

def test_deskew_image(skewed_image):
    """Tests the deskew_image function."""
    deskewed = deskew_image(skewed_image.copy(), threshold=0.1)
    
    # Check if the image was rotated (it should be different from the original)
    assert not np.array_equal(skewed_image, deskewed)

    # A simple check: a deskewed image should have straighter lines.
    # We can re-run the line detection and check the angle.
    gray = cv2.cvtColor(deskewed, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 100, minLineLength=100, maxLineGap=10)
    
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = np.rad2deg(np.arctan2(y2 - y1, x2 - x1))
            assert abs(angle) < 0.1 