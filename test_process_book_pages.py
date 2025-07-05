import pytest
import numpy as np
import cv2
from process_book_pages import split_pages, deskew_image, enhance_page_quality

@pytest.fixture
def sample_image() -> np.ndarray:
    """Creates a sample image for testing."""
    # Create an image with some content for enhancement testing
    img = np.zeros((100, 200, 3), dtype=np.uint8)
    # Add some white rectangles to simulate text/content
    cv2.rectangle(img, (10, 10), (90, 30), (255, 255, 255), -1)
    cv2.rectangle(img, (110, 10), (190, 30), (200, 200, 200), -1)
    cv2.rectangle(img, (10, 40), (90, 60), (150, 150, 150), -1)
    cv2.rectangle(img, (110, 40), (190, 60), (100, 100, 100), -1)
    return img

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

def test_enhance_page_quality(sample_image):
    """Tests the enhance_page_quality function."""
    enhanced = enhance_page_quality(sample_image.copy())
    
    # Check that the function returns an image
    assert enhanced is not None
    assert enhanced.shape == sample_image.shape
    
    # Check that enhancement is applied (image should be different)
    assert not np.array_equal(sample_image, enhanced)
    
    # Test with different parameters
    enhanced_custom = enhance_page_quality(sample_image.copy(), sharpness=2.0, contrast=1.5)
    assert enhanced_custom is not None
    assert enhanced_custom.shape == sample_image.shape 