import argparse
import cv2
import logging
import numpy as np
import os
import rawpy
from pathlib import Path
from typing import Tuple, Dict
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

def load_image(filepath: Path) -> np.ndarray:
    """Loads an image from the specified file path, converting RAW formats if necessary."""
    try:
        if filepath.suffix.lower() in ['.cr3', '.nef', '.arw', '.dng']:
            with rawpy.imread(str(filepath)) as raw:
                # Post-process to get an RGB image, then convert to BGR for OpenCV
                rgb = raw.postprocess(use_camera_wb=True)
                image = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        else:
            image = cv2.imread(str(filepath))

        if image is None:
            raise IOError(f"Could not read or convert image file: {filepath}")
        return image
    except Exception as e:
        raise IOError(f"Error loading image {filepath}: {e}")

def deskew_image(image: np.ndarray, threshold: float = 1.0) -> np.ndarray:
    """
    Detects and corrects skew in an image if the angle exceeds a threshold.
    This implementation uses the Hough Transform on a cropped central portion of the image
    to focus on text lines and avoid page edges.
    """
    if image is None:
        return None

    # Create a copy to find skew, but rotate the original image
    h, w = image.shape[:2]
    
    # Crop the image to focus on the text area, avoiding edges
    crop_border_h = int(h * 0.10)
    crop_border_w = int(w * 0.10)
    cropped_for_skew = image[crop_border_h:h-crop_border_h, crop_border_w:w-crop_border_w]

    gray = cv2.cvtColor(cropped_for_skew, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)

    # Use Hough transform with parameters tuned for text lines
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 100, minLineLength=50, maxLineGap=10)

    if lines is None:
        return image

    angles = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        angle = np.rad2deg(np.arctan2(y2 - y1, x2 - x1))
        # Filter for angles that are likely to be text
        if abs(angle) < 45:
            angles.append(angle)

    if not angles:
        return image

    median_angle = np.median(angles)

    if abs(median_angle) > threshold:
        # Rotate the original, uncropped image
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, float(median_angle), 1.0)
        rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
        return rotated
    
    return image

def crop_to_content(image: np.ndarray, padding: int = 20) -> np.ndarray:
    """
    Finds the largest bright object (the page) and crops the image to its
    bounding box, leaving a small padding. This is effective at removing
    dark backgrounds like a table.
    """
    if image is None:
        return None

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Use a larger blur kernel to smooth out text and focus on the page shape
    blurred = cv2.GaussianBlur(gray, (11, 11), 0)

    # Use Otsu's thresholding to automatically separate the bright page from the dark background
    try:
        _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    except cv2.error:
        # Otsu's method can fail on images with no contrast (e.g., all black/white)
        return image

    # Find the outermost contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return image  # No content found

    # Find the largest contour by area, which we assume is the page
    largest_contour = max(contours, key=cv2.contourArea)

    # Get the bounding box for the page
    x, y, w, h = cv2.boundingRect(largest_contour)

    # Apply padding to the bounding box
    x_start = max(x - padding, 0)
    y_start = max(y - padding, 0)
    x_end = min(x + w + padding, image.shape[1])
    y_end = min(y + h + padding, image.shape[0])

    cropped = image[y_start:y_end, x_start:x_end]

    # Return the cropped image only if it has a valid size
    if cropped.shape[0] > 0 and cropped.shape[1] > 0:
        return cropped
    else:
        return image

def split_pages(image: np.ndarray, offset_percentage: float = 0.0) -> Tuple[np.ndarray, np.ndarray]:
    """
    Splits an image vertically into two pages. An offset can be applied to
    shift the split point from the exact center. For example, an offset of 0.015
    shifts the split point 1.5% to the right of the center, which helps avoid the book spine.
    """
    if image is None:
        return None, None
    height, width, _ = image.shape
    
    # Calculate the split point, applying the offset from the center (0.5).
    # A positive offset shifts the split to the right.
    midpoint = int(width * (0.5 + offset_percentage))
    
    left_page = image[:, :midpoint]
    right_page = image[:, midpoint:]
    return left_page, right_page

def save_pages(left_page: np.ndarray, right_page: np.ndarray, output_dir: Path, filename: str) -> None:
    """Saves the left and right pages to the output directory, always as JPG."""
    try:
        if not output_dir.exists():
            output_dir.mkdir(parents=True, exist_ok=True)
        
        base_name, _ = os.path.splitext(filename)
        output_ext = ".jpg"  # Always save as JPG
        
        left_output_path = output_dir / f"{base_name}_1{output_ext}"
        right_output_path = output_dir / f"{base_name}_2{output_ext}"

        cv2.imwrite(str(left_output_path), left_page)
        cv2.imwrite(str(right_output_path), right_page)
    except Exception as e:
        raise IOError(f"Error saving pages for {filename}: {e}")

def process_image(image_path: Path, output_dir: Path, logger: logging.Logger):
    """
    Processes an image using a two-phase cropping approach for best results.
    Full pipeline: load -> rotate -> deskew -> coarse crop -> split -> fine crop -> save.
    """
    try:
        # Steps 1 & 2: Load the image and correct its orientation
        image = load_image(image_path)
        image = cv2.rotate(image, cv2.ROTATE_180)
        
        # Step 3: Deskew the entire image to align the text horizontally
        deskewed_image = deskew_image(image)

        # Step 4: Coarse crop - Isolate the book from the background (e.g., the table)
        # We use a larger padding here to ensure we don't clip the book itself.
        book_image = crop_to_content(deskewed_image, padding=50)
        
        # Step 5: Split the cropped book image into left and right pages
        # We apply a small offset to the right to ensure the split avoids the book's spine.
        left_page, right_page = split_pages(book_image, offset_percentage=0.015) # Split at 51.5%
        
        # Step 6: Fine crop - Tightly crop each page to its text content
        # A smaller, standard padding is used here for a clean final result.
        final_left_page = crop_to_content(left_page)
        final_right_page = crop_to_content(right_page)

        # Step 7: Save the processed pages
        save_pages(final_left_page, final_right_page, output_dir, image_path.name)
        return True
    except Exception as e:
        logger.error(f"Failed to process {image_path.name}: {e}")
        return False

def process_batch(batch_dir: Path, workers: int = 4) -> Dict[str, int]:
    """
    Processes a batch of images in parallel.
    """
    input_dir = batch_dir / "imagenes_juntas"
    output_dir = batch_dir / "imagenes_separadas"
    
    if not input_dir.is_dir():
        print(f"Input directory not found: {input_dir}")
        return {"processed": 0, "errors": 0}

    # Setup logging
    log_file = batch_dir / "errors.log"
    logger = logging.getLogger(str(batch_dir))
    if not logger.handlers:
        handler = logging.FileHandler(log_file)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.ERROR)

    SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".cr3", ".nef", ".arw", ".dng"}
    image_files = [f for f in input_dir.iterdir() if f.is_file() and f.suffix.lower() in SUPPORTED_EXTENSIONS]

    if not image_files:
        print(f"No supported image files (e.g., jpg, png, cr3) found in {input_dir}")
        return {"processed": 0, "errors": 0}

    processed_count = 0
    error_count = 0
    
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(process_image, image_path, output_dir, logger): image_path for image_path in image_files}
        
        with tqdm(total=len(image_files), desc=f"Processing Batch: {batch_dir.name}") as pbar:
            for future in as_completed(futures):
                if future.result():
                    processed_count += 1
                else:
                    error_count += 1
                pbar.update(1)

    return {"processed": processed_count, "errors": error_count}


def main() -> None:
    """
    Main function to handle command-line arguments and start the batch processing.
    
    Installation:
    pip install -r requirements.txt
    
    Usage:
    
    Process a single batch:
    python process_book_pages.py --path /path/to/contenedor/lote_X
    
    Process all batches in the container:
    python process_book_pages.py --path /path/to/contenedor
    """
    parser = argparse.ArgumentParser(description="Batch process book page images.")
    parser.add_argument("--path", type=Path, required=True, help="Path to a batch directory or a container of batches.")
    parser.add_argument("--workers", type=int, default=os.cpu_count(), help="Number of worker processes to use.")
    args = parser.parse_args()

    if not args.path.exists():
        print(f"Error: The path {args.path} does not exist.")
        return

    total_processed = 0
    total_errors = 0
    
    if args.path.name.startswith("lote_"):
        # Process a single batch
        stats = process_batch(args.path, args.workers)
        total_processed += stats["processed"]
        total_errors += stats["errors"]
    else:
        # Process all batches in the container
        batch_dirs = [d for d in args.path.iterdir() if d.is_dir() and d.name.startswith("lote_")]
        for batch_dir in batch_dirs:
            stats = process_batch(batch_dir, args.workers)
            total_processed += stats["processed"]
            total_errors += stats["errors"]
            
    print("\n--- Processing Complete ---")
    print(f"Total images processed: {total_processed}")
    print(f"Total errors: {total_errors}")

if __name__ == "__main__":
    main() 