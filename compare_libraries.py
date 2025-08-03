#!/usr/bin/env python3
"""
Library Comparison Script
Compares rawpy vs canon_cr3 for CR3 file processing
"""

import argparse
import time
from pathlib import Path
import cv2
import rawpy
from canon_cr3_wrapper import convert_cr3_to_jpg_canon_cr3

def compare_cr3_libraries(cr3_file: Path, output_dir: Path):
    """
    Compare rawpy vs canon_cr3 for CR3 file processing
    """
    output_dir.mkdir(exist_ok=True)
    
    print(f"Comparing libraries for: {cr3_file.name}")
    print("=" * 60)
    
    # Test rawpy
    print("1. Testing rawpy (current method)...")
    start_time = time.time()
    try:
        with rawpy.imread(str(cr3_file)) as raw:
            rgb = raw.postprocess(use_camera_wb=True)
            rawpy_image = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        
        rawpy_output = output_dir / f"{cr3_file.stem}_rawpy.jpg"
        cv2.imwrite(str(rawpy_output), rawpy_image)
        
        rawpy_time = time.time() - start_time
        rawpy_size = rawpy_output.stat().st_size
        
        print(f"   ✅ Success: {rawpy_time:.2f}s, Output: {rawpy_size:,} bytes")
        print(f"   📁 Saved: {rawpy_output}")
        
    except Exception as e:
        print(f"   ❌ Failed: {e}")
        rawpy_time = None
        rawpy_size = None
    
    print()
    
    # Test canon_cr3
    print("2. Testing canon_cr3 (experimental method)...")
    start_time = time.time()
    try:
        canon_output = output_dir / f"{cr3_file.stem}_canon_cr3.jpg"
        
        if convert_cr3_to_jpg_canon_cr3(cr3_file, canon_output):
            canon_time = time.time() - start_time
            canon_size = canon_output.stat().st_size if canon_output.exists() else 0
            
            print(f"   ✅ Success: {canon_time:.2f}s, Output: {canon_size:,} bytes")
            print(f"   📁 Saved: {canon_output}")
        else:
            print(f"   ❌ Failed: Conversion returned False")
            canon_time = None
            canon_size = None
            
    except Exception as e:
        print(f"   ❌ Failed: {e}")
        canon_time = None
        canon_size = None
    
    print()
    
    # Comparison summary
    print("📊 COMPARISON SUMMARY:")
    print("-" * 30)
    
    if rawpy_time and canon_time:
        speed_diff = ((rawpy_time - canon_time) / rawpy_time) * 100
        if speed_diff > 0:
            print(f"🏃 Speed: canon_cr3 is {speed_diff:.1f}% faster")
        else:
            print(f"🏃 Speed: rawpy is {abs(speed_diff):.1f}% faster")
    elif rawpy_time:
        print(f"🏃 Speed: Only rawpy succeeded ({rawpy_time:.2f}s)")
    elif canon_time:
        print(f"🏃 Speed: Only canon_cr3 succeeded ({canon_time:.2f}s)")
    else:
        print("🏃 Speed: Both methods failed")
    
    if rawpy_size and canon_size:
        size_diff = ((canon_size - rawpy_size) / rawpy_size) * 100
        if size_diff > 0:
            print(f"📦 Size: canon_cr3 output is {size_diff:.1f}% larger")
        else:
            print(f"📦 Size: rawpy output is {abs(size_diff):.1f}% larger")
    elif rawpy_size:
        print(f"📦 Size: Only rawpy produced output ({rawpy_size:,} bytes)")
    elif canon_size:
        print(f"📦 Size: Only canon_cr3 produced output ({canon_size:,} bytes)")
    
    print()
    
    # Recommendations
    print("💡 RECOMMENDATIONS:")
    print("-" * 20)
    
    if rawpy_time and not canon_time:
        print("✅ Use rawpy: More reliable and stable")
        print("❌ Avoid canon_cr3: Failed to process this file")
    elif canon_time and not rawpy_time:
        print("✅ Use canon_cr3: Only working option for this file")
        print("❌ rawpy failed for this file")
    elif rawpy_time and canon_time:
        if rawpy_time < canon_time:
            print("✅ Use rawpy: Faster and more reliable")
        else:
            print("🤔 Both work, but canon_cr3 is faster (test quality)")
    else:
        print("❌ Both methods failed - file may be corrupted")
    
    print()

def main():
    parser = argparse.ArgumentParser(description="Compare rawpy vs canon_cr3 libraries")
    parser.add_argument("--cr3-file", type=Path, required=True, help="CR3 file to test")
    parser.add_argument("--output-dir", type=Path, default=Path("library_comparison"), help="Output directory for comparison results")
    
    args = parser.parse_args()
    
    if not args.cr3_file.exists():
        print(f"Error: CR3 file not found: {args.cr3_file}")
        return
    
    if args.cr3_file.suffix.lower() != '.cr3':
        print(f"Error: File must be a CR3 file, got: {args.cr3_file.suffix}")
        return
    
    print("🔬 CR3 LIBRARY COMPARISON TOOL")
    print("=" * 60)
    print(f"Testing file: {args.cr3_file}")
    print(f"Output directory: {args.output_dir}")
    print()
    
    compare_cr3_libraries(args.cr3_file, args.output_dir)
    
    print("🏁 Comparison completed!")
    print(f"Check the results in: {args.output_dir}")

if __name__ == "__main__":
    main() 