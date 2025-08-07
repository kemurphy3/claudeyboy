# K8M AOP Utils TIFF Format Fix

## Problem
The `neonProductsTest.py` script was failing with a TIFF format error when trying to create RGB radiance mosaics. The error indicated an invalid TIFF format with `SampleFormat=IEEEFP` and `BitsPerSample=8`, which is an incompatible combination.

## Root Cause
The issue occurred in the `writeRasterToTif()` function in `aop_processing_utils_k8m_v1.py`. The function was using the input array's dtype directly without ensuring it was compatible with TIFF format specifications.

## Solution Implemented

### 1. Fixed TIFF Creation (Preventive)
Modified `writeRasterToTif()` function (line 3640-3656) to:
- Check the input data type before writing
- Ensure proper data type conversion for TIFF compatibility
- Handle special cases for radiance data (already converted to int16)
- Convert problematic float types to float32 instead of allowing invalid combinations

### 2. Enhanced Error Reporting (Diagnostic)
Modified `getMapExtentsForMosiac()` function (line 4299-4315) to:
- Catch the specific TIFF format error
- Provide clear error messages indicating which file is problematic
- Suggest remediation steps (delete and regenerate the file)
- Prevent silent failures by raising a RuntimeError

## How to Use

1. **If you encounter the error with existing files:**
   - Delete the problematic TIFF file mentioned in the error message
   - Re-run the script - it will regenerate the file with the correct format

2. **For new processing runs:**
   - The fix will automatically ensure proper TIFF format
   - No manual intervention needed

## Technical Details

The fix ensures that:
- uint8 data remains uint8 (no float conversion)
- Float data is converted to float32 (not uint8)
- Radiance data that's already converted to int16 is preserved
- The TIFF writer receives compatible dtype specifications

## Files Modified
- `aop_processing_utils_k8m_v1.py`:
  - `writeRasterToTif()` function: Added data type validation
  - `getMapExtentsForMosiac()` function: Added error handling with diagnostics

## Note
This is a minimal change approach as requested - it only modifies the essential parts to fix the TIFF format issue while maintaining all existing functionality.