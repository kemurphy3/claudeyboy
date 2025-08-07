# AOP TIFF Format Error Fix

## Problem
The `neonProductsTest.py` script was failing with the following error:
```
rasterio._err.CPLE_AppDefinedError: D:/2022/FullSite/D16/2022_ABBY_5/QA/Spectrometer/OrthoRadiance/2022071115/RGB_radiance_mosaic.tif: Cannot open TIFF file with SampleFormat=IEEEFP and BitsPerSample=8
```

This error occurs because the TIFF file has an incompatible format specification:
- `SampleFormat=IEEEFP` indicates IEEE floating-point format
- `BitsPerSample=8` indicates 8-bit samples

This combination is invalid because 8-bit samples cannot represent IEEE floating-point values.

## Solution Implemented

The fix involves three components:

### 1. Error Handling in `getMapExtentsForMosiac()`
Added try-catch block to handle `RasterioIOError` exceptions when opening TIFF files:
- Detects the specific "SampleFormat=IEEEFP and BitsPerSample=8" error
- Attempts to convert the problematic file
- Continues processing if conversion fails

### 2. New Function: `convertProblematicTiff()`
Created a conversion function with two approaches:
- **Primary method**: Uses GDAL to read and re-write the TIFF with proper format
- **Fallback method**: Uses PIL/Pillow to convert via numpy array

### 3. Validation Check
Added check to ensure at least one valid TIFF file can be processed before continuing with mosaic generation.

## Usage

The fix is automatic - when the script encounters a problematic TIFF file:
1. It will print a warning message
2. Attempt to convert the file
3. Use the converted file if successful
4. Skip the file if conversion fails

## Requirements

For best results, ensure one of these libraries is installed:
- `GDAL` (recommended): `pip install GDAL`
- `Pillow`: `pip install Pillow`

If neither is available, problematic files will be skipped.

## Files Modified
- `/home/nerds/Murph/claudeyboy/aop/aop_processing_utils.py`
  - Added `convertProblematicTiff()` function
  - Modified `getMapExtentsForMosiac()` function

## Testing
To test the fix:
1. Run `neonProductsTest.py` again
2. Watch for warning messages about file conversions
3. The script should continue processing instead of crashing