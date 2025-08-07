# K8M Radiance Processing Adaptation

## Problem
The k8m version of aop_processing_utils was missing the `rawNisLineClass` implementation, causing an AttributeError when trying to process radiance data.

## Solution
Adapted the MATLAB implementation that was commented out in the original code to replace the missing rawNisLineClass functionality.

## Changes Made

### In `processRadiance()` method (line 11600-11657)

1. **Removed dependency on rawNisLineClass**
   - The missing class was expected to handle DN to Radiance conversion
   - Replaced with direct MATLAB engine calls

2. **Added MATLAB implementation**
   - Uses `matlab.engine` to call `nisCalCode_noGUI`
   - Automatically searches for DN2Radiance directory in common locations
   - Provides clear error messages if MATLAB components are missing

3. **Directory search logic**
   - Checks environment variable `DN2RAD_DIR` first
   - Falls back to common installation paths:
     - D:/Gold_Pipeline/ProcessingPipelines/NIS/DN2Radiance
     - C:/Gold_Pipeline/ProcessingPipelines/NIS/DN2Radiance
     - D:/NIS/Processing/DN2Radiance
     - C:/NIS/Processing/DN2Radiance

## Usage

The adapted code will:
1. Check if radiance file already exists
2. If not, start MATLAB engine
3. Navigate to DN2Radiance directory
4. Call `nisCalCode_noGUI` with appropriate parameters
5. Close MATLAB engine when complete

## Requirements

- MATLAB installed with Python engine
- DN2Radiance MATLAB code directory
- `nisCalCode_noGUI.m` function available

## Note

The radiance QA generation step is skipped as it requires the rawNisLineClass. This shouldn't affect the main processing pipeline but means QA files won't be generated for the radiance step.