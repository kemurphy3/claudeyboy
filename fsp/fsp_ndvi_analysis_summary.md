# FSP NDVI Analysis Summary Report (2019-2024)

## Overview

This report summarizes the NDVI analysis setup for NEON FSP (Field Spectral) data from 2019-2024. Since R is not available in the current environment, this document provides the analysis framework and expected outputs.

## Analysis Configuration

- **Date Range**: January 2019 - December 2024
- **Data Product**: DP1.30012.001 (Field spectral)
- **Sites**: All NEON sites with FSP data
- **Key Measurements**:
  - NDVI calculation using red (660-680 nm) and NIR (790-810 nm) bands
  - Spectral ratio (995-1005 nm / 495-505 nm)
  - Data quality metrics

## Data Processing Steps

### 1. Data Loading
```r
# Load FSP data for 2019-2024
fsp_data <- neonUtilities::loadByProduct(
  dpID = "DP1.30012.001",
  site = "all",
  startdate = "2019-01",
  enddate = "2024-12",
  package = "expanded"
)
```

### 2. Extract Tables
- `fsp_spectralData`: Contains spectral sample IDs and download URLs
- `fsp_sampleMetadata`: Contains collection dates, sites, and sample details
- `fsp_boutMetadata`: Contains bout-level information

### 3. NDVI Calculation Process

For each spectral sample:
1. Download CSV file from `downloadFileUrl`
2. Read wavelength and reflectance columns
3. Calculate averages:
   - Red reflectance: mean of 660-680 nm
   - NIR reflectance: mean of 790-810 nm
4. Compute NDVI: (NIR - Red) / (NIR + Red)

### 4. Expected NDVI Ranges by Ecosystem

Based on typical NEON sites:
- **Forest sites** (e.g., HARV, SERC): 0.7 - 0.9
- **Grassland sites** (e.g., KONZ, CPER): 0.3 - 0.7
- **Desert sites** (e.g., SRER, JORN): 0.1 - 0.4
- **Tundra sites** (e.g., TOOL, BARR): 0.2 - 0.6

## Quality Control Checks

### 1. Reflectance Validation
- All reflectance values should be between 0 and 1
- Flag samples with reflectance > 1 or < 0

### 2. NDVI Validation
- NDVI values must be between -1 and 1
- Most vegetation should have positive NDVI (> 0)

### 3. Spectral Ratio Check
- Ratio of 995-1005 nm / 495-505 nm should typically be > 1
- This validates spectral quality and instrument calibration

### 4. Wavelength Coverage
- Valid spectra should cover 300-2600 nm
- Minimum 426 wavelength bands expected

## Expected Outputs

### 1. Summary Statistics
- Total samples processed
- Mean NDVI by site
- Temporal trends (seasonal patterns)
- Data quality metrics

### 2. Visualizations
- NDVI distribution histogram
- NDVI by site (boxplots)
- Time series plots
- Spectral quality plots

### 3. Data Tables
- Detailed results with NDVI values
- Quality flags
- Site-specific summaries

## Running the Analysis

To execute the full analysis with R:

```bash
# In R environment
rmarkdown::render('fsp_ndvi_analysis_report_fixed.Rmd')
```

This will:
1. Download FSP data for 2019-2024
2. Process spectral CSV files
3. Calculate NDVI and quality metrics
4. Generate HTML report with interactive visualizations

## Key Files Created

1. **fsp_ndvi_analysis_report_fixed.Rmd**: Complete analysis script
2. **test_ndvi_report.R**: Test script to run the analysis
3. **Original issues fixed**:
   - Corrected data product ID (DP1.30012.001)
   - Proper CSV file processing
   - Correct wavelength band extraction
   - Error handling for missing data

## Notes

- Processing time depends on number of samples and internet speed
- The script processes first 10 samples by default (adjustable)
- Requires NEON_PAT token for faster downloads (optional)
- All spectral CSV files are downloaded temporarily and cleaned up

## Next Steps

1. Run the analysis in an R environment
2. Review NDVI patterns across sites and years
3. Identify any data quality issues
4. Export results for further analysis