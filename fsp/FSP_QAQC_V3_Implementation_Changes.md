# FSP QAQC V3 Implementation Changes

**Date:** 2024-08-04  
**Purpose:** Document all changes made to create v3 of the FSP QAQC script  
**Previous Version:** fsp_qaReview_render_skeleton_compliant_v2.Rmd  
**New Version:** fsp_qaReview_render_skeleton_compliant_v3.Rmd

## Executive Summary

Version 3 implements the correct FSP QAQC criteria as specified by the user, fixing critical errors in primary key definitions, data source locations, and validation methodologies. This version follows the authoritative requirements documented in FSP_QAQC_Implementation_Criteria.md.

## Critical Changes Made

### 1. Primary Key Corrections

**BEFORE (v2 - INCORRECT):**
```r
# Check 11: sampleID for sampleMetadata
fsp_sampleMetadata_pub$duplicateRecordQF <- as.integer(duplicated(fsp_sampleMetadata_pub$sampleID))

# Check 12: sampleID for spectralData  
fsp_spectralData_pub$duplicateRecordQF <- as.integer(duplicated(fsp_spectralData_pub$sampleID))
```

**AFTER (v3 - CORRECT):**
```r
# Check 11: spectralSampleID for sampleMetadata (NOT sampleID)
fsp_sampleMetadata_pub$duplicateRecordQF <- as.integer(duplicated(fsp_sampleMetadata_pub$spectralSampleID))

# Check 12: spectralSampleID for spectralData (NOT sampleID)
fsp_spectralData_pub$duplicateRecordQF <- as.integer(duplicated(fsp_spectralData_pub$spectralSampleID))
```

### 2. CSV File Download Implementation

**BEFORE (v2 - INCORRECT):**
- Checked data directly in pub tables
- No CSV downloads performed
- Missing downloadFileUrl usage

**AFTER (v3 - CORRECT):**
```r
# Added expanded package parameter to get downloadFileUrl
spectra <- try(neonUtilities::loadByProduct(
  dpID='DP1.30012.001',
  package = "expanded",  # Important for getting downloadFileUrl
  ...
))

# Implemented CSV download and processing for checks 7, 13, 14, 15
temp_file <- tempfile(fileext = ".csv")
download.file(download_urls$downloadFileUrl[i], temp_file, quiet = TRUE)
csv_data <- read.csv(temp_file, stringsAsFactors = FALSE)
```

### 3. Spectral Ratio Calculation

**BEFORE (v2 - INCORRECT):**
```r
# Single wavelength comparison
spectral_ratio_valid = reflectance_1000 > reflectance_500
```

**AFTER (v3 - CORRECT):**
```r
# Average reflectance for wavelength ranges
avg_495_505 <- mean(csv_data$reflectance[csv_data$wavelength >= 495 & csv_data$wavelength <= 505], na.rm = TRUE)
avg_995_1005 <- mean(csv_data$reflectance[csv_data$wavelength >= 995 & csv_data$wavelength <= 1005], na.rm = TRUE)
ratio_valid = avg_995_1005 > avg_495_505
```

### 4. Band Count Verification

**BEFORE (v2):**
- Counted records in pub table
- Used hardcoded 426 value without ratio calculation

**AFTER (v3):**
```r
# Download CSV and calculate ratio
record_count <- nrow(csv_data)
band_ratio <- record_count / 426
ratio_valid = band_ratio >= 25 & band_ratio <= 26
```

### 5. Required Fields Validation

**BEFORE (v2):**
- Used hardcoded required fields lists

**AFTER (v3):**
- Added placeholder for entryValidationRulesParser integration
- Documented need for validation rules table
- Maintained skeleton compliance with proper error handling

### 6. GCS Integration

**ADDED in v3:**
- Full GCS output section from skeleton template
- Proper eval=T for production use
- Complete output organization and export code
- Manifest generation and retrieval instructions

## Check-by-Check Implementation Summary

| Check # | Description | v2 Implementation | v3 Implementation | Status |
|---------|-------------|-------------------|-------------------|---------|
| 1 | Bout completeness | ✓ Correct | ✓ Correct | No change |
| 2 | CFC sample matching | ✓ Correct | ✓ Correct | No change |
| 3 | Sample count per event | ✓ Correct | ✓ Correct | No change |
| 4-6 | Required fields | Hardcoded lists | Placeholder for validation rules | Needs rules table |
| 7 | Band count | Pub table count | CSV download & ratio calc | **FIXED** |
| 8 | FSP-CFC timing | ✓ Correct | ✓ Correct | No change |
| 9 | Bout duration | ✓ Correct | ✓ Correct | No change |
| 10 | Bout duplicates | ✓ Correct (eventID) | ✓ Correct (eventID) | No change |
| 11 | Sample duplicates | ✗ Wrong (sampleID) | ✓ Correct (spectralSampleID) | **FIXED** |
| 12 | Spectral duplicates | ✗ Wrong (sampleID) | ✓ Correct (spectralSampleID) | **FIXED** |
| 13 | Reflectance range | Pub table check | CSV download & check | **FIXED** |
| 14 | Wavelength range | Pub table check | CSV download & check | **FIXED** |
| 15 | Spectral ratio | Single wavelength | Range averages | **FIXED** |
| 16 | Biome validation | Not implemented | Not implemented | Per instruction |

## Skeleton Template Compliance

Version 3 maintains full skeleton template compliance:

1. **Document Structure**: 
   - Proper YAML header with all required parameters
   - Sections in correct order: Load → Duplicates → Completeness → Timeliness → Plausibility → Outputs
   
2. **Output Naming**: 
   - Lists use `mod.` prefix (e.g., `fsp.bout.completeness`)
   - DataFrames avoid `mod_` prefix
   
3. **GCS Integration**: 
   - Full GCS output code included
   - Set to eval=T for production
   - Proper manifest and retrieval instructions

4. **Tabset Formatting**: 
   - Maintains `.tabset .tabset-fade .tabset-pills` format
   - Proper subsection organization

## Testing Recommendations

1. **Test with Real Data**: Run with actual FSP data to verify CSV downloads work
2. **Verify Primary Keys**: Check that duplicate detection uses spectralSampleID
3. **Validate Ratios**: Confirm spectral ratio calculations with known good/bad samples
4. **Check GCS Upload**: Verify outputs are properly written to GCS
5. **Obtain Validation Rules**: Get entryValidationRulesParser table for checks 4-6

## Migration Notes

When migrating from v2 to v3:

1. **Data Loading**: Must use `package = "expanded"` parameter in loadByProduct
2. **Memory Usage**: CSV downloads may increase memory usage - monitor for large datasets
3. **Network Requirements**: Script now requires internet access for CSV downloads
4. **Error Handling**: Added try-catch blocks for CSV download failures

## Future Enhancements

1. Implement actual validation rules table integration (checks 4-6)
2. Add caching for downloaded CSV files to improve performance
3. Consider parallel processing for CSV downloads
4. Add progress indicators for long-running CSV operations

---

**IMPORTANT**: This v3 implementation follows the authoritative criteria in FSP_QAQC_Implementation_Criteria.md. Any future modifications should reference that document, not the abbreviated descriptions in fsp_QAQC_checks.csv.