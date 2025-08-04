# FSP QAQC Implementation Criteria - AUTHORITATIVE VERSION

**Date:** 2024-08-04  
**Purpose:** Document the correct implementation criteria for FSP QAQC checks  
**Status:** This document supersedes all previous interpretations

## CRITICAL: Use These Criteria, Not the CSV Column Names

The fsp_QAQC_checks.csv file contains abbreviated descriptions. The actual implementation requirements are detailed below.

---

## Primary Key Definitions

**IMPORTANT**: These are the correct primary keys for duplicate detection:

1. **fsp_boutMetadata_pub**: Primary key = `eventID`
2. **fsp_sampleMetadata_pub**: Primary key = `spectralSampleID` (NOT sampleID)
3. **fsp_spectralData_pub**: Primary key = `spectralSampleID` (NOT sampleID)

---

## Detailed Check Implementations

### Check 1: Bout Completeness
- **Table**: fsp_boutMetadata_pub
- **Implementation**: Match `eventID` to `boutID` in the fsp_complete_bout lookup table
- **Note**: This is matching eventID to boutID, not just checking completeness

### Check 2: CFC Sample Matching
- **Table**: fsp_sampleMetadata_pub
- **Implementation**: Match on `sampleID` between FSP and CFC tables
- **Note**: Uses sampleID field (not spectralSampleID) for CFC matching

### Check 3: Sample Count per Event
- **Table**: fsp_sampleMetadata_pub
- **Implementation**: Verify at least 1 record exists for each `eventID`
- **Note**: Minimum threshold is 1, not a specific expected count

### Checks 4-6: Required Fields Validation
- **Tables**: All three FSP tables
- **Implementation**: 
  1. Pull in validation table `entryValidationRulesParser`
  2. Check that all fields marked as 'required' have data
- **Note**: Do NOT use hardcoded required fields lists

### Check 7: Band Count Verification
- **Table**: fsp_spectralData_pub
- **Implementation**:
  1. Download CSV files using URLs from `downloadFileUrl` field
  2. Count total records in each CSV
  3. Divide by 426
  4. Result should be between 25 and 26
- **Note**: Must download and process actual CSV files, not count records in pub table

### Check 8: FSP-CFC Timing Window
- **Table**: fsp_sampleMetadata_pub
- **Implementation**: Verify FSP end date is within 31 days of CFC collection start date
- **Note**: This is cross-product timing, not internal FSP duration

### Check 9: FSP Bout Duration
- **Table**: fsp_sampleMetadata_pub
- **Implementation**: Verify ≤30 days between earliest and latest FSP sample
- **Note**: This is internal FSP duration check

### Checks 10-12: Duplicate Detection
- **Implementation**: Use primary keys defined above
- **Check 10**: eventID for boutMetadata
- **Check 11**: spectralSampleID for sampleMetadata
- **Check 12**: spectralSampleID for spectralData

### Check 13: Reflectance Range
- **Table**: Downloaded CSV files (not pub table)
- **Implementation**: Verify all values in `reflectance` field are between 0-1
- **Note**: Check performed on downloaded CSV data

### Check 14: Wavelength Range
- **Table**: Downloaded CSV files (not pub table)
- **Implementation**: Verify all `wavelength` values are between 300-2600
- **Note**: Check performed on downloaded CSV data

### Check 15: Spectral Ratio Validation
- **Table**: Downloaded CSV files
- **Implementation**:
  1. Calculate average reflectance for wavelengths 995-1005
  2. Calculate average reflectance for wavelengths 495-505
  3. Verify average(995-1005) > average(495-505)
- **Note**: Uses ranges, not single wavelengths

### Check 16: Biome-specific Validation
- **Implementation**: DO NOT IMPLEMENT (per instruction)
- **Status**: Future enhancement

---

## Key Implementation Notes

1. **CSV Downloads Required**: Checks 7, 13, 14, and 15 require downloading and processing actual CSV files from downloadFileUrl

2. **Validation Rules**: Checks 4-6 require the entryValidationRulesParser table (not hardcoded field lists)

3. **Primary Keys**: Use spectralSampleID (not sampleID) for checks 11 and 12

4. **Wavelength Ranges**: Check 15 uses average of ranges, not single wavelength comparisons

5. **Cross-Product Timing**: Check 8 compares FSP to CFC dates, not just FSP internal duration

---

## Script Versions

- **v2**: Previous implementation with incorrect primary keys and simplified checks
- **v3**: New implementation following these correct criteria
- **All future versions**: Must follow these criteria exactly

---

**IMPORTANT**: When implementing or reviewing FSP QAQC scripts, always refer to this document, not just the CSV descriptions.