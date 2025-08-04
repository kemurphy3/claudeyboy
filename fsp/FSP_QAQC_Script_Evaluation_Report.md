# FSP QAQC Script Evaluation Report
## fsp_qaReview_render_skeleton_compliant_v2.Rmd

**Date:** 2024-08-04  
**Evaluator:** Script Compliance Analyst

---

## 1. COMPLIANCE REPORT

### 1.1 Skeleton Compliance: **PARTIAL PASS**

#### Document Structure Compliance
- ✅ **YAML Header:** Matches skeleton format with proper params
- ✅ **CSS Styling:** Includes standard TOC and page width settings
- ✅ **Section Order:** Follows skeleton sequence (Load → Duplicates → Completeness → Timeliness → Plausibility → Outputs)
- ✅ **R Version Display:** Includes version string and neonOSqc version
- ✅ **Data Period Display:** Shows date range being checked
- ✅ **Metadata Requirements:** All tables have dpID, dpName, tableName, domainID added

#### Deviations from Skeleton
1. ❌ **Missing Section:** No "Instructions for use" section (though this should be deleted in production)
2. ⚠️ **Section Naming:** Uses "Band Count Completeness" instead of generic DP-specific section names
3. ⚠️ **Tabset Format:** Uses `{.tabset .tabset-fade .tabset-pills}` instead of just `{.tabset}`
4. ❌ **Output Section:** Missing GCS storage location information and write_output_gcs() implementation

### 1.2 Output Structure Compliance: **PARTIAL PASS**

#### Naming Convention Compliance
- ✅ **Lists:** Correctly use `fsp.` prefix (e.g., `fsp.bout.completeness`)
- ✅ **DataFrames:** Use `fsp_` prefix for input data
- ❌ **Inconsistent Tracking:** Not all outputs are tracked in listOuts/dfOuts

#### Output Tables Compliance
- ✅ All neonOSqc functions return standard output structure:
  - `*_all`: All data with flags
  - `*_flags`: Only flagged records
  - `*_summary`: Summary statistics
  - `*_figure`: Optional visualization
- ⚠️ Custom checks don't follow this exact pattern

### 1.3 QAQC Check Coverage Table

| Check # | Description | CSV Function | Script Implementation | Status |
|---------|-------------|--------------|----------------------|--------|
| 1 | Bout completeness | complete_bout | `complete_bout()` | ✅ PASS |
| 2 | CFC sample matching | complete_cross_table | `complete_cross_table()` | ✅ PASS |
| 3 | Sample count per eventID | complete.cross.table | `complete_within_bout()` | ✅ PASS* |
| 4 | Bout metadata fields complete | complete.within.rec | Custom implementation | ⚠️ PARTIAL |
| 5 | Sample metadata fields complete | complete.within.rec | Custom implementation | ⚠️ PARTIAL |
| 6 | Spectral data fields complete | complete.within.rec | Custom implementation | ⚠️ PARTIAL |
| 7 | 426 bands per sample | complete.within.bout | Custom implementation | ⚠️ PARTIAL |
| 8 | 31-day window check | timely.bout.duration | `timely_bout_duration()` | ✅ PASS |
| 9 | 30-day duration check | timely.bout.duration | `timely_bout_duration()` | ✅ PASS |
| 10 | Bout metadata duplicates | de-dupe function | `neonOS::removeDups()` or custom | ✅ PASS |
| 11 | Sample metadata duplicates | de-dupe function | `neonOS::removeDups()` or custom | ✅ PASS |
| 12 | Spectral data duplicates | de-dupe function | `neonOS::removeDups()` or custom | ✅ PASS |
| 13 | Reflectance range 0-1 | custom | Custom implementation | ✅ PASS |
| 14 | Wavelength range 0-3000 | custom | Custom implementation | ✅ PASS |
| 15 | Spectral ratio 1000>500 | custom | Custom implementation | ✅ PASS |
| 16 | Biome-specific ratios | future | Not implemented | ✅ PASS** |

*Note: Check #3 uses `complete_within_bout` instead of `complete_cross_table` as specified in CSV  
**Note: Check #16 marked as "future" in CSV, correctly not implemented

---

## 2. GAP ANALYSIS

### 2.1 Missing or Incomplete QAQC Checks

1. **Within-Record Completeness (Checks 4-6):**
   - **Issue:** Uses custom implementation instead of `complete_within_rec()`
   - **Reason:** FSP tables are not in long format required by the function
   - **Impact:** Results don't follow standard output structure

2. **Band Count Check (Check 7):**
   - **Issue:** Custom implementation instead of `complete_within_bout()`
   - **Reason:** No lookup table configured for band counts
   - **Impact:** Inconsistent with standard neonOSqc patterns

### 2.2 Redundant or Unapproved Functions

1. **Custom Duplicate Checking:**
   - Simple duplicate detection when `variables_30012` is not available
   - Should always use `neonOS::removeDups()` with proper variables file

2. **Custom Field Completeness:**
   - Manual checking of required fields
   - Should attempt to restructure data for `complete_within_rec()` or use approved alternative

### 2.3 Improper Product References

✅ **PASS:** No references to non-FSP/CFC products found

---

## 3. RECOMMENDATIONS

### 3.1 Skeleton Compliance Fixes

1. **Add Missing Sections:**
   ```r
   ## Instructions for use
   ::: {style="color: SlateBlue"}
   [Instructions to be removed in production]
   :::
   ```

2. **Fix Tabset Formatting:**
   - Change all `{.tabset .tabset-fade .tabset-pills}` to `{.tabset}`

3. **Complete Outputs Section:**
   ```r
   ## Outputs
   
   # Storage information
   cat("GCS project:", neonOSqc:::default_gcs_project_name, "\n")
   cat("GCS bucket:", neonOSqc:::default_gcs_bucket_name, "\n")
   cat("Filepath:", neonOSqc:::default_gcs_prefix, "\n")
   
   # [Add code to properly track and export all outputs]
   ```

4. **Fix Output Tracking:**
   - Ensure all created objects are properly added to listOuts or dfOuts
   - Use correct object names (e.g., actual flagged dataframes not "_dups")

### 3.2 Function Usage Recommendations

1. **Within-Record Completeness:**
   - Option A: Create a wrapper to convert FSP tables to long format for `complete_within_rec()`
   - Option B: Use `format_custom_outputs()` to standardize custom check outputs

2. **Band Count Check:**
   - Create appropriate lookup table entry for `complete_within_bout()` with:
     - tableName: "fsp_spectralData_pub"
     - minExpectedRecords: 426
     - maxExpectedRecords: 426

3. **Always Use removeDups():**
   - Ensure variables_30012 file is available
   - Never fall back to custom duplicate checking

### 3.3 Modularity Improvements

1. **Encapsulate Custom Checks:**
   ```r
   # Create functions for each custom check
   check_reflectance_range <- function(data) {
     # Implementation
     return(format_custom_outputs(...))
   }
   ```

2. **Standardize Custom Output Format:**
   - Use `format_custom_outputs()` for all custom implementations
   - Ensure consistent _all, _flags, _summary structure

3. **Create Helper Functions:**
   - `add_fsp_metadata()`: Add required columns to all tables
   - `check_fsp_required_fields()`: Standardized field checking

---

## 4. SUMMARY

### Overall Assessment: **REQUIRES MINOR REVISIONS**

The script successfully implements all required QAQC checks but needs adjustments for full skeleton compliance:

**Strengths:**
- All 16 active checks are implemented
- Proper use of most neonOSqc functions
- Good error handling and conditional logic
- Clear tabset organization

**Required Fixes:**
1. Complete the Outputs section with GCS integration
2. Standardize tabset formatting
3. Fix output tracking in listOuts/dfOuts
4. Consider restructuring custom checks to use approved functions

**Priority Actions:**
1. HIGH: Fix Outputs section for GCS compliance
2. MEDIUM: Standardize custom check outputs using `format_custom_outputs()`
3. LOW: Minor formatting adjustments (tabsets, section names)

The script is functionally complete but requires structural adjustments to achieve full skeleton compliance.