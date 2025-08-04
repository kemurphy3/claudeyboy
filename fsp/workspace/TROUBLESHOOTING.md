# FSP QAQC Script Troubleshooting Guide

## **NEW: Click and Go Functionality**

The `fsp_qaReview_render_v3.Rmd` script is now a **fully automated click-and-go script** that:
- ✅ Runs independently without external dependencies
- ✅ Defines its own parameters internally
- ✅ Uses NEON_PAT token for data access
- ✅ Includes restR2 fallback logic for data loading
- ✅ Can be run directly with `rmarkdown::render()`

### **How to Use**
```r
# Simply render the Rmd file directly
rmarkdown::render("fsp_qaReview_render_v3.Rmd")
```

## **Common Issues and Solutions**

### **1. Params Object Already Exists Error**
```
Error: params object already exists in knit environment so can't be overwritten by render params
```

**Solution**: ✅ **FIXED**
- Removed `params` definition from YAML header in `fsp_qaReview_render_v3.Rmd`
- The `rmarkdown::render()` function now handles params without conflict

### **2. Object 'params' Not Found Error**
```
Error: ! object 'params' not found
```

**Solution**: ✅ **FIXED**
- Moved title with `params$titleDate` from YAML header to R code chunk
- Split data loading into two chunks: `setup_packages` (no params) and `load_data_with_params` (requires params)
- Added conditional checks for `params` existence in all chunks that use params
- Title now appears in R code chunk AFTER data loading where `params` object definitely exists
- Placed title display before `rm(params)` to ensure params is still available
- All `params` references now wrapped in `if (exists("params"))` checks to handle early processing

### **3. Variable Naming Confusion**
**Issue**: Variables named `fsp1`, `fsp2`, `fsp3` etc. are not descriptive

**Solution**: ✅ **FIXED**
- All variables now have descriptive names:
  - `fsp_bout_completeness` (was `fsp1`)
  - `fsp_within_bout_samples` (was `fsp.2.samples`)
  - `fsp_boutMetadata_within_rec` (was `fsp3.boutMetadata`)
  - etc.

### **4. Spectral Data Processing Placeholder**
**Issue**: Spectral data checks were just placeholders

**Solution**: ✅ **IMPLEMENTED**
- Real spectral data processing now downloads and analyzes CSV files
- Checks for 426 bands, reflectance range (0-1), wavelength range (0-3000)

### **5. Missing CSV Lookup Tables**
**Issue**: LUTs not integrated for completeness checks

**Solution**: ✅ **INTEGRATED**
- `fsp_complete_bout.csv` - for bout completeness expectations
- `fsp_within_bout_nums.csv` - for sample count expectations

## **Testing the Script**

### **Local Testing**
```r
# Run the test script
source("fsp_test_render.R")
```

### **Production Deployment**
```r
# Run the production script for GCS upload
source("fsp_render.R")
```

## **Expected Outputs**

1. **HTML Report**: `fsp_test_report.html` (local) or GCS upload (production)
2. **Summary Tables**: For each check type (completeness, timeliness, plausibility)
3. **Flag Tables**: Records that fail quality checks
4. **Figures**: Visual summaries of data quality

## **File Structure**
```
fsp/
├── fsp_qaReview_render_v3.Rmd     # Main QC script
├── fsp_test_render.R              # Local testing script
├── fsp_render.R                   # Production deployment script
├── fsp_complete_bout.csv          # Bout completeness LUT
├── fsp_within_bout_nums.csv       # Sample count LUT
└── fsp_QAQC_checks.csv           # Requirements checklist
```

## **Support**

If you encounter issues:
1. Check this troubleshooting guide
2. Verify all required packages are installed (`neonOSqc`, `neonUtilities`, etc.)
3. Ensure GitHub PAT is set for LUT access
4. Check that FSP data is available for the specified date range 