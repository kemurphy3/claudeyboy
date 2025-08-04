# FSP QAQC Script Compliance Needs Checklist

**Script:** fsp_qaReview_render_skeleton_compliant_v2.Rmd  
**Purpose:** Gather all required information to bring script into full skeleton compliance  
**Instructions:** Please fill in all required fields. Optional fields may be left blank if not applicable.

---

## 1. SKELETON COMPLIANCE NEEDS

### 1.1 GCS (Google Cloud Storage) Integration
- [ ] **GCS Project Name** *(Required)*
  - Current value needed: ________________________________
  - Example: `"neon-os-data-qa"` or use `neonOSqc:::default_gcs_project_name`
  
- [ ] **GCS Bucket Name** *(Required)*
  - Current value needed: ________________________________
  - Example: `"neon-qa-outputs"` or use `neonOSqc:::default_gcs_bucket_name`
  
- [ ] **GCS File Path Prefix** *(Required)*
  - Current value needed: ________________________________
  - Example: `"qa-reports/fsp/"` or use `neonOSqc:::default_gcs_prefix`

- [ ] **Enable GCS Write?** *(Required)*
  - Choice (Y/N): _______
  - If N, outputs will be prepared but not uploaded

### 1.2 Instructional Text Section
- [ ] **Include "Instructions for use" section?** *(Required)*
  - Choice (Y/N): _______
  - Note: Standard practice is Y for templates, N for production

- [ ] **Custom instructional text** *(Optional)*
  - If yes above, provide text: ________________________________
  - Or use default: "To be removed in production scripts"

### 1.3 Output Tracking Method
- [ ] **Output detection method** *(Required)*
  - Choice: [ ] Dynamic (recommended) [ ] Hardcoded
  - If hardcoded, list all output object names: ________________________________

---

## 2. FUNCTION USAGE NEEDS

### 2.1 Variables File for Duplicate Detection
- [ ] **Path to variables_30012 file** *(Required)*
  - Full path: ________________________________
  - Example: `/path/to/qc_lookup_tables/variables_30012.csv`
  - Alternative: If not available, check here [ ] to use fallback method

- [ ] **Primary keys for duplicate detection** *(Required if no variables file)*
  - fsp_boutMetadata_pub: ________________________________ (default: `eventID`)
  - fsp_sampleMetadata_pub: ________________________________ (default: `sampleID`)
  - fsp_spectralData_pub: ________________________________ (default: `sampleID`)

### 2.2 Within-Record Completeness Implementation
- [ ] **Implementation method for checks 4-6** *(Required)*
  - Choice: [ ] Convert to long format [ ] Use format_custom_outputs [ ] Keep custom
  
- [ ] **Required fields lists** *(Required)*
  - Bout metadata required fields: ________________________________
    - Example: `c("eventID", "siteID", "collectDate", "samplingProtocolVersion")`
  - Sample metadata required fields: ________________________________
    - Example: `c("sampleID", "eventID", "siteID", "collectDate", "sampleType")`
  - Spectral data required fields: ________________________________
    - Example: `c("sampleID", "wavelength", "rawReflectance")`

### 2.3 Band Count Implementation
- [ ] **Band count check method** *(Required)*
  - Choice: [ ] Use complete_within_bout with lookup [ ] Keep custom implementation
  
- [ ] **If using complete_within_bout, provide lookup table values** *(Conditional)*
  - Expected band count per sample: _______ (default: 426)
  - Years applicable: From _______ to _______ (example: 2013 to 2999)
  - Apply to all sites? (Y/N): _______
  - If N, list specific sites: ________________________________

### 2.4 GitHub PAT for Lookup Tables
- [ ] **GitHub Personal Access Token available?** *(Required)*
  - Choice (Y/N): _______
  - If Y, environment variable name: ________________________________ (default: `GITHUB_PAT`)
  - If N, provide alternative lookup table paths:
    - complete_bout lookup: ________________________________
    - complete_within_bout lookup: ________________________________

---

## 3. FORMATTING AND PRESENTATION NEEDS

### 3.1 Tabset Styling
- [ ] **Tabset format preference** *(Required)*
  - Choice: [ ] Simple `{.tabset}` [ ] Enhanced `{.tabset .tabset-fade .tabset-pills}`
  - Apply to all tabsets? (Y/N): _______

### 3.2 Table Captions
- [ ] **Add captions to all DT::datatables?** *(Required)*
  - Choice (Y/N): _______
  - Caption format preference: [ ] Standard glue template [ ] Custom
  - If custom, provide template: ________________________________

### 3.3 Figure Settings
- [ ] **Include figures in output?** *(Required)*
  - complete_bout figure (Y/N): _______
  - timely_bout_duration figure (Y/N): _______
  - Custom check figures (Y/N): _______

### 3.4 CSS Customization
- [ ] **Keep current CSS styling?** *(Required)*
  - Choice (Y/N): _______
  - If N, use: [ ] Skeleton default CSS [ ] No CSS [ ] Custom CSS
  - If custom, provide CSS: ________________________________

---

## 4. TESTING AND VERIFICATION NEEDS

### 4.1 Test Data Availability
- [ ] **Test data location** *(Required)*
  - FSP data path: ________________________________
  - CFC data path (optional): ________________________________
  - Or use: [ ] Portal download [ ] Test server [ ] Local files

### 4.2 Date Range for Testing
- [ ] **Test date range** *(Required)*
  - Start month (YYYY-MM): ________________________________
  - End month (YYYY-MM): ________________________________

### 4.3 Environment Setup
- [ ] **R package versions to use** *(Required)*
  - neonOSqc version: ________________________________ (example: "1.0.0" or "latest")
  - neonOS version: ________________________________
  - neonUtilities version: ________________________________

- [ ] **NEON API token available?** *(Required)*
  - Choice (Y/N): _______
  - If Y, environment variable name: ________________________________ (default: `NEON_PAT`)

### 4.4 Output Verification
- [ ] **Verification requirements** *(Required - check all that apply)*
  - [ ] Verify all 16 checks produce output
  - [ ] Verify HTML renders without errors
  - [ ] Verify GCS upload (if enabled)
  - [ ] Verify output naming conventions
  - [ ] Generate test summary report

### 4.5 Error Handling Preferences
- [ ] **On missing CFC data** *(Required)*
  - Choice: [ ] Skip cross-table check [ ] Error and stop [ ] Warning and continue

- [ ] **On missing lookup tables** *(Required)*
  - Choice: [ ] Use defaults [ ] Error and stop [ ] Warning and continue

- [ ] **On GCS write failure** *(Required)*
  - Choice: [ ] Error and stop [ ] Warning and save locally [ ] Silent continue

---

## 5. ADDITIONAL INFORMATION

### 5.1 Custom Requirements
- [ ] **Any custom QAQC checks to add?** *(Optional)*
  - If yes, describe: ________________________________

- [ ] **Any checks to skip?** *(Optional)*
  - If yes, list check numbers: ________________________________
  - Justification: ________________________________

### 5.2 Output Preferences
- [ ] **Additional output formats needed?** *(Optional)*
  - [ ] CSV exports
  - [ ] PDF report
  - [ ] Excel workbook
  - [ ] JSON for API

- [ ] **Report naming convention** *(Required)*
  - Use default: [ ] Yes [ ] No
  - If no, provide format: ________________________________
  - Example: `"fsp_qa_report_YYYYMMDD_HHMMSS"`

### 5.3 Deployment Information
- [ ] **Deployment environment** *(Required)*
  - Choice: [ ] Local [ ] Server [ ] Cloud [ ] Container
  - Special requirements: ________________________________

- [ ] **Automated execution?** *(Required)*
  - Choice (Y/N): _______
  - If Y, execution schedule: ________________________________

---

## CONFIRMATION

By completing this checklist, I confirm that:
- [ ] All required fields have been filled
- [ ] File paths have been verified as accessible
- [ ] Environment variables are properly set
- [ ] Test data is available for verification

**Completed by:** ________________________________  
**Date:** ________________________________  
**Contact for questions:** ________________________________

---

*Note: Once this checklist is complete, the script can be updated to full compliance without additional clarification.*