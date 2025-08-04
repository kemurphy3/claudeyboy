# FSP QAQC Script Compliance Needs Checklist - STREAMLINED

**Script:** fsp_qaReview_render_skeleton_compliant_v2.Rmd  
**Purpose:** Gather ONLY the missing information needed for full skeleton compliance  
**Note:** This streamlined version excludes information already defined in the script

---

## 1. SKELETON COMPLIANCE NEEDS

### 1.1 GCS Integration *(All Required)*
- [ ] **Enable GCS Write?** 
  - Choice (Y/N): _______
  - If Y, use default GCS settings from neonOSqc package? (Y/N): _______
  - If using custom GCS settings, provide:
    - Project: ________________________________
    - Bucket: ________________________________
    - Path prefix: ________________________________

### 1.2 Instructional Section
- [ ] **Include "Instructions for use" section?**
  - Choice (Y/N): _______ (typically N for production)

---

## 2. FUNCTION USAGE NEEDS

### 2.1 Variables File
- [ ] **Path to variables_30012 file**
  - Full path: ________________________________
  - Or check here [ ] if not available (script will use existing fallback)

### 2.2 Within-Record Completeness
- [ ] **Implementation preference for within-record checks**
  - Choice: [ ] Keep current custom implementation (recommended)
          [ ] Convert to use complete_within_rec() function
  - *Note: Current implementation already checks the correct required fields*

### 2.3 Band Count Check
- [ ] **Implementation preference for band count check**
  - Choice: [ ] Keep current custom implementation (recommended)
          [ ] Create lookup table for complete_within_bout()
  - *Note: Current implementation correctly checks for 426 bands*

### 2.4 GitHub PAT
- [ ] **Is GITHUB_PAT environment variable set?**
  - Choice (Y/N): _______
  - *Required for automatic lookup table loading*

---

## 3. FORMATTING PREFERENCES

### 3.1 Tabsets
- [ ] **Standardize tabset format?**
  - Current: `{.tabset .tabset-fade .tabset-pills}`
  - Change to simple `{.tabset}`? (Y/N): _______

### 3.2 Table Captions
- [ ] **Add standard captions to all tables?**
  - Choice (Y/N): _______

### 3.3 CSS Styling
- [ ] **Keep current CSS?**
  - Choice (Y/N): _______ (current CSS positions TOC and widens page)

---

## 4. ENVIRONMENT SETUP

### 4.1 Package Availability
- [ ] **Confirm package availability:**
  - [ ] neonOSqc package is installed
  - [ ] neonOS package is installed (for removeDups)
  - [ ] neonUtilities package is installed

### 4.2 API Access
- [ ] **NEON API Token (NEON_PAT) set?**
  - Choice (Y/N): _______
  - *Required for data downloads*

### 4.3 Testing Requirements
- [ ] **Test with real FSP/CFC data?**
  - Choice (Y/N): _______
  - If Y, date range: _______ to _______
  - If N, will use example/mock data

---

## 5. OUTPUT PREFERENCES

### 5.1 Error Handling
- [ ] **If CFC data is unavailable:**
  - Choice: [X] Skip cross-table check gracefully (current behavior)
          [ ] Error and stop

### 5.2 Report Output
- [ ] **Additional outputs beyond HTML report?**
  - [ ] Save intermediate CSVs
  - [ ] Save plots as separate files
  - [ ] None (HTML only)

### 5.3 Final Script Version
- [ ] **Script naming:**
  - [ ] Keep current name (fsp_qaReview_render_skeleton_compliant_v2.Rmd)
  - [ ] Rename to: ________________________________

---

## QUICK REFERENCE - Already Defined in Script

The following are already correctly implemented and need NO changes:

✅ **Primary Keys:**
- boutMetadata: `eventID`
- sampleMetadata: `sampleID`  
- spectralData: `sampleID`

✅ **Required Fields:**
- boutMetadata: `eventID`, `siteID`, `collectDate`
- sampleMetadata: `sampleID`, `eventID`, `siteID`, `collectDate`
- spectralData: `sampleID`, `wavelength`, `rawReflectance`

✅ **Check Parameters:**
- Band count: 426 bands expected
- Bout duration: 30 days maximum
- Reflectance range: 0-1
- Wavelength range: 300-2600 nm (though CSV says 0-3000)
- Spectral ratio: 1000nm > 500nm

✅ **Data Product IDs:**
- FSP: DP1.30012.001
- CFC: DP1.10026.001

---

**Total items to complete: 13** (compared to 40+ in original checklist)

**Completed by:** ________________________________  
**Date:** ________________________________