# FSP QAQC Script - Required Fixes for Full Compliance

## Priority 1: Critical Skeleton Compliance Issues

### 1. Complete the Outputs Section
**Current State:** Missing GCS integration code  
**Required Fix:**
```r
## Outputs

# Storage location information
cat("GCS project:", neonOSqc:::default_gcs_project_name, "\n")
cat("GCS bucket:", neonOSqc:::default_gcs_bucket_name, "\n")
cat("Filepath:", neonOSqc:::default_gcs_prefix, "\n")

# Get all lists from environment
listOuts <- ls(pattern = "^fsp\\.")
listOuts <- listOuts[sapply(listOuts, function(x) is.list(get(x)))]

# Unpack lists to create uniquely named dataframes
for(i in listOuts){
  list_to_unpack <- get(i)
  for(j in names(list_to_unpack)){
    df_name <- paste0(i, "_", j)
    assign(df_name, list_to_unpack[[j]])
  }
}

# Get all dataframes
dfOuts <- ls(pattern = "^fsp")
dfOuts <- dfOuts[sapply(dfOuts, function(x) is.data.frame(get(x)))]
dfOuts <- dfOuts[!grepl("_pub$|_noDups$", dfOuts)]  # Exclude input data

# Create mega-list for export
all_outputs <- mget(c(listOuts, dfOuts))

# Remove empty tables
all_outputs <- all_outputs[sapply(all_outputs, function(x) nrow(x) > 0 || !is.data.frame(x))]

# Write to GCS
neonOSqc::write_output_gcs(all_outputs, reportName)
```

### 2. Fix Output Tracking Lists
**Current State:** Incomplete tracking of created objects  
**Required Fix:**
- Remove hardcoded `listOuts` and `dfOuts` assignments
- Use dynamic pattern matching as shown above
- Ensure all custom check results follow naming conventions

## Priority 2: Function Usage Compliance

### 1. Within-Record Completeness Checks (Checks 4-6)
**Current State:** Custom implementation  
**Options for Fix:**

**Option A - Convert to Long Format:**
```r
# Convert FSP tables to long format for complete_within_rec
fsp_boutMetadata_long <- fsp_boutMetadata_noDups %>%
  pivot_longer(cols = all_of(required_fields_bout),
               names_to = "measure_var",
               values_to = "measure_value") %>%
  mutate(measure_value = as.character(measure_value))

fsp.boutMetadata.within.rec <- complete_within_rec(
  input.df = fsp_boutMetadata_long,
  num.expected.records.per.sample = length(required_fields_bout),
  measure.var = "measure_var",
  measure.value.var = "measure_value",
  id.var = "eventID",
  site.var = "siteID",
  date.var = "collectDate"
)
```

**Option B - Use format_custom_outputs:**
```r
# Standardize custom check output
fsp.boutMetadata.within.rec <- format_custom_outputs(
  all_data = fsp_boutMetadata_noDups,
  flagged_data = fsp.boutMetadata.missing,
  function_name = "within_record_completeness"
)
```

### 2. Band Count Check (Check 7)
**Current State:** Custom implementation  
**Required Fix:**
1. Create lookup table entry in `fsp_within_bout_nums.csv`:
   ```csv
   eventID,mod,scale,siteID,minExpectedRecords,yearFirstApplicable,yearLastApplicable,tableName
   all,fsp,bands per sample,all,426,2013,2999,fsp_spectralData_pub
   ```

2. Use complete_within_bout:
   ```r
   fsp.band.count.completeness <- complete_within_bout(
     input.df = fsp_spectralData_noDups,
     mod = "fsp",
     event.var = "sampleID",
     table.name = "fsp_spectralData_pub",
     type = "record"
   )
   ```

## Priority 3: Minor Formatting Issues

### 1. Standardize Tabset Formatting
**Current:** `{.tabset .tabset-fade .tabset-pills}`  
**Fix:** Change all to `{.tabset}`

### 2. Add Missing Instructional Section
Add after CSS chunk:
```markdown
::: {style="color: SlateBlue"}

## Instructions for use

[This section to be deleted in production scripts]

:::
```

### 3. Standardize DT::datatable Calls
Add captions to all tables:
```r
DT::datatable(fsp.bout.completeness$complete_bout_summary,
              extensions = "Buttons",
              caption = glue::glue("Summary table of 'complete_bout' results for 'fsp_boutMetadata_pub', records created between {startDate_checked} and {endDate_checked}."),
              options = dtOptions)
```

## Implementation Checklist

- [ ] Fix Outputs section with full GCS integration
- [ ] Replace hardcoded listOuts/dfOuts with dynamic detection
- [ ] Choose approach for within-record completeness (long format or format_custom_outputs)
- [ ] Create lookup table for band count check
- [ ] Implement complete_within_bout for band counts
- [ ] Standardize all tabset formatting
- [ ] Add missing instructional section
- [ ] Add captions to all DT::datatable calls
- [ ] Test with actual neonOSqc package and variables file
- [ ] Verify GCS write functionality

## Testing Requirements

1. **With Full neonOSqc Environment:**
   - Confirm all neonOSqc functions work as expected
   - Verify lookup tables load from GitHub
   - Test removeDups with proper variables file

2. **Output Verification:**
   - All outputs follow _all, _flags, _summary pattern
   - All outputs are captured in final GCS write
   - No input data (_pub, _noDups) included in outputs

3. **Report Generation:**
   - HTML renders without errors
   - All tabsets display correctly
   - Summary tables show accurate counts