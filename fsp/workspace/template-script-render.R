##############################################################################################
#' @title HELPER SCRIPT TO RENDER OS QC TEMPLATE REPORT  

#' @author
#' Zachary Nickerson \email{nickerson@battelleecology.org} \cr

#' @return HTML report progromatically pushed to GCS

# changelog and author contributions / copyrights
#   Zachary Nickerson (2024-07-17)
#     original creation
##############################################################################################

# Load libraries
library(neonOSqc)

# Set local filepath to the neonOSqc package
if(file.exists('C:/Users/nickerson')){
gitDir <- "C:/Users/nickerson/Documents/GitHub/neonOSqc/"
}

if(file.exists('/Users/sweintraub')){
gitDir <- "/Users/sweintraub/Documents/GitHub/neonOSqc/"
}

if(file.exists('/Users/kmurphy')){
  gitDir <- "/Users/kmurphy/Documents/Git/neonOSqc/"
}

# Run the render_qaqc_report() function to render the template report to GCS
render_qaqc_report(
  mod = "fsp", # mod should be the abbreviated code for a portal-level data product (e.g., 'asi' for Aquatic Stable Isotopes)
  reportDir = paste0(gitDir,"inst/rmarkdown/templates/neonOSqcReport/skeleton/"), # reportDir should the the direct file path to the directory that contains the QC Rmd script
  rmdName = "skeleton", # The name of the QC .Rmd script
  reportMonth = "2025-04", # Start date for the report formatted as YYYY-MM
  monthlyAnnual = "annual", # either 'monthly' or 'annual' depending on the type of run you want to initiate
  titleDate = "Calendar Year 2025 (with some exceptions)", # This entry is flexible and will be appended to the end of the title of the HTML report
  labData = F, # For use only with scripts that can be run in two modes: field data only, field data + lab data. If this does not fit your script, leave this variable as FALSE
  customParams = NA # For use with any script that takes custom parameters to determine the checks to run. See the helper file for more information by entering ?render_qaqc_report()
)

# Running this function will render the QC template script directly to a GCS bucket:
# https://console.cloud.google.com/storage/browser/neon-nonprod-os-data-quality/os_qc_testing_cert/reports;tab=objects?project=neon-nonprod-storage&prefix=&forceOnObjectsSortingFiltering=true

# Running render_qaqc_report() for your own data product will render your QC Script to the same GCS bucket with the following standardized naming convention
# 'mod_monthly_YYYYMM_REPORTTIMESTAMP.html' <- for a monthly script
# 'mod_yearly_YYYY_REPORTTIMESTAMP.html' <- for an annual script