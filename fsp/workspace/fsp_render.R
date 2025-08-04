##############################################################################################
#' @title HELPER SCRIPT TO RENDER FSP QC REPORT  

#' @author
#' Kate Murphy \email{kmurphy@battelleecology.org} \cr

#' @return HTML report programmatically pushed to GCS

# changelog and author contributions / copyrights
#   Kate Murphy (2025-01-XX)
#     original creation for FSP
##############################################################################################

# Load libraries
library(neonOSqc)

# Set local filepath to the FSP script directory
scriptDir <- "C:/Users/kmurphy/Documents/Git/os-data-quality-review/fsp/"

# Run the render_qaqc_report() function to render the FSP report to GCS
render_qaqc_report(
  mod = "fsp", # FSP module code
  reportDir = scriptDir, # directory containing the FSP QC Rmd script
  rmdName = "fsp_qaReview_render_v3", # The name of the FSP QC .Rmd script (without extension)
  reportMonth = "2025-01", # Start date for the report formatted as YYYY-MM
  monthlyAnnual = "monthly", # either 'monthly' or 'annual' depending on the type of run
  titleDate = "January 2025", # This entry is flexible and will be appended to the end of the title of the HTML report
  labData = FALSE, # FSP doesn't have lab data, so this is FALSE
  customParams = NA # No custom parameters needed for FSP
)

# Running this function will render the FSP QC script directly to a GCS bucket:
# https://console.cloud.google.com/storage/browser/neon-nonprod-os-data-quality/os_qc_testing_cert/reports;tab=objects?project=neon-nonprod-storage&prefix=&forceOnObjectsSortingFiltering=true

# The rendered report will be named:
# 'fsp_monthly_202501_REPORTTIMESTAMP.html' <- for a monthly script
# 'fsp_yearly_2025_REPORTTIMESTAMP.html' <- for an annual script 