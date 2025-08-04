##############################################################################################
#' @title ALTERNATIVE TEST SCRIPT TO RENDER FSP QC REPORT  

#' @author
#' Kate Murphy \email{kmurphy@battelleecology.org} \cr

#' @return HTML report for testing FSP QC script

# changelog and author contributions / copyrights
#   Kate Murphy (2025-01-XX)
#     original creation for FSP testing
##############################################################################################

# Load libraries
library(neonOSqc)

# Set local filepath to the FSP script directory
scriptDir <- "C:/Users/kmurphy/Documents/Git/os-data-quality-review/fsp/"

# Alternative approach: Create params in the environment first
params <- list(
  titleDate = "Test Run - 2022",
  startMonth = "2022-01",
  endMonth = "2022-12", 
  monthlyAnnual = "annual",
  labData = FALSE,
  customParams = TRUE,
  reportTimestamp = "20250101000000",
  reportName = "fsp_annual_2022_20250101000000"
)

# Test rendering the FSP script with params already in environment
rmarkdown::render(
  input = paste0(scriptDir, "fsp_qaReview_render_v3.Rmd"),
  output_format = "html_document",
  output_file = paste0(scriptDir, "fsp_test_report_v2.html")
)

print("FSP test report v2 rendered successfully!")
print("Check the output file: fsp_test_report_v2.html") 