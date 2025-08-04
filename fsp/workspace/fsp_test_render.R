##############################################################################################
#' @title TEST SCRIPT TO RENDER FSP QC REPORT  

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

# For testing, we can run the Rmd directly with params
# This allows testing without the full render_qaqc_report function

# Set up params for testing
test_params <- list(
  titleDate = "Test Run - 2024",
  startMonth = "2024-01",
  endMonth = "2024-12", 
  monthlyAnnual = "annual",
  labData = FALSE,
  customParams = TRUE,
  reportTimestamp = "20250101000000",
  reportName = "fsp_annual_2024_20250101000000"
)

# Test rendering the FSP script
rmarkdown::render(
  input = paste0(scriptDir, "fsp_qaReview_render_v3.Rmd"),
  params = test_params,
  output_format = "html_document",
  output_file = paste0(scriptDir, "fsp_test_report.html")
)

print("FSP test report rendered successfully!")
print("Check the output file: fsp_test_report.html") 