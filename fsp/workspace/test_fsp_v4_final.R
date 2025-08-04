# Test script for fsp_qaReview_render_v4.Rmd
# This script runs the updated FSP QC report with all fixes applied

# Set working directory (adjust if needed)
# setwd("C:/Users/kmurphy/Documents/Git/os-data-quality-review/fsp")

# Load required packages
library(rmarkdown)
library(neonOSqc)
library(neonUtilities)
library(neonOS)
library(DT)
library(glue)
library(dplyr)
library(tidyr)
library(ggplot2)
library(restR2)

# Check if NEON_PAT is set
if (Sys.getenv('NEON_PAT') == "") {
  stop("NEON_PAT environment variable is not set. Please set it before running this script.")
}

# Set parameters for the report
test_params <- list(
  titleDate = paste("Test Run -", format(Sys.Date(), "%Y")),
  startMonth = "2022-01",
  endMonth = "2022-12",
  monthlyAnnual = "annual",
  labData = FALSE,
  customParams = NULL,
  reportTimestamp = format(Sys.time(), "%Y%m%d%H%M%S"),
  reportName = paste0("fsp_annual_", format(Sys.Date(), "%Y"), "_", format(Sys.time(), "%Y%m%d%H%M%S"))
)

# Create params object in the environment (required for rmarkdown::render)
params <- test_params

# Render the RMarkdown document
cat("Starting FSP QC report generation...\n")
cat("Using fsp_qaReview_render_v4.Rmd\n")
cat("Date range:", test_params$startMonth, "to", test_params$endMonth, "\n")
cat("Report name:", test_params$reportName, "\n\n")

# Render the document
tryCatch({
  render_result <- rmarkdown::render(
    input = "fsp_qaReview_render_v4.Rmd",
    output_format = "html_document",
    params = test_params,
    output_file = paste0("fsp_qaReview_v4_", format(Sys.time(), "%Y%m%d_%H%M%S"), ".html"),
    quiet = FALSE
  )
  
  cat("\n✅ SUCCESS! Report generated successfully.\n")
  cat("Output file:", render_result, "\n")
  
}, error = function(e) {
  cat("\n❌ ERROR: Report generation failed.\n")
  cat("Error message:", e$message, "\n")
  cat("Please check the error details above.\n")
})

cat("\nTest completed.\n") 