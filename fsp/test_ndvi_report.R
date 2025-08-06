# Test script for NDVI analysis report
# This script tests the fixed NDVI report with a small date range

# Render the fixed NDVI report
rmarkdown::render(
  input = "fsp_ndvi_analysis_report_fixed.Rmd",
  output_file = "fsp_ndvi_analysis_test_output.html",
  output_dir = ".",
  quiet = FALSE
)

cat("\nReport generated successfully! Check fsp_ndvi_analysis_test_output.html\n")