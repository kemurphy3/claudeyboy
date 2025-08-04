# Test script for FSP v4 - FSP/CFC Only
# This tests the cleaned version with only FSP and CFC relevant checks

cat("Testing FSP v4 script (FSP/CFC only)...\n")
cat("This version contains only FSP and CFC relevant checks from fsp_QAQC_checks.csv\n\n")

# Render the FSP v4 script
rmarkdown::render(
  "fsp_qaReview_render_v4.Rmd", 
  output_format = "html_document",
  output_file = "fsp_v4_test.html"
)

cat("\n✅ FSP v4 script completed successfully!\n")
cat("Check the output file: fsp_v4_test.html\n")
cat("\nThis version contains only FSP and CFC relevant checks!\n") 