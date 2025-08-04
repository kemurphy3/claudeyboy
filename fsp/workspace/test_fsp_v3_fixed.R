# Test script to verify the fixed FSP script
# This tests if the data loading and variable definitions work correctly

cat("Testing fixed FSP script...\n")
cat("Checking if fsp_boutMetadata is properly defined...\n\n")

# Render the fixed FSP script
rmarkdown::render(
  "fsp_qaReview_render_v3.Rmd", 
  output_format = "html_document",
  output_file = "fsp_v3_fixed_test.html"
)

cat("\n✅ FSP script test completed!\n")
cat("Check the output file: fsp_v3_fixed_test.html\n") 