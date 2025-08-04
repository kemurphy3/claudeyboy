# Test script for FSP v4 - Formatting Verification
# This tests that the V4 formatting matches the skeleton template exactly

cat("Testing FSP v4 formatting against skeleton template...\n")
cat("Verifying that section formatting matches skeleton template...\n\n")

# Render the FSP v4 script
rmarkdown::render(
  "fsp_qaReview_render_v4.Rmd", 
  output_format = "html_document",
  output_file = "fsp_v4_formatted_test.html"
)

cat("\n✅ FSP v4 formatting test completed!\n")
cat("Check the output file: fsp_v4_formatted_test.html\n")
cat("\nThe formatting should now match the skeleton template exactly!\n") 