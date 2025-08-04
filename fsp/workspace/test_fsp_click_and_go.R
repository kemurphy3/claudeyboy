# Test script to verify FSP click and go functionality
# This script tests if fsp_qaReview_render_v3.Rmd can run independently

cat("Testing FSP click and go script...\n")
cat("This should work without any external dependencies!\n\n")

# Test rendering the FSP script independently
rmarkdown::render(
  "fsp_qaReview_render_v3.Rmd", 
  output_format = "html_document",
  output_file = "fsp_click_and_go_test.html"
)

cat("\n✅ FSP click and go script completed successfully!\n")
cat("Check the output file: fsp_click_and_go_test.html\n")
cat("\nThe script is now fully automated and self-contained!\n") 