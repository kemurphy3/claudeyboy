# Manual script to run fsp_qaReview_render_v3.Rmd
# This uses the working approach: create params in environment first

# Set up params in environment first (same as working fsp_test_render_v2.R)
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

# Print params to verify
cat("Using params:\n")
print(params)
cat("\n")

# Render the FSP QC report (same approach as working fsp_test_render_v2.R)
cat("Rendering fsp_qaReview_render_v3.Rmd...\n")
rmarkdown::render(
  "fsp_qaReview_render_v3.Rmd", 
  output_format = "html_document",
  output_file = "fsp_manual_test_report.html"
)

cat("\n✅ FSP QC report rendered successfully!\n")
cat("Check the output file: fsp_manual_test_report.html\n") 