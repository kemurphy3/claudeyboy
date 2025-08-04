# Click and Go Script for fsp_qaReview_render_v4.Rmd
# This script runs the FSP QC report with minimal setup required

# Load required packages
library(rmarkdown)

# Check if NEON_PAT is set
if (Sys.getenv('NEON_PAT') == "") {
  cat("❌ ERROR: NEON_PAT environment variable is not set.\n")
  cat("Please set your NEON Personal Access Token before running this script.\n")
  cat("You can set it in R using: Sys.setenv(NEON_PAT = 'your_token_here')\n")
  stop("NEON_PAT not found")
}

# No external files required - everything is self-contained

cat("🚀 Starting FSP QC Report Generation...\n")
cat("📁 Using: fsp_qaReview_render_v4.Rmd\n")
cat("📅 Date range: 2022-01 to 2022-12\n")
cat("🔑 NEON_PAT: Found ✓\n")
cat("🔧 Primary keys: Self-contained in script ✓\n\n")

# Render the document
tryCatch({
  render_result <- rmarkdown::render(
    input = "fsp_qaReview_render_v4.Rmd",
    output_format = "html_document",
    output_file = paste0("fsp_qaReview_v4_", format(Sys.time(), "%Y%m%d_%H%M%S"), ".html"),
    quiet = FALSE
  )
  
  cat("\n✅ SUCCESS! FSP QC Report generated successfully.\n")
  cat("📄 Output file:", render_result, "\n")
  cat("🌐 Open the HTML file in your browser to view the report.\n")
  
}, error = function(e) {
  cat("\n❌ ERROR: Report generation failed.\n")
  cat("Error message:", e$message, "\n")
  cat("Please check the error details above.\n")
})

cat("\n🏁 Script completed.\n") 