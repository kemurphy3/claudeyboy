# Simple test to verify params handling

test_params <- list(
  titleDate = "Test Run - 2024",
  startMonth = "2024-01",
  endMonth = "2024-12"
)

# Test rendering the simple script
rmarkdown::render(
  input = "simple_test.Rmd",
  params = test_params,
  output_format = "html_document"
)

print("Simple test completed!") 