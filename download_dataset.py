import kagglehub

# Download latest version
path = kagglehub.dataset_download("washingtongold/exdark-dataset")

print("Path to dataset files:", path)