import zipfile
import os


# Get the uploaded zip filename
zip_files = [f for f in os.listdir('.') if f.endswith('.zip')]
if zip_files:
    dataset_zip = zip_files[0]  # Use first zip file found
    print(f"Using zip file: {dataset_zip}")
else:
    print("No zip file found. Make sure you uploaded a zip file.")
    dataset_zip = "your-dataset.zip"  # fallback

# Step 6: Extract the uploaded zip file
# Extract the zip file
with zipfile.ZipFile(dataset_zip, 'r') as zip_ref:
    zip_ref.extractall('Facial Expressions Dataset/')

print(f"Extracted {dataset_zip} to 'Facial Expressions Dataset/' folder")
