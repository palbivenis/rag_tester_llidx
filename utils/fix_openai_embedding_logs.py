import os

def rename_files_in_directory(directory, old_word, new_word):
    for filename in os.listdir(directory):
        if old_word in filename:
            new_filename = filename.replace(old_word, new_word)
            old_file = os.path.join(directory, filename)
            new_file = os.path.join(directory, new_filename)
            os.rename(old_file, new_file)

# Example usage:
directory = "F:/rag_sdk/evaluations/data/kai_nw_plan"
rename_files_in_directory(directory, "large", "small")

import os
import pandas as pd


# Iterate through all files in the directory
for filename in os.listdir(directory):
    # Check if "small" is in the filename and if the file is an Excel file
    if "small" in filename and filename.endswith('.xlsx'):
        # Get the full path of the file
        file_path = os.path.join(directory, filename)
        
        # Load the Excel file
        excel_file = pd.ExcelFile(file_path)
        
        # Create a dictionary to store the updated data for each sheet
        updated_sheets = {}
        
        # Iterate through each sheet in the Excel file
        for sheet_name in excel_file.sheet_names:
            # Read the sheet into a DataFrame
            df = pd.read_excel(excel_file, sheet_name=sheet_name)
            
            # Replace all occurrences of the word "large" with "small" in the DataFrame
            df = df.applymap(lambda x: x.replace('-large', '-small') if isinstance(x, str) else x)
            
            # Save the updated DataFrame back to the dictionary
            updated_sheets[sheet_name] = df
        
        # Write the updated data back to the same Excel file
        with pd.ExcelWriter(file_path, engine='openpyxl') as writer:
            # Write each updated sheet back to the Excel file
            for sheet_name, df in updated_sheets.items():
                df.to_excel(writer, sheet_name=sheet_name, index=False)

print("All replacements and file updates are complete!")


