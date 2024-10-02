import os

# Specify the directory to scan (current directory)
directory = 'F:/rag_sdk/datasets/cmp_proc_guide/files/md'

# Specify the output file
output_file = 'commands.bat'

# Open the output file in write mode
with open(output_file, 'w') as f_out:
    # Loop over each file in the directory
    for filename in os.listdir(directory):
        # Check if the file has a .md extension
        if filename.lower().endswith('.md'):
            # Generate the pandoc command without the full path
            cmd = f'pandoc "{filename}" --pdf-engine=xelatex -o "{filename[:-3]}.pdf"'
            # Write the command to the output file
            f_out.write(cmd + '\n')

print(f'All commands have been written to {output_file}')


