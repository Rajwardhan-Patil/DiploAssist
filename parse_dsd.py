import pandas as pd
import fitz  # PyMuPDF
import re
import os
from pathlib import Path

def parse_dsd_pdf(pdf_path):
    """Parse single DSD PDF for college, cutoff, caste, branch data."""
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
    doc.close()
    
    rows = []
    # Common patterns in DSD PDFs (Maharashtra CET): tables like "College Name | Merit No | Perc | Category | Branch"
    # Adjust regex based on actual PDF structure - inspect first!
    
    # Example patterns (customize after inspection):
    # College: uppercase names, e.g. "COLLEGE OF ENGINEERING PUNE"
    # Cutoff: \d+\.\d+ or \d+
    # Castes: OPEN, OBC, SC, ST, VJNT, etc.
    # Branches: CS, IT, EXTC, MECH, CIVIL, etc. (abbrev)
    
    lines = text.split('\n')
    current_college = None
    for line in lines:
        line = line.strip()
        if re.match(r'^[A-Z\s\.\-]{10,}$', line) and len(line) > 20:  # Likely college name
            current_college = line
        elif current_college and re.search(r'\d+\.?\d*.*(OPEN|OBC|SC|ST|VJNT)', line):
            # Parse cutoff, caste, branch - heuristic
            cutoff_match = re.search(r'(\d+\.?\d*)', line)
            caste_match = re.search(r'(OPEN|OBC|SC|ST|VJNT)', line)
            branch_match = re.search(r'(CS|IT|EXTC|EX|ME|MECH|CIVIL|CHEM)', line)
            if cutoff_match and caste_match and branch_match:
                cutoff = float(cutoff_match.group(1))
                caste = caste_match.group(1)
                branch = branch_match.group(1)
                rows.append({'college_name': current_college, 'cutoff_percentage': cutoff, 'caste': caste, 'branch': branch})
    
    return rows

def main():
    pdf_dir = 'data'
    pdf_files = [f for f in os.listdir(pdf_dir) if f.startswith('dsd') and f.endswith('.pdf')]
    all_rows = []
    
    for pdf_file in pdf_files:
        pdf_path = os.path.join(pdf_dir, pdf_file)
        print(f"Parsing {pdf_file}...")
        rows = parse_dsd_pdf(pdf_path)
        all_rows.extend(rows)
        print(f"Extracted {len(rows)} rows from {pdf_file}")
    
    if all_rows:
        df = pd.DataFrame(all_rows)
        output_path = 'data/real_colleges.csv'
        df.to_csv(output_path, index=False)
        print(f"Real dataset saved: {output_path} ({len(df)} rows)")
        print(df.head())
        print("\nUnique colleges:", df['college_name'].nunique())
        print("Unique castes:", df['caste'].unique())
        print("Unique branches:", df['branch'].unique())
    else:
        print("No data extracted. PDFs may have tables/images - inspect manually or use tabula-py.")

if __name__ == '__main__':
    main()

