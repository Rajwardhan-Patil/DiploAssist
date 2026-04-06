"""
Synthetic DSE-style rows: (college, cutoff%, caste, branch, gender, quota).
Each college has a notional base merit; adjustments match Maharashtra-style patterns.
"""
import os

import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_CSV = os.path.join(BASE_DIR, 'data', 'huge_colleges.csv')

# (display_name, base_cutoff) — higher base ≈ tougher admission; spread across ~65–93
COLLEGE_BASES = [
    ('College of Engineering Pune (COEP)', 92.5),
    ('Veermata Jijabai Technological Institute (VJTI)', 91.0),
    ('Sardar Patel Institute of Technology (SPIT)', 89.5),
    ('Vishwakarma Institute of Technology (VIT)', 88.5),
    ('Pune Institute of Computer Technology (PICT)', 87.8),
    ('Army Institute of Technology (AIT)', 87.2),
    ('DJ Sanghvi College of Engineering', 86.5),
    ('K J Somaiya College of Engineering', 86.0),
    ('Fr Conceicao Rodrigues Institute of Technology (FrCRIT)', 85.5),
    ('Thakur College of Engineering and Technology', 85.0),
    ('Vivekanand Education Society Institute of Technology (VESIT)', 84.6),
    ('Dwarkadas J Sanghvi College of Engineering', 84.2),
    ('Shah and Anchor Kutchhi Engineering College', 83.8),
    ('Lokmanya Tilak College of Engineering', 83.4),
    ('Agnel Technical College', 83.0),
    ('Xavier Institute of Engineering', 82.6),
    ('St Francis Institute of Technology', 82.2),
    ('Rizvi College of Engineering', 81.8),
    ('Vasantdada Patil College of Engineering', 81.4),
    ('Terna Engineering College', 81.0),
    ('AISSMS College of Engineering Pune', 80.6),
    ('JSPM Bhivarabai Sawant College of Engineering', 80.2),
    ('Trinity Academy of Engineering Pune', 79.8),
    ('MIT World Peace University', 79.4),
    ('Dr D Y Patil Institute of Technology', 79.0),
    ('Sinhgad College of Engineering', 78.6),
    ('Cummins College of Engineering for Women', 78.2),
    ('Bharati Vidyapeeth College of Engineering', 77.8),
    ('DY Patil College of Engineering Akurdi', 77.4),
    ('Zeal College of Engineering and Research', 77.0),
    ('PCCoER Pimpri Chinchwad College of Engineering and Research', 76.6),
    ('Sanjay Bhole College of Engineering', 76.2),
    ('Walchand College of Engineering Sangli', 75.8),
    ('Government College of Engineering Karad', 75.4),
    ('KIT College of Engineering Kolhapur', 75.0),
    ('KIT Kolhapur', 74.6),
    ('Government College of Engineering Aurangabad', 74.2),
    ('Jawaharlal Nehru College of Engineering Aurangabad', 73.8),
    ('Shri Ramdeobaba College of Engineering and Management Nagpur', 73.4),
    ('Yeshwantrao Chavan College of Engineering Nagpur', 72.8),
    ('GH Raisoni College of Engineering Nagpur', 72.4),
    ('Priyadarshini Institute of Engineering and Technology Nagpur', 72.0),
    ('Gokhale Education Societys College of Engineering', 71.6),
    ('Maharashtra Institute of Technology (MIT)', 71.2),
    ('MGM College of Engineering Nanded', 70.8),
    ('MGM Institute of Technology Mumbai', 70.4),
    ('ISBM College of Engineering Pune', 70.0),
    ('RSCOE Pune', 69.5),
    ('AISSMS Institute of Information Technology', 69.0),
    ('RAIT Navi Mumbai', 68.5),
    ('ACE College of Engineering', 68.0),
    ('SFIS Nagpur', 67.5),
    ('Tulsiramji Gaikwad Patil College of Engineering', 67.0),
    ('Dr Babasaheb Ambedkar Technological University (DBATU) COE', 66.5),
    ('Pimpri Chinchwad University COE', 66.0),
    ('GH Raisoni Institute of Engineering and Technology Pune', 65.5),
    ('Symbiosis Institute of Technology', 88.0),
    ('NMIMS Mukesh Patel School of Technology', 87.0),
]

castes = ['OPEN', 'OBC', 'SC', 'ST', 'VJNT']
branches = ['CS', 'IT', 'EXTC', 'ME', 'CIVIL']
caste_adj = {'OPEN': 0, 'OBC': -4.5, 'SC': -12, 'ST': -20, 'VJNT': -7}
branch_adj = {'CS': 0.5, 'IT': 0, 'EXTC': -0.5, 'ME': -1, 'CIVIL': -2}

genders = ['M', 'F']
gender_adj = {'M': 0, 'F': -0.5}

quotas = ['MS', 'AI']
quota_adj = {'MS': 0, 'AI': 1.5}

rows = []
# 10 replications × N colleges × 5 × 5 × 2 × 2
REPS = 10
for rep in range(REPS):
    for college_name, base_cutoff in COLLEGE_BASES:
        base = base_cutoff + np.random.uniform(-1.0, 1.0)
        for caste in castes:
            adj_c = caste_adj[caste]
            for br in branches:
                adj_b = branch_adj[br]
                for gender in genders:
                    adj_g = gender_adj[gender]
                    for quota in quotas:
                        adj_q = quota_adj[quota]
                        noise = np.random.normal(0, 1.5)
                        perc = max(50.0, base + adj_c + adj_b + adj_g + adj_q + noise)
                        rows.append([college_name, round(perc, 1), caste, br, gender, quota])

df = pd.DataFrame(rows, columns=['college_name', 'cutoff_percentage', 'caste', 'branch', 'gender', 'quota'])
df.to_csv(OUT_CSV, index=False)
print(f'Generated {len(df)} rows, {df["college_name"].nunique()} colleges -> {OUT_CSV}')
