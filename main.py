import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors
import pickle

# Load job and user CSV files (use the correct paths to your files)
jobs = pd.read_csv('/Users/amalmr/Desktop/Freelancing AI 2/archive/jobs_rows.csv')
users = pd.read_csv('/Users/amalmr/Desktop/Freelancing AI 2/archive/user_large.csv')

# Check the structure of the data and print column names to inspect
print("Jobs columns:", jobs.columns)  # To check what the column names are
print("Users columns:", users.columns)  # To check user file column names

# Data preprocessing - clean and filter
# Ensure the 'skills' column is a list of skills by splitting it based on commas
jobs['skills'] = jobs['skills'].apply(lambda x: x.split(',') if isinstance(x, str) else [])
users['skills'] = users['skills'].apply(lambda x: x.split(',') if isinstance(x, str) else [])

# Filter users based on specific conditions (e.g., users with at least 5 years of experience)
filtered_users = users[users['experience_years'] >= 5]

# Function to compute skill match score (intersection of job and user skills)
def skill_match(job_skills, user_skills):
    return len(set(job_skills) & set(user_skills))

# Add a new column for skill match score between jobs and filtered users
user_job_match = []
for _, job in jobs.iterrows():
    for _, user in filtered_users.iterrows():
        match_score = skill_match(job['skills'], user['skills'])
        user_job_match.append([job['id'], user['user_id'], match_score])  # Adjust column names based on the actual data

# If 'Job_ID' is not correct, you will need to replace it with the actual job ID column name from the jobs dataset.
# Similarly, if 'user_id' is incorrect, replace it with the correct column name from the users dataset.

# Create a DataFrame for the matches
matches = pd.DataFrame(user_job_match, columns=['id', 'User_ID', 'Skill_Match'])

# Filter for significant matches (e.g., skill match score of at least 2)
matches = matches[matches['Skill_Match'] >= 2]

# Merge job and user data with the match information
merged_data = matches.merge(jobs, left_on='id', right_on='id').merge(users, left_on='User_ID', right_on='user_id')

# Create a pivot table for recommendations
pivot_table = merged_data.pivot_table(columns='User_ID', index='id', values='Skill_Match')
pivot_table.fillna(0, inplace=True)  # Replace NaN values with 0

# Machine learning part: Use NearestNeighbors for clustering and finding similar job recommendations
job_sparse = csr_matrix(pivot_table)

model = NearestNeighbors(algorithm='brute')
model.fit(job_sparse)

# Example: finding the nearest jobs for a specific user (you can adjust the user and job indices here)
distance, suggestion = model.kneighbors(pivot_table.iloc[0, :].values.reshape(1, -1), n_neighbors=6)

# Save the model and data for future use
pickle.dump(model, open('artifacts/skill_match_model.pkl', 'wb'))
pickle.dump(pivot_table.index, open('artifacts/job_ids.pkl', 'wb'))
pickle.dump(matches, open('artifacts/skill_matches.pkl', 'wb'))

print("Recommendation system model created and saved!")
