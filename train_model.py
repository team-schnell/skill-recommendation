import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors
import pickle
import os

def train_model():
    print("Starting model training...")

    # Create artifacts directory if it doesn't exist
    os.makedirs('artifacts', exist_ok=True)

    # Load job and user CSV files (adjust paths as needed)
    jobs = pd.read_csv('archive/jobs_rows.csv')
    users = pd.read_csv('archive/user_large.csv')

    print("Jobs columns:", jobs.columns)
    print("Users columns:", users.columns)

    # Data preprocessing
    jobs['skills'] = jobs['skills'].apply(lambda x: x.split(',') if isinstance(x, str) else [])
    users['skills'] = users['skills'].apply(lambda x: x.split(',') if isinstance(x, str) else [])

    # Filter users with at least 5 years of experience
    filtered_users = users[users['experience_years'] >= 5]

    # Function to compute skill match score
    def skill_match(job_skills, user_skills):
        return len(set(job_skills) & set(user_skills))

    # Add a new column for skill match score between jobs and filtered users
    user_job_match = []
    for _, job in jobs.iterrows():
        for _, user in filtered_users.iterrows():
            match_score = skill_match(job['skills'], user['skills'])
            user_job_match.append([job['id'], user['user_id'], match_score])

    # Create a DataFrame for the matches
    matches = pd.DataFrame(user_job_match, columns=['id', 'User_ID', 'Skill_Match'])

    # Filter for significant matches
    matches = matches[matches['Skill_Match'] >= 2]

    # Merge job and user data with the match information
    merged_data = matches.merge(jobs, left_on='id', right_on='id').merge(users, left_on='User_ID', right_on='user_id')

    # Create a pivot table for recommendations
    pivot_table = merged_data.pivot_table(columns='User_ID', index='id', values='Skill_Match')
    pivot_table.fillna(0, inplace=True)

    # Machine learning: Use NearestNeighbors
    job_sparse = csr_matrix(pivot_table)

    model = NearestNeighbors(algorithm='brute')
    model.fit(job_sparse)

    # Save the model and data
    pickle.dump(model, open('artifacts/skill_match_model.pkl', 'wb'))
    pickle.dump(pivot_table.index, open('artifacts/job_ids.pkl', 'wb'))
    pickle.dump(matches, open('artifacts/skill_matches.pkl', 'wb'))

    print("Recommendation system model created and saved!")

if __name__ == "__main__":
    train_model()
