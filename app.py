import pickle
import streamlit as st
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors

# Load models and data
st.header('Job Recommender System Using Machine Learning')

# Loading pre-trained models and data
model = pickle.load(open('artifacts/skill_match_model.pkl', 'rb'))
job_ids = pickle.load(open('artifacts/job_ids.pkl', 'rb'))
skill_matches = pickle.load(open('artifacts/skill_matches.pkl', 'rb'))
jobs = pd.read_csv('archive/jobs_rows.csv')

# Load users data
users = pd.read_csv('archive/user_large.csv')  # Adjust the path if needed

# Streamlit user interface
st.write("Job recommendations will be displayed automatically based on the predefined user profile.")

# Function to get user skills and experience based on user ID
def get_user_profile(user_id):
    user_data = users[users['user_id'] == user_id]
    if not user_data.empty:
        skills = user_data.iloc[0]['skills'].split(',')  # Assuming the column is named 'skills'
        experience = user_data.iloc[0]['experience_years']
        return [skill.strip().lower() for skill in skills], experience
    return [], 0

# Use the predefined user ID
predefined_user_id = 'U003'  # Use the actual user ID

# Get user profile
user_skills_list, user_experience = get_user_profile(predefined_user_id)

if user_skills_list:
    st.write(f"User Skills: {', '.join(user_skills_list)}")
    st.write(f"User Experience: {user_experience} years")

    # Function to compute skill match score
    def skill_match(user_skills_list, job_skills):
        user_skills_set = set(skill.strip().lower() for skill in user_skills_list)
        job_skills_set = set(skill.strip().lower() for skill in job_skills)
        return len(user_skills_set & job_skills_set)

    # Prepare job data for recommendations
    job_matches = []
    for _, job in jobs.iterrows():
        job_skills = job['skills'].split(',') if isinstance(job['skills'], str) else []
        match_score = skill_match(user_skills_list, job_skills)
        
        if match_score >= 1:  # Adjust threshold if needed
            job_matches.append({
                'id': job['id'],
                'name': job['name'],
                'skills': job['skills'],
                'match_score': match_score
            })
    
    # Sort by match score in descending order and select top jobs
    job_matches.sort(key=lambda x: x['match_score'], reverse=True)
    top_jobs = job_matches[:5]  # Show top 5 jobs
    
    # Display results
    if top_jobs:
        st.write("Here are some job recommendations for you:")
        for job in top_jobs:
            st.write(f"**{job['name']}**")
            st.write(f"Skills Required: {job['skills']}")
            st.write(f"Match Score: {job['match_score']}")
            st.write("-----------")
    else:
        st.write("No suitable jobs found based on the user profile.")
else:
    st.write("No profile found for the user ID U003.")
