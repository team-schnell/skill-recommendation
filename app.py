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
jobs = pd.read_csv('archive/jobs_rows.csv')

# Load users data
users = pd.read_csv('archive/user_large.csv')  # Adjust the path if needed

# Streamlit user interface
st.write("Input the skills you know and get job recommendations.")

# Function to get user profile based on input skills
def get_user_profile_from_input(skills_input):
    skills = [skill.strip().lower() for skill in skills_input.split(',')]
    return skills

# Function to compute skill match score
def skill_match(user_skills_list, job_skills):
    user_skills_set = set(user_skills_list)
    job_skills_set = set(skill.strip().lower() for skill in job_skills)
    return len(user_skills_set & job_skills_set)

# Input skills from user
skills_input = st.text_input("Enter your skills (comma-separated):")

if skills_input:
    user_skills_list = get_user_profile_from_input(skills_input)

    if user_skills_list:
        st.write(f"User Skills: {', '.join(user_skills_list)}")

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
            st.write("No suitable jobs found based on the skills provided.")
    else:
        st.write("No skills entered.")
else:
    st.write("Please enter your skills to get job recommendations.")
