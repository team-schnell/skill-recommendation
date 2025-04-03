from flask import Flask, request, jsonify
import pickle
import pandas as pd
from flask_cors import CORS
import os

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes to allow frontend access

# Global variables to store models and data
model = None
job_ids = None
jobs = None
users = None

# Load models and data
def load_model_and_data():
    global model, job_ids, jobs, users

    # Define relative paths for the model and data files
    model_path = os.path.join('artifacts', 'skill_match_model.pkl')
    job_ids_path = os.path.join('artifacts', 'job_ids.pkl')
    jobs_path = os.path.join('archive', 'jobs_rows.csv')
    users_path = os.path.join('archive', 'user_large.csv')

    try:
        # Load the model and data
        model = pickle.load(open(model_path, 'rb'))
        job_ids = pickle.load(open(job_ids_path, 'rb'))
        jobs = pd.read_csv(jobs_path)
        users = pd.read_csv(users_path)
        print("Models and data loaded successfully")
        return True
    except Exception as e:
        print(f"Error loading models and data: {e}")
        return False

# Function to compute skill match score
def skill_match(user_skills_list, job_skills):
    user_skills_set = set(user_skills_list)
    job_skills_set = set(skill.strip().lower() for skill in job_skills)
    return len(user_skills_set & job_skills_set)

@app.route('/api/recommendations', methods=['POST'])
def get_recommendations():
    global model, job_ids, jobs, users

    # Make sure the models are loaded
    if model is None or job_ids is None or jobs is None or users is None:
        success = load_model_and_data()
        if not success:
            return jsonify({
                'error': 'Failed to load recommendation models',
                'recommendations': []
            }), 500

    data = request.get_json()
    skills_input = data.get('skills', '')

    if not skills_input:
        return jsonify({
            'error': 'No skills provided',
            'recommendations': []
        })

    # Process user skills input
    user_skills_list = [skill.strip().lower() for skill in skills_input.split(',')]

    # Find matching jobs
    job_matches = []
    for _, job in jobs.iterrows():
        job_skills = job['skills'].split(',') if isinstance(job['skills'], str) else []
        match_score = skill_match(user_skills_list, job_skills)

        if match_score >= 1:  # Adjust threshold if needed
            job_matches.append({
                'id': job['id'],  # Don't try to convert to int, keep the original value
                'name': job['name'],
                'skills': job['skills'],
                'match_score': match_score
            })

    # Sort by match score and select top jobs
    job_matches.sort(key=lambda x: x['match_score'], reverse=True)
    top_jobs = job_matches[:5]  # Return top 5 jobs

    return jsonify({
        'user_skills': user_skills_list,
        'recommendations': top_jobs
    })

# Health check endpoint
@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'ok'})

if __name__ == '__main__':
    # Load model and data at startup
    load_model_and_data()

    # Run the Flask app
    app.run(debug=True, host='0.0.0.0', port=5000)
