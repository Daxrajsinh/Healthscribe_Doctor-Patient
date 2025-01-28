import boto3
import time
import requests
import json
import os
from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv


# Flask app setup
app = Flask(__name__)
CORS(app)
load_dotenv()

# AWS clients
transcribe_medical = boto3.client('transcribe', region_name=os.getenv('AWS_REGION', 'us-east-1'))
s3_client = boto3.client('s3')
brt = boto3.client("bedrock-runtime", region_name=os.getenv('AWS_REGION', 'us-east-1'))

# Global variable to store transcription summary
transcription_summary = None

# Default settings from environment variables
BUCKET_NAME = os.getenv('BUCKET_NAME', 'default-bucket-name')
DATA_ACCESS_ROLE_ARN = os.getenv('DATA_ACCESS_ROLE_ARN', 'default-role-arn')
AUDIO_FILE_URL = os.getenv('AUDIO_FILE_URL', 'default-audio-url')

def start_transcription(job_name, audio_file_uri):
    """
    Ensures only one transcription job runs at a time.
    If an active job exists, it waits for that job to finish before starting a new one.
    """
    # Check for any active transcription jobs
    try:
        existing_jobs = transcribe_medical.list_medical_scribe_jobs(
            Status='IN_PROGRESS',  # Only fetch jobs that are currently in progress
            MaxResults=5
        )
        
        # If any job is in progress, wait for it to complete
        active_jobs = existing_jobs.get('MedicalScribeJobSummaries', [])
        if active_jobs:
            active_job = active_jobs[0]  # Get the first active job (if multiple, wait for any one)
            print(f"An active transcription job is in progress: {active_job['MedicalScribeJobName']}")
            return poll_transcription_job(active_job['MedicalScribeJobName'])

    except Exception as e:
        raise Exception(f"Error checking active transcription jobs: {e}")

    # No active jobs, start a new transcription job
    try:
        transcribe_medical.start_medical_scribe_job(
            MedicalScribeJobName=job_name,
            Media={'MediaFileUri': audio_file_uri},
            OutputBucketName=BUCKET_NAME,
            DataAccessRoleArn=DATA_ACCESS_ROLE_ARN,
            Settings={
                'ShowSpeakerLabels': True,  # Enable speaker partitioning
                'MaxSpeakerLabels': 2      # Set the maximum number of speakers
            }
        )
        print(f"Started a new transcription job: {job_name}")
    except Exception as e:
        raise Exception(f"Error starting transcription job: {e}")

    # Poll the new job until it is completed
    return poll_transcription_job(job_name)


def poll_transcription_job(job_name):
    """
    Polls the transcription job status until it is completed or failed.
    """
    while True:
        try:
            response = transcribe_medical.get_medical_scribe_job(MedicalScribeJobName=job_name)
            print(response)  # Debugging: Check job response

            status = response['MedicalScribeJob']['MedicalScribeJobStatus']
            if status == 'COMPLETED':
                print(f"Job '{job_name}' completed successfully.")
                return response['MedicalScribeJob']['MedicalScribeOutput']
            elif status == 'FAILED':
                raise Exception(f"Job '{job_name}' failed.")
            
            # Wait before polling again
            time.sleep(15)
        except Exception as e:
            raise Exception(f"Error checking job status: {e}")

def convert_json_to_text(transcript_json):
    """
    Converts the transcript JSON into a formatted text for patient-doctor dialogue and symptoms.
    """
    dialogue = []
    symptoms = []
    diagnosed_diseases = []

    # Parse JSON content
    for item in transcript_json.get('Conversation', {}).get('ClinicalInsights', []):
        content = item.get('Spans', [{}])[0].get('Content', '')
        category = item.get('Category', '')
        if category == 'MEDICAL_CONDITION':
            if item.get('Type') == 'DX_NAME':
                diagnosed_diseases.append(content)
            else:
                symptoms.append(content)
        elif category == 'BEHAVIORAL_ENVIRONMENTAL_SOCIAL':
            dialogue.append(f"Patient: {content}")
        elif category == 'ANATOMY':
            dialogue.append(f"Doctor: What about {content}?")

    # Combine results into text format
    formatted_text = "Patient-Doctor Dialogue:\n\n"
    formatted_text += "\n".join(dialogue)
    formatted_text += "\n\nDiagnosed Diseases:\n"
    formatted_text += "\n".join(diagnosed_diseases)
    formatted_text += "\n\nSymptoms:\n"
    formatted_text += "\n".join(symptoms)

    return formatted_text


def save_to_s3(content, file_name, bucket_name):
    """
    Saves the content as a file to the specified S3 bucket.
    """
    try:
        s3_client.put_object(Body=content, Bucket=bucket_name, Key=file_name)
        print(f"File saved to S3: {file_name}")
    except Exception as e:
        raise Exception(f"Error saving file to S3: {e}")

def generate_actual_uri(bucket_name, object_key, region='us-east-1'):
    """
    Generate the actual URI for an S3 object.
    """
    return f"https://{bucket_name}.s3.{region}.amazonaws.com/{object_key}"

def fetch_summary(summary_uri):
    """
    Fetches the summary.json file using the actual S3 URI.
    """
    response = requests.get(summary_uri)
    if response.status_code == 200:
        return response.json()
    else:
        raise Exception(f"Failed to fetch summary.json: {response.status_code}, {response.text}")

def ask_claude(question, summary):
    """
    Queries Claude using the Conversation API.
    """
    conversation = [
        {
            "role": "user",
            "content": [{"text": f"Here is the summary of the medical transcription:\n{json.dumps(summary, indent=2)}\n\nNow, based on this summary, please answer the following question:\n{question}"}]
        }
    ]

    try:
        response = brt.converse(
            modelId=os.getenv('BEDROCK_MODEL_ID', 'anthropic.claude-3-haiku-20240307-v1:0'),
            messages=conversation,
            inferenceConfig={
                "maxTokens": 512,
                "temperature": 0.5,
                "topP": 0.9
            }
        )
        response_text = response["output"]["message"]["content"][0]["text"]
        return response_text
    except Exception as e:
        raise Exception(f"Error querying Claude: {e}")

@app.route('/question-ans', methods=['POST'])
def question_answer():
    """
    Flask API endpoint for question answering.
    """
    global transcription_summary
    if not transcription_summary:
        return jsonify({"error": "Transcription summary not available. Complete transcription first."}), 400

    data = request.json
    question = data.get('question')
    if not question:
        return jsonify({"error": "No question provided."}), 400

    try:
        answer = ask_claude(question, transcription_summary)
        return jsonify({"question": question, "answer": answer})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

def main():
    """
    Main function to process the transcription job.
    """
    global transcription_summary

    # Create a unique job name based on timestamp
    job_name = f"medical_transcription_job_{int(time.time())}"
    audio_file_uri = AUDIO_FILE_URL

    try:
        medical_scribe_output = start_transcription(job_name, audio_file_uri)
        summary_uri = medical_scribe_output['ClinicalDocumentUri']
        print(f"Summary URI: {summary_uri}")

        transcription_summary = fetch_summary(summary_uri)
        print("Transcription summary fetched successfully.")
        print(json.dumps(transcription_summary, indent=2))

        app.run(debug=True, port=5000)
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
