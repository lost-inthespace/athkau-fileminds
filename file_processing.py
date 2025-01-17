import os
import json
import uuid
from datetime import datetime
from PyPDF2 import PdfReader
import docx
import openpyxl
import pdfplumber
import requests
from dotenv import load_dotenv

from azure.cognitiveservices.vision.computervision import ComputerVisionClient
from azure.cognitiveservices.vision.computervision.models import VisualFeatureTypes
from msrest.authentication import CognitiveServicesCredentials
import azure.cognitiveservices.speech as speechsdk

load_dotenv()
cog_key = os.getenv("cog_key")
cog_endpoint = os.getenv("cog_endpoint")

vision_client = ComputerVisionClient(cog_endpoint, CognitiveServicesCredentials(cog_key))

speech_config = speechsdk.SpeechConfig(subscription=cog_key, region="eastus")

def extract_image_description(file_path):
    """Analyze an image using Azure Computer Vision and return a descriptive string."""

    if not os.path.exists(file_path):
        return "Image file not found."

    with open(file_path, "rb") as image_file:
        # Specify the features you want (description, tags, you can add more if desired)
        features = [VisualFeatureTypes.description, VisualFeatureTypes.tags]

        # Analyze the image
        analysis = vision_client.analyze_image_in_stream(image_file, visual_features=features)

    description_parts = []

    # Use captions from description
    if analysis.description and analysis.description.captions:
        # Captions are generally the best textual description of the image
        for caption in analysis.description.captions:
            description_parts.append(f"{caption.text} (conf: {caption.confidence:.2f})")

    # Add tags
    if analysis.tags:
        tag_names = [f"{tag.name} (conf: {tag.confidence:.2f})" for tag in analysis.tags]
        description_parts.append("Tags: " + ", ".join(tag_names))

    # Combine everything into a single descriptive text
    if description_parts:
        return " | ".join(description_parts)
    else:
        return "No detailed description available."

def extract_audio_transcript(file_path):
    """
    Transcribe audio from a file using Azure Speech-to-Text.
    Relies on SPEECH_KEY and SPEECH_REGION environment variables.
    """
    # speech_config.speech_recognition_language = "en-US"

    # Configure audio from file
    audio_config = speechsdk.audio.AudioConfig(filename=file_path)
    speech_recognizer = speechsdk.SpeechRecognizer(speech_config=speech_config, audio_config=audio_config)

    # Perform recognition once
    speech_recognition_result = speech_recognizer.recognize_once_async().get()

    # Handle different results
    if speech_recognition_result.reason == speechsdk.ResultReason.RecognizedSpeech:
        # Successfully recognized speech
        return speech_recognition_result.text
    elif speech_recognition_result.reason == speechsdk.ResultReason.NoMatch:
        return "No speech could be recognized."
    elif speech_recognition_result.reason == speechsdk.ResultReason.Canceled:
        cancellation_details = speech_recognition_result.cancellation_details
        error_message = "Speech Recognition canceled: {}".format(cancellation_details.reason)
        if cancellation_details.reason == speechsdk.CancellationReason.Error:
            error_message += f" Error details: {cancellation_details.error_details}"
            error_message += " Check your speech resource key and region."
        return error_message

    # If some unexpected reason occurs
    return "Audio transcription failed for an unknown reason."

def get_file_metadata(file_path):
    stats = os.stat(file_path)
    creation_time = datetime.fromtimestamp(stats.st_ctime).isoformat()
    modification_time = datetime.fromtimestamp(stats.st_mtime).isoformat()
    file_type = os.path.splitext(file_path)[1].lower()
    return creation_time, modification_time, file_type

def clean_duplicates_in_json(json_path='documents.json'):
    """Remove duplicates from documents.json, keeping only the newest version based on modification_time."""
    if not os.path.exists(json_path) or os.path.getsize(json_path) == 0:
        return
    with open(json_path, 'r', encoding='utf-8') as f:
        try:
            data = json.load(f)
        except json.JSONDecodeError:
            return

    # Group by file_path
    docs_by_path = {}
    for doc in data:
        path = doc['file_path']
        # If there's a doc with same path, keep the one with newer modification_time
        if path not in docs_by_path:
            docs_by_path[path] = doc
        else:
            # Compare modification_time
            existing_doc = docs_by_path[path]
            if doc['modification_time'] > existing_doc['modification_time']:
                docs_by_path[path] = doc

    cleaned_data = list(docs_by_path.values())
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(cleaned_data, f, ensure_ascii=False, indent=4)


def process_files(file_paths, json_path='documents.json', progress_callback=None):
    # Load existing data if the JSON file exists
    if os.path.exists(json_path) and os.path.getsize(json_path) > 0:
        with open(json_path, 'r', encoding='utf-8') as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError:
                data = []
    else:
        data = []

    # Create a mapping of file paths to document data
    existing_docs = {doc['file_path']: doc for doc in data}

    # Prepare a new dictionary to hold updated documents
    updated_docs = existing_docs.copy()

    total_files = len(file_paths)
    processed_files = 0

    # Remove entries for files that no longer exist on disk
    for file_path in list(existing_docs.keys()):
        if not os.path.exists(file_path):
            del existing_docs[file_path]

    # Extract content for different file types
    def extract_content(file_path):
        extension = os.path.splitext(file_path)[1].lower()
        content = ''
        try:
            if extension == '.txt':
                with open(file_path, 'r', encoding='utf-8') as file:
                    content = file.read()
            elif extension == '.pdf':
                with pdfplumber.open(file_path) as pdf:
                    content = '\n'.join(page.extract_text() for page in pdf.pages if page.extract_text())
            elif extension == '.docx':
                doc_obj = docx.Document(file_path)
                content = '\n'.join(para.text for para in doc_obj.paragraphs)
            elif extension == '.xlsx':
                wb = openpyxl.load_workbook(file_path)
                for sheet in wb.worksheets:
                    for row in sheet.iter_rows(values_only=True):
                        content += ' '.join(str(cell) for cell in row if cell) + '\n'
            elif extension in ['.jpg', '.jpeg', '.png']:
                content = extract_image_description(file_path)
            elif extension in ['.mp3', '.wav', '.m4a']:
                content = extract_audio_transcript(file_path)
            elif extension in ['.mp4', '.avi', '.mkv', '.mov']:
                content = "This feature (video analysis) is still in development."
            else:
                content = ''
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
        return content

    # Process each file
    for i, file_path in enumerate(file_paths):
        creation_time, modification_time, file_type = get_file_metadata(file_path)
        file_modified = False
        if file_path in existing_docs:
            stored_doc = existing_docs[file_path]
            if stored_doc['modification_time'] == modification_time:
                # No changes, skip re-processing
                continue
            else:
                file_modified = True
        else:
            file_modified = True

        if file_modified:
            file_content = extract_content(file_path)
            file_id = existing_docs[file_path]['id'] if file_path in existing_docs else str(uuid.uuid4())
            file_data = {
                'id': file_id,
                'file_path': file_path,
                'file_type': file_type,
                'creation_time': creation_time,
                'modification_time': modification_time,
                'content': file_content
            }
            updated_docs[file_path] = file_data
            processed_files += 1

        if progress_callback:
            progress_percent = int(((i + 1) / total_files) * 100)
            progress_callback(progress_percent)

    # Convert updated_docs back to a list
    updated_data = list(updated_docs.values())

    # Save updated data to JSON file
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(updated_data, f, ensure_ascii=False, indent=4)

    # Clean duplicates after updating
    clean_duplicates_in_json(json_path=json_path)

    return updated_data