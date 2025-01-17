import os
from azure.cognitiveservices.vision.computervision import ComputerVisionClient
from azure.cognitiveservices.vision.computervision.models import VisualFeatureTypes
from msrest.authentication import CognitiveServicesCredentials

# Azure credentials
endpoint = "https://athakucog.cognitiveservices.azure.com/"
key = "3spGeY9CEQI7Cqa95lnBUmkHbpy53N0uO4GnNh47gugFfBcqpp1aJQQJ99ALACYeBjFXJ3w3AAAEACOGVu3o"

# Create a client
client = ComputerVisionClient(endpoint, CognitiveServicesCredentials(key))


def analyze_uploaded_image(image_path):
    """Analyze the uploaded image."""
    # Open the image file
    with open(image_path, "rb") as image_file:
        # Specify the features you want
        features = [VisualFeatureTypes.description, VisualFeatureTypes.tags]

        # Analyze the image
        analysis = client.analyze_image_in_stream(image_file, visual_features=features)

        # Print descriptions
        if analysis.description and analysis.description.captions:
            print("Description:")
            for caption in analysis.description.captions:
                print(f" - {caption.text} (confidence: {caption.confidence})")

        # Print tags
        if analysis.tags:
            print("\nTags:")
            for tag in analysis.tags:
                print(f" - {tag.name} (confidence: {tag.confidence})")


if __name__ == "__main__":
    # Prompt the user to upload an image file
    image_path = input("Enter the path to the image you want to analyze: ").strip()

    if os.path.exists(image_path):
        analyze_uploaded_image(image_path)
    else:
        print("The file path does not exist. Please try again.")
