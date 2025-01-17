import os
import sys
from setuptools import setup, find_packages, Command

class WipeJSONCommand(Command):
    """Custom command to wipe documents.json content."""
    description = 'Wipe the content of documents.json'
    user_options = []

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self):
        json_path = 'documents.json'
        if os.path.exists(json_path):
            with open(json_path, 'w', encoding='utf-8') as f:
                f.write("[]")
            print("documents.json has been wiped.")
        else:
            print("documents.json does not exist, nothing to wipe.")

requirements = [
    "PyQt5",
    "PyPDF2",
    "pdfplumber",
    "python-docx",
    "openpyxl",
    "torch",
    "transformers",
    "sentence-transformers",
    "openai",
    "requests"
]

setup(
    name="file-minds-assistant",
    version="1.0.0",
    packages=find_packages(),
    install_requires=requirements,
    entry_points={
        "console_scripts": []
    },
    author="Your Name",
    author_email="youremail@example.com",
    description="A document management and AI assistant application.",
    long_description=open("README.md").read() if os.path.exists("README.md") else "",
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/file-minds-assistant",
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    cmdclass={
        'wipe_json': WipeJSONCommand
    },
)
