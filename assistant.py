# assistant.py
import base64
import re
import json
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
from PyQt5.QtCore import QUrl
from nlp import summarize_text, semantic_search, generate_embeddings
from openai import AzureOpenAI
import logging
import os
from enum import Enum
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()
AZURE_API_KEY = os.getenv("AZURE_API_KEY")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")

class QueryIntent(Enum):
    """Enumeration for different types of query intents"""
    SEARCH = "search_file"
    SUMMARIZE = "summarize"
    CREATE = "create_content"
    DELETE = "delete_content"
    UPDATE = "update_content"
    HELP = "help"
    ASK = "ask_question"
    DOCUMENT = "ask_document"
    GENERAL = "general"


@dataclass
class Document:
    """Data class for document information"""
    id: str
    file_path: str
    file_type: str
    creation_time: str
    modification_time: str
    content: str


class Assistant:
    """Enhanced Assistant class with improved functionality"""

    def __init__(self, json_path='documents.json'):
        self.json_path = json_path
        self.documents: List[Document] = []
        self.selected_document: Optional[Document] = None
        self.embeddings = None

        # Setup logging
        self._setup_logging()

        # Initialize components
        self.load_documents()
        self._initialize_embeddings()
        self.azure_client = self._initialize_azure_client()

    def _setup_logging(self):
        """Setup logging configuration"""
        log_dir = "logs"
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

        log_file = f'logs/assistant_{datetime.now().strftime("%Y%m%d")}.log'
        self.logger = logging.getLogger(__name__)
        handler = logging.FileHandler(log_file)
        handler.setFormatter(logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        ))
        self.logger.addHandler(handler)
        self.logger.setLevel(logging.INFO)

    def _initialize_azure_client(self) -> AzureOpenAI:
        """Initialize Azure OpenAI client"""
        try:
            return AzureOpenAI(
                azure_endpoint=AZURE_OPENAI_ENDPOINT,
                api_key=AZURE_API_KEY,
                api_version="2024-02-15-preview"
            )
        except Exception as e:
            self.logger.error(f"Failed to initialize Azure client: {e}")
            raise

    def _initialize_embeddings(self):
        """Initialize document embeddings"""
        if self.documents:
            try:
                self.embeddings = generate_embeddings([doc.content for doc in self.documents])
                self.logger.info("Embeddings generated successfully")
            except Exception as e:
                self.logger.error(f"Failed to generate embeddings: {e}")
                self.embeddings = None

    def load_documents(self):
        """Load documents from JSON file"""
        try:
            if os.path.exists(self.json_path) and os.path.getsize(self.json_path) > 0:
                with open(self.json_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.documents = []
                    for doc in data:
                        self.documents.append(Document(
                            id=doc['id'],
                            file_path=doc['file_path'],
                            file_type=doc['file_type'],
                            creation_time=doc['creation_time'],
                            modification_time=doc['modification_time'],
                            content=doc['content']
                        ))
                self.logger.info(f"Loaded {len(self.documents)} documents")
            else:
                self.documents = []
                self.logger.info("No documents to load")
        except Exception as e:
            self.logger.error(f"Error loading documents: {e}")
            self.documents = []

    def interpret_query(self, query: str) -> QueryIntent:
        """Enhanced query interpretation with keyword scoring"""
        query_lower = query.lower()

        # Keyword patterns with weights
        keyword_patterns = {
            QueryIntent.SEARCH: [
                (r'\b(find|search|locate|get|retrieve)\b', 1.0),
                (r'\b(look\s+for|explore|seek)\b', 0.8),
                (r'\b(where|show|list)\b', 0.6)
            ],
            QueryIntent.SUMMARIZE: [
                (r'\b(summarize|summary|brief|overview)\b', 1.0),
                (r'\b(condense|outline|digest|recap)\b', 0.8),
                (r'\b(shorten|simplify)\b', 0.6)
            ],
            QueryIntent.CREATE: [
                (r'\b(create|make|generate|build)\b', 1.0),
                (r'\b(new|start|begin)\b', 0.7)
            ],
            QueryIntent.DELETE: [
                (r'\b(delete|remove|erase)\b', 1.0),
                (r'\b(clear|clean|wipe)\b', 0.8)
            ],
            QueryIntent.HELP: [
                (r'\b(help|assist|guide|how)\b', 1.0),
                (r'\b(explain|show|tell)\b', 0.7),
                (r'\?$', 0.5)
            ]
        }

        # Calculate scores for each intent
        scores = {intent: 0.0 for intent in QueryIntent}
        for intent, patterns in keyword_patterns.items():
            for pattern, weight in patterns:
                if re.search(pattern, query_lower):
                    scores[intent] += weight

        # Get highest scoring intent
        max_score = max(scores.values())
        if max_score > 0:
            return max(scores.items(), key=lambda x: x[1])[0]

        # Default handling
        if self.selected_document:
            return QueryIntent.DOCUMENT
        return QueryIntent.GENERAL

    def handle_query(self, query: str) -> str:
        """Handle user query with improved response handling"""
        try:
            self.logger.info(f"Processing query: {query}")
            intent = self.interpret_query(query)

            # Map intents to handlers
            handlers = {
                QueryIntent.SUMMARIZE: self.summarize_selected_document,
                QueryIntent.DOCUMENT: lambda: self.answer_question_about_document(query),
                QueryIntent.SEARCH: lambda: self.search_files(query),
                QueryIntent.HELP: self.get_help_text,
                QueryIntent.CREATE: lambda: "Feature under development: Content creation",
                QueryIntent.DELETE: lambda: "Feature under development: Content deletion"
            }

            if intent in handlers:
                response = handlers[intent]()
                self.logger.info(f"Query handled with intent: {intent}")
                return response

            return "I'm not sure how to help with that. Type 'help' for more information."

        except Exception as e:
            self.logger.error(f"Error handling query: {e}")
            return "Sorry, I encountered an error while processing your request."

    def summarize_selected_document(self) -> str:
        """Summarize the selected document"""
        try:
            if not self.selected_document:
                return "No document selected for summarization."

            content = self.selected_document.content
            if not content.strip():
                return "Selected document has no content to summarize."

            summary = summarize_text(content)
            return f"[summarizer] {summary}"

        except Exception as e:
            self.logger.error(f"Error in summarization: {e}")
            return "Failed to generate summary. Please try again."

    def answer_question_about_document(self, query: str) -> str:
        """Generate answer about document using Azure OpenAI"""
        try:
            if not self.selected_document:
                return "No document selected. Please select a document first."

            content = self.selected_document.content
            if not content.strip():
                return "Selected document has no content to analyze."

            try:
                response = self.azure_client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {
                            "role": "system",
                            "content": f"The following is the document content: {content}",
                        },
                        {
                            "role": "user",
                            "content": f"My question about the document is: {query}",
                        }
                    ],
                    temperature=0.7,
                    max_tokens=500
                )

                return f"[Azure AI] {response.choices[0].message.content.strip()}"

            except Exception as e:
                self.logger.error(f"Azure API error: {e}")
                return "Sorry, I encountered an error while processing your question."

        except Exception as e:
            self.logger.error(f"Error in document Q&A: {e}")
            return "Sorry, I encountered an error while processing your request."

    def search_files(self, query: str) -> str:
        """Search files with enhanced results formatting"""
        try:
            if not self.embeddings or not self.documents:
                return "No documents available for search."

            results = semantic_search(query, self.documents, self.embeddings)

            if not results:
                return "No matching files found."

            # Format results with HTML
            response = "<div>Here are the files that match your query:<br>"
            for doc, score in results:
                file_name = os.path.basename(doc.file_path)
                file_url = QUrl.fromLocalFile(doc.file_path).toString()
                response += f"<a href='{file_url}'>{file_name}</a> (Score: {score:.2f})<br>"
            response += "</div>"

            return response

        except Exception as e:
            self.logger.error(f"Error in file search: {e}")
            return "An error occurred while searching files."

    def get_help_text(self) -> str:
        """Generate help text with examples"""
        return '''
    [HELP MENU]
    This assistant can help you with the following tasks:
    1. **Searching for Files**:
       - Use keywords like: **find, search, locate, get, retrieve**.
       - Ensure you have **not selected a file** for this operation.
       - Example: "Find all files related to project X."

    2. **Summarizing Documents**:
       - Use keywords like: **summarize, summary, brief, overview**.
       - Ensure you have **selected a file** for this operation.
       - Example: "Give me a brief summary of the selected document."

    3. **Asking Questions About a Document**:
       - Make sure a file is **selected** for this operation.
       - To confirm, look for this message: "**Assistant: Selected file: [document_name]**".
       - Example: "What is the main conclusion in this document?"

    [Note]:
    - If you're unsure about the file selection, type: "Show selected file."
    - For further assistance, feel free to ask!
    '''

    def select_document(self, document_id: str) -> bool:
        """Select a document by ID"""
        try:
            for doc in self.documents:
                if doc.id == document_id:
                    self.selected_document = doc
                    self.logger.info(f"Selected document: {doc.file_path}")
                    return True
            return False
        except Exception as e:
            self.logger.error(f"Error selecting document: {e}")
            return False