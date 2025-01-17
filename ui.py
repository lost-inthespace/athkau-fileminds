import sys
import os
import shutil
import json
import torch
import logging
from datetime import datetime
from typing import Optional, Dict, List

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QFileDialog, QTreeView,
    QVBoxLayout, QHBoxLayout, QPushButton, QTextBrowser, QLineEdit, QSplitter,
    QListWidget, QListWidgetItem, QLabel, QProgressBar, QToolBar, QMessageBox,
    QDialog, QInputDialog, QMenu, QFrame, QComboBox, QScrollArea, QStatusBar
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QSize, QStandardPaths, QUrl, QTimer
from PyQt5.QtGui import QIcon, QColor, QPalette, QFont

from assistant import Assistant
from file_processing import process_files
from nlp import cluster_documents, incorporate_user_feedback, generate_embeddings

import subprocess


class FileProcessingThread(QThread):
    """Thread for handling file processing operations"""
    progress = pyqtSignal(int)
    finished = pyqtSignal()
    error = pyqtSignal(str)

    def __init__(self, file_paths: List[str]):
        super().__init__()
        self.file_paths = file_paths

    def run(self):
        try:
            process_files(self.file_paths, json_path='documents.json',
                          progress_callback=self.progress.emit)
            self.finished.emit()
        except Exception as e:
            self.error.emit(str(e))


class OrganizationDialog(QDialog):
    """Dialog for organizing files into categories"""

    def __init__(self, parent, categories: Dict):
        super().__init__(parent)
        self.setWindowTitle("Review and Approve Organization")
        self.categories = categories
        self.setMinimumSize(600, 400)
        self.init_ui()

    def init_ui(self):
        """Initialize the dialog UI"""
        from PyQt5.QtWidgets import QTreeWidget, QTreeWidgetItem
        layout = QVBoxLayout()

        # Add search functionality
        search_layout = QHBoxLayout()
        self.search_input = QLineEdit()
        self.search_input.setPlaceholderText("Search files...")
        self.search_input.textChanged.connect(self.filter_items)
        search_layout.addWidget(self.search_input)
        layout.addLayout(search_layout)

        # Tree widget setup
        self.tree = QTreeWidget()
        self.tree.setColumnCount(2)
        self.tree.setHeaderLabels(["Name", "Type"])
        self.tree.setDragDropMode(self.tree.InternalMove)
        self.tree.setDefaultDropAction(Qt.MoveAction)
        self.tree.setAlternatingRowColors(True)
        self.populate_tree()
        layout.addWidget(self.tree)

        # Buttons
        button_layout = QHBoxLayout()
        self.approve_btn = QPushButton("Approve")
        self.cancel_btn = QPushButton("Cancel")
        self.rename_btn = QPushButton("Rename Category")

        self.approve_btn.clicked.connect(self.accept)
        self.cancel_btn.clicked.connect(self.reject)
        self.rename_btn.clicked.connect(self.rename_category)

        button_layout.addWidget(self.rename_btn)
        button_layout.addWidget(self.cancel_btn)
        button_layout.addWidget(self.approve_btn)
        layout.addLayout(button_layout)

        self.setLayout(layout)

    def filter_items(self, text: str):
        """Filter tree items based on search text"""
        for i in range(self.tree.topLevelItemCount()):
            category_item = self.tree.topLevelItem(i)
            show_category = False

            for j in range(category_item.childCount()):
                file_item = category_item.child(j)
                show_file = text.lower() in file_item.text(0).lower()
                file_item.setHidden(not show_file)
                show_category = show_category or show_file

            category_item.setHidden(not show_category)

    def populate_tree(self):
        """Populate the tree widget with categories and files"""
        self.tree.clear()
        from PyQt5.QtWidgets import QTreeWidgetItem

        for cat_id, cat_info in self.categories.items():
            cat_item = QTreeWidgetItem([cat_info["name"], "Category"])
            cat_item.setData(0, Qt.UserRole, ("category", cat_id))

            for doc in cat_info["files"]:
                file_item = QTreeWidgetItem([
                    os.path.basename(doc['file_path']),
                    doc['file_type']
                ])
                file_item.setData(0, Qt.UserRole, ("file", doc['file_path']))
                cat_item.addChild(file_item)

            self.tree.addTopLevelItem(cat_item)

        self.tree.expandAll()

    def rename_category(self):
        """Rename the selected category"""
        item = self.tree.currentItem()
        if item and item.data(0, Qt.UserRole) and item.data(0, Qt.UserRole)[0] == "category":
            new_name, ok = QInputDialog.getText(
                self, "Rename Category",
                "New Category Name:",
                text=item.text(0)
            )
            if ok and new_name.strip():
                item.setText(0, new_name.strip())

    def get_final_structure(self) -> Dict:
        """Get the final organization structure"""
        final_cats = {}
        for i in range(self.tree.topLevelItemCount()):
            cat_item = self.tree.topLevelItem(i)
            cat_name = cat_item.text(0)
            files = []

            for j in range(cat_item.childCount()):
                f_item = cat_item.child(j)
                f_data = f_item.data(0, Qt.UserRole)
                if f_data and f_data[0] == "file":
                    files.append(f_data[1])

            final_cats[i] = {"name": cat_name, "files": files}

        return final_cats


class AIDocumentAssistant(QMainWindow):
    """Main application window"""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("File Minds Assistant")
        self.resize(1200, 800)
        self.is_day_mode = True

        # Initialize components
        self.init_colors()
        self.setup_logging()
        self.assistant = Assistant()

        # Set up UI
        self.init_ui()
        self.setup_auto_save()

        self.logger.info("Application initialized successfully")

    def setup_logging(self):
        """Set up application logging"""
        log_dir = "logs"
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

        self.logger = logging.getLogger(__name__)
        handler = logging.FileHandler(
            f'logs/ui_{datetime.now().strftime("%Y%m%d")}.log'
        )
        handler.setFormatter(
            logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        )
        self.logger.addHandler(handler)
        self.logger.setLevel(logging.INFO)

    def setup_auto_save(self):
        """Set up automatic saving of application state"""
        self.auto_save_timer = QTimer(self)
        self.auto_save_timer.timeout.connect(self.save_application_state)
        self.auto_save_timer.start(300000)  # Save every 5 minutes

    def save_application_state(self):
        """Save current application state"""
        try:
            state = {
                'current_folder': getattr(self, 'current_folder_path', None),
                'selected_files': self.get_selected_files(),
                'theme': 'day' if self.is_day_mode else 'night',
                'window_geometry': self.saveGeometry().toHex().data().decode()
            }

            with open('app_state.json', 'w') as f:
                json.dump(state, f)

            self.logger.info("Application state saved successfully")
        except Exception as e:
            self.logger.error(f"Error saving application state: {e}")

    def load_application_state(self):
        """Load saved application state"""
        try:
            if os.path.exists('app_state.json'):
                with open('app_state.json', 'r') as f:
                    state = json.load(f)

                if state.get('current_folder'):
                    self.load_folder(state['current_folder'])

                if state.get('theme'):
                    self.is_day_mode = state['theme'] == 'day'
                    self.apply_styles()

                if state.get('window_geometry'):
                    self.restoreGeometry(bytes.fromhex(state['window_geometry']))

                self.logger.info("Application state loaded successfully")
        except Exception as e:
            self.logger.error(f"Error loading application state: {e}")

    def init_colors(self):
        """Initialize color schemes"""
        self.day_colors = {
            "primary_color": "#FFFFFF",
            "secondary_color": "#F7F7F7",
            "border_color": "#E0E0E0",
            "accent_color": "#5B8DEF",
            "text_color": "#333333",
            "hover_color": "#F0F0F0",
            "selection_color": "#E3F2FD"
        }

        self.night_colors = {
            "primary_color": "#1E1E1E",
            "secondary_color": "#2C2C2C",
            "border_color": "#3C3C3C",
            "accent_color": "#5B8DEF",
            "text_color": "#CCCCCC",
            "hover_color": "#383838",
            "selection_color": "#264F78"
        }

        self.colors = self.day_colors

    def init_ui(self):
        """Initialize the user interface"""
        # Central widget setup
        central_widget = QWidget()
        main_layout = QHBoxLayout()
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)
        central_widget.setLayout(main_layout)
        self.setCentralWidget(central_widget)

        # Main splitter setup
        splitter_main = QSplitter(Qt.Horizontal)
        splitter_main.setHandleWidth(1)
        splitter_main.setStyleSheet(
            f"QSplitter::handle {{ background-color: {self.colors['border_color']}; }}"
        )

        # Navigation panel
        self.setup_navigation_panel(splitter_main)

        # Inner splitter setup
        splitter_inner = QSplitter(Qt.Horizontal)
        splitter_inner.setHandleWidth(1)
        splitter_inner.setStyleSheet(
            f"QSplitter::handle {{ background-color: {self.colors['border_color']}; }}"
        )

        # File list panel
        self.setup_file_list_panel(splitter_inner)

        # Assistant panel
        self.setup_assistant_panel(splitter_inner)

        # Set splitter proportions
        splitter_inner.setStretchFactor(0, 2)
        splitter_inner.setStretchFactor(1, 3)

        splitter_main.addWidget(splitter_inner)
        main_layout.addWidget(splitter_main)

        # Toolbar and status bar
        self.setup_toolbar()
        self.setup_status_bar()

        # Apply styles
        self.apply_styles()

        # Load saved state
        self.load_application_state()

    def open_file_cross_platform(self, file_path: str):
        """Open a file using the system's default application"""
        try:
            if sys.platform.startswith('win'):
                os.startfile(file_path)
            elif sys.platform.startswith('darwin'):
                subprocess.call(['open', file_path])
            elif sys.platform.startswith('linux'):
                subprocess.call(['xdg-open', file_path])
            else:
                raise OSError("Unsupported platform")

        except Exception as e:
            self.logger.error(f"Cross-platform file open error: {e}")
            raise

    def open_file_from_link(self, url):
        """Open a file from a link in the chat"""
        file_path = url.toLocalFile()
        if os.path.exists(file_path):
            try:
                self.open_file_cross_platform(file_path)
            except Exception as e:
                QMessageBox.warning(self, "Error", f"Could not open file: {e}")
        else:
            QMessageBox.warning(self, "Error", "File not found.")

    def setup_navigation_panel(self, parent):
        """Set up the navigation panel"""
        nav_widget = QWidget()
        nav_layout = QVBoxLayout()
        nav_layout.setContentsMargins(0, 0, 0, 0)

        # Search box for navigation
        self.nav_search = QLineEdit()
        self.nav_search.setPlaceholderText("Search locations...")
        self.nav_search.textChanged.connect(self.filter_navigation)
        nav_layout.addWidget(self.nav_search)

        # Navigation list
        self.navigation_list = QListWidget()
        self.navigation_list.setFixedWidth(250)
        self.populate_navigation_list()
        self.navigation_list.itemClicked.connect(self.navigation_item_clicked)
        nav_layout.addWidget(self.navigation_list)

        nav_widget.setLayout(nav_layout)
        parent.addWidget(nav_widget)

    def setup_file_list_panel(self, parent):
        """Set up the file list panel"""
        file_widget = QWidget()
        file_layout = QVBoxLayout()
        file_layout.setContentsMargins(8, 8, 8, 8)

        # Search box for files
        self.file_search = QLineEdit()
        self.file_search.setPlaceholderText("Search files...")
        self.file_search.textChanged.connect(self.filter_files)
        file_layout.addWidget(self.file_search)

        # File list
        self.file_list = QListWidget()
        self.file_list.setAlternatingRowColors(True)
        self.file_list.itemClicked.connect(self.file_selected)
        file_layout.addWidget(self.file_list)

        file_widget.setLayout(file_layout)
        parent.addWidget(file_widget)

    def setup_assistant_panel(self, parent):
        """Set up the assistant panel"""
        assistant_widget = QWidget()
        assistant_layout = QVBoxLayout()
        assistant_layout.setContentsMargins(16, 16, 16, 16)
        assistant_layout.setSpacing(8)

        # Chat display
        self.chat_display = QTextBrowser()
        self.chat_display.setOpenLinks(False)
        self.chat_display.anchorClicked.connect(self.open_file_from_link)
        self.chat_display.setStyleSheet(
            f"""
            QTextBrowser {{
                background-color: {self.colors['secondary_color']};
                border: 1px solid {self.colors['border_color']};
                border-radius: 6px;
                padding: 8px;
            }}
            """
        )

        # Input area
        input_container = QWidget()
        input_layout = QHBoxLayout()
        input_layout.setContentsMargins(0, 0, 0, 0)

        self.chat_input = QLineEdit()
        self.chat_input.setPlaceholderText("Type your message here or type Help...")
        self.chat_input.returnPressed.connect(self.process_query)

        send_button = QPushButton("Send")
        send_button.clicked.connect(self.process_query)

        input_layout.addWidget(self.chat_input)
        input_layout.addWidget(send_button)
        input_container.setLayout(input_layout)

        # Continue from setup_assistant_panel
        assistant_layout.addWidget(self.chat_display)
        assistant_layout.addWidget(input_container)

        assistant_widget.setLayout(assistant_layout)
        parent.addWidget(assistant_widget)

    def setup_toolbar(self):
        """Set up the application toolbar"""
        toolbar = QToolBar("Main Toolbar")
        toolbar.setMovable(False)
        toolbar.setIconSize(QSize(24, 24))
        self.addToolBar(Qt.TopToolBarArea, toolbar)

        # Folder path display
        self.folder_path_label = QLabel("No folder selected")
        self.folder_path_label.setStyleSheet(f"padding: 0 10px;")
        toolbar.addWidget(self.folder_path_label)

        toolbar.addSeparator()

        # Toolbar buttons
        buttons = [
            ("Select Folder", "folder-open.png", self.select_folder),
            ("Organize Files", "files-sorting.png", self.organize_files),
            ("Toggle Theme", "toggle-day-night.png", self.toggle_theme),
            ("Reset App", "reset.png", self.reset_app_data)
        ]

        for text, icon, callback in buttons:
            button = QPushButton()
            button.setIcon(QIcon(f"icons/{icon}"))
            button.setToolTip(text)
            button.clicked.connect(callback)
            toolbar.addWidget(button)

    def setup_status_bar(self):
        """Set up the application status bar"""
        status_bar = QStatusBar()
        self.setStatusBar(status_bar)

        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setMaximumWidth(200)
        self.progress_bar.setVisible(False)
        status_bar.addPermanentWidget(self.progress_bar)

        # Status message
        self.status_message = QLabel("")
        status_bar.addWidget(self.status_message)

    def filter_navigation(self, text: str):
        """Filter navigation items based on search text"""
        for i in range(self.navigation_list.count()):
            item = self.navigation_list.item(i)
            item.setHidden(text.lower() not in item.text().lower())

    def filter_files(self, text: str):
        """Filter file list based on search text"""
        for i in range(self.file_list.count()):
            item = self.file_list.item(i)
            item.setHidden(text.lower() not in item.text().lower())

    def populate_navigation_list(self):
        """Populate the navigation list with system locations"""
        locations = [
            ("Desktop", QStandardPaths.DesktopLocation),
            ("Downloads", QStandardPaths.DownloadLocation),
            ("Documents", QStandardPaths.DocumentsLocation),
            ("Pictures", QStandardPaths.PicturesLocation),
            ("Music", QStandardPaths.MusicLocation),
            ("Videos", QStandardPaths.MoviesLocation)
        ]

        for name, location in locations:
            path = QStandardPaths.writableLocation(location)
            if path:
                item = QListWidgetItem(name)
                item.setData(Qt.UserRole, path)
                item.setIcon(QIcon(f"icons/{name.lower()}.png"))
                self.navigation_list.addItem(item)

    def navigation_item_clicked(self, item: QListWidgetItem):
        """Handle navigation item selection"""
        try:
            folder_path = item.data(Qt.UserRole)
            if folder_path and os.path.exists(folder_path):
                self.load_folder(folder_path)
                self.folder_path_label.setText(f"Folder: {folder_path}")
                self.status_message.setText(f"Loaded folder: {folder_path}")
            else:
                QMessageBox.warning(self, "Error", "Selected location is not accessible")
        except Exception as e:
            self.logger.error(f"Navigation error: {e}")
            QMessageBox.warning(self, "Error", f"Error accessing location: {str(e)}")

    def select_folder(self):
        """Open folder selection dialog"""
        try:
            folder_path = QFileDialog.getExistingDirectory(
                self,
                "Select Folder",
                os.path.expanduser("~"),
                QFileDialog.ShowDirsOnly
            )

            if folder_path:
                self.load_folder(folder_path)
                self.folder_path_label.setText(f"Folder: {folder_path}")
                self.status_message.setText(f"Selected folder: {folder_path}")
        except Exception as e:
            self.logger.error(f"Folder selection error: {e}")
            QMessageBox.warning(self, "Error", f"Error selecting folder: {str(e)}")

    def load_folder(self, folder_path: str):
        """Load and process files from selected folder"""
        try:
            self.current_folder_path = folder_path
            self.file_paths = []

            # Collect all file paths
            for root, _, files in os.walk(folder_path):
                for file in files:
                    self.file_paths.append(os.path.join(root, file))

            # Start processing in thread
            self.progress_bar.setVisible(True)
            self.file_processing_thread = FileProcessingThread(self.file_paths)
            self.file_processing_thread.progress.connect(self.update_progress)
            self.file_processing_thread.finished.connect(self.processing_finished)
            self.file_processing_thread.error.connect(self.processing_error)
            self.file_processing_thread.start()

            self.status_message.setText("Processing files...")

        except Exception as e:
            self.logger.error(f"Folder loading error: {e}")
            QMessageBox.warning(self, "Error", f"Error loading folder: {str(e)}")

    def update_progress(self, value: int):
        """Update progress bar value"""
        self.progress_bar.setValue(value)

    def processing_finished(self):
        """Handle completion of file processing"""
        try:
            self.progress_bar.setVisible(False)
            self.assistant.load_documents()

            if self.assistant.documents:
                self.assistant.embeddings = generate_embeddings(
                    [doc.content for doc in self.assistant.documents]
                )
            else:
                self.assistant.embeddings = None

            self.load_files()
            self.status_message.setText("File processing completed")

        except Exception as e:
            self.logger.error(f"Processing completion error: {e}")
            QMessageBox.warning(self, "Error", "Error finishing file processing")

    def processing_error(self, error_message: str):
        """Handle file processing errors"""
        self.logger.error(f"File processing error: {error_message}")
        self.progress_bar.setVisible(False)
        QMessageBox.warning(self, "Error", f"Error processing files: {error_message}")
        self.status_message.setText("File processing failed")

    def load_files(self):
        """Load processed files into file list"""
        try:
            self.file_list.clear()

            if hasattr(self, 'current_folder_path'):
                for doc in self.assistant.documents:
                    if doc.file_path.startswith(self.current_folder_path):
                        item = QListWidgetItem(os.path.basename(doc.file_path))
                        item.setData(Qt.UserRole, doc.id)
                        item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
                        item.setCheckState(Qt.Unchecked)

                        # Add file type and size info
                        file_info = os.stat(doc.file_path)
                        size_mb = file_info.st_size / (1024 * 1024)
                        item.setToolTip(
                            f"Name: {os.path.basename(doc.file_path)}\n"
                            f"Type: {doc.file_type}\n"
                            f"Size: {size_mb:.2f} MB"
                        )

                        self.file_list.addItem(item)

        except Exception as e:
            self.logger.error(f"File loading error: {e}")
            QMessageBox.warning(self, "Error", f"Error loading files: {str(e)}")

    def file_selected(self, item: QListWidgetItem):
        """Handle file selection"""
        try:
            if item.checkState() == Qt.Checked:
                # Uncheck other items
                for i in range(self.file_list.count()):
                    other_item = self.file_list.item(i)
                    if other_item != item:
                        other_item.setCheckState(Qt.Unchecked)

                # Select document
                document_id = item.data(Qt.UserRole)
                if self.assistant.select_document(document_id):
                    self.chat_display.append(
                        f"<b>Assistant:</b> Selected file: {item.text()}"
                    )
                    self.status_message.setText(f"Selected file: {item.text()}")
            else:
                # Deselect document
                document_id = item.data(Qt.UserRole)
                if (self.assistant.selected_document and
                        self.assistant.selected_document.id == document_id):
                    self.assistant.selected_document = None
                    self.chat_display.append(
                        f"<b>Assistant:</b> Deselected file: {item.text()}"
                    )
                    self.status_message.setText("No file selected")

        except Exception as e:
            self.logger.error(f"File selection error: {e}")
            QMessageBox.warning(self, "Error", f"Error selecting file: {str(e)}")

    def process_query(self):
        """Process user queries"""
        try:
            query = self.chat_input.text().strip()
            if not query:
                return

            self.chat_input.clear()

            # Add user message to chat
            user_message = f"<p><b>You:</b> {query}</p>"
            self.chat_display.append(user_message)

            # Get assistant response
            response = self.assistant.handle_query(query)
            assistant_message = f"<p><b>Assistant:</b><br>{response}</p>"
            self.chat_display.append(assistant_message)

            # Scroll to bottom
            self.chat_display.verticalScrollBar().setValue(
                self.chat_display.verticalScrollBar().maximum()
            )

        except Exception as e:
            self.logger.error(f"Query processing error: {e}")
            self.chat_display.append(
                "<p><b>Assistant:</b><br>Sorry, I encountered an error while "
                "processing your request.</p>"
            )

    def organize_files(self):
        """Organize files into categories"""
        try:
            if not hasattr(self, 'current_folder_path'):
                QMessageBox.information(self, "Info", "No folder selected.")
                return

            folder_docs = [
                doc for doc in self.assistant.documents
                if doc.file_path.startswith(self.current_folder_path)
            ]

            if not folder_docs:
                QMessageBox.information(self, "Info", "No documents in selected folder.")
                return

            # Generate embeddings and cluster documents
            texts = [doc.content for doc in folder_docs]
            embeddings = generate_embeddings(texts)
            n_clusters = min(3, len(folder_docs)) if folder_docs else 1
            categories = cluster_documents(folder_docs, embeddings, n_clusters)

            # Show organization dialog
            dlg = OrganizationDialog(self, categories)
            if dlg.exec_() == QDialog.Accepted:
                self._apply_organization(dlg.get_final_structure())

        except Exception as e:
            self.logger.error(f"File organization error: {e}")
            QMessageBox.warning(self, "Error", f"Error organizing files: {str(e)}")

    def _apply_organization(self, categories: Dict):
        """Apply the file organization structure"""
        try:
            for cat_id, cat_info in categories.items():
                # Create category folder
                cat_folder_name = cat_info["name"].replace(" ", "_")
                target_dir = os.path.join(self.current_folder_path, cat_folder_name)

                if not os.path.exists(target_dir):
                    os.makedirs(target_dir)

                # Move files to category folder
                for file_path in cat_info["files"]:
                    base_name = os.path.basename(file_path)
                    new_path = os.path.join(target_dir, base_name)

                    if file_path != new_path:
                        try:
                            shutil.move(file_path, new_path)
                            # Update document paths
                            for doc in self.assistant.documents:
                                if doc.file_path == file_path:
                                    doc.file_path = new_path
                        except Exception as e:
                            self.logger.error(f"Error moving file {file_path}: {e}")

            # Save updated documents
            with open('documents.json', 'w', encoding='utf-8') as f:
                json.dump(
                    [doc.__dict__ for doc in self.assistant.documents],
                    f,
                    ensure_ascii=False,
                    indent=4
                )

            # Update user preferences
            incorporate_user_feedback(categories, {})

            QMessageBox.information(self, "Done", "Files organized successfully!")
            self.load_files()
            self.status_message.setText("Files organized successfully")

        except Exception as e:
            self.logger.error(f"Organization application error: {e}")
            QMessageBox.warning(self, "Error", f"Error applying organization: {str(e)}")

    def toggle_theme(self):
        """Toggle between day and night themes"""
        try:
            self.is_day_mode = not self.is_day_mode
            self.colors = self.day_colors if self.is_day_mode else self.night_colors
            self.apply_styles()
            theme_name = "day" if self.is_day_mode else "night"
            self.status_message.setText(f"Switched to {theme_name} theme")

        except Exception as e:
            self.logger.error(f"Theme toggle error: {e}")
            QMessageBox.warning(self, "Error", "Error changing theme")

    def reset_app_data(self):
        """Reset the application to its initial state"""
        try:
            # Remove data files
            if os.path.exists('documents.json'):
                os.remove('documents.json')
            if os.path.exists('user_preferences.json'):
                os.remove('user_preferences.json')

            # Reset assistant
            self.assistant.documents = []
            self.assistant.selected_document = None
            self.assistant.embeddings = None

            # Reset UI
            if hasattr(self, 'current_folder_path'):
                del self.current_folder_path

            self.file_list.clear()
            self.chat_display.clear()
            self.folder_path_label.setText("No folder selected")
            QMessageBox.information(self, "Reset", "All data has been wiped and the app has been reset.")
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Error resetting application: {str(e)}")

    def apply_styles(self):
        """Apply the current theme styles"""
        try:
            colors = self.colors
            stylesheet = f"""
                    QMainWindow {{
                        background-color: {colors['primary_color']};
                    }}
                    QWidget {{
                        background-color: {colors['primary_color']};
                        color: {colors['text_color']};
                        font-family: 'Segoe UI', 'Helvetica Neue', Arial, sans-serif;
                        font-size: 10pt;
                    }}
                    QToolBar {{
                        background-color: {colors['secondary_color']};
                        border-bottom: 1px solid {colors['border_color']};
                        padding: 4px;
                    }}
                    QLabel {{
                        color: {colors['text_color']};
                    }}
                    QToolBar QPushButton {{
            # Continuing the stylesheet from apply_styles method
                            background-color: {colors['accent_color']};
                            color: #FFFFFF;
                            border: none;
                            border-radius: 6px;
                            padding: 6px 12px;
                            margin: 0 8px;
                        }}
                        QToolBar QPushButton:hover {{
                            background-color: {colors['hover_color']};
                        }}
                        QLineEdit {{
                            background-color: {colors['secondary_color']};
                            color: {colors['text_color']};
                            border: 1px solid {colors['border_color']};
                            border-radius: 6px;
                            padding: 6px;
                        }}
                        QTextBrowser {{
                            background-color: {colors['secondary_color']};
                            color: {colors['text_color']};
                            border: 1px solid {colors['border_color']};
                            border-radius: 6px;
                            padding: 8px;
                        }}
                        QListWidget {{
                            background-color: {colors['secondary_color']};
                            color: {colors['text_color']};
                            border: 1px solid {colors['border_color']};
                            border-radius: 6px;
                            padding: 4px;
                        }}
                        QListWidget::item {{
                            padding: 4px;
                            border-radius: 4px;
                        }}
                        QListWidget::item:hover {{
                            background-color: {colors['hover_color']};
                        }}
                        QListWidget::item:selected {{
                            background-color: {colors['selection_color']};
                        }}
                        QProgressBar {{
                            border: 1px solid {colors['border_color']};
                            border-radius: 4px;
                            text-align: center;
                        }}
                        QProgressBar::chunk {{
                            background-color: {colors['accent_color']};
                            width: 10px;
                        }}
                        QStatusBar {{
                            background-color: {colors['secondary_color']};
                            color: {colors['text_color']};
                            border-top: 1px solid {colors['border_color']};
                        }}
                        QSplitter::handle {{
                            background-color: {colors['border_color']};
                        }}
                        """
            self.setStyleSheet(stylesheet)

        except Exception as e:
            self.logger.error(f"Style application error: {e}")
            QMessageBox.warning(self, "Error", "Error applying styles")

        def open_file_from_link(self, url: QUrl):
            """Open a file from a chat link"""
            try:
                file_path = url.toLocalFile()
                if os.path.exists(file_path):
                    self.open_file_cross_platform(file_path)
                else:
                    QMessageBox.warning(self, "Error", "File not found.")

            except Exception as e:
                self.logger.error(f"File opening error: {e}")
                QMessageBox.warning(self, "Error", f"Could not open file: {str(e)}")

        def reset_app_data(self):
            """Reset the application to its initial state"""
            try:
                reply = QMessageBox.question(
                    self,
                    "Confirm Reset",
                    "Are you sure you want to reset all application data? This cannot be undone.",
                    QMessageBox.Yes | QMessageBox.No,
                    QMessageBox.No
                )

                if reply == QMessageBox.Yes:
                    # Remove data files
                    files_to_remove = ['documents.json', 'user_preferences.json', 'app_state.json']
                    for file in files_to_remove:
                        if os.path.exists(file):
                            os.remove(file)

                    # Reset assistant
                    self.assistant.documents = []
                    self.assistant.selected_document = None
                    self.assistant.embeddings = None

                    # Reset UI
                    if hasattr(self, 'current_folder_path'):
                        del self.current_folder_path
                    self.file_list.clear()
                    self.chat_display.clear()
                    self.folder_path_label.setText("No folder selected")

                    # Clear logs
                    log_dir = "logs"
                    if os.path.exists(log_dir):
                        for log_file in os.listdir(log_dir):
                            os.remove(os.path.join(log_dir, log_file))

                    QMessageBox.information(
                        self,
                        "Reset Complete",
                        "All data has been wiped and the app has been reset."
                    )
                    self.status_message.setText("Application reset completed")

            except Exception as e:
                self.logger.error(f"App reset error: {e}")
                QMessageBox.warning(self, "Error", f"Error resetting application: {str(e)}")

        def get_selected_files(self) -> List[str]:
            """Get list of currently selected files"""
            selected = []
            for i in range(self.file_list.count()):
                item = self.file_list.item(i)
                if item.checkState() == Qt.Checked:
                    selected.append(item.data(Qt.UserRole))
            return selected

        def closeEvent(self, event):
            """Handle application close event"""
            try:
                # Save application state
                self.save_application_state()

                # Clean up resources
                if hasattr(self, 'file_processing_thread'):
                    self.file_processing_thread.quit()
                    self.file_processing_thread.wait()

                self.logger.info("Application closed successfully")
                event.accept()

            except Exception as e:
                self.logger.error(f"Close event error: {e}")
                event.accept()

        if __name__ == "__main__":
            try:
                app = QApplication(sys.argv)
                window = AIDocumentAssistant()
                window.show()
                sys.exit(app.exec_())
            except Exception as e:
                logging.error(f"Application startup error: {e}")
                sys.exit(1)