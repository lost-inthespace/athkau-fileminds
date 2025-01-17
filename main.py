# main.py

import sys
import os
from PyQt5.QtWidgets import QApplication
from ui import AIDocumentAssistant

# Disable tokenizers parallelism to avoid the warning
os.environ['TOKENIZERS_PARALLELISM'] = 'false'


def main():
    try:
        # Create QApplication instance first
        app = QApplication(sys.argv)

        # Then create the main window
        window = AIDocumentAssistant()
        window.show()

        # Start the event loop
        return app.exec_()
    except Exception as e:
        print(f"Error starting application: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())