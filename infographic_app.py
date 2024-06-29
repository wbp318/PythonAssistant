print("Starting the application.....")

import sys
import os
from PyQt6.QtWidgets import QApplication, QMainWindow, QPushButton, QFileDialog, QVBoxLayout, QWidget, QLabel
from PyQt6.QtGui import QPixmap
from PyQt6.QtCore import Qt

# Import the function from your original script
from your_original_script import create_detailed_infographic

class InfographicApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.setWindowTitle('Infographic Generator')
        self.setGeometry(100, 100, 400, 200)  # Reduced window size

        layout = QVBoxLayout()

        self.upload_button = QPushButton('Upload File', self)
        self.upload_button.clicked.connect(self.upload_file)
        layout.addWidget(self.upload_button)

        self.status_label = QLabel('No file uploaded', self)
        layout.addWidget(self.status_label)

        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

    def upload_file(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Open File", "", "Excel Files (*.xlsx);;PDF Files (*.pdf)")
        if file_path:
            self.status_label.setText(f'Processing file: {os.path.basename(file_path)}')
            QApplication.processEvents()  # Update the GUI
            
            try:
                create_detailed_infographic(file_path)
                self.status_label.setText('Infographic created successfully!')
                self.display_infographic()
            except Exception as e:
                self.status_label.setText(f'Error: {str(e)}')

    def display_infographic(self):
        image_path = 'detailed_infographic.png'
        if os.path.exists(image_path):
            if sys.platform.startswith('darwin'):  # macOS
                os.system(f'open "{image_path}"')
            elif sys.platform.startswith('win'):  # Windows
                os.system(f'start "" "{image_path}"')
            else:  # Linux and other Unix-like
                os.system(f'xdg-open "{image_path}"')
        else:
            self.status_label.setText('Error: Infographic image not found')

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = InfographicApp()
    ex.show()
    sys.exit(app.exec())