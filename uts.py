import sys
import cv2
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QPushButton, QVBoxLayout, QWidget, QHBoxLayout, \
    QFileDialog, QMessageBox
from PyQt5.QtGui import QPixmap, QImage, QIcon
from PyQt5.QtCore import Qt
import numpy as np


class UTS(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("UTS PENGOLAHAN CITRA")

        self.images = []
        self.current_index = 0

        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)

        self.layout = QVBoxLayout()

        self.image_label = QLabel()
        self.layout.addWidget(self.image_label, alignment=Qt.AlignCenter)

        # Button Layout
        self.button_layout = QVBoxLayout()  # Changed to QVBoxLayout
        self.layout.addLayout(self.button_layout)

        # Previous and Next Buttons in One Row
        self.navigation_layout = QHBoxLayout()
        self.button_layout.addLayout(self.navigation_layout)

        self.left_button = QPushButton("←", clicked=self.prev_image)
        self.navigation_layout.addWidget(self.left_button)

        self.right_button = QPushButton("→", clicked=self.next_image)
        self.navigation_layout.addWidget(self.right_button)

        # Other Buttons Below Navigation
        self.add_button = QPushButton("Tambah Gambar", clicked=self.add_image)
        self.button_layout.addWidget(self.add_button)

        self.resize_button = QPushButton("Resize Gambar", clicked=self.resize_images)
        self.button_layout.addWidget(self.resize_button)

        self.grayscale_button = QPushButton("Change Grayscale", clicked=self.convert_to_grayscale)
        self.button_layout.addWidget(self.grayscale_button)

        self.reduce_noise_button_gray = QPushButton("Reduce Noise Grayscale", clicked=self.reduce_noise_gray)
        self.button_layout.addWidget(self.reduce_noise_button_gray)

        self.reduce_noise_button_color = QPushButton("Reduce Noise Color", clicked=self.reduce_noise_color)
        self.button_layout.addWidget(self.reduce_noise_button_color)

        self.normalization_button = QPushButton("Normalization", clicked=self.normalize)
        self.button_layout.addWidget(self.normalization_button)

        self.edge_button = QPushButton("Edge Detection", clicked=self.detect_edges)
        self.button_layout.addWidget(self.edge_button)

        self.save_one_picture_button = QPushButton("Save in 1 Picture", clicked=self.save_in_one_picture)
        self.button_layout.addWidget(self.save_one_picture_button)

        self.save_picture_button = QPushButton("Save Picture", clicked=self.save_picture)
        self.button_layout.addWidget(self.save_picture_button)

        self.reset_button = QPushButton("Restart", clicked=self.restart)
        self.button_layout.addWidget(self.reset_button)

        self.central_widget.setLayout(self.layout)

        self.setWindowIcon(QIcon('logo-udb.png'))

    def add_image(self):
        file_paths, _ = QFileDialog.getOpenFileNames(self, "Tambah Gambar", "", "Image Files (*.png *.jpg *.jpeg)")
        for file_path in file_paths:
            image = cv2.imread(file_path)
            self.images.append(image)
        self.show_image()

    def show_image(self):
        if self.images:
            image = self.images[self.current_index]
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = QImage(image.data, image.shape[1], image.shape[0], QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(image)
            self.image_label.setPixmap(pixmap)
        else:
            self.image_label.clear()

    def next_image(self):
        if self.current_index < len(self.images) - 1:
            self.current_index += 1
            self.show_image()

    def prev_image(self):
        if self.current_index > 0:
            self.current_index -= 1
            self.show_image()

    def resize_images(self):
        if self.images:
            self.images = [cv2.resize(img, (256, 256)) for img in self.images]
            self.show_image()

    def convert_to_grayscale(self):
        if self.images:
            self.images = [cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img for img in self.images]
            self.show_image()

    def reduce_noise_gray(self):
        if self.images:
            self.images = [cv2.fastNlMeansDenoising(img, None, 20, 10, 10) for img in self.images]
            self.show_image()

    def reduce_noise_color(self):
        if self.images:
            self.images = [cv2.fastNlMeansDenoisingColored(img, None, 20, 10, 5, 20) for img in self.images]
            self.show_image()

    def normalize(self):
        if self.images:
            normalized_images = []
            for img in self.images:
                if len(img.shape) == 3:
                    img = cv2.normalize(img, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
                else:
                    min_val, max_val = np.min(img), np.max(img)
                    if min_val != max_val:
                        img = (img - min_val) / (max_val - min_val) * 255
                normalized_images.append(img)
            self.images = normalized_images
            self.show_image()

    def detect_edges(self):
        if self.images:
            self.images = [cv2.Canny(img, 100, 200) if len(img.shape) == 2 else cv2.Canny(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), 100, 200) for img in self.images]
            self.show_image()

    # def save_in_one_picture(self):
    #     if self.images:
    #         collage = np.zeros((1080, 1920, 3), dtype=np.uint8)
    #         row, col = 0, 0
    #         for img in self.images:
    #             img = cv2.resize(img, (256, 256))
    #             collage[row:row+256, col:col+256] = img
    #             col += 256
    #             if col >= 1920:
    #                 col = 0
    #                 row += 256
    #                 if row >= 1080:
    #                     break
    #
    #         save_path, _ = QFileDialog.getSaveFileName(self, "Save Collage", "", "Image Files (*.png *.jpg *.jpeg)")
    #         if save_path:
    #             cv2.imwrite(save_path, collage)
    #             QMessageBox.information(self, "Information", "Collage saved successfully!")

    def save_in_one_picture(self):
        if self.images:
            num_images = len(self.images)
            num_cols = 4  # set column jml
            num_rows = (num_images + num_cols - 1) // num_cols  # count line
            collage_width = num_cols * 256  # set collage width
            collage_height = num_rows * 256  # set collage height
            collage = np.zeros((collage_height, collage_width, 3), dtype=np.uint8)
            row, col = 0, 0
            for img in self.images:
                img = cv2.resize(img, (256, 256))  # set img scale
                collage[row:row + 256, col:col + 256] = img
                col += 256
                if col >= collage_width:
                    col = 0
                    row += 256
            save_path, _ = QFileDialog.getSaveFileName(self, "Save Collage", "", "Image Files (*.png *.jpg *.jpeg)")
            if save_path:
                cv2.imwrite(save_path, collage)
                QMessageBox.information(self, "Information", "Collage saved successfully!")

    def save_picture(self):
        if self.images:
            for i, img in enumerate(self.images):
                save_path, _ = QFileDialog.getSaveFileName(self, f"Save Image {i+1}", "", "Image Files (*.png *.jpg *.jpeg)")
                if save_path:
                    cv2.imwrite(save_path, img)
            QMessageBox.information(self, "Information", "Images saved successfully!")

    def restart(self):
        self.images = []
        self.show_image()


def main():
    app = QApplication(sys.argv)
    window = UTS()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
