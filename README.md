# Complete OCR System (Advanced Version)

![Python](https://img.shields.io/badge/Python-3.11-blue)
![OpenCV](https://img.shields.io/badge/OpenCV-Computer%20Vision-green)
![Tesseract](https://img.shields.io/badge/Tesseract-OCR-red)
![License](https://img.shields.io/badge/License-MIT-yellow)
![Status](https://img.shields.io/badge/Project-Academic%20Lab%20Project-purple)

------------------------------------------------------------------------

## 🎬 Demo

![Watch Demo Video](docs/demo.mp4)


## 📌 Overview

This project is an **advanced, CLI-based Optical Character Recognition
(OCR) system** implemented in **Python** using **OpenCV, Tesseract OCR,
img2table, and Rich**. It is designed as an **Object-Oriented academic
lab project** with emphasis on **modularity, extensibility, and
real-world document processing challenges**.

The system handles: - Image orientation correction - Printed and
handwritten text extraction - Bordered and borderless table detection -
Mixed-content documents (text + tables) - Multi-language OCR support

------------------------------------------------------------------------

## 🎯 Problem Statement

Manual extraction of textual and tabular information from scanned
documents is time-consuming and error-prone. Existing OCR systems often
fail when images are rotated, contain mixed content, or include
borderless tables. This project addresses these challenges by designing
a structured OCR pipeline using computer vision and OCR techniques.

------------------------------------------------------------------------

## 🎯 Objectives

-   Design a modular OCR system using Object-Oriented Programming
-   Automatically correct image orientation
-   Extract text, tables, and mixed content accurately
-   Support multiple OCR languages
-   Provide a user-friendly CLI interface
-   Improve maintainability and scalability of OCR workflows

------------------------------------------------------------------------

## ✨ Key Features

-   **Smart Orientation Detection**
-   **Printed Text OCR**
-   **Handwritten Text OCR (Experimental)**
-   **Bordered & Borderless Table Extraction**
-   **Mixed Content Handling**
-   **Auto Column Detection**
-   **Interactive Rich-based CLI**
-   **Debug & Comparison Images**

------------------------------------------------------------------------

## 🧠 System Architecture (OOP Design)

![System Architecture](docs/architecture.png)

  Class                         Responsibility
  ----------------------------- --------------------------------
  `OCRApp`                      CLI interface and control flow
  `OCRProcessor`                Central processing pipeline
  `SmartOrientationCorrector`   Image rotation detection
  `TextExtractor`               Printed text extraction
  `ImprovedHandwrittenOCR`      Handwritten text OCR
  `FixedTableExtractor`         Table OCR
  `Img2TableExtractor`          Structured table extraction
  `FixedMixedContentHandler`    Mixed document handler
  `ContentDetector`             Content classification

------------------------------------------------------------------------

## 🔍 System Workflow

1.  User provides image path
2.  Orientation correction (optional)
3.  Content type detection
4.  OCR execution based on detected content
5.  Output saving and terminal display

------------------------------------------------------------------------

## 🧮 Algorithm Overview (Pseudocode)

    Read image path
    If orientation enabled:
        Rotate image at 0°, 90°, 180°, 270°
        Score each using word detection
        Select best orientation
    Detect content type
    Run corresponding OCR pipeline
    Save and display results

------------------------------------------------------------------------

## 🖼️ Debug & Comparison Outputs

-   original_image.png
-   corrected_orientation.png
-   debug_edges.png

------------------------------------------------------------------------

## 🌍 Supported Languages

-   English (eng)
-   Spanish (spa)
-   French (fra)
-   German (deu)

------------------------------------------------------------------------

## 🛠️ Requirements

- Python **3.11**
- Tesseract OCR (installed separately)
- Python libraries:
  ```txt
  numpy
  opencv-python
  pillow
  pytesseract
  rich
  img2table
  os
  re
  tempfile
  ```

------------------------------------------------------------------------

## ▶️ How to Run

``` bash
python main.py
```

------------------------------------------------------------------------

## ⚠️ Limitations

-   Limited handwritten OCR accuracy
-   No PDF support
-   Performance degrades on noisy images

------------------------------------------------------------------------

## 🚀 Future Improvements

-   GUI (Tkinter / PyQt)
-   PDF OCR support
-   Deep learning handwriting recognition
-   Automatic language detection
-   CSV / Excel export

------------------------------------------------------------------------

## 👤 Authors

-   Muhammad Inshal Saqib Siddiqui
-   Muhammad Hamza Latif Khan
-   Muhammad Abdullah Sohail
-   Marwan Ali

------------------------------------------------------------------------

## 📜 License

MIT License
