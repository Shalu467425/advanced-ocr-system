# Complete OCR System (Advanced Version)

![Python](https://img.shields.io/badge/Python-3.11-blue)
![OpenCV](https://img.shields.io/badge/OpenCV-Computer%20Vision-green)
![Tesseract](https://img.shields.io/badge/Tesseract-OCR-red)
![License](https://img.shields.io/badge/License-MIT-yellow)
![Status](https://img.shields.io/badge/Project-Academic%20Lab%20Project-purple)


## 📌 Overview

This project is an **advanced, CLI-based Optical Character Recognition (OCR) system** implemented in **Python** using **OpenCV, Tesseract OCR, img2table, and Rich**. It is an improved and extended version of a basic OCR system and is designed as an **OOP-focused academic / lab project**.

The system is capable of:
- Automatically correcting image orientation using **word-based scoring**
- Extracting **plain text**, **tables**, and **mixed content (text + tables)**
- Handling **bordered and borderless tables**
- Performing OCR on **handwritten text (with limited accuracy)**
- Auto-detecting **table columns**
- Supporting **multiple languages**
- Providing **comparison images** for analysis (original vs processed)

---

## ✨ Key Features

- ✅ **Smart Orientation Detection**  
  Automatically tests 0°, 90°, 180°, and 270° rotations and selects the best orientation based on real English word detection.

- ✅ **Text OCR**  
  Standard OCR for printed text using OpenCV preprocessing and Tesseract LSTM engine.

- ✅ **Table Extraction**  
  - Border-based table detection using OpenCV morphology
  - Borderless table OCR using word-position clustering
  - Optional integration with `img2table` for structured tables

- ✅ **Mixed Content Handling (Text + Table)**  
  Detects table regions, extracts:
  - Text above the table
  - Table content
  - Text below the table

- ✅ **Handwritten Text Support**  
  Uses multiple preprocessing strategies and selects the best OCR output based on character count.

- ✅ **Auto Column Detection**  
  Automatically determines the number of columns in tables (manual override supported).

- ✅ **Image Comparison Output**  
  Allows users to visually compare:
  - Original input image
  - Orientation-corrected image
  - Debug images for table detection

- ✅ **Interactive CLI (Rich-based)**  
  User-friendly terminal UI with menus, tables, panels, and progress indicators.

---

## 🧠 System Architecture (OOP Design)

The project follows **Object-Oriented Programming principles** with clear separation of responsibilities:

| Class | Responsibility |
|------|---------------|
| `OCRApp` | CLI interface and user interaction |
| `OCRProcessor` | Main processing pipeline controller |
| `SmartOrientationCorrector` | Detects and fixes image rotation |
| `TextExtractor` | Extracts printed text |
| `ImprovedHandwrittenOCR` | Handwritten text extraction |
| `FixedTableExtractor` | OCR-based table extraction |
| `Img2TableExtractor` | Structured table extraction using img2table |
| `FixedMixedContentHandler` | Handles text + table documents |
| `ContentDetector` | Auto-detects document type |

This modular design improves **maintainability, reusability, and testability**.

---

## 🔍 How the System Works

1. **Image Input**  
   User selects an image file via CLI.

2. **Orientation Correction (Optional)**  
   The image is rotated at multiple angles and scored using detected real words. The best orientation is selected.

3. **Content Detection**  
   The system determines whether the image contains:
   - Only text
   - Only tables
   - Mixed content (text + table)

4. **OCR Processing**  
   Depending on the mode:
   - Text → `TextExtractor`
   - Table → `FixedTableExtractor` / `Img2TableExtractor`
   - Mixed → `FixedMixedContentHandler`
   - Handwritten → `ImprovedHandwrittenOCR`

5. **Result Saving & Display**  
   Extracted text and tables are saved as `.txt` files and displayed in the terminal.

---

## 🖼️ Comparison Images

When enabled, the system saves the following files:

- `original_image.png` → Original uploaded image
- `corrected_orientation.png` → Orientation-corrected image (if rotated)
- `debug_edges.png` → Edge detection for table debugging

These images help in **visual verification and debugging**, especially useful for viva demonstrations.

---

## 🌍 Supported Languages

- English (`eng`)
- Spanish (`spa`)
- French (`fra`)
- German (`deu`)

> ⚠️ The same language must be used consistently for text and tables.

---

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
  re
  os
  tempfile
  ```

---

## ▶️ How to Run

```bash
python main.py
```

Follow the interactive prompts to:
- Select image
- Choose language
- Select OCR mode
- Enable orientation correction
- Enable image comparison

---

## ⚠️ Limitations

- Handwritten OCR accuracy is limited
- Very complex or noisy tables may fail
- Orientation detection relies on known words
- Non-Latin scripts require additional tuning

---

## 🚀 Future Improvements

- GUI version (Tkinter / PyQt)
- PDF OCR support
- Better handwritten recognition using deep learning
- Automatic language detection
- Export tables as CSV / Excel

---

## 👤 Authors

- **Muhammad Inshal Saqib Siddiqui**
- **Muhammad Hamza Latif Khan**
- **Muhammad Abdullah Sohail**
- **Marwan Ali**

---

## 📜 License

This project is licensed under the **MIT License**. See the `LICENSE` file for details.

