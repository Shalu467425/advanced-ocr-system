# 📄 OCR Terminal Application (OOP Lab Project)

## Overview
This project is a **command-line based Optical Character Recognition (OCR) system** implemented in **Python** using **Object-Oriented Programming (OOP)** principles. The application allows users to:

- Select an image file from the system
- Choose an OCR language interactively
- Preprocess the image for better accuracy
- Extract text using **Tesseract OCR**
- Display results in a rich, formatted terminal interface
- Save OCR output to a text file automatically

The project is designed specifically for **academic lab evaluation**, demonstrating clean OOP design, modularity, and integration of third‑party libraries.

---

## Key Features

- 📌 Object-Oriented design (separation of logic & UI)
- 🖥️ Interactive CLI using **Rich**
- 🌍 Multi-language OCR support
- 🧠 Image preprocessing with OpenCV
- 💾 Auto-saving OCR output to file
- ⚠️ Robust error handling

---

## Technologies & Libraries Used

| Component | Purpose |
|---------|--------|
| **Python 3.11** | Core programming language |
| **OpenCV (cv2)** | Image preprocessing |
| **NumPy** | Image matrix manipulation |
| **Pillow (PIL)** | Image format conversion |
| **pytesseract** | Python wrapper for Tesseract OCR |
| **Tesseract OCR Engine** | Actual OCR engine |
| **Rich** | Styled terminal UI |
| **OS Module** | File handling and system interaction |

---

## Project Structure

```
OCR/
│
├── main.py               # Main application file
├── requirements.txt      # Python dependencies
├── .venv/                # Virtual environment
├── OCR_Result.txt        # Auto-generated output file
└── README.md             # Project documentation
```

---

## How the Application Works

The application is divided into **two main classes**, each with a distinct responsibility:

### 1️⃣ `OCRProcessor` — Core Logic Class

This class handles **image processing and text extraction**.

**Responsibilities:**
- Load image from disk
- Preprocess image for OCR
- Invoke Tesseract OCR
- Return extracted text

**Workflow:**
1. Image is loaded using OpenCV
2. Image is upscaled for better recognition
3. Converted to grayscale
4. Noise is reduced using Gaussian Blur
5. Adaptive thresholding is applied
6. Processed image is passed to Tesseract

---

### 2️⃣ `OCRTerminalApp` — User Interface Class

This class manages **user interaction and program flow**.

**Responsibilities:**
- Display program header
- Prompt user for image path
- Show language selection menu
- Execute OCR process
- Display and save results

The UI is implemented using the **Rich** library to enhance readability and usability in the terminal.

---

## Supported Languages

| Language | Tesseract Code |
|--------|---------------|
| English | `eng` |
| Spanish | `spa` |
| French  | `fra` |
| German  | `deu` |

⚠️ Corresponding language packs must be installed in Tesseract.

---

## How to Run the Project

### Step 1: Install Python
Ensure **Python 3.11** is installed and added to PATH.

### Step 2: Install Tesseract OCR
- Download from official source
- Install to default location:
  ```
  C:\Program Files\Tesseract-OCR\
  ```

### Step 3: Create Virtual Environment

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### Step 4: Install Dependencies

```powershell
pip install -r requirements.txt
```

### Step 5: Run the Application

```powershell
python main.py
```

---

## Output

- Extracted text is:
  - Displayed in terminal
  - Saved to `OCR_Result.txt`
  - Automatically opened after completion

---

## Important Terminologies Explained

### OCR (Optical Character Recognition)
Technology used to convert images containing text into machine-readable text.

### Tesseract OCR
An open-source OCR engine developed by Google, responsible for actual text recognition.

### pytesseract
A Python wrapper that allows Python programs to communicate with Tesseract.

### OpenCV
A computer vision library used here for image preprocessing to improve OCR accuracy.

### Adaptive Thresholding
A technique that converts grayscale images to binary images by calculating thresholds locally, improving recognition under uneven lighting.

### Gaussian Blur
A noise-reduction technique that smooths images before thresholding.

### PSM (Page Segmentation Mode)
Controls how Tesseract splits the image into text blocks.
- `PSM 6`: Assumes a single uniform block of text

### OEM (OCR Engine Mode)
Specifies which OCR engine Tesseract should use.
- `OEM 3`: Default LSTM-based engine

### Rich Library
A Python library that enhances terminal output using colors, tables, panels, and status indicators.

---

## OOP Concepts Demonstrated

- ✅ Encapsulation
- ✅ Separation of concerns
- ✅ Modular class design
- ✅ Reusability
- ✅ Clean abstraction

---

## Academic Relevance

This project is suitable for:
- OOP Lab Assessment
- Python Programming Lab
- Image Processing Fundamentals
- Software Engineering Demonstration

---

## Author

- **Muhammad Inshal Saqib Siddiqui**  
- **Muhammad Hamza Latif Khan**  
- **Muhammad Abdullah Sohail**  
- **Marwan Ali**

---

## Limitations

- OCR accuracy depends heavily on image quality and resolution
- Handwritten text is not supported
- Non‑Latin scripts require language‑specific preprocessing
- Lighting variations may affect thresholding results

---

## Future Improvements

- GUI version using Tkinter or PyQt
- Batch OCR for multiple images
- PDF document OCR support
- Automatic language detection
- Advanced preprocessing for Arabic/Urdu scripts

---

## License

This project is released under the **MIT License**. See the `LICENSE` file for details.

