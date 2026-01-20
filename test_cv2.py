import cv2
import pytesseract
import numpy as np
from PIL import Image
import os
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt
from rich.status import Status
from rich.table import Table as RichTable
import re
import tempfile
from img2table.document import Image as Img2TableImage
from img2table.ocr import TesseractOCR

console = Console()
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

def ocr_preprocess(image):
    """
    Final OCR preprocessing:
    - 2x upscale
    - grayscale
    - gaussian blur
    - adaptive threshold (B/W)
    """
    img = cv2.resize(image, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)

    thresh = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31, 11
    )

    return thresh


def show_preprocessed_image(cv2_img, title="Preprocessed Image"):
    """
    Displays the final preprocessed image to the user.
    Works reliably on Windows.
    """
    if not isinstance(cv2_img, np.ndarray):
        return

    # Handle both grayscale and color images
    if len(cv2_img.shape) == 2:
        # Grayscale image
        pil_img = Image.fromarray(cv2_img)
    else:
        # Convert BGR → RGB
        rgb = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb)

    # Save temp image so OS viewer opens it
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
    pil_img.save(tmp.name)
    tmp.close()

    # Open with default image viewer
    os.startfile(tmp.name)

# IMG2TABLE TABLE EXTRACTOR

class Img2TableExtractor:
    def _init_(self, language="eng"):
        self.language = language
        self.ocr_backend = TesseractOCR(lang=language)

    def extract(self, image):
        """
        image: OpenCV image (numpy array or file path)
        returns: list of rows (list of lists of strings)
        """
        if isinstance(image, str):
            img_array = cv2.imread(image)
        else:
            img_array = image.copy()
        
        # Convert BGR -> RGB
        img_rgb = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
        document = Img2TableImage(img_rgb)

        #Extract tables using latest API
        tables = document.extract_tables(
            ocr=self.ocr_backend,
            implicit_rows=False,
            implicit_columns=False,
            borderless_tables=False,
            min_confidence=50
        )

        if not tables:
            return []

        # Use the first table
        table = tables[0].df.fillna("").values.tolist()
        return table

# SMART ORIENTATION CORRECTOR

class SmartOrientationCorrector:
    COMMON_WORDS = {
        'the', 'be', 'to', 'of', 'and', 'a', 'in', 'that', 'have', 'i',
        'it', 'for', 'not', 'on', 'with', 'he', 'as', 'you', 'do', 'at',
        'this', 'but', 'his', 'by', 'from', 'they', 'we', 'say', 'her', 'she',
        'or', 'an', 'will', 'my', 'one', 'all', 'would', 'there', 'their', 'what',
        'so', 'up', 'out', 'if', 'about', 'who', 'get', 'which', 'go', 'me',
        'when', 'make', 'can', 'like', 'time', 'no', 'just', 'him', 'know', 'take',
        'is', 'was', 'are', 'hello', 'name', 'yes', 'why', 'join', 'us', 'write',
        'example', 'good', 'birthday', 'invited', 'bash', 'celebrating'
    }
    
    @staticmethod
    def score_orientation(img, language="eng"):
        """Score based on REAL WORDS"""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        text = pytesseract.image_to_string(gray, lang=language, config='--oem 3 --psm 6')
        
        if not text.strip():
            return 0
        
        words = text.lower().split()
        valid_words = sum(1 for w in words if re.sub(r'[^\w]', '', w) in SmartOrientationCorrector.COMMON_WORDS)
        score = valid_words * 100 + len(words) * 5
        
        return score
    
    @staticmethod
    def fix(image, language="eng"):
        console.print("[yellow]Testing orientations...[/yellow]")
        
        versions = {
            0: image.copy(),
            90: cv2.rotate(image.copy(), cv2.ROTATE_90_CLOCKWISE),
            180: cv2.rotate(image.copy(), cv2.ROTATE_180),
            270: cv2.rotate(image.copy(), cv2.ROTATE_90_COUNTERCLOCKWISE)
        }
        
        results = []
        for angle, img in versions.items():
            score = SmartOrientationCorrector.score_orientation(img, language)
            results.append((angle, img, score))
            console.print(f"[dim]{angle:3d}°: score={score}[/dim]")
        
        best_angle, best_img, best_score = max(results, key=lambda x: x[2])
        
        if best_score < 50:
            console.print("[yellow]⚠ Low confidence, keeping original[/yellow]")
            return image, 0
        
        console.print(f"[green]✓ Selected: {best_angle}° (score: {best_score})[/green]")
        return best_img, best_angle


class ImprovedHandwrittenOCR:
    def _init_(self, image, language="eng"):
        if isinstance(image, str):
            self.image = cv2.imread(image)
        else:
            self.image = image.copy()
        self.language = language
        self.preprocessed_img = None
    
    def extract(self):
        console.print("[cyan]Processing handwritten text...[/cyan]")

        results = []
        preprocessed_images = []

        # -------------------------------
        # Method 1: Reliable 2x upscale
        # -------------------------------
        img1 = cv2.resize(self.image, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
        gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        gray1 = cv2.GaussianBlur(gray1, (3, 3), 0)
        thresh1 = cv2.adaptiveThreshold(
            gray1, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 31, 11
        )
        text1 = pytesseract.image_to_string(thresh1, lang=self.language, config='--oem 1 --psm 6')
        results.append(text1.strip())
        preprocessed_images.append(thresh1)
        console.print(f"[dim]Method 1: {len(text1.strip())} chars[/dim]")

        # Method 2: 3x upscale + contrast

        img2 = cv2.resize(self.image, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
        gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.GaussianBlur(gray2, (3, 3), 0)
        gray2 = cv2.convertScaleAbs(gray2, alpha=1.3, beta=0)
        thresh2 = cv2.adaptiveThreshold(
            gray2, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 31, 11
        )
        text2 = pytesseract.image_to_string(thresh2, lang=self.language, config='--oem 1 --psm 6')
        results.append(text2.strip())
        preprocessed_images.append(thresh2)
        console.print(f"[dim]Method 2: {len(text2.strip())} chars[/dim]")

        # Pick the longest text (most characters)
     
        best_idx = max(range(len(results)), key=lambda i: len(results[i]))
        best = results[best_idx]
        self.preprocessed_img = preprocessed_images[best_idx]
        console.print(f"[green]✓ Best result: {len(best)} characters (Method {best_idx + 1})[/green]")

        return best

# FixedTableExtractor

class FixedTableExtractor:
    def _init_(self, image):
        if isinstance(image, str):
            self.image = cv2.imread(image)
        else:
            self.image = image.copy()
        self.preprocessed_img = None

    def extract(self, num_columns=None):
        """OCR-only table extraction using y-coordinate clustering (borderless safe)"""
        gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        
        # Store preprocessed version
        self.preprocessed_img = ocr_preprocess(self.image)
        
        data = pytesseract.image_to_data(gray, output_type=pytesseract.Output.DICT, config='--oem 3 --psm 6')

        words = []
        for i, txt in enumerate(data['text']):
            txt = txt.strip()
            if txt:
                words.append({'text': txt, 'x': data['left'][i], 'y': data['top'][i], 'cx': data['left'][i]+data['width'][i]//2})

        if not words:
            return []

        # Group words by rows
        words.sort(key=lambda w: w['y'])
        rows = []
        current_row = [words[0]]
        for w in words[1:]:
            if abs(w['y'] - current_row[0]['y']) < 30:
                current_row.append(w)
            else:
                rows.append(current_row)
                current_row = [w]
        rows.append(current_row)

        # Assign words to columns
        if num_columns is None:
            num_columns = max(1, max(len(r) for r in rows))

        table = []
        for row in rows:
            row_data = [''] * num_columns
            for idx, w in enumerate(row):
                col_idx = min(idx, num_columns-1)
                row_data[col_idx] = w['text'] if not row_data[col_idx] else row_data[col_idx] + ' ' + w['text']
            table.append(row_data)
        return table

# TEXT EXTRACTOR

class TextExtractor:
    def _init_(self, image, language="eng"):
        if isinstance(image, str):
            self.image = cv2.imread(image)
        else:
            self.image = image.copy()
        self.language = language
        self.preprocessed_img = None
    
    def extract(self):
        img = cv2.resize(self.image, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        thresh = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 31, 11
        )
        
        # Store preprocessed version
        self.preprocessed_img = thresh
        
        text = pytesseract.image_to_string(thresh, lang=self.language, config='--oem 3 --psm 6')
        return text.strip()

# FIXED MIXED CONTENT HANDLER

class FixedMixedContentHandler:
    def _init_(self, image, language="eng"):
        if isinstance(image, str):
            self.image = cv2.imread(image)
        else:
            self.image = image.copy()
        self.language = language
        self.preprocessed_img = None
    
    def detect_table_region(self):
        gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)

        # Detect lines
        h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (50, 1))
        v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 50))
        h_lines = cv2.morphologyEx(edges, cv2.MORPH_OPEN, h_kernel, iterations=2)
        v_lines = cv2.morphologyEx(edges, cv2.MORPH_OPEN, v_kernel, iterations=2)
        table_mask = cv2.add(h_lines, v_lines)
        table_mask = cv2.dilate(table_mask, cv2.getStructuringElement(cv2.MORPH_RECT, (3,3)), iterations=2)

        # Count how many pixels are part of lines
        line_density = np.sum(table_mask > 0) / (self.image.shape[0]*self.image.shape[1])
        if line_density < 0.005:
            return None

        # Contour detection
        contours, _ = cv2.findContours(table_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None

        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)
        area = w*h
        img_area = self.image.shape[0]*self.image.shape[1]

        # Require both size and line density
        if area > img_area*0.05 and line_density > 0.01:
            return {'x': x, 'y': y, 'w': w, 'h': h}

        return None

    
    def extract_mixed(self, num_columns=None):
        result = {'text_above': None, 'table': None, 'text_below': None, 'full_text': None}
        table_region = self.detect_table_region()
        
        # Store preprocessed version of full image
        self.preprocessed_img = ocr_preprocess(self.image)
        
        if table_region:
            x, y, w, h = table_region['x'], table_region['y'], table_region['w'], table_region['h']
            padding = 10
            y_start = max(0, y-padding)
            y_end = min(self.image.shape[0], y+h+padding)
            table_img = self.image[y_start:y_end, x:x+w]
            
            # Run Img2Table extractor
            table = Img2TableExtractor(self.language).extract(table_img)
            if table:
                result['table'] = table
            
            # Text above
            if y_start>10:
                text_above_img = self.image[0:y_start, :]
                result['text_above'] = TextExtractor(text_above_img, self.language).extract()
            
            # Text below
            if y_end < self.image.shape[0]-10:
                text_below_img = self.image[y_end:, :]
                result['text_below'] = TextExtractor(text_below_img, self.language).extract()
        else:
            # No table region → treat as pure text
            result['text_above'] = TextExtractor(self.image, self.language).extract()
        
        # Full text
        result['full_text'] = TextExtractor(self.image, self.language).extract()
        return result


# CONTENT DETECTOR


class ContentDetector:
    @staticmethod
    def detect(image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Table signal
        edges = cv2.Canny(gray, 50, 150)
        lines = cv2.HoughLinesP(
            edges, 1, np.pi/180,
            threshold=100,
            minLineLength=60,
            maxLineGap=10
        )

        # Text signal
        text = pytesseract.image_to_string(gray, config="--psm 6").strip()

        has_table = lines is not None and len(lines) > 8
        has_text = len(text) > 30

        if has_table and has_text:
            return "mixed"
        if has_table:
            return "table"
        return "text"

# MAIN PROCESSOR

class OCRProcessor:
    def _init_(self, image_path, language="eng"):
        self.image_path = image_path
        self.language = language
        self.final_preprocessed_img = None
    
    def process(self, mode="auto", num_columns=None, fix_orientation=True, save_comparison=False):
        img_original = cv2.imread(self.image_path)
        if img_original is None:
            raise FileNotFoundError(f"Image not found: {self.image_path}")
        
        img = img_original.copy()
        
        # Save original for comparison
        if save_comparison:
            cv2.imwrite("original_image.png", img_original)
            console.print("[dim]Saved: original_image.png[/dim]")
        
        if fix_orientation:
            console.print("[bold cyan]═══ ORIENTATION CHECK ═══[/bold cyan]")
            img, angle = SmartOrientationCorrector.fix(img, self.language)
            if save_comparison and angle != 0:
                cv2.imwrite("corrected_orientation.png", img)
                console.print("[dim]Saved: corrected_orientation.png[/dim]")
        
        if mode == "auto":
            mode = ContentDetector.detect(img)
        
        console.print(f"[bold cyan]═══ MODE: {mode.upper()} ═══[/bold cyan]\n")
        
        result = {'mode': mode, 'text': None, 'table': None, 'text_above': None, 'text_below': None}
        
        if mode == "handwritten":
            hw = ImprovedHandwrittenOCR(img, self.language)
            result['text'] = hw.extract()
            self.final_preprocessed_img = hw.preprocessed_img
            
        elif mode == "mixed":
            handler = FixedMixedContentHandler(img, self.language)
            mixed = handler.extract_mixed(num_columns)
            result.update(mixed)
            self.final_preprocessed_img = handler.preprocessed_img
            
        elif mode == "table":
            extractor = FixedTableExtractor(img)
            result['table'] = extractor.extract(num_columns)
            txt_ext = TextExtractor(img, self.language)
            result['text'] = txt_ext.extract()
            self.final_preprocessed_img = extractor.preprocessed_img
            
        else:  # text
            txt_ext = TextExtractor(img, self.language)
            result['text'] = txt_ext.extract()
            self.final_preprocessed_img = txt_ext.preprocessed_img
        
        return result

# APP

class OCRApp:
    LANGUAGES = {
        "1": ("English", "eng"),
        "2": ("Spanish", "spa"),
        "3": ("French", "fra"),
        "4": ("German", "deu")
    }
    
    def run(self):
        console.print(Panel(
            "[bold cyan]COMPLETE OCR SYSTEM - FINAL VERSION[/bold cyan]\n"
            "✓ Smart orientation (word-based)\n"
            "✓ Improved handwritten (includes standard OCR)\n"
            "✓ Fixed table extraction (lower threshold)\n"
            "✓ Auto column detection (dual method)\n"
            "✓ Fixed mixed content (better detection)\n"
            "✓ Comparison images (see original)\n"
            "✓ Multi-language support\n"
            "✓ Preprocessed image display",
            expand=False
        ))
        
        while True:
            path = Prompt.ask("\nImage path").strip('"')
            if os.path.exists(path):
                console.print("[green]✓ Opening image...[/green]")
                os.startfile(path)
                break
            console.print("[red]File not found[/red]")
        
        console.print("\n[bold]Languages:[/bold]")
        for k, (name, _) in self.LANGUAGES.items():
            console.print(f"  {k}. {name}")
        
        lang_choice = Prompt.ask("Language", choices=self.LANGUAGES.keys(), default="1")
        language = self.LANGUAGES[lang_choice][1]
        
        console.print("\n[bold]Modes:[/bold]")
        console.print("  1. Auto-detect")
        console.print("  2. Table")
        console.print("  3. Text")
        console.print("  4. Handwritten")
        console.print("  5. Mixed (Text + Table)")
        
        mode_choice = Prompt.ask("Mode", choices=["1", "2", "3", "4", "5"], default="1")
        modes = {"1": "auto", "2": "table", "3": "text", "4": "handwritten", "5": "mixed"}
        mode = modes[mode_choice]
        
        num_cols = None
        if mode in ["table", "mixed"]:
            auto = Prompt.ask("Auto-detect columns?", choices=["y", "n"], default="y")
            if auto == "n":
                num_cols = int(Prompt.ask("Number of columns", default="6"))
        
        fix_orient = Prompt.ask("Check orientation?", choices=["y", "n"], default="y") == "y"
        save_comparison = Prompt.ask("Save comparison images?", choices=["y", "n"], default="y") == "y"
        
        console.print()
        with Status("[green]Processing...[/green]"):
            processor = OCRProcessor(path, language)
            result = processor.process(
                mode=mode, 
                num_columns=num_cols, 
                fix_orientation=fix_orient,
                save_comparison=save_comparison
            )
        
        console.print()
        
        # Display results
        if result['text_above']:
            console.print("[yellow]📄 TEXT ABOVE TABLE:[/yellow]")
            console.print(Panel(result['text_above'][:300], expand=False))
        
        if result['table']:
            display = RichTable(show_header=False, show_lines=True, title="📊 TABLE")
            
            max_cols = len(result['table'][0])
            for i in range(max_cols):
                display.add_column(f"Col{i+1}", style="cyan", overflow="fold", max_width=30)
            
            for row in result['table']:
                display.add_row(*row)
            
            console.print(display)
            console.print(f"[green]✓ {len(result['table'])} rows × {max_cols} columns[/green]\n")
        
        if result['text_below']:
            console.print("[yellow]📄 TEXT BELOW TABLE:[/yellow]")
            console.print(Panel(result['text_below'][:300], expand=False))
        
        if result['text'] and not result['table']:
            console.print("[yellow]📄 TEXT:[/yellow]")
            preview = result['text'][:500] + ("..." if len(result['text']) > 500 else "")
            console.print(Panel(preview, expand=False))
        
        # Save files
        base = os.path.splitext(os.path.basename(path))[0]
        saved = []
        
        if result['table']:
            f = f"{base}_table.txt"
            with open(f, 'w', encoding='utf-8') as file:
                for row in result['table']:
                    file.write(' | '.join(row) + '\n')
            saved.append(f)
        
        if result['text_above']:
            f = f"{base}_text_above.txt"
            with open(f, 'w', encoding='utf-8') as file:
                file.write(result['text_above'])
            saved.append(f)
        
        if result['text_below']:
            f = f"{base}_text_below.txt"
            with open(f, 'w', encoding='utf-8') as file:
                file.write(result['text_below'])
            saved.append(f)
        
        if result['text']:
            f = f"{base}_text.txt"
            with open(f, 'w', encoding='utf-8') as file:
                file.write(result['text'])
            saved.append(f)
        
        if result.get('full_text'):
            f = f"{base}_full_text.txt"
            with open(f, 'w', encoding='utf-8') as file:
                file.write(result['full_text'])
            saved.append(f)
        
        if saved:
            console.print(f"\n[green]✓ Saved: {', '.join(saved)}[/green]")
        
        # Show comparison images info
        if save_comparison:
            console.print("\n[bold cyan]Comparison Images:[/bold cyan]")
            console.print("  • original_image.png - Your uploaded image")
            if fix_orient:
                console.print("  • corrected_orientation.png - After rotation (if rotated)")
            if mode == "mixed":
                console.print("  • debug_table_region.png - Detected table region")
        
        # Show final preprocessed image (what OCR actually saw)
        if hasattr(processor, "final_preprocessed_img") and processor.final_preprocessed_img is not None:
            console.print("\n[bold green]Opening preprocessed image (what OCR actually read)...[/bold green]")
            show_preprocessed_image(processor.final_preprocessed_img, "Final Preprocessed Image")
        
        if saved:
            os.startfile(saved[0])


if __name__ == "_main_":
    try:
        app = OCRApp()
        app.run()
    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted[/yellow]")
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        import traceback
        traceback.print_exc()