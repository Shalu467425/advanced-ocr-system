import cv2
import pytesseract
import numpy as np
from PIL import Image
import os
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt
from rich.status import Status
from rich.table import Table

console = Console()

# =========================================================
# CLASS 1: OCRProcessor (Logic)
# =========================================================
class OCRProcessor:
    def __init__(self, image_path, language="eng"):
        self.image_path = image_path
        self.language = language
        pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

    def process_image(self):
        """Simple preprocessing for Latin-based text."""
        img = cv2.imread(self.image_path)
        if img is None:
            raise FileNotFoundError(f"Could not load image at {self.image_path}")

        # Upscale 2x (sufficient for Latin text)
        img = cv2.resize(img, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (3, 3), 0)

        # Adaptive threshold
        thresh = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY, 31, 11
        )

        return thresh

    def get_text(self):
        processed_img = self.process_image()
        pil_img = Image.fromarray(cv2.cvtColor(processed_img, cv2.COLOR_GRAY2RGB))

        # PSM 6: Uniform block
        custom_config = "--oem 3 --psm 6 -c preserve_interword_spaces=1"
        text = pytesseract.image_to_string(pil_img, lang=self.language, config=custom_config)
        return text.strip()

# =========================================================
# CLASS 2: OCRTerminalApp (UI)
# =========================================================
class OCRTerminalApp:
    def __init__(self):
        self.selected_path = ""
        self.selected_lang = "eng"

    def display_header(self):
        console.print(Panel(
            "[bold cyan]SIMPLE OCR SYSTEM[/bold cyan]\n"
            "[white]Languages: English, Spanish, French, German[/white]", 
            expand=False
        ))

    def get_user_input(self):
        while True:
            path = Prompt.ask("Enter image path").strip('"')
            if os.path.exists(path):
                self.selected_path = path
                break
            console.print("[red]Error: File not found.[/red]")

        # Language menu
        lang_table = Table(title="Available Languages")
        lang_table.add_column("ID", style="cyan", justify="center")
        lang_table.add_column("Language", style="green")

        lang_options = {
            "1": ("English", "eng"),
            "2": ("Spanish", "spa"),
            "3": ("French", "fra"),
            "4": ("German", "deu")
        }

        for key, (name, _) in lang_options.items():
            lang_table.add_row(key, name)
        
        console.print(lang_table)

        choice = Prompt.ask("Select Language ID", choices=lang_options.keys())
        self.selected_lang = lang_options[choice][1]

    def start_process(self):
        try:
            with Status(f"[bold green]Processing {self.selected_lang.upper()}...[/bold green]"):
                processor = OCRProcessor(self.selected_path, self.selected_lang)
                result_text = processor.get_text()

            console.print("\n[bold yellow]OCR RESULT:[/bold yellow]")
            console.print(Panel(result_text if result_text else "[No text detected]"))

            # Save and auto-open
            filename = "OCR_Result.txt"
            with open(filename, "w", encoding="utf-8") as f:
                f.write(result_text)
            
            console.print(f"\n[green]Saved to {filename}. Opening now...[/green]")
            os.startfile(filename)

        except Exception as e:
            console.print(f"[bold red]System Error:[/bold red] {e}")

    def run(self):
        self.display_header()
        self.get_user_input()
        self.start_process()

if __name__ == "__main__":
    app = OCRTerminalApp()
    app.run()
