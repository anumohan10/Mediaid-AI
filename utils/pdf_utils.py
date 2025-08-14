# utils/pdf_utils.py
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib.units import inch

def text_to_pdf(text: str, out_path: str):
    c = canvas.Canvas(out_path, pagesize=letter)
    width, height = letter
    x, y = inch * 0.75, height - inch * 0.75
    max_w = width - inch * 1.5
    line_height = 14

    for raw in text.splitlines():
        line = raw
        while line:
            # simple wrap by characters
            i = len(line)
            while i and c.stringWidth(line[:i]) > max_w:
                i -= 1
            draw = line if i == len(line) else line[:i]
            c.drawString(x, y, draw)
            y -= line_height
            line = "" if i == len(line) else line[i:]
            if y < inch:
                c.showPage(); y = height - inch * 0.75
    c.save()
