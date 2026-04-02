import pytesseract
from PIL import Image, ImageDraw

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

img = Image.new("RGB", (250, 70), "white")
d = ImageDraw.Draw(img)
d.text((10, 10), "HELLO OCR", fill="black")

print(pytesseract.image_to_string(img))
