from tkinter import *
from tkinter import filedialog
import PyPDF2
import fitz 
import os

root = Tk()
root.geometry("1000x750")

# Cadre pour les boutons et la zone de texte
control_frame = Frame(root)
control_frame.pack(pady=10)

# Zone de texte pour afficher le texte extrait du PDF
my_text = Text(root, height=10, width=80)
my_text.pack(pady=10)

# Cadre pour afficher le PDF 
pdf_frame = Frame(root, width=800, height=700)
pdf_frame.pack(pady=10)

canvas = Canvas(pdf_frame)
scroll_y = Scrollbar(pdf_frame, orient="vertical", command=canvas.yview)
scroll_x = Scrollbar(pdf_frame, orient="horizontal", command=canvas.xview)
canvas.configure(yscrollcommand=scroll_y.set, xscrollcommand=scroll_x.set)
scroll_y.pack(side="right", fill="y")
scroll_x.pack(side="bottom", fill="x")
canvas.pack(side="left", expand=True, fill="both")


# Fonction pour ouvrir et afficher un PDF
def open_pdf():
    open_file = filedialog.askopenfilename(
        initialdir=os.getcwd(),
        title="Open PDF File",
        filetypes=(("PDF Files", "*.pdf"), ("All Files", "*.*"))
    )
    if open_file:
        # Affichage du PDF
        display_pdf(open_file)

        # Extraction du texte du PDF
        extract_text(open_file)


def display_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    page_count = doc.page_count

    for i in range(page_count):
        page = doc.load_page(i)
        pix = page.get_pixmap()
        img_path = f"page_{i}.png"
        pix.save(img_path)

        img = PhotoImage(file=img_path)
        canvas.create_image(0, i * pix.height, anchor="nw", image=img)
        canvas.image = img

    canvas.config(scrollregion=canvas.bbox("all"))

# Fonction pour l'extraction du texte
def extract_text(pdf_path):
    with open(pdf_path, "rb") as f:
        pdf_reader = PyPDF2.PdfReader(f)
        all_text = ""
        for page_num in range(len(pdf_reader.pages)):
            page = pdf_reader.pages[page_num]
            page_text = page.extract_text()
            all_text += page_text

        my_text.delete(1.0, END) 
        my_text.insert(1.0, all_text) 


# Bouton
Button(control_frame, text="Open", command=open_pdf, width=40, bd=4, bg="purple").pack()

root.mainloop()