import sys

pdf_path = "/home/pablo/Bureau/Master 2/Memoire/dist2/distributed_learning_2/distributed_learning/Xie_JointSQ_Joint_Sparsification-Quantization_for_Distributed_Learning_CVPR_2024_paper.pdf"
out_path = "/home/pablo/Bureau/Master 2/Memoire/dist2/distributed_learning_2/distributed_learning/scratch/jointsq_text.txt"

# Try pypdf
try:
    import pypdf
    reader = pypdf.PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text() or ""
    with open(out_path, "w") as f:
        f.write(text)
    print("Success with pypdf")
    sys.exit(0)
except Exception as e:
    print("pypdf failed:", e)

# Try PyPDF2
try:
    import PyPDF2
    reader = PyPDF2.PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text() or ""
    with open(out_path, "w") as f:
        f.write(text)
    print("Success with PyPDF2")
    sys.exit(0)
except Exception as e:
    print("PyPDF2 failed:", e)

# Try pdfminer
try:
    from pdfminer.high_level import extract_text
    text = extract_text(pdf_path)
    with open(out_path, "w") as f:
        f.write(text)
    print("Success with pdfminer")
    sys.exit(0)
except Exception as e:
    print("pdfminer failed:", e)

# Try fitz (PyMuPDF)
try:
    import fitz
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
    with open(out_path, "w") as f:
        f.write(text)
    print("Success with fitz")
    sys.exit(0)
except Exception as e:
    print("fitz failed:", e)

print("All PDF extraction libraries failed")
