from pypdf import PdfReader

def text_extractor_pdf(file_path):
    try:
        pdf_file = PdfReader(file_path)
        pdf_text = []
        
        for page in pdf_file.pages:
            text_only = page.extract_text()
            if text_only:
                pdf_text.append(text_only)
        
        # Join with newlines for readability
        return "\n".join(pdf_text)
    
    except Exception as e:
        print(f"Error reading PDF: {e}")
        return ""