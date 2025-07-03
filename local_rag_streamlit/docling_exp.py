from docling.document_converter import DocumentConverter

FILE_PATH = "CBP-7735.pdf"
converter = DocumentConverter()
result = converter.convert(FILE_PATH)
print(result.document.export_to_markdown())
