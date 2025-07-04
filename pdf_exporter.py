# utils/exporters.py
from reportlab.lib.pagesizes import letter, A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
import io
from datetime import datetime

class PDFExporter:
    def __init__(self):
        self.styles = getSampleStyleSheet()
    
    def create_text_export(self, summary, flashcards, quiz):
        """Create text export of all study materials"""
        content = f"Study Notes - Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        content += "=" * 50 + "\n\n"
        
        if summary:
            content += "SUMMARY:\n"
            content += "-" * 20 + "\n"
            content += summary + "\n\n"
        
        if flashcards:
            content += "FLASHCARDS:\n"
            content += "-" * 20 + "\n"
            for i, card in enumerate(flashcards):
                content += f"Card {i+1}:\n"
                content += f"Q: {card.get('question', 'N/A')}\n"
                content += f"A: {card.get('answer', 'N/A')}\n\n"
        
        if quiz:
            content += "QUIZ:\n"
            content += "-" * 20 + "\n"
            for i, q in enumerate(quiz):
                content += f"Question {i+1}: {q.get('question', 'N/A')}\n"
                for j, option in enumerate(q.get('options', [])):
                    content += f"  {chr(65+j)}) {option}\n"
                content += f"Answer: {q.get('correct_answer', 'N/A')}\n\n"
        
        return content
    
    def create_pdf_export(self, summary, flashcards, quiz):
        """Create PDF export of study materials"""
        try:
            buffer = io.BytesIO()
            doc = SimpleDocTemplate(buffer, pagesize=A4)
            elements = []
            
            # Title
            title = Paragraph("Study Notes", self.styles['Title'])
            elements.append(title)
            elements.append(Spacer(1, 20))
            
            # Summary
            if summary:
                summary_title = Paragraph("Summary", self.styles['Heading1'])
                elements.append(summary_title)
                summary_para = Paragraph(summary, self.styles['Normal'])
                elements.append(summary_para)
                elements.append(Spacer(1, 20))
            
            # Flashcards
            if flashcards:
                flashcards_title = Paragraph("Flashcards", self.styles['Heading1'])
                elements.append(flashcards_title)
                
                for i, card in enumerate(flashcards):
                    question = Paragraph(f"Q{i+1}: {card.get('question', 'N/A')}", self.styles['Heading2'])
                    answer = Paragraph(f"A: {card.get('answer', 'N/A')}", self.styles['Normal'])
                    elements.append(question)
                    elements.append(answer)
                    elements.append(Spacer(1, 10))
            
            doc.build(elements)
            return buffer.getvalue()
        except Exception as e:
            return None
    
    def create_flashcard_pdf(self, flashcards):
        """Create printable flashcard PDF"""
        try:
            buffer = io.BytesIO()
            doc = SimpleDocTemplate(buffer, pagesize=A4)
            elements = []
            
            title = Paragraph("Flashcards", self.styles['Title'])
            elements.append(title)
            elements.append(Spacer(1, 30))
            
            for i, card in enumerate(flashcards):
                # Question side
                question = Paragraph(f"Card {i+1} - Question", self.styles['Heading2'])
                question_text = Paragraph(card.get('question', 'N/A'), self.styles['Normal'])
                elements.append(question)
                elements.append(question_text)
                elements.append(Spacer(1, 20))
                
                # Answer side
                answer = Paragraph(f"Card {i+1} - Answer", self.styles['Heading2'])
                answer_text = Paragraph(card.get('answer', 'N/A'), self.styles['Normal'])
                elements.append(answer)
                elements.append(answer_text)
                elements.append(Spacer(1, 30))
            
            doc.build(elements)
            return buffer.getvalue()
        except Exception as e:
            return None
