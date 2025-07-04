'''from fpdf import FPDF
import io
from datetime import datetime

class StudyNotesExporter:
    def __init__(self):
        self.pdf = None
    
    def create_text_export(self, summary=None, flashcards=None, quiz=None):
        """Create a text export of all study materials"""
        content = f"📚 Study Notes - Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        content += "=" * 60 + "\n\n"
        
        if summary:
            content += "📝 SUMMARY\n"
            content += "-" * 20 + "\n"
            content += summary + "\n\n"
        
        if flashcards:
            content += "🗂️ FLASHCARDS\n"
            content += "-" * 20 + "\n"
            for i, card in enumerate(flashcards, 1):
                content += f"Card {i}:\n"
                content += f"Q: {card.get('question', 'N/A')}\n"
                content += f"A: {card.get('answer', 'N/A')}\n\n"
        
        if quiz:
            content += "📋 QUIZ\n"
            content += "-" * 20 + "\n"
            for i, q in enumerate(quiz, 1):
                content += f"Question {i}: {q.get('question', 'N/A')}\n"
                options = q.get('options', [])
                for j, option in enumerate(options):
                    letter = chr(65 + j)  # A, B, C, D
                    content += f"{letter}. {option}\n"
                content += f"Correct Answer: {q.get('correct_answer', 'N/A')}\n"
                content += f"Explanation: {q.get('explanation', 'N/A')}\n\n"
        
        return content
    
    def create_pdf_export(self, summary=None, flashcards=None, quiz=None):
        """Create a PDF export of all study materials"""
        try:
            pdf = FPDF()
            pdf.add_page()
            pdf.set_font("Arial", size=16)
            
            # Title
            pdf.cell(200, 10, txt="Study Notes", ln=True, align='C')
            pdf.set_font("Arial", size=10)
            pdf.cell(200, 10, txt=f"Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", ln=True, align='C')
            pdf.ln(10)
            
            # Summary
            if summary:
                pdf.set_font("Arial", 'B', 14)
                pdf.cell(200, 10, txt="Summary", ln=True)
                pdf.set_font("Arial", size=10)
                
                # Split summary into lines that fit
                lines = self._split_text(summary, 80)
                for line in lines:
                    pdf.cell(200, 6, txt=line.encode('latin-1', 'replace').decode('latin-1'), ln=True)
                pdf.ln(5)
            
            # Flashcards
            if flashcards:
                pdf.set_font("Arial", 'B', 14)
                pdf.cell(200, 10, txt="Flashcards", ln=True)
                pdf.set_font("Arial", size=10)
                
                for i, card in enumerate(flashcards, 1):
                    pdf.set_font("Arial", 'B', 10)
                    pdf.cell(200, 6, txt=f"Card {i}:", ln=True)
                    pdf.set_font("Arial", size=10)
                    
                    # Question
                    q_lines = self._split_text(f"Q: {card.get('question', 'N/A')}", 80)
                    for line in q_lines:
                        pdf.cell(200, 6, txt=line.encode('latin-1', 'replace').decode('latin-1'), ln=True)
                    
                    # Answer
                    a_lines = self._split_text(f"A: {card.get('answer', 'N/A')}", 80)
                    for line in a_lines:
                        pdf.cell(200, 6, txt=line.encode('latin-1', 'replace').decode('latin-1'), ln=True)
                    pdf.ln(3)
            
            # Quiz
            if quiz:
                pdf.add_page()  # New page for quiz
                pdf.set_font("Arial", 'B', 14)
                pdf.cell(200, 10, txt="Quiz", ln=True)
                pdf.set_font("Arial", size=10)
                
                for i, q in enumerate(quiz, 1):
                    pdf.set_font("Arial", 'B', 10)
                    pdf.cell(200, 6, txt=f"Question {i}:", ln=True)
                    pdf.set_font("Arial", size=10)
                    
                    # Question text
                    q_lines = self._split_text(q.get('question', 'N/A'), 80)
                    for line in q_lines:
                        pdf.cell(200, 6, txt=line.encode('latin-1', 'replace').decode('latin-1'), ln=True)
                    
                    # Options
                    options = q.get('options', [])
                    for j, option in enumerate(options):
                        letter = chr(65 + j)  # A, B, C, D
                        opt_lines = self._split_text(f"{letter}. {option}", 75)
                        for line in opt_lines:
                            pdf.cell(200, 6, txt=line.encode('latin-1', 'replace').decode('latin-1'), ln=True)
                    
                    # Answer and explanation
                    pdf.cell(200, 6, txt=f"Answer: {q.get('correct_answer', 'N/A')}", ln=True)
                    exp_lines = self._split_text(f"Explanation: {q.get('explanation', 'N/A')}", 80)
                    for line in exp_lines:
                        pdf.cell(200, 6, txt=line.encode('latin-1', 'replace').decode('latin-1'), ln=True)
                    pdf.ln(5)
            
            # Return PDF as bytes
            return pdf.output(dest='S').encode('latin-1')
        
        except Exception as e:
            print(f"Error creating PDF: {e}")
            return None
    
    def _split_text(self, text, max_length):
        """Split text into lines that fit within the specified length"""
        words = text.split()
        lines = []
        current_line = ""
        
        for word in words:
            if len(current_line + " " + word) <= max_length:
                current_line += " " + word if current_line else word
            else:
                if current_line:
                    lines.append(current_line)
                current_line = word
        
        if current_line:
            lines.append(current_line)
        
        return lines
    
    def create_flashcard_pdf(self, flashcards):
        """Create a PDF specifically for flashcards with a nice layout"""
        try:
            pdf = FPDF()
            
            for i, card in enumerate(flashcards):
                # Question page
                pdf.add_page()
                pdf.set_font("Arial", 'B', 20)
                pdf.cell(200, 30, txt=f"Flashcard {i+1}", ln=True, align='C')
                
                pdf.set_font("Arial", 'B', 16)
                pdf.cell(200, 20, txt="Question:", ln=True, align='C')
                
                pdf.set_font("Arial", size=14)
                q_lines = self._split_text(card.get('question', 'N/A'), 60)
                for line in q_lines:
                    pdf.cell(200, 10, txt=line.encode('latin-1', 'replace').decode('latin-1'), ln=True, align='C')
                
                # Answer page
                pdf.add_page()
                pdf.set_font("Arial", 'B', 20)
                pdf.cell(200, 30, txt=f"Flashcard {i+1} - Answer", ln=True, align='C')
                
                pdf.set_font("Arial", 'B', 16)
                pdf.cell(200, 20, txt="Answer:", ln=True, align='C')
                
                pdf.set_font("Arial", size=14)
                a_lines = self._split_text(card.get('answer', 'N/A'), 60)
                for line in a_lines:
                    pdf.cell(200, 10, txt=line.encode('latin-1', 'replace').decode('latin-1'), ln=True, align='C')
            
            return pdf.output(dest='S').encode('latin-1')
        
        except Exception as e:
            print(f"Error creating flashcard PDF: {e}")
            return None'''

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