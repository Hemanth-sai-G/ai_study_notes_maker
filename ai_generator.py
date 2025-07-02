'''import openai
import streamlit as st
from config import OPENAI_API_KEY, DEFAULT_MODEL, MAX_TOKENS, TEMPERATURE
import json
import re

class AIGenerator:
    def __init__(self):
        if not OPENAI_API_KEY:
            st.error("⚠️ OpenAI API key not found! Please set your OPENAI_API_KEY environment variable.")
            st.stop()
        
        openai.api_key = OPENAI_API_KEY
        self.client = openai.OpenAI(api_key=OPENAI_API_KEY)
    
    def generate_summary(self, text, style="concise"):
        """Generate a summary of the input text"""
        prompt = f"""
        Create a comprehensive yet {style} summary of the following text. 
        Focus on the key concepts, main ideas, and important details that would be useful for studying.
        
        Structure your summary with:
        - Main topic/subject
        - Key concepts (bullet points)
        - Important details
        - Conclusion/takeaways
        
        Text to summarize:
        {text[:4000]}  # Limit text to avoid token limits
        """
        
        try:
            response = self.client.chat.completions.create(
                model=DEFAULT_MODEL,
                messages=[
                    {"role": "system", "content": "You are an expert study assistant who creates clear, comprehensive summaries for students."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=MAX_TOKENS,
                temperature=TEMPERATURE
            )
            
            return response.choices[0].message.content
        
        except Exception as e:
            st.error(f"Error generating summary: {str(e)}")
            return None
    
    def generate_flashcards(self, text, num_cards=10):
        """Generate flashcards from the input text"""
        prompt = f"""
        Create {num_cards} flashcards from the following text. Each flashcard should have a clear question and a comprehensive answer.
        
        Format your response as a JSON array with objects containing 'question' and 'answer' fields.
        Make questions that test understanding, not just memorization.
        
        Example format:
        [
            {{"question": "What is the main concept of...?", "answer": "The main concept is..."}},
            {{"question": "How does... work?", "answer": "It works by..."}}
        ]
        
        Text for flashcards:
        {text[:3000]}
        """
        
        try:
            response = self.client.chat.completions.create(
                model=DEFAULT_MODEL,
                messages=[
                    {"role": "system", "content": "You are an expert educator who creates effective flashcards for students. Always return valid JSON."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=MAX_TOKENS,
                temperature=TEMPERATURE
            )
            
            # Parse the JSON response
            flashcards_text = response.choices[0].message.content
            
            # Try to extract JSON from the response
            try:
                # Look for JSON array in the response
                json_match = re.search(r'\[.*\]', flashcards_text, re.DOTALL)
                if json_match:
                    flashcards_json = json_match.group()
                    flashcards = json.loads(flashcards_json)
                else:
                    # Fallback: try to parse the entire response
                    flashcards = json.loads(flashcards_text)
                
                return flashcards
            
            except json.JSONDecodeError:
                # Fallback: parse manually
                return self._parse_flashcards_manually(flashcards_text)
        
        except Exception as e:
            st.error(f"Error generating flashcards: {str(e)}")
            return None
    
    def generate_quiz(self, text, num_questions=5):
        """Generate a multiple-choice quiz from the input text"""
        prompt = f"""
        Create a {num_questions}-question multiple choice quiz from the following text.
        Each question should have 4 options (A, B, C, D) with only one correct answer.
        
        Format your response as a JSON array with objects containing:
        - 'question': the question text
        - 'options': array of 4 options
        - 'correct_answer': the letter of the correct option (A, B, C, or D)
        - 'explanation': brief explanation of why the answer is correct
        
        Example format:
        [
            {{
                "question": "What is...?",
                "options": ["Option A", "Option B", "Option C", "Option D"],
                "correct_answer": "A",
                "explanation": "Option A is correct because..."
            }}
        ]
        
        Text for quiz:
        {text[:3000]}
        """
        
        try:
            response = self.client.chat.completions.create(
                model=DEFAULT_MODEL,
                messages=[
                    {"role": "system", "content": "You are an expert quiz creator who makes engaging multiple-choice questions for students. Always return valid JSON."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=MAX_TOKENS,
                temperature=TEMPERATURE
            )
            
            # Parse the JSON response
            quiz_text = response.choices[0].message.content
            
            try:
                # Look for JSON array in the response
                json_match = re.search(r'\[.*\]', quiz_text, re.DOTALL)
                if json_match:
                    quiz_json = json_match.group()
                    quiz = json.loads(quiz_json)
                else:
                    quiz = json.loads(quiz_text)
                
                return quiz
            
            except json.JSONDecodeError:
                # Fallback: parse manually
                return self._parse_quiz_manually(quiz_text)
        
        except Exception as e:
            st.error(f"Error generating quiz: {str(e)}")
            return None
    
    def _parse_flashcards_manually(self, text):
        """Fallback method to parse flashcards manually"""
        flashcards = []
        lines = text.split('\n')
        current_card = {}
        
        for line in lines:
            line = line.strip()
            if line.startswith('Q:') or line.startswith('Question:'):
                if current_card:
                    flashcards.append(current_card)
                current_card = {'question': line.replace('Q:', '').replace('Question:', '').strip()}
            elif line.startswith('A:') or line.startswith('Answer:'):
                if current_card:
                    current_card['answer'] = line.replace('A:', '').replace('Answer:', '').strip()
        
        if current_card:
            flashcards.append(current_card)
        
        return flashcards if flashcards else [{"question": "Sample Question", "answer": "Sample Answer"}]
    
    def _parse_quiz_manually(self, text):
        """Fallback method to parse quiz manually"""
        # This is a simplified fallback - in a real app, you'd want more robust parsing
        return [
            {
                "question": "Sample Question?",
                "options": ["Option A", "Option B", "Option C", "Option D"],
                "correct_answer": "A",
                "explanation": "This is a sample explanation."
            }
        ]'''

# utils/ai_generators.py
'''import openai
import os
import json
from dotenv import load_dotenv

load_dotenv()

class AIGenerator:
    def __init__(self):
        self.client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    
    def generate_summary(self, text, model="gpt-3.5-turbo"):
        """Generate a summary of the text"""
        try:
            response = self.client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that creates concise, well-structured summaries of academic content."},
                    {"role": "user", "content": f"Please create a comprehensive summary of the following text:\n\n{text[:4000]}"}
                ]
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Error generating summary: {str(e)}"
    
    def generate_flashcards(self, text, num_cards=10, model="gpt-3.5-turbo"):
        """Generate flashcards from the text"""
        try:
            response = self.client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": f"Create {num_cards} flashcards from the provided text. Return as JSON array with 'question' and 'answer' fields."},
                    {"role": "user", "content": f"Text: {text[:4000]}"}
                ]
            )
            
            content = response.choices[0].message.content
            # Try to parse JSON
            if content.startswith('```json'):
                content = content.replace('```json', '').replace('```', '').strip()
            
            flashcards = json.loads(content)
            return flashcards
        except Exception as e:
            # Fallback format
            return [{"question": f"Sample Question {i+1}", "answer": f"Sample Answer {i+1}"} for i in range(num_cards)]
    
    def generate_quiz(self, text, num_questions=5, model="gpt-3.5-turbo"):
        """Generate a multiple choice quiz"""
        try:
            response = self.client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": f"Create {num_questions} multiple choice questions from the text. Return as JSON array with 'question', 'options' (array of 4 choices), 'correct_answer' (A/B/C/D), and 'explanation' fields."},
                    {"role": "user", "content": f"Text: {text[:4000]}"}
                ]
            )
            
            content = response.choices[0].message.content
            if content.startswith('```json'):
                content = content.replace('```json', '').replace('```', '').strip()
            
            quiz = json.loads(content)
            return quiz
        except Exception as e:
            # Fallback format
            return [
                {
                    "question": f"Sample Question {i+1}",
                    "options": ["Option A", "Option B", "Option C", "Option D"],
                    "correct_answer": "A",
                    "explanation": "Sample explanation"
                } for i in range(num_questions)
            ]'''
'''import json ##worked.................
import re
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
import torch
from typing import List, Dict, Any

class AIGenerator:
    def __init__(self):
        """Initialize the AI generator with local models"""
        self.device = 0 if torch.cuda.is_available() else -1
        
        # Initialize text generation pipeline with a smaller, efficient model
        print("Loading AI models... This may take a few minutes on first run.")
        
        # Using Microsoft's DialoGPT or similar lightweight model for text generation
        # You can also use "microsoft/DialoGPT-medium" or "gpt2" for faster loading
        model_name = "microsoft/DialoGPT-small"  # Lightweight and fast
        
        try:
            self.generator = pipeline(
                "text-generation",
                model=model_name,
                tokenizer=model_name,
                device=self.device,
                max_length=512,
                do_sample=True,
                temperature=0.7,
                pad_token_id=50256  # GPT-2 pad token
            )
            print("AI models loaded successfully!")
        except Exception as e:
            print(f"Error loading model: {e}")
            # Fallback to a simpler model
            self.generator = pipeline(
                "text-generation",
                model="gpt2",
                device=self.device,
                max_length=256
            )
    
    def _clean_generated_text(self, text: str, prompt: str) -> str:
        """Clean the generated text by removing the prompt and unwanted tokens"""
        # Remove the original prompt from the generated text
        if prompt in text:
            text = text.replace(prompt, "").strip()
        
        # Clean up common artifacts
        text = re.sub(r'<\|.*?\|>', '', text)  # Remove special tokens
        text = re.sub(r'\n+', '\n', text)  # Remove multiple newlines
        text = text.strip()
        
        return text
    
    def generate_summary(self, text: str, model: str = "default") -> str:
        """Generate a summary of the text using local transformer model"""
        try:
            # Truncate text to fit model context
            max_input_length = 400
            if len(text) > max_input_length:
                text = text[:max_input_length]
            
            prompt = f"Summarize the following text concisely:\n\n{text}\n\nSummary:"
            
            # Generate summary
            result = self.generator(
                prompt,
                max_length=len(prompt.split()) + 100,
                num_return_sequences=1,
                temperature=0.7,
                do_sample=True,
                pad_token_id=50256
            )
            
            generated_text = result[0]['generated_text']
            summary = self._clean_generated_text(generated_text, prompt)
            
            # If summary is too short or empty, provide a fallback
            if len(summary.strip()) < 20:
                summary = self._create_extractive_summary(text)
            
            return summary
            
        except Exception as e:
            print(f"Error generating summary: {e}")
            return self._create_extractive_summary(text)
    
    def _create_extractive_summary(self, text: str) -> str:
        """Create a simple extractive summary as fallback"""
        sentences = text.split('.')
        # Take first few sentences as summary
        summary_sentences = sentences[:3]
        return '. '.join(summary_sentences).strip() + '.'
    
    def generate_flashcards(self, text: str, num_cards: int = 10, model: str = "default") -> List[Dict[str, str]]:
        """Generate flashcards from the text"""
        try:
            # Split text into chunks for processing
            chunks = self._split_text_into_chunks(text, max_length=300)
            flashcards = []
            
            cards_per_chunk = max(1, num_cards // len(chunks))
            
            for i, chunk in enumerate(chunks[:num_cards]):
                try:
                    prompt = f"Create a question and answer from this text:\n\n{chunk}\n\nQuestion:"
                    
                    result = self.generator(
                        prompt,
                        max_length=len(prompt.split()) + 50,
                        num_return_sequences=1,
                        temperature=0.8,
                        pad_token_id=50256
                    )
                    
                    generated = result[0]['generated_text']
                    qa_text = self._clean_generated_text(generated, prompt)
                    
                    # Parse question and answer
                    question, answer = self._parse_qa_text(qa_text, chunk)
                    
                    flashcards.append({
                        "question": question,
                        "answer": answer
                    })
                    
                except Exception as e:
                    # Fallback flashcard
                    flashcards.append({
                        "question": f"What is the main topic discussed in section {i+1}?",
                        "answer": chunk[:100] + "..." if len(chunk) > 100 else chunk
                    })
            
            # Fill remaining cards if needed
            while len(flashcards) < num_cards:
                flashcards.append({
                    "question": f"Review Question {len(flashcards) + 1}",
                    "answer": "Please review the source material for this topic."
                })
            
            return flashcards[:num_cards]
            
        except Exception as e:
            print(f"Error generating flashcards: {e}")
            return self._create_fallback_flashcards(text, num_cards)
    
    def _split_text_into_chunks(self, text: str, max_length: int = 300) -> List[str]:
        """Split text into manageable chunks"""
        words = text.split()
        chunks = []
        current_chunk = []
        
        for word in words:
            if len(' '.join(current_chunk + [word])) > max_length:
                if current_chunk:
                    chunks.append(' '.join(current_chunk))
                    current_chunk = [word]
                else:
                    chunks.append(word)
            else:
                current_chunk.append(word)
        
        if current_chunk:
            chunks.append(' '.join(current_chunk))
        
        return chunks
    
    def _parse_qa_text(self, qa_text: str, original_chunk: str) -> tuple:
        """Parse generated text to extract question and answer"""
        lines = qa_text.strip().split('\n')
        question = ""
        answer = ""
        
        for line in lines:
            line = line.strip()
            if line and not question:
                question = line.rstrip('?') + '?'
                break
        
        if not question:
            question = f"What is discussed in this passage?"
        
        # Use original chunk as answer if no clear answer generated
        answer = original_chunk[:200] + "..." if len(original_chunk) > 200 else original_chunk
        
        return question, answer
    
    def _create_fallback_flashcards(self, text: str, num_cards: int) -> List[Dict[str, str]]:
        """Create simple flashcards as fallback"""
        chunks = self._split_text_into_chunks(text, max_length=200)
        flashcards = []
        
        for i, chunk in enumerate(chunks[:num_cards]):
            flashcards.append({
                "question": f"What is the main point of section {i+1}?",
                "answer": chunk
            })
        
        # Fill remaining cards
        while len(flashcards) < num_cards:
            flashcards.append({
                "question": f"Review Question {len(flashcards) + 1}",
                "answer": "Please review the source material."
            })
        
        return flashcards[:num_cards]
    
    def generate_quiz(self, text: str, num_questions: int = 5, model: str = "default") -> List[Dict[str, Any]]:
        """Generate a multiple choice quiz"""
        try:
            chunks = self._split_text_into_chunks(text, max_length=300)
            quiz_questions = []
            
            for i, chunk in enumerate(chunks[:num_questions]):
                try:
                    # Generate question
                    prompt = f"Create a multiple choice question about this text:\n\n{chunk}\n\nQuestion:"
                    
                    result = self.generator(
                        prompt,
                        max_length=len(prompt.split()) + 40,
                        num_return_sequences=1,
                        temperature=0.8,
                        pad_token_id=50256
                    )
                    
                    generated = result[0]['generated_text']
                    question_text = self._clean_generated_text(generated, prompt)
                    
                    # Create options based on the text
                    correct_answer = self._extract_key_concept(chunk)
                    options = self._generate_options(correct_answer, chunk)
                    
                    quiz_questions.append({
                        "question": question_text if question_text else f"What is the main concept in this passage?",
                        "options": options,
                        "correct_answer": "A",  # Correct answer is always first option
                        "explanation": f"The correct answer is based on the key concept: {correct_answer}"
                    })
                    
                except Exception as e:
                    # Fallback question
                    quiz_questions.append(self._create_fallback_question(chunk, i+1))
            
            # Fill remaining questions if needed
            while len(quiz_questions) < num_questions:
                quiz_questions.append(self._create_fallback_question("", len(quiz_questions) + 1))
            
            return quiz_questions[:num_questions]
            
        except Exception as e:
            print(f"Error generating quiz: {e}")
            return self._create_fallback_quiz(text, num_questions)
    
    def _extract_key_concept(self, text: str) -> str:
        """Extract a key concept from the text"""
        # Simple keyword extraction
        words = text.split()
        # Filter out common words and take meaningful terms
        meaningful_words = [w for w in words if len(w) > 4 and w.lower() not in ['the', 'and', 'that', 'this', 'with', 'from', 'they', 'have', 'were', 'been', 'their']]
        
        if meaningful_words:
            return meaningful_words[0]
        return "key concept"
    
    def _generate_options(self, correct_answer: str, context: str) -> List[str]:
        """Generate multiple choice options"""
        options = [correct_answer]  # Correct answer first
        
        # Add some plausible distractors
        words = context.split()
        potential_options = [w for w in words if len(w) > 3 and w != correct_answer]
        
        # Add 3 more options
        for word in potential_options[:3]:
            options.append(word)
        
        # Fill with generic options if needed
        while len(options) < 4:
            options.append(f"Option {len(options)}")
        
        return options[:4]
    
    def _create_fallback_question(self, chunk: str, question_num: int) -> Dict[str, Any]:
        """Create a fallback quiz question"""
        return {
            "question": f"Question {question_num}: What is the main topic discussed?",
            "options": ["Main topic", "Secondary topic", "Related concept", "Other topic"],
            "correct_answer": "A",
            "explanation": "Based on the provided text content."
        }
    
    def _create_fallback_quiz(self, text: str, num_questions: int) -> List[Dict[str, Any]]:
        """Create a simple fallback quiz"""
        return [self._create_fallback_question("", i+1) for i in range(num_questions)]


# Example usage and testing
if __name__ == "__main__":
    # Test the AI generator
    ai_gen = AIGenerator()
    
    sample_text = """
    Artificial Intelligence (AI) is a branch of computer science that aims to create intelligent machines 
    that work and react like humans. AI systems can perform tasks that typically require human intelligence, 
    such as visual perception, speech recognition, decision-making, and language translation. Machine learning 
    is a subset of AI that enables computers to learn and improve from experience without being explicitly programmed.
    """
    
    print("Testing Summary Generation...")
    summary = ai_gen.generate_summary(sample_text)
    print(f"Summary: {summary}\n")
    
    print("Testing Flashcard Generation...")
    flashcards = ai_gen.generate_flashcards(sample_text, num_cards=3)
    for i, card in enumerate(flashcards):
        print(f"Card {i+1}: Q: {card['question']}")
        print(f"A: {card['answer']}\n")
    
    print("Testing Quiz Generation...")
    quiz = ai_gen.generate_quiz(sample_text, num_questions=2)
    for i, q in enumerate(quiz):
        print(f"Q{i+1}: {q['question']}")
        for j, option in enumerate(q['options']):
            print(f"  {chr(65+j)}) {option}")
        print(f"Correct: {q['correct_answer']}\n")'''

'''import json
import re
import random
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
import torch
from typing import List, Dict, Any, Tuple
import nltk
from collections import Counter
import spacy

# Download required NLTK data
try:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)
except:
    pass

class AIGenerator:
    def __init__(self):
        """Initialize the AI generator with optimized models for better performance"""
        self.device = 0 if torch.cuda.is_available() else -1
        
        print("Loading AI models... This may take a few minutes on first run.")
        
        try:
            # Use T5 for better text-to-text generation
            self.summarizer = pipeline(
                "summarization",
                model="t5-small",
                device=self.device,
                max_length=150,
                min_length=30,
                do_sample=False
            )
            
            # Use BART for question generation
            self.question_generator = pipeline(
                "text2text-generation",
                model="t5-small",
                device=self.device,
                max_length=100
            )
            
            print("AI models loaded successfully!")
            
        except Exception as e:
            print(f"Error loading models: {e}")
            # Fallback to basic models
            self.summarizer = None
            self.question_generator = None
    
    def extract_key_concepts(self, text: str) -> List[str]:
        """Extract key concepts and terms from text"""
        # Simple keyword extraction using frequency analysis
        words = re.findall(r'\b[A-Za-z]{4,}\b', text.lower())
        
        # Common stopwords to exclude
        stopwords = {'this', 'that', 'with', 'have', 'will', 'from', 'they', 'been', 
                    'were', 'said', 'each', 'which', 'their', 'time', 'would', 'there',
                    'could', 'other', 'after', 'first', 'well', 'way', 'many', 'must',
                    'before', 'here', 'through', 'when', 'where', 'why', 'what', 'how'}
        
        # Filter out stopwords and get most frequent terms
        filtered_words = [word for word in words if word not in stopwords]
        word_freq = Counter(filtered_words)
        
        # Return top 10 most frequent meaningful words
        key_concepts = [word for word, count in word_freq.most_common(15)]
        return key_concepts
    
    def generate_summary(self, text: str, model: str = "default") -> str:
        """Generate an insightful summary with key takeaways"""
        try:
            # Clean and prepare text
            clean_text = self._clean_text(text)
            
            # If text is too long, break it into chunks
            chunks = self._split_text_into_chunks(clean_text, max_length=500)
            
            summaries = []
            key_insights = []
            
            for chunk in chunks[:3]:  # Process first 3 chunks
                if self.summarizer and len(chunk) > 50:
                    try:
                        # Generate summary using transformer model
                        summary_result = self.summarizer(
                            chunk,
                            max_length=80,
                            min_length=20,
                            do_sample=False
                        )
                        summary = summary_result[0]['summary_text']
                        summaries.append(summary)
                    except:
                        # Fallback to extractive summary
                        summary = self._create_extractive_summary(chunk)
                        summaries.append(summary)
                else:
                    summary = self._create_extractive_summary(chunk)
                    summaries.append(summary)
            
            # Extract key concepts for insights
            key_concepts = self.extract_key_concepts(text)
            
            # Create comprehensive summary with insights
            final_summary = self._create_comprehensive_summary(summaries, key_concepts, text)
            
            return final_summary
            
        except Exception as e:
            print(f"Error generating summary: {e}")
            return self._create_fallback_summary_with_insights(text)
    
    def _create_comprehensive_summary(self, summaries: List[str], key_concepts: List[str], original_text: str) -> str:
        """Create a comprehensive summary with insights"""
        
        # Combine summaries
        main_points = "\n".join([f"• {summary}" for summary in summaries if summary])
        
        # Create insights section
        insights = []
        if key_concepts:
            insights.append(f"Key concepts discussed: {', '.join(key_concepts[:5])}")
        
        # Analyze text structure for additional insights
        sentences = original_text.split('.')
        if len(sentences) > 10:
            insights.append(f"The document contains {len(sentences)} main points across multiple sections")
        
        # Add contextual insights
        if any(word in original_text.lower() for word in ['process', 'method', 'approach', 'technique']):
            insights.append("This content focuses on processes and methodologies")
        
        if any(word in original_text.lower() for word in ['important', 'significant', 'crucial', 'critical']):
            insights.append("The material emphasizes important concepts that require attention")
        
        if any(word in original_text.lower() for word in ['example', 'instance', 'case', 'illustration']):
            insights.append("Contains practical examples and case studies for better understanding")
        
        # Combine everything
        comprehensive_summary = f"""
📋 SUMMARY:
{main_points}

💡 KEY INSIGHTS:
{chr(10).join([f"• {insight}" for insight in insights])}

🎯 MAIN TAKEAWAYS:
• Focus on understanding the core concepts: {', '.join(key_concepts[:3]) if key_concepts else 'the main topics'}
• This material is suitable for {self._determine_difficulty_level(original_text)} level study
• Recommended review time: {self._estimate_study_time(original_text)} minutes
        """
        
        return comprehensive_summary.strip()
    
    def _determine_difficulty_level(self, text: str) -> str:
        """Determine the difficulty level of the content"""
        # Simple heuristic based on vocabulary complexity
        words = text.split()
        complex_words = [w for w in words if len(w) > 8]
        complexity_ratio = len(complex_words) / len(words) if words else 0
        
        if complexity_ratio > 0.15:
            return "advanced"
        elif complexity_ratio > 0.08:
            return "intermediate"
        else:
            return "beginner"
    
    def _estimate_study_time(self, text: str) -> int:
        """Estimate study time based on content length"""
        word_count = len(text.split())
        # Assume 200 words per minute reading + processing time
        return max(5, word_count // 150)
    
    def generate_diverse_quiz(self, text: str, num_questions: int = 5, model: str = "default") -> List[Dict[str, Any]]:
        """Generate diverse quiz questions with different types and difficulties"""
        try:
            key_concepts = self.extract_key_concepts(text)
            chunks = self._split_text_into_chunks(text, max_length=300)
            
            quiz_questions = []
            question_types = ['definition', 'application', 'comparison', 'analysis', 'recall']
            
            for i in range(min(num_questions, len(chunks))):
                chunk = chunks[i]
                question_type = question_types[i % len(question_types)]
                
                question_data = self._generate_question_by_type(
                    chunk, question_type, key_concepts, i + 1
                )
                quiz_questions.append(question_data)
            
            # Fill remaining questions if needed
            while len(quiz_questions) < num_questions:
                remaining_type = question_types[len(quiz_questions) % len(question_types)]
                fallback_q = self._generate_fallback_question_by_type(
                    text, remaining_type, len(quiz_questions) + 1, key_concepts
                )
                quiz_questions.append(fallback_q)
            
            return quiz_questions[:num_questions]
            
        except Exception as e:
            print(f"Error generating quiz: {e}")
            return self._create_diverse_fallback_quiz(text, num_questions)
    
    def _generate_question_by_type(self, chunk: str, question_type: str, 
                                 key_concepts: List[str], question_num: int) -> Dict[str, Any]:
        """Generate specific question types"""
        
        concept = key_concepts[0] if key_concepts else "concept"
        
        if question_type == 'definition':
            question = f"What does '{concept}' refer to in this context?"
            correct_answer = self._extract_definition(chunk, concept)
            distractors = self._generate_definition_distractors(concept)
            
        elif question_type == 'application':
            question = f"How would you apply the concept of '{concept}' in practice?"
            correct_answer = self._extract_application(chunk)
            distractors = self._generate_application_distractors()
            
        elif question_type == 'comparison':
            question = f"What is the main difference between the concepts discussed in this passage?"
            correct_answer = self._extract_comparison(chunk)
            distractors = self._generate_comparison_distractors()
            
        elif question_type == 'analysis':
            question = f"Why is '{concept}' important according to the text?"
            correct_answer = self._extract_importance(chunk, concept)
            distractors = self._generate_importance_distractors()
            
        else:  # recall
            question = f"According to the passage, what is mentioned about '{concept}'?"
            correct_answer = self._extract_key_fact(chunk, concept)
            distractors = self._generate_recall_distractors(concept)
        
        # Combine correct answer with distractors
        all_options = [correct_answer] + distractors[:3]
        random.shuffle(all_options)
        
        correct_index = all_options.index(correct_answer)
        correct_letter = chr(65 + correct_index)  # A, B, C, D
        
        return {
            "question": question,
            "options": all_options,
            "correct_answer": correct_letter,
            "explanation": f"The correct answer focuses on {concept} as described in the source material.",
            "type": question_type,
            "difficulty": self._assign_difficulty(question_num)
        }
    
    def _extract_definition(self, text: str, concept: str) -> str:
        """Extract or generate a definition from text"""
        sentences = text.split('.')
        for sentence in sentences:
            if concept.lower() in sentence.lower():
                return sentence.strip()[:80] + "..."
        return f"{concept} is a key concept discussed in the material"
    
    def _extract_application(self, text: str) -> str:
        """Extract application information"""
        if any(word in text.lower() for word in ['use', 'apply', 'implement', 'practice']):
            return "Through practical implementation as described in the text"
        return "By following the methods outlined in the passage"
    
    def _extract_comparison(self, text: str) -> str:
        """Extract comparison information"""
        if any(word in text.lower() for word in ['differ', 'unlike', 'whereas', 'however']):
            return "The key differences are outlined in the comparative analysis"
        return "The main distinction lies in their fundamental approaches"
    
    def _extract_importance(self, text: str, concept: str) -> str:
        """Extract importance information"""
        if any(word in text.lower() for word in ['important', 'significant', 'crucial', 'essential']):
            return f"{concept} is important because it forms the foundation of the discussed topic"
        return f"{concept} plays a crucial role in the overall framework"
    
    def _extract_key_fact(self, text: str, concept: str) -> str:
        """Extract a key fact about the concept"""
        sentences = text.split('.')
        for sentence in sentences:
            if concept.lower() in sentence.lower() and len(sentence) > 20:
                return sentence.strip()[:80] + "..."
        return f"The text discusses {concept} in detail"
    
    def _generate_definition_distractors(self, concept: str) -> List[str]:
        """Generate plausible but incorrect definitions"""
        return [
            f"{concept} is primarily used for data storage",
            f"{concept} refers to a theoretical framework only",
            f"{concept} is an outdated methodology"
        ]
    
    def _generate_application_distractors(self) -> List[str]:
        """Generate application distractors"""
        return [
            "By ignoring the theoretical aspects",
            "Through trial and error only",
            "By avoiding practical implementation"
        ]
    
    def _generate_comparison_distractors(self) -> List[str]:
        """Generate comparison distractors"""
        return [
            "There are no significant differences",
            "They are completely unrelated concepts",
            "The differences are purely semantic"
        ]
    
    def _generate_importance_distractors(self) -> List[str]:
        """Generate importance distractors"""
        return [
            "It has no practical significance",
            "It's only relevant in theoretical contexts",
            "It's a minor supporting detail"
        ]
    
    def _generate_recall_distractors(self, concept: str) -> List[str]:
        """Generate recall distractors"""
        return [
            f"The text dismisses {concept} as irrelevant",
            f"{concept} is mentioned only in passing",
            f"The passage contradicts the importance of {concept}"
        ]
    
    def _assign_difficulty(self, question_num: int) -> str:
        """Assign difficulty levels to questions"""
        if question_num <= 2:
            return "Easy"
        elif question_num <= 4:
            return "Medium"
        else:
            return "Hard"
    
    def _generate_fallback_question_by_type(self, text: str, question_type: str, 
                                          question_num: int, key_concepts: List[str]) -> Dict[str, Any]:
        """Generate fallback questions by type"""
        concept = key_concepts[0] if key_concepts else "the main topic"
        
        fallback_questions = {
            'definition': {
                'question': f"How would you define {concept} based on the passage?",
                'options': [
                    f"A key concept central to the discussion",
                    "An unimportant detail",
                    "A contradictory element",
                    "An outdated theory"
                ]
            },
            'application': {
                'question': f"In what context would {concept} be most applicable?",
                'options': [
                    "In the scenarios described in the text",
                    "Only in theoretical discussions",
                    "Never in practical situations",
                    "Only in historical contexts"
                ]
            },
            'comparison': {
                'question': f"How does {concept} relate to other concepts in the passage?",
                'options': [
                    "It connects with and supports other key ideas",
                    "It contradicts all other concepts",
                    "It's completely isolated",
                    "It's less important than other concepts"
                ]
            },
            'analysis': {
                'question': f"What makes {concept} significant in this context?",
                'options': [
                    "Its role in supporting the main argument",
                    "Its historical importance only",
                    "Its lack of practical value",
                    "Its controversial nature"
                ]
            },
            'recall': {
                'question': f"What specific information is provided about {concept}?",
                'options': [
                    "Detailed explanation of its characteristics",
                    "Only a brief mention",
                    "Criticism of its validity",
                    "No substantial information"
                ]
            }
        }
        
        question_data = fallback_questions.get(question_type, fallback_questions['recall'])
        
        return {
            "question": question_data['question'],
            "options": question_data['options'],
            "correct_answer": "A",
            "explanation": f"Based on the content analysis of {concept}",
            "type": question_type,
            "difficulty": self._assign_difficulty(question_num)
        }
    
    def _create_diverse_fallback_quiz(self, text: str, num_questions: int) -> List[Dict[str, Any]]:
        """Create diverse fallback quiz questions"""
        key_concepts = self.extract_key_concepts(text)
        question_types = ['definition', 'application', 'comparison', 'analysis', 'recall']
        
        questions = []
        for i in range(num_questions):
            question_type = question_types[i % len(question_types)]
            question = self._generate_fallback_question_by_type(
                text, question_type, i + 1, key_concepts
            )
            questions.append(question)
        
        return questions
    
    def _clean_text(self, text: str) -> str:
        """Clean and preprocess text"""
        # Remove extra whitespace and clean up text
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'[^\w\s\.\,\!\?\:\;\-\(\)]', '', text)
        return text.strip()
    
    def _split_text_into_chunks(self, text: str, max_length: int = 300) -> List[str]:
        """Split text into manageable chunks"""
        sentences = text.split('.')
        chunks = []
        current_chunk = []
        current_length = 0
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
                
            if current_length + len(sentence) > max_length and current_chunk:
                chunks.append('. '.join(current_chunk) + '.')
                current_chunk = [sentence]
                current_length = len(sentence)
            else:
                current_chunk.append(sentence)
                current_length += len(sentence)
        
        if current_chunk:
            chunks.append('. '.join(current_chunk) + '.')
        
        return chunks
    
    def _create_extractive_summary(self, text: str) -> str:
        """Create an improved extractive summary"""
        sentences = [s.strip() for s in text.split('.') if s.strip()]
        
        # Score sentences based on key terms and position
        scored_sentences = []
        key_concepts = self.extract_key_concepts(text)
        
        for i, sentence in enumerate(sentences):
            score = 0
            # Position scoring (first and last sentences are important)
            if i == 0 or i == len(sentences) - 1:
                score += 2
            
            # Key concept scoring
            for concept in key_concepts[:5]:
                if concept.lower() in sentence.lower():
                    score += 1
            
            # Length scoring (prefer moderate length sentences)
            word_count = len(sentence.split())
            if 10 <= word_count <= 25:
                score += 1
            
            scored_sentences.append((sentence, score))
        
        # Select top scoring sentences
        scored_sentences.sort(key=lambda x: x[1], reverse=True)
        top_sentences = [s[0] for s in scored_sentences[:3]]
        
        return '. '.join(top_sentences) + '.'
    
    def _create_fallback_summary_with_insights(self, text: str) -> str:
        """Create fallback summary with insights"""
        key_concepts = self.extract_key_concepts(text)
        summary = self._create_extractive_summary(text)
        
        insights_summary = f"""
📋 SUMMARY:
{summary}

💡 KEY INSIGHTS:
• Main concepts: {', '.join(key_concepts[:5]) if key_concepts else 'Various topics discussed'}
• Content type: {self._determine_difficulty_level(text)} level material
• Estimated study time: {self._estimate_study_time(text)} minutes

🎯 STUDY RECOMMENDATIONS:
• Focus on understanding the key concepts identified
• Review the material systematically
• Test your understanding with practice questions
        """
        
        return insights_summary.strip()


# Example usage and testing
if __name__ == "__main__":
    # Test the improved AI generator
    ai_gen = AIGenerator()
    
    sample_text = """
    Project management is the application of knowledge, skills, tools, and techniques to project activities 
    to meet project requirements. It involves planning, executing, monitoring, and closing projects effectively. 
    Key project management principles include scope management, time management, cost management, quality management, 
    and stakeholder management. Successful project managers must possess strong leadership skills, communication abilities, 
    and technical expertise. Modern project management methodologies include Agile, Scrum, and Waterfall approaches, 
    each suited to different types of projects and organizational contexts.
    """
    
    print("Testing Enhanced Summary Generation...")
    summary = ai_gen.generate_summary_with_insights(sample_text)
    print(f"Summary:\n{summary}\n")
    
    print("Testing Diverse Quiz Generation...")
    quiz = ai_gen.generate_diverse_quiz(sample_text, num_questions=5)
    for i, q in enumerate(quiz):
        print(f"Q{i+1} ({q['type'].title()}, {q['difficulty']}): {q['question']}")
        for j, option in enumerate(q['options']):
            print(f"  {chr(65+j)}) {option}")
        print(f"Correct: {q['correct_answer']}")
        print(f"Explanation: {q['explanation']}\n")'''
#...........................................................................................
import json## worked well generated summary and quiz well,but flashcards are same.
import re
import random
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
import torch
from typing import List, Dict, Any, Tuple
import nltk
from collections import Counter
import spacy

# Download required NLTK data
try:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)
except:
    pass

class AIGenerator:
    def __init__(self):
        """Initialize the AI generator with optimized models for better performance"""
        self.device = 0 if torch.cuda.is_available() else -1
        
        print("Loading AI models... This may take a few minutes on first run.")
        
        try:
            # Use T5 for better text-to-text generation
            self.summarizer = pipeline(
                "summarization",
                model="t5-small",
                device=self.device,
                max_length=150,
                min_length=30,
                do_sample=False
            )
            
            # Use BART for question generation
            self.question_generator = pipeline(
                "text2text-generation",
                model="t5-small",
                device=self.device,
                max_length=100
            )
            
            print("AI models loaded successfully!")
            
        except Exception as e:
            print(f"Error loading models: {e}")
            # Fallback to basic models
            self.summarizer = None
            self.question_generator = None
    
    def _clean_generated_text(self, text: str, prompt: str) -> str:
        """Clean the generated text by removing the prompt and unwanted tokens"""
        # Remove the original prompt from the generated text
        if prompt in text:
            text = text.replace(prompt, "").strip()
        
        # Clean up common artifacts
        text = re.sub(r'<\|.*?\|>', '', text)  # Remove special tokens
        text = re.sub(r'\n+', '\n', text)  # Remove multiple newlines
        text = text.strip()
        
        return text
    
    def extract_key_concepts(self, text: str) -> List[str]:
        """Extract key concepts and terms from text"""
        # Simple keyword extraction using frequency analysis
        words = re.findall(r'\b[A-Za-z]{4,}\b', text.lower())
        
        # Common stopwords to exclude
        stopwords = {'this', 'that', 'with', 'have', 'will', 'from', 'they', 'been', 
                    'were', 'said', 'each', 'which', 'their', 'time', 'would', 'there',
                    'could', 'other', 'after', 'first', 'well', 'way', 'many', 'must',
                    'before', 'here', 'through', 'when', 'where', 'why', 'what', 'how'}
        
        # Filter out stopwords and get most frequent terms
        filtered_words = [word for word in words if word not in stopwords]
        word_freq = Counter(filtered_words)
        
        # Return top 10 most frequent meaningful words
        key_concepts = [word for word, count in word_freq.most_common(15)]
        return key_concepts
    
    def generate_summary(self, text: str, model: str = "default") -> str:
        """Generate an insightful summary with key takeaways"""
        try:
            # Clean and prepare text
            clean_text = self._clean_text(text)
            
            # If text is too long, break it into chunks
            chunks = self._split_text_into_chunks(clean_text, max_length=500)
            
            summaries = []
            key_insights = []
            
            for chunk in chunks[:3]:  # Process first 3 chunks
                if self.summarizer and len(chunk) > 50:
                    try:
                        # Generate summary using transformer model
                        summary_result = self.summarizer(
                            chunk,
                            max_length=80,
                            min_length=20,
                            do_sample=False
                        )
                        summary = summary_result[0]['summary_text']
                        summaries.append(summary)
                    except:
                        # Fallback to extractive summary
                        summary = self._create_extractive_summary(chunk)
                        summaries.append(summary)
                else:
                    summary = self._create_extractive_summary(chunk)
                    summaries.append(summary)
            
            # Extract key concepts for insights
            key_concepts = self.extract_key_concepts(text)
            
            # Create comprehensive summary with insights
            final_summary = self._create_comprehensive_summary(summaries, key_concepts, text)
            
            return final_summary
            
        except Exception as e:
            print(f"Error generating summary: {e}")
            return self._create_fallback_summary_with_insights(text)
    
    def _create_comprehensive_summary(self, summaries: List[str], key_concepts: List[str], original_text: str) -> str:
        """Create a comprehensive summary with insights"""
        
        # Combine summaries
        main_points = "\n".join([f"• {summary}" for summary in summaries if summary])
        
        # Create insights section
        insights = []
        if key_concepts:
            insights.append(f"Key concepts discussed: {', '.join(key_concepts[:5])}")
        
        # Analyze text structure for additional insights
        sentences = original_text.split('.')
        if len(sentences) > 10:
            insights.append(f"The document contains {len(sentences)} main points across multiple sections")
        
        # Add contextual insights
        if any(word in original_text.lower() for word in ['process', 'method', 'approach', 'technique']):
            insights.append("This content focuses on processes and methodologies")
        
        if any(word in original_text.lower() for word in ['important', 'significant', 'crucial', 'critical']):
            insights.append("The material emphasizes important concepts that require attention")
        
        if any(word in original_text.lower() for word in ['example', 'instance', 'case', 'illustration']):
            insights.append("Contains practical examples and case studies for better understanding")
        
        # Combine everything
        comprehensive_summary = f"""
📋 SUMMARY:
{main_points}

💡 KEY INSIGHTS:
{chr(10).join([f"• {insight}" for insight in insights])}

🎯 MAIN TAKEAWAYS:
• Focus on understanding the core concepts: {', '.join(key_concepts[:3]) if key_concepts else 'the main topics'}
• This material is suitable for {self._determine_difficulty_level(original_text)} level study
• Recommended review time: {self._estimate_study_time(original_text)} minutes
        """
        
        return comprehensive_summary.strip()
    
    def _determine_difficulty_level(self, text: str) -> str:
        """Determine the difficulty level of the content"""
        # Simple heuristic based on vocabulary complexity
        words = text.split()
        complex_words = [w for w in words if len(w) > 8]
        complexity_ratio = len(complex_words) / len(words) if words else 0
        
        if complexity_ratio > 0.15:
            return "advanced"
        elif complexity_ratio > 0.08:
            return "intermediate"
        else:
            return "beginner"
    
    def _estimate_study_time(self, text: str) -> int:
        """Estimate study time based on content length"""
        word_count = len(text.split())
        # Assume 200 words per minute reading + processing time
        return max(5, word_count // 150)
    
    def generate_flashcards(self, text: str, num_cards: int = 10, model: str = "default") -> List[Dict[str, str]]:
        """Generate flashcards from the text"""
        try:
            # Split text into chunks for processing
            chunks = self._split_text_into_chunks(text, max_length=300)
            flashcards = []
            
            cards_per_chunk = max(1, num_cards // len(chunks))
            
            for i, chunk in enumerate(chunks[:num_cards]):
                try:
                    # Use key concepts for better questions
                    key_concepts = self.extract_key_concepts(chunk)
                    concept = key_concepts[0] if key_concepts else "concept"
                    
                    question = f"What is {concept} and how does it relate to the main topic?"
                    answer = chunk[:200] + "..." if len(chunk) > 200 else chunk
                    
                    flashcards.append({
                        "question": question,
                        "answer": answer
                    })
                    
                except Exception as e:
                    # Fallback flashcard
                    flashcards.append({
                        "question": f"What is the main topic discussed in section {i+1}?",
                        "answer": chunk[:100] + "..." if len(chunk) > 100 else chunk
                    })
            
            # Fill remaining cards if needed
            while len(flashcards) < num_cards:
                flashcards.append({
                    "question": f"Review Question {len(flashcards) + 1}",
                    "answer": "Please review the source material for this topic."
                })
            
            return flashcards[:num_cards]
            
        except Exception as e:
            print(f"Error generating flashcards: {e}")
            return self._create_fallback_flashcards(text, num_cards)
    
    def generate_quiz(self, text: str, num_questions: int = 5, model: str = "default") -> List[Dict[str, Any]]:
        """Generate diverse quiz questions with different types and difficulties"""
        try:
            key_concepts = self.extract_key_concepts(text)
            chunks = self._split_text_into_chunks(text, max_length=300)
            
            quiz_questions = []
            question_types = ['definition', 'application', 'comparison', 'analysis', 'recall']
            
            for i in range(min(num_questions, len(chunks))):
                chunk = chunks[i]
                question_type = question_types[i % len(question_types)]
                
                question_data = self._generate_question_by_type(
                    chunk, question_type, key_concepts, i + 1
                )
                quiz_questions.append(question_data)
            
            # Fill remaining questions if needed
            while len(quiz_questions) < num_questions:
                remaining_type = question_types[len(quiz_questions) % len(question_types)]
                fallback_q = self._generate_fallback_question_by_type(
                    text, remaining_type, len(quiz_questions) + 1, key_concepts
                )
                quiz_questions.append(fallback_q)
            
            return quiz_questions[:num_questions]
            
        except Exception as e:
            print(f"Error generating quiz: {e}")
            return self._create_fallback_quiz(text, num_questions)
    
    def _generate_question_by_type(self, chunk: str, question_type: str, 
                                 key_concepts: List[str], question_num: int) -> Dict[str, Any]:
        """Generate specific question types"""
        
        concept = key_concepts[0] if key_concepts else "concept"
        
        if question_type == 'definition':
            question = f"What does '{concept}' refer to in this context?"
            correct_answer = self._extract_definition(chunk, concept)
            distractors = self._generate_definition_distractors(concept)
            
        elif question_type == 'application':
            question = f"How would you apply the concept of '{concept}' in practice?"
            correct_answer = self._extract_application(chunk)
            distractors = self._generate_application_distractors()
            
        elif question_type == 'comparison':
            question = f"What is the main difference between the concepts discussed in this passage?"
            correct_answer = self._extract_comparison(chunk)
            distractors = self._generate_comparison_distractors()
            
        elif question_type == 'analysis':
            question = f"Why is '{concept}' important according to the text?"
            correct_answer = self._extract_importance(chunk, concept)
            distractors = self._generate_importance_distractors()
            
        else:  # recall
            question = f"According to the passage, what is mentioned about '{concept}'?"
            correct_answer = self._extract_key_fact(chunk, concept)
            distractors = self._generate_recall_distractors(concept)
        
        # Combine correct answer with distractors
        all_options = [correct_answer] + distractors[:3]
        random.shuffle(all_options)
        
        correct_index = all_options.index(correct_answer)
        correct_letter = chr(65 + correct_index)  # A, B, C, D
        
        return {
            "question": question,
            "options": all_options,
            "correct_answer": correct_letter,
            "explanation": f"The correct answer focuses on {concept} as described in the source material.",
            "type": question_type,
            "difficulty": self._assign_difficulty(question_num)
        }
    
    def _extract_definition(self, text: str, concept: str) -> str:
        """Extract or generate a definition from text"""
        sentences = text.split('.')
        for sentence in sentences:
            if concept.lower() in sentence.lower():
                return sentence.strip()[:80] + "..."
        return f"{concept} is a key concept discussed in the material"
    
    def _extract_application(self, text: str) -> str:
        """Extract application information"""
        if any(word in text.lower() for word in ['use', 'apply', 'implement', 'practice']):
            return "Through practical implementation as described in the text"
        return "By following the methods outlined in the passage"
    
    def _extract_comparison(self, text: str) -> str:
        """Extract comparison information"""
        if any(word in text.lower() for word in ['differ', 'unlike', 'whereas', 'however']):
            return "The key differences are outlined in the comparative analysis"
        return "The main distinction lies in their fundamental approaches"
    
    def _extract_importance(self, text: str, concept: str) -> str:
        """Extract importance information"""
        if any(word in text.lower() for word in ['important', 'significant', 'crucial', 'essential']):
            return f"{concept} is important because it forms the foundation of the discussed topic"
        return f"{concept} plays a crucial role in the overall framework"
    
    def _extract_key_fact(self, text: str, concept: str) -> str:
        """Extract a key fact about the concept"""
        sentences = text.split('.')
        for sentence in sentences:
            if concept.lower() in sentence.lower() and len(sentence) > 20:
                return sentence.strip()[:80] + "..."
        return f"The text discusses {concept} in detail"
    
    def _generate_definition_distractors(self, concept: str) -> List[str]:
        """Generate plausible but incorrect definitions"""
        return [
            f"{concept} is primarily used for data storage",
            f"{concept} refers to a theoretical framework only",
            f"{concept} is an outdated methodology"
        ]
    
    def _generate_application_distractors(self) -> List[str]:
        """Generate application distractors"""
        return [
            "By ignoring the theoretical aspects",
            "Through trial and error only",
            "By avoiding practical implementation"
        ]
    
    def _generate_comparison_distractors(self) -> List[str]:
        """Generate comparison distractors"""
        return [
            "There are no significant differences",
            "They are completely unrelated concepts",
            "The differences are purely semantic"
        ]
    
    def _generate_importance_distractors(self) -> List[str]:
        """Generate importance distractors"""
        return [
            "It has no practical significance",
            "It's only relevant in theoretical contexts",
            "It's a minor supporting detail"
        ]
    
    def _generate_recall_distractors(self, concept: str) -> List[str]:
        """Generate recall distractors"""
        return [
            f"The text dismisses {concept} as irrelevant",
            f"{concept} is mentioned only in passing",
            f"The passage contradicts the importance of {concept}"
        ]
    
    def _assign_difficulty(self, question_num: int) -> str:
        """Assign difficulty levels to questions"""
        if question_num <= 2:
            return "Easy"
        elif question_num <= 4:
            return "Medium"
        else:
            return "Hard"
    
    def _generate_fallback_question_by_type(self, text: str, question_type: str, 
                                          question_num: int, key_concepts: List[str]) -> Dict[str, Any]:
        """Generate fallback questions by type"""
        concept = key_concepts[0] if key_concepts else "the main topic"
        
        fallback_questions = {
            'definition': {
                'question': f"How would you define {concept} based on the passage?",
                'options': [
                    f"A key concept central to the discussion",
                    "An unimportant detail",
                    "A contradictory element",
                    "An outdated theory"
                ]
            },
            'application': {
                'question': f"In what context would {concept} be most applicable?",
                'options': [
                    "In the scenarios described in the text",
                    "Only in theoretical discussions",
                    "Never in practical situations",
                    "Only in historical contexts"
                ]
            },
            'comparison': {
                'question': f"How does {concept} relate to other concepts in the passage?",
                'options': [
                    "It connects with and supports other key ideas",
                    "It contradicts all other concepts",
                    "It's completely isolated",
                    "It's less important than other concepts"
                ]
            },
            'analysis': {
                'question': f"What makes {concept} significant in this context?",
                'options': [
                    "Its role in supporting the main argument",
                    "Its historical importance only",
                    "Its lack of practical value",
                    "Its controversial nature"
                ]
            },
            'recall': {
                'question': f"What specific information is provided about {concept}?",
                'options': [
                    "Detailed explanation of its characteristics",
                    "Only a brief mention",
                    "Criticism of its validity",
                    "No substantial information"
                ]
            }
        }
        
        question_data = fallback_questions.get(question_type, fallback_questions['recall'])
        
        return {
            "question": question_data['question'],
            "options": question_data['options'],
            "correct_answer": "A",
            "explanation": f"Based on the content analysis of {concept}",
            "type": question_type,
            "difficulty": self._assign_difficulty(question_num)
        }
    
    def _create_fallback_quiz(self, text: str, num_questions: int) -> List[Dict[str, Any]]:
        """Create diverse fallback quiz questions"""
        key_concepts = self.extract_key_concepts(text)
        question_types = ['definition', 'application', 'comparison', 'analysis', 'recall']
        
        questions = []
        for i in range(num_questions):
            question_type = question_types[i % len(question_types)]
            question = self._generate_fallback_question_by_type(
                text, question_type, i + 1, key_concepts
            )
            questions.append(question)
        
        return questions
    
    def _create_fallback_flashcards(self, text: str, num_cards: int) -> List[Dict[str, str]]:
        """Create simple flashcards as fallback"""
        chunks = self._split_text_into_chunks(text, max_length=200)
        flashcards = []
        
        for i, chunk in enumerate(chunks[:num_cards]):
            flashcards.append({
                "question": f"What is the main point of section {i+1}?",
                "answer": chunk
            })
        
        # Fill remaining cards
        while len(flashcards) < num_cards:
            flashcards.append({
                "question": f"Review Question {len(flashcards) + 1}",
                "answer": "Please review the source material."
            })
        
        return flashcards[:num_cards]
    
    def _clean_text(self, text: str) -> str:
        """Clean and preprocess text"""
        # Remove extra whitespace and clean up text
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'[^\w\s\.\,\!\?\:\;\-\(\)]', '', text)
        return text.strip()
    
    def _split_text_into_chunks(self, text: str, max_length: int = 300) -> List[str]:
        """Split text into manageable chunks"""
        sentences = text.split('.')
        chunks = []
        current_chunk = []
        current_length = 0
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
                
            if current_length + len(sentence) > max_length and current_chunk:
                chunks.append('. '.join(current_chunk) + '.')
                current_chunk = [sentence]
                current_length = len(sentence)
            else:
                current_chunk.append(sentence)
                current_length += len(sentence)
        
        if current_chunk:
            chunks.append('. '.join(current_chunk) + '.')
        
        return chunks
    
    def _create_extractive_summary(self, text: str) -> str:
        """Create an improved extractive summary"""
        sentences = [s.strip() for s in text.split('.') if s.strip()]
        
        # Score sentences based on key terms and position
        scored_sentences = []
        key_concepts = self.extract_key_concepts(text)
        
        for i, sentence in enumerate(sentences):
            score = 0
            # Position scoring (first and last sentences are important)
            if i == 0 or i == len(sentences) - 1:
                score += 2
            
            # Key concept scoring
            for concept in key_concepts[:5]:
                if concept.lower() in sentence.lower():
                    score += 1
            
            # Length scoring (prefer moderate length sentences)
            word_count = len(sentence.split())
            if 10 <= word_count <= 25:
                score += 1
            
            scored_sentences.append((sentence, score))
        
        # Select top scoring sentences
        scored_sentences.sort(key=lambda x: x[1], reverse=True)
        top_sentences = [s[0] for s in scored_sentences[:3]]
        
        return '. '.join(top_sentences) + '.'
    
    def _create_fallback_summary_with_insights(self, text: str) -> str:
        """Create fallback summary with insights"""
        key_concepts = self.extract_key_concepts(text)
        summary = self._create_extractive_summary(text)
        
        insights_summary = f"""
📋 SUMMARY:
{summary}

💡 KEY INSIGHTS:
• Main concepts: {', '.join(key_concepts[:5]) if key_concepts else 'Various topics discussed'}
• Content type: {self._determine_difficulty_level(text)} level material
• Estimated study time: {self._estimate_study_time(text)} minutes

🎯 STUDY RECOMMENDATIONS:
• Focus on understanding the key concepts identified
• Review the material systematically
• Test your understanding with practice questions
        """
        
        return insights_summary.strip()


# Example usage and testing
if __name__ == "__main__":
    # Test the improved AI generator
    ai_gen = AIGenerator()
    
    sample_text = """
    Project management is the application of knowledge, skills, tools, and techniques to project activities 
    to meet project requirements. It involves planning, executing, monitoring, and closing projects effectively. 
    Key project management principles include scope management, time management, cost management, quality management, 
    and stakeholder management. Successful project managers must possess strong leadership skills, communication abilities, 
    and technical expertise. Modern project management methodologies include Agile, Scrum, and Waterfall approaches, 
    each suited to different types of projects and organizational contexts.
    """
    
    print("Testing Enhanced Summary Generation...")
    summary = ai_gen.generate_summary(sample_text)
    print(f"Summary:\n{summary}\n")
    
    print("Testing Flashcard Generation...")
    flashcards = ai_gen.generate_flashcards(sample_text, num_cards=3)
    for i, card in enumerate(flashcards):
        print(f"Card {i+1}: Q: {card['question']}")
        print(f"A: {card['answer']}\n")
    
    print("Testing Diverse Quiz Generation...")
    quiz = ai_gen.generate_quiz(sample_text, num_questions=5)
    for i, q in enumerate(quiz):
        print(f"Q{i+1} ({q['type'].title()}, {q['difficulty']}): {q['question']}")
        for j, option in enumerate(q['options']):
            print(f"  {chr(65+j)}) {option}")
        print(f"Correct: {q['correct_answer']}")
        print(f"Explanation: {q['explanation']}\n")

