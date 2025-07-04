import json
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

