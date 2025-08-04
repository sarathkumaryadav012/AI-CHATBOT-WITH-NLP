#!/usr/bin/env python3
"""
AI Chatbot with Natural Language Processing
Uses NLTK and spaCy for intelligent conversation handling

Installation Requirements:
pip install nltk spacy scikit-learn numpy
python -m spacy download en_core_web_sm

Author: AI Assistant
Created: 2025
"""

import nltk
import spacy
import random
import re
import sys
from collections import defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Download required NLTK data
print("📦 Checking NLTK dependencies...")
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    print("Downloading NLTK punkt tokenizer...")
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    print("Downloading NLTK stopwords...")
    nltk.download('stopwords')

try:
    nltk.data.find('taggers/averaged_perceptron_tagger')
except LookupError:
    print("Downloading NLTK POS tagger...")
    nltk.download('averaged_perceptron_tagger')

from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tag import pos_tag

class NLPChatbot:
    def __init__(self):
        """Initialize the chatbot with NLP models and knowledge base"""
        print("🤖 Initializing NLP Chatbot...")
        
        # Load spaCy model (install with: python -m spacy download en_core_web_sm)
        try:
            self.nlp = spacy.load("en_core_web_sm")
            self.spacy_available = True
            print("✓ spaCy model loaded successfully")
        except OSError:
            print("⚠ spaCy model not found. Install with: python -m spacy download en_core_web_sm")
            print("⚠ Continuing with NLTK-only functionality")
            self.spacy_available = False
        
        # Initialize NLTK components
        self.stemmer = PorterStemmer()
        try:
            self.stop_words = set(stopwords.words('english'))
        except:
            print("⚠ NLTK stopwords not available, using basic set")
            self.stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
        
        # Knowledge base and responses
        self.knowledge_base = {
            'greetings': [
                "Hello! How can I help you today?",
                "Hi there! What would you like to know?",
                "Greetings! I'm here to assist you.",
                "Hey! What's on your mind?",
                "Welcome! How may I assist you?"
            ],
            'goodbyes': [
                "Goodbye! Have a great day!",
                "See you later!",
                "Take care!",
                "Until next time!",
                "Farewell! Thanks for chatting!"
            ],
            'thanks': [
                "You're welcome!",
                "Happy to help!",
                "No problem at all!",
                "Glad I could assist!",
                "My pleasure!"
            ],
            'default': [
                "That's interesting! Can you tell me more?",
                "I understand. What else would you like to discuss?",
                "Could you provide more details about that?",
                "I'm here to help. What specific information do you need?",
                "Tell me more about what you're thinking."
            ]
        }
        
        # Topic-specific responses
        self.topic_responses = {
            'weather': [
                "I don't have access to real-time weather data, but you can check local weather services!",
                "Weather can be quite unpredictable! What's it like where you are?",
                "I'd recommend checking a weather app for the most accurate forecast.",
                "Weather is always changing! Are you planning something outdoors?"
            ],
            'technology': [
                "Technology is fascinating! What specific aspect interests you?",
                "I love discussing tech topics. What would you like to know?",
                "Technology evolves so quickly these days. What's caught your attention?",
                "The world of tech is amazing! Which area are you curious about?"
            ],
            'science': [
                "Science is amazing! Which field of science are you curious about?",
                "There's so much to explore in science. What would you like to discuss?",
                "Scientific discoveries happen every day. What interests you most?",
                "Science helps us understand the world! What topic intrigues you?"
            ],
            'food': [
                "Food is one of life's great pleasures! What's your favorite cuisine?",
                "I love talking about food! Are you looking for recipes or recommendations?",
                "Cooking can be so creative. What kind of dishes do you enjoy?",
                "Food brings people together! What are you in the mood for?"
            ],
            'health': [
                "Health is so important! What aspect of wellness interests you?",
                "Taking care of yourself is key! What health topic would you like to discuss?",
                "I'm not a medical professional, but I can share general health information."
            ],
            'education': [
                "Learning is a lifelong journey! What subject are you studying?",
                "Education opens so many doors! What would you like to learn about?",
                "Knowledge is power! What educational topic interests you?"
            ]
        }
        
        # Initialize TF-IDF vectorizer for similarity matching
        self.vectorizer = TfidfVectorizer(stop_words='english', lowercase=True)
        self.response_vectors = None
        self.setup_similarity_matching()
        
        print("✓ Chatbot initialized successfully!")
    
    def setup_similarity_matching(self):
        """Set up similarity matching for better responses"""
        # Create a corpus of possible responses for similarity matching
        self.response_corpus = []
        self.response_mapping = {}
        
        # Add responses with their categories
        for category, responses in self.topic_responses.items():
            for response in responses:
                self.response_corpus.append(response)
                self.response_mapping[response] = category
    
    def preprocess_text_nltk(self, text):
        """Preprocess text using NLTK"""
        try:
            # Tokenize
            tokens = word_tokenize(text.lower())
            
            # Remove stopwords and punctuation
            tokens = [token for token in tokens if token.isalnum() and token not in self.stop_words]
            
            # Stem words
            stemmed_tokens = [self.stemmer.stem(token) for token in tokens]
            
            return stemmed_tokens
        except:
            # Fallback to basic preprocessing
            words = text.lower().split()
            return [word.strip('.,!?;:"()[]{}') for word in words if word.strip('.,!?;:"()[]{}')]
    
    def preprocess_text_spacy(self, text):
        """Preprocess text using spaCy"""
        if not self.spacy_available:
            return self.preprocess_text_nltk(text)
        
        try:
            doc = self.nlp(text.lower())
            
            # Extract lemmatized tokens, removing stopwords and punctuation
            tokens = [token.lemma_ for token in doc 
                     if not token.is_stop and not token.is_punct and token.is_alpha]
            
            return tokens
        except:
            return self.preprocess_text_nltk(text)
    
    def extract_entities_spacy(self, text):
        """Extract named entities using spaCy"""
        if not self.spacy_available:
            return []
        
        try:
            doc = self.nlp(text)
            entities = [(ent.text, ent.label_) for ent in doc.ents]
            return entities
        except:
            return []
    
    def get_pos_tags(self, text):
        """Get part-of-speech tags using NLTK"""
        try:
            tokens = word_tokenize(text)
            pos_tags = pos_tag(tokens)
            return pos_tags
        except:
            # Fallback - return basic word list
            words = text.split()
            return [(word, 'UNKNOWN') for word in words]
    
    def detect_intent(self, text):
        """Detect user intent from the input text"""
        text_lower = text.lower()
        
        # Greeting patterns
        greeting_patterns = ['hello', 'hi', 'hey', 'greetings', 'good morning', 'good afternoon', 'good evening']
        if any(pattern in text_lower for pattern in greeting_patterns):
            return 'greeting'
        
        # Goodbye patterns
        goodbye_patterns = ['bye', 'goodbye', 'see you', 'farewell', 'exit', 'quit', 'later']
        if any(pattern in text_lower for pattern in goodbye_patterns):
            return 'goodbye'
        
        # Thank you patterns
        thank_patterns = ['thank', 'thanks', 'appreciate', 'grateful']
        if any(pattern in text_lower for pattern in thank_patterns):
            return 'thanks'
        
        # Question patterns
        question_words = ['what', 'when', 'where', 'why', 'how', 'who', 'which', 'can', 'could', 'would', 'should']
        if any(word in text_lower for word in question_words) or text.endswith('?'):
            return 'question'
        
        # Help patterns
        help_patterns = ['help', 'assist', 'support', 'guide']
        if any(pattern in text_lower for pattern in help_patterns):
            return 'help_request'
        
        return 'statement'
    
    def detect_topic(self, text):
        """Detect the topic of conversation"""
        text_lower = text.lower()
        
        # Weather keywords
        weather_keywords = ['weather', 'rain', 'sunny', 'temperature', 'forecast', 'climate', 'snow', 'wind', 'storm']
        if any(keyword in text_lower for keyword in weather_keywords):
            return 'weather'
        
        # Technology keywords
        tech_keywords = ['computer', 'software', 'programming', 'ai', 'technology', 'internet', 'code', 'python', 'java', 'javascript', 'machine learning', 'algorithm', 'data']
        if any(keyword in text_lower for keyword in tech_keywords):
            return 'technology'
        
        # Science keywords
        science_keywords = ['science', 'research', 'experiment', 'discovery', 'biology', 'physics', 'chemistry', 'mathematics', 'laboratory', 'theory']
        if any(keyword in text_lower for keyword in science_keywords):
            return 'science'
        
        # Food keywords
        food_keywords = ['food', 'recipe', 'cooking', 'restaurant', 'meal', 'cuisine', 'dish', 'eat', 'drink', 'chef', 'kitchen']
        if any(keyword in text_lower for keyword in food_keywords):
            return 'food'
        
        # Health keywords
        health_keywords = ['health', 'fitness', 'exercise', 'diet', 'medicine', 'doctor', 'hospital', 'wellness', 'medical']
        if any(keyword in text_lower for keyword in health_keywords):
            return 'health'
        
        # Education keywords
        education_keywords = ['education', 'school', 'university', 'study', 'learn', 'teaching', 'student', 'course', 'lesson']
        if any(keyword in text_lower for keyword in education_keywords):
            return 'education'
        
        return 'general'
    
    def generate_response(self, user_input):
        """Generate an appropriate response based on user input"""
        # Detect intent and topic
        intent = self.detect_intent(user_input)
        topic = self.detect_topic(user_input)
        
        # Extract entities if spaCy is available
        entities = self.extract_entities_spacy(user_input)
        
        # Generate response based on intent
        if intent == 'greeting':
            response = random.choice(self.knowledge_base['greetings'])
        elif intent == 'goodbye':
            response = random.choice(self.knowledge_base['goodbyes'])
        elif intent == 'thanks':
            response = random.choice(self.knowledge_base['thanks'])
        elif intent == 'help_request':
            response = "I'm here to help! You can ask me about various topics, or use commands like 'stats', 'entities', or 'sentiment'."
        elif topic in self.topic_responses:
            response = random.choice(self.topic_responses[topic])
        else:
            response = random.choice(self.knowledge_base['default'])
        
        # Add entity-specific information if available
        if entities and len(entities) > 0:
            entity_info = ", ".join([f"{ent[0]} ({ent[1]})" for ent in entities[:2]])
            response += f" I noticed you mentioned: {entity_info}."
        
        return response
    
    def analyze_sentiment(self, text):
        """Simple sentiment analysis using keyword matching"""
        positive_words = ['good', 'great', 'excellent', 'amazing', 'wonderful', 'fantastic', 'love', 'like', 'happy', 'joy', 'awesome', 'brilliant', 'perfect', 'beautiful']
        negative_words = ['bad', 'terrible', 'awful', 'hate', 'dislike', 'sad', 'angry', 'frustrated', 'disappointed', 'horrible', 'disgusting', 'annoying', 'boring']
        
        text_lower = text.lower()
        positive_count = sum(1 for word in positive_words if word in text_lower)
        negative_count = sum(1 for word in negative_words if word in text_lower)
        
        if positive_count > negative_count:
            return 'positive'
        elif negative_count > positive_count:
            return 'negative'
        else:
            return 'neutral'
    
    def get_conversation_stats(self, user_input):
        """Get linguistic statistics about the input"""
        try:
            # Basic stats
            word_count = len(word_tokenize(user_input))
            sentence_count = len(sent_tokenize(user_input))
        except:
            # Fallback counting
            word_count = len(user_input.split())
            sentence_count = user_input.count('.') + user_input.count('!') + user_input.count('?') + 1
        
        # POS tags
        pos_tags = self.get_pos_tags(user_input)
        noun_count = sum(1 for word, pos in pos_tags if pos.startswith('NN'))
        verb_count = sum(1 for word, pos in pos_tags if pos.startswith('VB'))
        
        # Sentiment
        sentiment = self.analyze_sentiment(user_input)
        
        return {
            'words': word_count,
            'sentences': sentence_count,
            'nouns': noun_count,
            'verbs': verb_count,
            'sentiment': sentiment
        }
    
    def chat(self):
        """Main chat loop"""
        print("\n" + "="*50)
        print("🤖 NLP CHATBOT - Ready to Chat!")
        print("="*50)
        print("Type 'quit', 'exit', or 'bye' to end the conversation")
        print("Type 'stats' to see conversation statistics")
        print("Type 'help' for available commands")
        print("-"*50 + "\n")
        
        conversation_history = []
        
        while True:
            try:
                user_input = input("You: ").strip()
                
                if not user_input:
                    print("Bot: Please say something!")
                    continue
                
                # Handle special commands
                if user_input.lower() in ['quit', 'exit', 'bye']:
                    print("Bot: " + random.choice(self.knowledge_base['goodbyes']))
                    break
                
                if user_input.lower() == 'help':
                    print("Bot: Available commands:")
                    print("  - 'stats': Show conversation statistics")
                    print("  - 'entities': Show named entities in your last message")
                    print("  - 'sentiment': Analyze sentiment of your last message")
                    print("  - 'analysis': Show detailed NLP analysis")
                    print("  - 'quit/exit/bye': End conversation")
                    continue
                
                if user_input.lower() == 'stats':
                    if conversation_history:
                        last_input = conversation_history[-1]
                        stats = self.get_conversation_stats(last_input)
                        print(f"Bot: Statistics for your last message:")
                        print(f"  Words: {stats['words']}, Sentences: {stats['sentences']}")
                        print(f"  Nouns: {stats['nouns']}, Verbs: {stats['verbs']}")
                        print(f"  Sentiment: {stats['sentiment']}")
                    else:
                        print("Bot: No conversation history yet!")
                    continue
                
                if user_input.lower() == 'entities':
                    if conversation_history:
                        entities = self.extract_entities_spacy(conversation_history[-1])
                        if entities:
                            print("Bot: Named entities found:")
                            for entity, label in entities:
                                print(f"  - {entity} ({label})")
                        else:
                            print("Bot: No named entities found in your last message.")
                    else:
                        print("Bot: No conversation history yet!")
                    continue
                
                if user_input.lower() == 'sentiment':
                    if conversation_history:
                        sentiment = self.analyze_sentiment(conversation_history[-1])
                        print(f"Bot: Sentiment of your last message: {sentiment}")
                    else:
                        print("Bot: No conversation history yet!")
                    continue
                
                if user_input.lower() == 'analysis':
                    if conversation_history:
                        last_input = conversation_history[-1]
                        print(f"Bot: Detailed analysis of: '{last_input}'")
                        
                        # Show preprocessing results
                        nltk_tokens = self.preprocess_text_nltk(last_input)
                        spacy_tokens = self.preprocess_text_spacy(last_input)
                        
                        print(f"  NLTK tokens: {nltk_tokens[:10]}...")  # Show first 10
                        print(f"  spaCy tokens: {spacy_tokens[:10]}...")
                        
                        # Show intent and topic
                        intent = self.detect_intent(last_input)
                        topic = self.detect_topic(last_input)
                        print(f"  Intent: {intent}, Topic: {topic}")
                        
                        # Show entities
                        entities = self.extract_entities_spacy(last_input)
                        if entities:
                            print(f"  Entities: {entities}")
                    else:
                        print("Bot: No conversation history yet!")
                    continue
                
                # Store conversation history
                conversation_history.append(user_input)
                
                # Generate and display response
                response = self.generate_response(user_input)
                print(f"Bot: {response}")
                
            except KeyboardInterrupt:
                print("\n\nBot: Goodbye! Thanks for chatting!")
                break
            except Exception as e:
                print(f"Bot: Sorry, I encountered an error: {str(e)}")

def demo_conversation():
    """Demonstrate chatbot output with sample conversations"""
    print("\n" + "="*60)
    print("🎬 CHATBOT OUTPUT DEMONSTRATION")
    print("="*60)
    
    # Sample conversation inputs
    demo_inputs = [
        "Hello there!",
        "How's the weather today?",
        "I love programming and artificial intelligence",
        "What do you think about machine learning?",
        "Can you tell me about cooking pasta?",
        "I'm feeling great today!",
        "Thank you for your help"
    ]
    
    print("Creating demo conversation...\n")
    chatbot = NLPChatbot()
    conversation_history = []
    
    for i, user_input in enumerate(demo_inputs, 1):
        print(f"You: {user_input}")
        
        conversation_history.append(user_input)
        response = chatbot.generate_response(user_input)
        print(f"Bot: {response}")
        print()  # Add spacing between exchanges
    
    print("="*60)
    print("🔍 TECHNICAL ANALYSIS EXAMPLE")
    print("="*60)
    
    # Show detailed analysis for a sample input
    sample_text = "I love programming and artificial intelligence"
    print(f"Analyzing: '{sample_text}'\n")
    
    # NLTK preprocessing
    nltk_tokens = chatbot.preprocess_text_nltk(sample_text)
    print(f"NLTK Preprocessing: {nltk_tokens}")
    
    # spaCy preprocessing
    spacy_tokens = chatbot.preprocess_text_spacy(sample_text)
    print(f"spaCy Preprocessing: {spacy_tokens}")
    
    # POS tagging
    pos_tags = chatbot.get_pos_tags(sample_text)
    print(f"POS Tags: {pos_tags}")
    
    # Intent and topic detection
    intent = chatbot.detect_intent(sample_text)
    topic = chatbot.detect_topic(sample_text)
    print(f"Detected Intent: {intent}")
    print(f"Detected Topic: {topic}")
    
    # Sentiment analysis
    sentiment = chatbot.analyze_sentiment(sample_text)
    print(f"Sentiment: {sentiment}")
    
    # Named entities
    entities = chatbot.extract_entities_spacy(sample_text)
    print(f"Named Entities: {entities}")
    
    # Statistics
    stats = chatbot.get_conversation_stats(sample_text)
    print(f"Statistics: {stats}")

def main():
    """Main function to run the chatbot"""
    print("🚀 NLP Chatbot - Natural Language Processing Demo")
    print("\nRequired libraries:")
    print("- nltk: pip install nltk")
    print("- spacy: pip install spacy")
    print("- scikit-learn: pip install scikit-learn")
    print("- For spaCy model: python -m spacy download en_core_web_sm")
    print("\n" + "-"*50)
    
    try:
        # Initialize and run chatbot
        chatbot = NLPChatbot()
        chatbot.chat()
        
    except Exception as e:
        print(f"❌ Error starting chatbot: {str(e)}")
        print("\nMake sure you have installed all required libraries:")
        print("pip install nltk spacy scikit-learn numpy")
        print("python -m spacy download en_core_web_sm")

if __name__ == "__main__":
    print("🤖 NLP Chatbot Menu")
    print("="*30)
    print("1. Run interactive chatbot")
    print("2. Show demo output")
    print("3. Exit")
    
    try:
        choice = input("\nEnter your choice (1-3): ").strip()
        
        if choice == "1":
            main()
        elif choice == "2":
            demo_conversation()
        elif choice == "3":
            print("Goodbye!")
            sys.exit(0)
        else:
            print("Invalid choice. Running interactive chatbot...")
            main()
    except KeyboardInterrupt:
        print("\nGoodbye!")
        sys.exit(0)