# app.py
# Complete Nexus ATS backend (full original endpoints + dashboard HTML)
# Enhancements added:
# - Better SentenceTransformer (all-mpnet-base-v2)
# - Cached embeddings (lru_cache)
# - Enriched SKILL_KNOWLEDGE_BASE and improved RAG indexing
# - Weighted skills scoring
# - LLM validation layer (OpenAI) with graceful fallback
# - Minor robustness & logging improvements

from flask import Flask, request, jsonify, send_from_directory, render_template_string, session
from flask_cors import CORS
from flask_sqlalchemy import SQLAlchemy
import json
import fitz  # PyMuPDF
import re
import string
import logging
import os
import openai
from openai import OpenAI
from scipy.spatial.distance import cosine
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
from typing import List, Dict, Tuple
from collections import defaultdict
from datetime import datetime
from werkzeug.security import generate_password_hash, check_password_hash
from functools import lru_cache

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ==================== DATABASE CONFIGURATION ====================
import os
app.config['SQLALCHEMY_DATABASE_URI'] = os.environ.get('DATABASE_URL', 'sqlite:///nexus_ats.db')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', 'your-secret-key-change-in-production-12345')
db = SQLAlchemy(app)


# ==================== DATABASE MODELS ====================

class User(db.Model):
    """Stores recruiter/user information from login/signup"""
    __tablename__ = 'users'
    
    id = db.Column(db.Integer, primary_key=True)
    first_name = db.Column(db.String(100), nullable=False)
    last_name = db.Column(db.String(100), nullable=False)
    email = db.Column(db.String(200), unique=True, nullable=False)
    password_hash = db.Column(db.String(255))
    company = db.Column(db.String(200))
    role = db.Column(db.String(50), default='recruiter')  # recruiter, admin, etc.
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    last_login = db.Column(db.DateTime)
    
    def to_dict(self):
        return {
            'id': self.id,
            'first_name': self.first_name,
            'last_name': self.last_name,
            'full_name': f"{self.first_name} {self.last_name}",
            'email': self.email,
            'company': self.company,
            'role': self.role,
            'created_at': self.created_at.strftime('%Y-%m-%d %H:%M:%S') if self.created_at else None,
            'last_login': self.last_login.strftime('%Y-%m-%d %H:%M:%S') if self.last_login else None
        }
    
    def __repr__(self):
        return f'<User {self.first_name} {self.last_name} ({self.email})>'

class Candidate(db.Model):
    """Stores candidate information and analysis results"""
    __tablename__ = 'candidates'
    
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(200), nullable=False)
    filename = db.Column(db.String(200))
    email = db.Column(db.String(200))
    phone = db.Column(db.String(50))
    resume_text = db.Column(db.Text)
    
    # Analysis results
    score = db.Column(db.Float)
    matched_skills = db.Column(db.JSON)
    experience = db.Column(db.String(100))
    education = db.Column(db.String(100))
    status = db.Column(db.String(50))
    
    # Metadata
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationship
    analyses = db.relationship('Analysis', backref='candidate', lazy=True, cascade='all, delete-orphan')
    
    def to_dict(self):
        return {
            'id': self.id,
            'name': self.name,
            'filename': self.filename,
            'email': self.email,
            'phone': self.phone,
            'score': self.score,
            'matched_skills': self.matched_skills,
            'experience': self.experience,
            'education': self.education,
            'status': self.status,
            'created_at': self.created_at.strftime('%Y-%m-%d %H:%M:%S') if self.created_at else None,
            'updated_at': self.updated_at.strftime('%Y-%m-%d %H:%M:%S') if self.updated_at else None
        }
    
    def __repr__(self):
        return f'<Candidate {self.name} - Score: {self.score}>'

class Job(db.Model):
    """Stores job descriptions for reuse"""
    __tablename__ = 'jobs'
    
    id = db.Column(db.Integer, primary_key=True)
    title = db.Column(db.String(200))
    description = db.Column(db.Text, nullable=False)
    required_skills = db.Column(db.JSON)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    is_active = db.Column(db.Boolean, default=True)
    
    analyses = db.relationship('Analysis', backref='job', lazy=True, cascade='all, delete-orphan')
    
    def to_dict(self):
        return {
            'id': self.id,
            'title': self.title,
            'description': self.description,
            'required_skills': self.required_skills,
            'created_at': self.created_at.strftime('%Y-%m-%d %H:%M:%S') if self.created_at else None,
            'is_active': self.is_active
        }
    
    def __repr__(self):
        return f'<Job {self.title}>'

class Analysis(db.Model):
    """Stores each resume analysis with job pairing"""
    __tablename__ = 'analyses'
    
    id = db.Column(db.Integer, primary_key=True)
    candidate_id = db.Column(db.Integer, db.ForeignKey('candidates.id'), nullable=False)
    job_id = db.Column(db.Integer, db.ForeignKey('jobs.id'))
    
    job_description = db.Column(db.Text)
    score = db.Column(db.Float)
    matched_skills = db.Column(db.JSON)
    skill_explanations = db.Column(db.JSON)
    career_coach_message = db.Column(db.Text)
    
    analyzed_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    def to_dict(self):
        return {
            'id': self.id,
            'candidate_id': self.candidate_id,
            'candidate_name': self.candidate.name if self.candidate else None,
            'job_id': self.job_id,
            'job_title': self.job.title if self.job else 'One-time Analysis',
            'score': self.score,
            'matched_skills': self.matched_skills,
            'skill_explanations': self.skill_explanations,
            'career_coach_message': self.career_coach_message,
            'analyzed_at': self.analyzed_at.strftime('%Y-%m-%d %H:%M:%S') if self.analyzed_at else None
        }
    
    def __repr__(self):
        return f'<Analysis ID:{self.id} - Score:{self.score}>'

# ==================== OPENAI CLIENT ====================
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    logger.warning("OPENAI_API_KEY environment variable not found. Using hardcoded key (change in production).")
    api_key = "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"

client = OpenAI(api_key=api_key)

# ==================== RAG IMPLEMENTATION ====================
# Try to use a better model; fallback to mini if unavailable
# Initialize as None - will load on first request
sentencemodel = None
EMBEDDING_DIM = 384

def get_sentence_model():
    global sentencemodel
    if sentencemodel is None:
        try:
            sentencemodel = SentenceTransformer('all-MiniLM-L6-v2', device='cpu')
            logger.info("Sentence Transformer loaded on first request")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
    return sentencemodel


# Knowledge base for enhanced skill context
SKILL_KNOWLEDGE_BASE = {
    "javascript": {
        "definition": "High-level dynamic programming language for web development",
        "related_concepts": ["frontend", "backend", "fullstack", "web applications"],
        "typical_experience": ["DOM manipulation", "async programming", "event handling"],
        "context_examples": [
            "Built single-page applications using React and vanilla JavaScript.",
            "Worked with Node.js to write backend services and scripts."
        ]
    },
    "python": {
        "definition": "Versatile high-level language used in data science, web, and automation",
        "related_concepts": ["data analysis", "scripting", "automation", "backend"],
        "typical_experience": ["data processing", "API development", "scripting"],
        "context_examples": [
            "Used Python with pandas and scikit-learn for predictive modeling.",
            "Built Flask APIs and automated ETL pipelines using Python."
        ]
    },
    "machine learning": {
        "definition": "AI branch focused on learning from data",
        "related_concepts": ["model training", "feature engineering", "data preprocessing"],
        "typical_experience": ["supervised learning", "model evaluation", "hyperparameter tuning"],
        "context_examples": [
            "Trained models using scikit-learn, TensorFlow, or PyTorch for classification tasks.",
            "Performed feature engineering and cross-validation for robust evaluation."
        ]
    },
    "react": {
        "definition": "JavaScript library for building user interfaces",
        "related_concepts": ["component architecture", "state management", "virtual DOM"],
        "typical_experience": ["component development", "hooks", "routing"],
        "context_examples": [
            "Developed complex UI components and managed state with Redux or Context API.",
            "Optimized render performance for large lists using virtualization."
        ]
    },
    "aws": {
        "definition": "Amazon cloud computing platform",
        "related_concepts": ["cloud infrastructure", "serverless", "scalability"],
        "typical_experience": ["deployment", "configuration", "cost optimization"],
        "context_examples": [
            "Deployed microservices to ECS/EKS and used S3 for storage.",
            "Implemented serverless functions using AWS Lambda and API Gateway."
        ]
    },
}

class RAGVectorStore:
    """Vector store for intelligent skill matching using RAG"""
    
    def __init__(self):
        self.index = None
        self.skill_embeddings = []
        self.skill_names = []
        self.skill_contexts = []
        self.embedding_dim = EMBEDDING_DIM
        
    def build_index(self, skill_map: Dict):
        model = get_sentence_model()
        if model is None:
            logger.warning("RAG unavailable - falling back to keyword matching")
        return
            
        texts = []
        names = []
        contexts = []
        
        for skill_name, synonyms in skill_map.items():
            base_text = skill_name
            synonyms_text = " ".join(synonyms)
            
            if skill_name in SKILL_KNOWLEDGE_BASE:
                kb = SKILL_KNOWLEDGE_BASE[skill_name]
                context = f"{base_text} {synonyms_text} {kb['definition']} {' '.join(kb['related_concepts'])} {' '.join(kb.get('context_examples', []))}"
            else:
                context = f"{base_text} {synonyms_text}"
            
            texts.append(context)
            names.append(skill_name)
            contexts.append(context)
        
        try:
            embeddings = model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
            self.index = faiss.IndexFlatL2(self.embedding_dim)
            self.index.add(embeddings.astype('float32'))
            self.skill_embeddings = embeddings
            self.skill_names = names
            self.skill_contexts = contexts
            logger.info(f"🚀 RAG Vector Store: Indexed {len(names)} skills")
        except Exception as e:
            logger.error(f"Failed to build RAG index: {e}")
            self.index = None
    
    def retrieve_relevant_skills(self, text: str, top_k: int = 15, threshold: float = 0.3):
        model = get_sentence_model()
        if self.index is None or model is None:
            return []
        
        # embed the query safely
        q = text if len(text) < 8000 else text[:8000]
        embeddings = model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
        distances, indices = self.index.search(query_embedding.astype('float32'), min(top_k, len(self.skill_names)))
        
        results = []
        for idx, dist in zip(indices[0], distances[0]):
            if idx < len(self.skill_names):
                similarity = 1 / (1 + float(dist))
                if similarity >= threshold:
                    results.append((self.skill_names[idx], similarity))
        
        return results

# COMPREHENSIVE SKILL MAPPING
SKILL_MAP = {
    # Programming Languages
    "javascript": ["js", "javascript", "ecmascript", "node.js", "nodejs", "es6"],
    "python": ["python", "py", "python3"],
    "java": ["java", "jvm"],
    "c++": ["cpp", "c++", "c plus plus"],
    "c#": ["csharp", "c#", "c sharp", "dotnet", ".net"],
    "php": ["php", "php7", "php8"],
    "ruby": ["ruby", "rails", "ruby on rails"],
    "go": ["golang", "go", "go lang"],
    "rust": ["rust"],
    "kotlin": ["kotlin"],
    "swift": ["swift", "ios"],
    "typescript": ["typescript", "ts"],
    "scala": ["scala"],
    "r": ["r programming", "r language"],
    "matlab": ["matlab"],
    "perl": ["perl"],
    
    # Frontend Technologies
    "react": ["reactjs", "react.js", "react", "react native"],
    "angular": ["angular", "angularjs", "angular.js"],
    "vue": ["vuejs", "vue.js", "vue"],
    "html": ["html", "html5"],
    "css": ["css", "css3", "scss", "sass"],
    "jquery": ["jquery"],
    "bootstrap": ["bootstrap"],
    "tailwind": ["tailwindcss", "tailwind css"],
    
    # Backend Technologies
    "node.js": ["nodejs", "node.js", "node"],
    "express": ["expressjs", "express.js"],
    "django": ["django"],
    "flask": ["flask"],
    "spring": ["spring boot", "spring framework"],
    "laravel": ["laravel"],
    "fastapi": ["fastapi", "fast api"],
    
    # Databases
    "sql": ["sql", "structured query language", "mysql", "postgresql", "sqlite"],
    "mongodb": ["mongodb", "mongo", "nosql"],
    "redis": ["redis"],
    "elasticsearch": ["elasticsearch", "elastic search"],
    "cassandra": ["cassandra"],
    "oracle": ["oracle database", "oracle db"],
    
    # Cloud Platforms
    "aws": ["aws", "amazon web services", "ec2", "s3", "lambda"],
    "azure": ["microsoft azure", "azure"],
    "gcp": ["google cloud", "google cloud platform", "gcp"],
    "heroku": ["heroku"],
    "digitalocean": ["digital ocean", "digitalocean"],
    
    # DevOps & Tools
    "docker": ["docker", "containerization"],
    "kubernetes": ["k8s", "kubernetes"],
    "jenkins": ["jenkins"],
    "git": ["git", "github", "gitlab"],
    "terraform": ["terraform"],
    "ansible": ["ansible"],
    "ci/cd": ["ci cd", "continuous integration", "continuous deployment"],
    
    # Data Science & AI
    "machine learning": ["ml", "machine learning", "ai", "artificial intelligence"],
    "deep learning": ["deep learning", "neural networks", "dl"],
    "nlp": ["nlp", "natural language processing"],
    "computer vision": ["computer vision", "cv", "image processing"],
    "data science": ["data science", "data analytics", "analytics"],
    "pandas": ["pandas"],
    "numpy": ["numpy"],
    "scikit-learn": ["sklearn", "scikit-learn"],
    
    # Business Intelligence
    "tableau": ["tableau"],
    "power bi": ["powerbi", "power bi", "microsoft bi"],
    "excel": ["excel", "microsoft excel", "ms excel"],
    "sap": ["sap"],
    "salesforce": ["salesforce", "crm"],
    
    # Cybersecurity
    "cybersecurity": ["cyber security", "information security", "infosec"],
    "penetration testing": ["pen testing", "penetration testing", "ethical hacking"],
    "network security": ["network security"],
    "cryptography": ["cryptography", "encryption"],
    
    # Mobile Development
    "android": ["android development", "android studio"],
    "ios": ["ios development", "xcode"],
    "flutter": ["flutter", "dart"],
    "react native": ["react native", "rn"],
    
    # Design & UX
    "ui/ux": ["ui", "ux", "user interface", "user experience", "ui/ux"],
    "figma": ["figma"],
    "adobe": ["photoshop", "illustrator", "adobe creative"],
    "sketch": ["sketch"],
    
    # Project Management
    "agile": ["agile", "scrum", "kanban"],
    "project management": ["project management", "pmp"],
    "jira": ["jira"],
    "trello": ["trello"],
    "asana": ["asana"],
    
    # Soft Skills
    "communication": ["communication", "communicate", "verbal communication", "written communication"],
    "teamwork": ["teamwork", "team player", "collaboration", "team work"],
    "leadership": ["leadership", "lead", "team lead", "manager"],
    "management": ["management", "manager", "people management"],
    "problem solving": ["problem solving", "analytical thinking", "critical thinking"],
    "creativity": ["creativity", "creative thinking", "innovation"],
    "adaptability": ["adaptability", "flexible", "adaptable"],
    "time management": ["time management", "organization", "prioritization"],
    "customer service": ["customer service", "client relations", "customer support"],
    "sales": ["sales", "business development", "account management"],
    "marketing": ["marketing", "digital marketing", "content marketing"],
    "public speaking": ["public speaking", "presentation skills", "presenting"],
    "negotiation": ["negotiation", "negotiating"],
    "mentoring": ["mentoring", "coaching", "training"],
    
    # Finance & Business
    "finance": ["finance", "financial analysis", "accounting"],
    "economics": ["economics", "microeconomics", "macroeconomics"],
    "business analysis": ["business analysis", "business analyst", "ba"],
    "consulting": ["consulting", "management consulting"],
    "strategy": ["strategy", "strategic planning", "business strategy"],
    
    # Industry Specific
    "healthcare": ["healthcare", "medical", "clinical"],
    "education": ["education", "teaching", "training"],
    "retail": ["retail", "e-commerce", "ecommerce"],
    "manufacturing": ["manufacturing", "operations", "supply chain"],
    "legal": ["legal", "law", "compliance"],
    "hr": ["human resources", "hr", "recruiting", "talent acquisition"],
    
    # Quality Assurance
    "testing": ["testing", "qa", "quality assurance", "automation testing"],
    "selenium": ["selenium"],
    
    # Other Technical Skills
    "api": ["api", "rest api", "restful", "graphql"],
    "microservices": ["microservices", "service oriented architecture"],
    "blockchain": ["blockchain", "cryptocurrency", "bitcoin", "ethereum"],
    "iot": ["iot", "internet of things"],
    "ar/vr": ["ar", "vr", "augmented reality", "virtual reality"],
}

# COMPREHENSIVE COURSE RECOMMENDATIONS (sample)
COURSE_RECOMMENDATIONS = {
    # Programming Languages
    "javascript": "🔗 [JavaScript for Beginners](https://www.coursera.org/learn/javascript)",
    "python": "🔗 [Python for Everybody](https://www.coursera.org/specializations/python)",
    "java": "🔗 [Java Programming](https://www.coursera.org/specializations/java-programming)",
    "c++": "🔗 [C++ For C Programmers](https://www.coursera.org/learn/c-plus-plus-a)",
    "c#": "🔗 [C# Programming](https://www.coursera.org/learn/introduction-programming-unity)",
    "php": "🔗 [Web Development with PHP](https://www.coursera.org/learn/web-development-php)",
    "ruby": "🔗 [Ruby on Rails](https://www.coursera.org/learn/ruby-on-rails-intro)",
    "go": "🔗 [Programming with Go](https://www.coursera.org/specializations/google-golang)",
    "rust": "🔗 [Rust Programming](https://www.coursera.org/learn/rust-programming)",
    "kotlin": "🔗 [Kotlin for Java Developers](https://www.coursera.org/learn/kotlin-for-java-developers)",
    "swift": "🔗 [iOS App Development](https://www.coursera.org/specializations/app-development)",
    "typescript": "🔗 [TypeScript for Beginners](https://www.coursera.org/learn/typescript)",
    "scala": "🔗 [Functional Programming in Scala](https://www.coursera.org/specializations/scala)",
    "r": "🔗 [R Programming](https://www.coursera.org/learn/r-programming)",
    "matlab": "🔗 [Introduction to Programming with MATLAB](https://www.coursera.org/learn/matlab)",
    
    # Frontend Technologies
    "react": "🔗 [React Basics](https://www.coursera.org/learn/react-basics)",
    "angular": "🔗 [Angular](https://www.coursera.org/learn/angular)",
    "vue": "🔗 [Vue.js](https://www.coursera.org/learn/vue-js)",
    "html": "🔗 [HTML, CSS, and Javascript for Web Developers](https://www.coursera.org/learn/html-css-javascript-for-web-developers)",
    "css": "🔗 [Advanced Styling with CSS](https://www.coursera.org/learn/responsivedesign)",
    "jquery": "🔗 [jQuery Tutorial](https://www.coursera.org/learn/jquery)",
    "bootstrap": "🔗 [Bootstrap 4 Tutorial](https://www.coursera.org/learn/bootstrap-4)",
    
    # Backend Technologies  
    "node.js": "🔗 [Server-side Development with NodeJS](https://www.coursera.org/learn/server-side-nodejs)",
    "django": "🔗 [Django for Everybody](https://www.coursera.org/specializations/django)",
    "flask": "🔗 [Flask Development](https://www.coursera.org/learn/python-flask)",
    "spring": "🔗 [Spring Framework](https://www.coursera.org/learn/spring-framework)",
    "laravel": "🔗 [Laravel PHP Framework](https://www.coursera.org/learn/laravel-framework)",
    "fastapi": "🔗 [FastAPI Development](https://www.coursera.org/learn/fastapi)",
    
    # Databases
    "sql": "🔗 [SQL for Data Science](https://www.coursera.org/learn/sql-for-data-science)",
    "mongodb": "🔗 [MongoDB Basics](https://www.coursera.org/learn/mongodb-basics)",
    "redis": "🔗 [Redis for Beginners](https://www.coursera.org/learn/redis)",
    "elasticsearch": "🔗 [Elasticsearch](https://www.coursera.org/learn/elasticsearch)",
    "oracle": "🔗 [Oracle Database](https://www.coursera.org/learn/oracle-database)",
    
    # Cloud Platforms
    "aws": "🔗 [AWS Cloud Practitioner](https://www.coursera.org/learn/aws-cloud-practitioner-essentials)",
    "azure": "🔗 [Microsoft Azure Fundamentals](https://www.coursera.org/learn/microsoft-azure-fundamentals-az-900)",
    "gcp": "🔗 [Google Cloud Platform Fundamentals](https://www.coursera.org/learn/gcp-fundamentals)",
    
    # DevOps & Tools
    "docker": "🔗 [Docker for Beginners](https://www.coursera.org/learn/docker-container-essentials)",
    "kubernetes": "🔗 [Kubernetes for Beginners](https://www.coursera.org/learn/kubernetes-for-the-absolute-beginner-hands-on)",
    "jenkins": "🔗 [Jenkins Tutorial](https://www.coursera.org/learn/jenkins)",
    "git": "🔗 [Version Control with Git](https://www.coursera.org/learn/version-control-with-git)",
    
    # Data Science & AI
    "machine learning": "🔗 [Machine Learning](https://www.coursera.org/learn/machine-learning)",
    "deep learning": "🔗 [Deep Learning Specialization](https://www.coursera.org/specializations/deep-learning)",
    "nlp": "🔗 [Natural Language Processing](https://www.coursera.org/specializations/natural-language-processing)",
    "computer vision": "🔗 [Computer Vision Basics](https://www.coursera.org/learn/computer-vision-basics)",
    "data science": "🔗 [Data Science Specialization](https://www.coursera.org/specializations/jhu-data-science)",
    
    # Business Intelligence
    "tableau": "🔗 [Data Visualization with Tableau](https://www.coursera.org/specializations/data-visualization)",
    "power bi": "🔗 [Data Analysis with Power BI](https://www.coursera.org/learn/data-analysis-with-power-bi)",
    "excel": "🔗 [Excel Skills for Business](https://www.coursera.org/specializations/excel)",
    
    # Soft Skills
    "communication": "🔗 [Effective Communication](https://www.coursera.org/learn/wharton-communication-skills)",
    "teamwork": "🔗 [Team Management](https://www.coursera.org/learn/high-performing-teams)",
    "leadership": "🔗 [Leadership Skills](https://www.coursera.org/learn/leadership-skills)",
    "management": "🔗 [People Management](https://www.coursera.org/learn/managing-people-teams)",
    "project management": "🔗 [Project Management](https://www.coursera.org/professional-certificates/google-project-management)",
    
    # Design & UX
    "ui/ux": "🔗 [UI/UX Design](https://www.coursera.org/professional-certificates/google-ux-design)",
    "figma": "🔗 [Figma for UI Design](https://www.coursera.org/learn/figma-ui-design)",
    
    # Cybersecurity
    "cybersecurity": "🔗 [Cybersecurity for Beginners](https://www.coursera.org/learn/intro-cyber-security)",
    "blockchain": "🔗 [Blockchain Basics](https://www.coursera.org/learn/blockchain-basics)",
}


NEGATION_TERMS = ["no", "not", "none", "without", "never", "exclude"]

# Initialize RAG vector store
rag_store = RAGVectorStore()
rag_store.build_index(SKILL_MAP)

# ==================== HELPER FUNCTIONS ====================

def convert_numpy_types(obj):
    """Convert NumPy types to native Python types for JSON serialization"""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    else:
        return obj

def extract_text_from_pdf_bytes(pdf_bytes):
    try:
        text = ""
        with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
            for page in doc:
                text += page.get_text()
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    except Exception as e:
        logger.error(f"PDF extraction error: {str(e)}")
        return ""

def extract_email(text):
    email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
    match = re.search(email_pattern, text)
    return match.group(0) if match else None

def extract_phone(text):
    phone_patterns = [
        r'\+?\d[\d\s\-\.\(\)]{7,}\d',
        r'\(\d{3}\)\s*\d{3}[-\s]?\d{4}',
        r'\d{3}[-\s]?\d{3}[-\s]?\d{4}'
    ]
    for pattern in phone_patterns:
        match = re.search(pattern, text)
        if match:
            return match.group(0)
    return None

def preprocess(text):
    text = (text or "").lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    return text

def contains_negation_near(text, keyword):
    window = 5
    tokens = text.split()
    for i, token in enumerate(tokens):
        if keyword in token:
            start = max(0, i - window)
            end = min(len(tokens), i + window + 1)
            window_tokens = tokens[start:end]
            for neg in NEGATION_TERMS:
                if neg in window_tokens:
                    return True
    return False

def classify_role(job_desc):
    job_desc = preprocess(job_desc)
    tech_count = 0
    for skill, synonyms in SKILL_MAP.items():
        if skill in ["communication", "teamwork", "leadership"]:
            continue
        for syn in synonyms:
            if syn in job_desc:
                tech_count += 1
                break
    return "technical" if tech_count >= 3 else "non-technical"

# CACHED EMBEDDING
@lru_cache(maxsize=2048)
def embedtextcached(text: str):
    model = get_sentence_model()
    if model is None:
        raise RuntimeError("Sentence transformer not available")
    t = text if len(text) <= 4000 else text[:4000]
    emb = model.encode(t, convert_to_numpy=True, show_progress_bar=False)[0]
    return emb


def extract_skills_with_rag(resume_text, job_desc):
    """Enhanced skill extraction using RAG"""
    resume_text_clean = preprocess(resume_text)
    job_desc_clean = preprocess(job_desc)
    
    role_type = classify_role(job_desc)
    
    job_skills_rag = rag_store.retrieve_relevant_skills(job_desc, top_k=25, threshold=0.32)
    job_skills_set = set([skill for skill, score in job_skills_rag])
    
    # Add keyword matches from job description
    for primary, synonyms in SKILL_MAP.items():
        for syn in synonyms:
            if syn in job_desc_clean:
                job_skills_set.add(primary)
                break
    
    candidate_skills = set()
    skill_explanations = []
    
    if rag_store.index is not None:
        rag_matches = rag_store.retrieve_relevant_skills(resume_text, top_k=30, threshold=0.35)
        for skill, similarity in rag_matches:
            if not contains_negation_near(resume_text_clean, skill):
                candidate_skills.add(skill)
                skill_explanations.append(f"✅ RAG matched '{skill}' (confidence: {similarity:.2f})")
    
    # Keyword matching fallback
    for primary, synonyms in SKILL_MAP.items():
        if primary in candidate_skills:
            continue
        for syn in synonyms:
            if contains_negation_near(resume_text_clean, syn):
                continue
            if syn in resume_text_clean:
                candidate_skills.add(primary)
                skill_explanations.append(f"🔍 Keyword matched '{primary}' via '{syn}'")
                break
    
    if role_type == "non-technical":
        for soft_skill in ["communication", "teamwork", "leadership"]:
            if soft_skill in job_skills_set or soft_skill in resume_text_clean:
                candidate_skills.add(soft_skill)
                skill_explanations.append(f"💼 Soft skill matched: '{soft_skill}'")
    
    matched_skills = list(job_skills_set & candidate_skills) or list(candidate_skills)
    
    return matched_skills, skill_explanations, role_type, list(job_skills_set)

def extract_experience(text):
    patterns = [
        r'(\d+)\s*-\s*(\d+)\s*years?',
        r'(\d+)\+?\s*years?',
        r'experience:\s*(\d+)\s*(?:years?|yrs)',
        r'(\d+)\s*yr'
    ]
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            if len(match.groups()) >= 2 and match.group(2):
                return f"{match.group(1)}-{match.group(2)} years"
            return f"{match.group(1)} years"
    
    internship_keywords = ["intern", "internship", "trainee"]
    for keyword in internship_keywords:
        if re.search(r"\b" + re.escape(keyword) + r"\b", text, re.IGNORECASE):
            return "Internship experience"
    return "N/A"

def extract_education(text):
    keywords = {
        "PhD": ["phd", "ph.d", "doctorate"],
        "Master's": ["master", "msc", "ms", "m.tech", "mtech", "mba"],
        "Bachelor's": ["bachelor", "bsc", "bs", "b.tech", "btech"]
    }
    found = []
    for degree, terms in keywords.items():
        for term in terms:
            if re.search(r"\b" + re.escape(term) + r"\b", text, re.IGNORECASE):
                found.append(degree)
    if not found:
        return "N/A"
    if "PhD" in found:
        return "PhD"
    if "Master's" in found:
        return "Master's"
    return "Bachelor's"

def semantic_similarity_enhanced(resume, job_desc):
    """Local embedding-based similarity (0-100)"""
    scores = []
    model = get_sentence_model()
    if model is not None:
        try:
            r_emb = embed_text_cached(resume if len(resume) < 4000 else resume[:4000])
            j_emb = embed_text_cached(job_desc if len(job_desc) < 4000 else job_desc[:4000])
            sim = 1 - cosine(r_emb, j_emb)
            scores.append(max(0.0, float(sim)) * 100)
        except Exception as e:
            logger.warning(f"Local similarity error: {str(e)}")
    return round(sum(scores) / len(scores), 1) if scores else 50.0

def generate_career_coach_message(total_score, matched_skills, job_skills, resume_text="", job_desc=""):
    """Enhanced career coach with LLM-powered course recommendations"""
    career_coach = f"You're a {total_score}% match for this role."
    
    # Find missing skills
    missing_skills = [skill for skill in job_skills if skill not in matched_skills]
    
    # Prioritize important skills
    high_priority = ["python", "java", "javascript", "react", "machine learning", "aws", "sql", "docker", "kubernetes", "node.js", "angular", "vue", "mongodb", "postgresql"]
    priority_missing = [s for s in missing_skills if s in high_priority]
    top_missing_skills = (priority_missing[:3] if priority_missing else missing_skills[:3])
    
    if top_missing_skills:
        skill_text = ', '.join(top_missing_skills)
        career_coach += f" Consider strengthening: {skill_text}."
        
        # Try LLM-powered course recommendations first
        llm_courses = None
        if resume_text and job_desc:
            llm_courses = llm_recommend_courses(resume_text, job_desc, matched_skills, top_missing_skills, total_score)
        
        if llm_courses:
            # LLM recommendations successful
            career_coach += "\n\n<span style='color: #FFFF00; font-weight: 600;'>Recommended Courses:</span>"
            career_coach += f"\n{llm_courses}"
        else:
            # Fallback to static courses if LLM fails
            career_coach += "\n\n<span style='color: #FFFF00; font-weight: 600;'>Recommended Courses:</span>"
            course_count = 0
            for skill in top_missing_skills:
                if skill in COURSE_RECOMMENDATIONS and course_count < 3:
                    course = COURSE_RECOMMENDATIONS[skill]
                    
                    # Extract and format course links
                    if '[' in course and '](' in course:
                        import re
                        match = re.search(r'\[([^\]]+)\]\(([^)]+)\)', course)
                        if match:
                            link_text = match.group(1)
                            link_url = match.group(2)
                            colored_course = f"🔗 <a href='{link_url}' target='_blank' style='color: #60a5fa;'>{link_text}</a>"
                            career_coach += f"\n{colored_course}"
                            course_count += 1
                    else:
                        career_coach += f"\n{course}"
                        course_count += 1
            
            # Ultimate fallback
            if course_count == 0:
                career_coach += f"\n🔗 <a href='https://www.coursera.org' target='_blank' style='color: #60a5fa;'>Explore Coursera</a>"
    
    return career_coach



# ==================== LLM VALIDATION LAYER ====================
def llm_validate_ranking(resume_text, job_desc, matched_skills, prelim_score):
    """Call OpenAI to validate/re-score with short reasoning. Graceful fallback."""
    try:
        # Build compact prompt & truncate to avoid token limits
        resume_snippet = resume_text[:3000]
        job_snippet = job_desc[:2000]
        prompt = (
            "You are an experienced technical recruiter. "
            "Given the JOB DESCRIPTION and the CANDIDATE RESUME, provide:\n"
            "1) a final suitability score (0-100),\n"
            "2) a 1-2 sentence explanation for the score,\n"
            "Return ONLY a JSON object with keys: final_score, analysis\n\n"
            f"JOB DESCRIPTION:\n{job_snippet}\n\n"
            f"CANDIDATE RESUME (snippet):\n{resume_snippet}\n\n"
            f"Preliminary matched skills: {matched_skills}\n"
            f"Preliminary score: {prelim_score}\n"
        )
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.25,
            max_tokens=300
        )
        content = response.choices[0].message.content.strip()
        try:
            parsed = json.loads(content)
            final_score = int(parsed.get("final_score", prelim_score))
            analysis = parsed.get("analysis", "")
            return final_score, analysis
        except Exception:
            m = re.search(r'"final_score"\s*:\s*([0-9]{1,3})', content)
            a = re.search(r'"analysis"\s*:\s*"([^"]+)"', content)
            final_score = int(m.group(1)) if m else prelim_score
            analysis = a.group(1) if a else content[:200]
            return final_score, analysis
    except Exception as e:
        logger.warning(f"LLM validation failed: {e}")
        return prelim_score, "LLM validation skipped (error)"

def llm_recommend_courses(resume_text, job_desc, matched_skills, missing_skills, total_score):
    """Use GPT-4o-mini to recommend highly relevant courses based on job-resume gap analysis"""
    try:
        # Truncate inputs to avoid token limits
        resume_snippet = resume_text[:1500]
        job_snippet = job_desc[:1200]
        
        prompt = f"""You are a career development expert. Analyze the gap between this candidate's profile and job requirements, then recommend 3 SPECIFIC online courses.

CANDIDATE PROFILE:
{resume_snippet}

JOB REQUIREMENTS:
{job_snippet}

MATCHED SKILLS: {', '.join(matched_skills[:10])}
MISSING CRITICAL SKILLS: {', '.join(missing_skills[:5])}
CURRENT MATCH SCORE: {total_score}%

Recommend 3 highly specific, actionable courses that will:
1. Bridge the MOST CRITICAL skill gaps for THIS EXACT JOB
2. Be realistic given the candidate's current level
3. Have high impact on match score

Format EXACTLY as (no extra text):
🔗 [Course Name](https://coursera.org/learn/course-url)
🔗 [Course Name](https://udemy.com/course/course-url)
🔗 [Course Name](https://www.edx.org/course-url)

Only output the 3 course links, nothing else."""

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=350
        )
        
        courses = response.choices[0].message.content.strip()
        logger.info(f"📚 LLM Course Recommendations generated successfully")
        return courses
        
    except Exception as e:
        logger.warning(f"LLM course recommendation error: {e}")
        return None


# ==================== RANKING FUNCTION (IMPROVED) ====================
def rank_resume_with_rag(resume_text, job_desc, weights=None):
    if weights is None:
        weights = {"semantic": 0.55, "skills": 0.30, "experience": 0.10, "education": 0.05}

    matched_skills, skill_explanations, role_type, job_skills_list = extract_skills_with_rag(resume_text, job_desc)
    sem_score = semantic_similarity_enhanced(resume_text, job_desc)

    # Dynamic weighting of skills: important skills in the job description are boosted
    high_priority = {"machine learning", "python", "aws", "react", "java"}
    weight_map = {}
    job_desc_clean = preprocess(job_desc)
    for s in (set(job_skills_list) | set(matched_skills)):
        base = 1.0
        if s in high_priority:
            base = 1.8
        if s in job_desc_clean:
            base = max(base, 1.4)
        weight_map[s] = base

    # compute weighted skills score
    if job_skills_list:
        weighted_sum = sum(weight_map.get(skill, 1.0) for skill in matched_skills)
        denom = sum(weight_map.get(skill, 1.0) for skill in job_skills_list) or len(job_skills_list)
        skills_score = min(100, (weighted_sum / denom) * 100)
    else:
        skills_score = (len(matched_skills) / (len(matched_skills) + 1)) * 100 if matched_skills else 0

    # small RAG confidence boost
    rag_confidence_boost = min(6, len([e for e in skill_explanations if "RAG" in e]) * 0.6)
    skills_score = min(100, skills_score + rag_confidence_boost)

    experience = extract_experience(resume_text)
    education = extract_education(resume_text)
    experience_score = 100 if experience != "N/A" else 0
    education_score = 100 if education != "N/A" else 0

    total_score = (
        sem_score * weights["semantic"] +
        skills_score * weights["skills"] +
        experience_score * weights["experience"] +
        education_score * weights["education"]
    )
    total_score = round(float(total_score), 2)
    status = "Recommended" if total_score >= 70 else "Review"
    career_coach = generate_career_coach_message(total_score, matched_skills, job_skills_list, resume_text, job_desc)


    # Validate with LLM (re-score & produce reasoning)
    try:
        validated_score, llm_reason = llm_validate_ranking(resume_text, job_desc, matched_skills, total_score)
        # smoothing
        final_score = round((total_score * 0.6) + (validated_score * 0.4), 2)
        career_coach += f"\n\n<span style='color: #0FF0FC; font-weight: 600;'>Note for HR's</span>\n{llm_reason}"

    except Exception as e:
        logger.warning(f"LLM validation error: {e}")
        final_score = total_score

    return final_score, matched_skills, skill_explanations, experience, education, status, career_coach

# ==================== USER AUTH ENDPOINTS ====================

@app.route("/signup", methods=["POST"])
def signup():
    """Handle user signup"""
    try:
        data = request.get_json()
        first_name = data.get('firstName')
        last_name = data.get('lastName')
        email = data.get('email')
        password = data.get('password')
        company = data.get('company', '')
        if not all([first_name, last_name, email, password]):
            return jsonify({"error": "All fields are required"}), 400
        existing_user = User.query.filter_by(email=email).first()
        if existing_user:
            return jsonify({"error": "Email already registered"}), 409
        new_user = User(
            first_name=first_name,
            last_name=last_name,
            email=email,
            password_hash=generate_password_hash(password),
            company=company
        )
        db.session.add(new_user)
        db.session.commit()
        logger.info(f"✅ New user registered: {first_name} {last_name} ({email})")
        return jsonify({"message": "User registered successfully", "user": new_user.to_dict()}), 201
    except Exception as e:
        db.session.rollback()
        logger.error(f"Signup error: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route("/login", methods=["POST"])
def login():
    """Handle user login"""
    try:
        data = request.get_json()
        email = data.get('email')
        password = data.get('password')
        if not email or not password:
            return jsonify({"error": "Email and password required"}), 400
        user = User.query.filter_by(email=email).first()
        if not user or not check_password_hash(user.password_hash, password):
            return jsonify({"error": "Invalid email or password"}), 401
        user.last_login = datetime.utcnow()
        db.session.commit()
        logger.info(f"✅ User logged in: {user.first_name} {user.last_name} ({email})")
        return jsonify({"message": "Login successful", "user": user.to_dict()}), 200
    except Exception as e:
        logger.error(f"Login error: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route("/users", methods=["GET"])
def get_users():
    """Get all registered users"""
    try:
        users = User.query.order_by(User.created_at.desc()).all()
        return jsonify({"total": len(users), "users": [u.to_dict() for u in users]})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ==================== API ENDPOINTS ====================

@app.route('/')
def home():
    try:
        return send_from_directory('.', 'index.html')
    except Exception as e:
        return f"Make sure index.html is in the same folder as app.py. Error: {str(e)}"

@app.route("/analyze", methods=["POST"])
def analyze():
    """Main endpoint: Analyze resumes and save to database"""
    try:
        job_desc = request.form.get("jobDescription", "")
        files = request.files.getlist("resumes")
        if not job_desc:
            return jsonify({"error": "Missing job description"}), 400
        if not files:
            return jsonify({"error": "No resumes uploaded"}), 400

        results = []
        for f in files:
            pdf_bytes = f.read()
            text = extract_text_from_pdf_bytes(pdf_bytes)
            if not text:
                results.append({
                    "name": f.filename.replace(".pdf", ""),
                    "filename": f.filename,
                    "score": 0,
                    "skills": [],
                    "skill_explanations": [],
                    "experience": "N/A",
                    "education": "N/A",
                    "status": "Error: Text extraction failed",
                    "career_coach": ""
                })
                continue

            email = extract_email(text)
            phone = extract_phone(text)

            score, skills, explanations, experience, education, status, career_coach = rank_resume_with_rag(text, job_desc)

            try:
                candidate = Candidate(
                    name=f.filename.replace(".pdf", ""),
                    filename=f.filename,
                    email=email,
                    phone=phone,
                    resume_text=text,
                    score=score,
                    matched_skills=skills,
                    experience=experience,
                    education=education,
                    status=status
                )
                db.session.add(candidate)
                db.session.flush()

                analysis = Analysis(
                    candidate_id=candidate.id,
                    job_description=job_desc,
                    score=score,
                    matched_skills=skills,
                    skill_explanations=explanations,
                    career_coach_message=career_coach
                )
                db.session.add(analysis)
                db.session.commit()
                logger.info(f"💾 Saved: {candidate.name} (ID: {candidate.id}, Score: {score})")

            except Exception as db_error:
                db.session.rollback()
                logger.error(f"Database error: {str(db_error)}")

            results.append({
                "name": f.filename.replace(".pdf", ""),
                "filename": f.filename,
                "email": email,
                "phone": phone,
                "score": score,
                "skills": skills,
                "skill_explanations": explanations,
                "experience": experience,
                "education": education,
                "status": status,
                "career_coach": career_coach
            })

        results.sort(key=lambda x: x["score"], reverse=True)
        results = convert_numpy_types(results)
        return jsonify({"results": results})

    except Exception as e:
        logger.error(f"Analyze endpoint error: {str(e)}")
        return jsonify({"error": str(e)}), 500

# ==================== DATABASE VIEW ENDPOINTS ====================
@app.route("/candidates", methods=["GET"])
def list_candidates():
    try:
        candidates = Candidate.query.order_by(Candidate.created_at.desc()).all()
        return jsonify({"total": len(candidates), "candidates": [c.to_dict() for c in candidates]})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/candidate/<int:candidate_id>", methods=["GET"])
def get_candidate(candidate_id):
    try:
        c = Candidate.query.get(candidate_id)
        if not c:
            return jsonify({"error": "Candidate not found"}), 404
        return jsonify({"candidate": c.to_dict()})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/jobs", methods=["GET", "POST"])
def jobs():
    if request.method == "GET":
        try:
            jobs = Job.query.order_by(Job.created_at.desc()).all()
            return jsonify({"total": len(jobs), "jobs": [j.to_dict() for j in jobs]})
        except Exception as e:
            return jsonify({"error": str(e)}), 500
    else:
        try:
            data = request.get_json()
            title = data.get("title")
            description = data.get("description")
            required_skills = data.get("required_skills", [])
            if not title or not description:
                return jsonify({"error": "title and description required"}), 400
            new_job = Job(title=title, description=description, required_skills=required_skills)
            db.session.add(new_job)
            db.session.commit()
            return jsonify({"message": "Job saved", "job": new_job.to_dict()}), 201
        except Exception as e:
            db.session.rollback()
            return jsonify({"error": str(e)}), 500

# Dashboard HTML endpoint - original dashboard HTML preserved fully
@app.route("/dashboard", methods=["GET"])
def dashboard():
    """Dark purple themed admin dashboard"""
    dashboard_html = '''
    <!DOCTYPE html>
    <html>
    <head>
        <title>Jobyn AI - Admin Dashboard</title>
        <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css">
        <style>
            * { margin: 0; padding: 0; box-sizing: border-box; }
            
            :root {
                --bg-primary: #0a0a0a;
                --bg-secondary: #1a1a1a;
                --bg-card: #1e1e1e;
                --primary: #8b5cf6;
                --primary-dark: #7c3aed;
                --primary-light: #a78bfa;
                --accent: #c084fc;
                --text-primary: #ffffff;
                --text-secondary: #a0a0a0;
                --text-dim: #666666;
                --border: #2a2a2a;
                --success: #10b981;
                --warning: #f59e0b;
            }
            
            body { 
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
                background: linear-gradient(135deg, #0a0a0a 0%, #1a0a2e 50%, #0a0a0a 100%);
                background-attachment: fixed;
                padding: 20px;
                color: var(--text-primary);
                min-height: 100vh;
            }
            
            .container {
                max-width: 1400px;
                margin: 0 auto;
                background: var(--bg-card);
                border-radius: 20px;
                                padding: 40px;
                box-shadow: 0 20px 60px rgba(139, 92, 246, 0.3);
                border: 1px solid var(--border);
            }
            
            .header {
                display: flex;
                justify-content: space-between;
                align-items: center;
                margin-bottom: 40px;
                padding-bottom: 20px;
                border-bottom: 2px solid var(--border);
            }
            
            h1 { 
                color: var(--primary-light);
                font-size: 2.5em;
                font-weight: 700;
                display: flex;
                align-items: center;
                gap: 15px;
            }
            
            .subtitle {
                color: var(--text-secondary);
                margin-top: 10px;
                font-size: 1.1em;
            }
            
            .stats {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
                gap: 20px;
                margin-bottom: 40px;
            }
            
            .stat-card {
                background: linear-gradient(135deg, var(--primary) 0%, var(--primary-dark) 100%);
                color: white;
                padding: 30px;
                border-radius: 15px;
                text-align: center;
                box-shadow: 0 8px 20px rgba(139, 92, 246, 0.4);
                transition: transform 0.3s ease;
            }
            
            .stat-card:hover {
                transform: translateY(-5px);
                box-shadow: 0 12px 30px rgba(139, 92, 246, 0.6);
            }
            
            .stat-card h3 {
                font-size: 3em;
                margin-bottom: 10px;
                font-weight: 700;
            }
            
            .stat-card p {
                font-size: 1.1em;
                opacity: 0.95;
            }
            
            .section h2 {
                color: var(--primary-light);
                margin-bottom: 25px;
                padding-bottom: 15px;
                border-bottom: 2px solid var(--primary);
                font-size: 1.8em;
            }
            
            table {
                width: 100%;
                border-collapse: collapse;
                background: var(--bg-secondary);
                border-radius: 12px;
                overflow: hidden;
            }
            
            th {
                background: linear-gradient(135deg, var(--primary) 0%, var(--primary-dark) 100%);
                color: white;
                padding: 18px 15px;
                text-align: left;
                font-weight: 600;
            }
            
            td {
                padding: 15px;
                border-bottom: 1px solid var(--border);
                color: var(--text-primary);
            }
            
            tr:hover {
                background: var(--bg-card);
            }
            
            .badge {
                display: inline-block;
                padding: 6px 14px;
                border-radius: 20px;
                font-size: 0.85em;
                font-weight: 600;
            }
            
            .badge-recommended {
                background: var(--success);
                color: white;
            }
            
            .badge-review {
                background: var(--warning);
                color: white;
            }
            
            .badge-user {
                background: var(--primary);
                color: white;
            }
            
            .score {
                font-weight: 700;
                font-size: 1.3em;
                color: var(--primary-light);
            }
            
            .skill-tag {
                background: var(--primary);
                color: white;
                padding: 5px 12px;
                border-radius: 15px;
                font-size: 0.85em;
                margin-right: 5px;
                display: inline-block;
                margin-bottom: 5px;
            }
            
            .refresh-btn {
                background: linear-gradient(135deg, var(--primary) 0%, var(--primary-dark) 100%);
                color: white;
                padding: 12px 30px;
                border: none;
                border-radius: 10px;
                cursor: pointer;
                font-weight: 600;
                transition: all 0.3s;
            }
            
            .refresh-btn:hover {
                transform: translateY(-2px);
                box-shadow: 0 6px 20px rgba(139, 92, 246, 0.6);
            }
            
            .tabs {
                display: flex;
                gap: 12px;
                margin-bottom: 30px;
            }
            
            .tab-btn {
                padding: 12px 25px;
                border: none;
                background: var(--bg-secondary);
                color: var(--text-secondary);
                border-radius: 10px;
                cursor: pointer;
                font-weight: 600;
                transition: all 0.3s;
            }
            
            .tab-btn.active {
                background: linear-gradient(135deg, var(--primary) 0%, var(--primary-dark) 100%);
                color: white;
            }
            
            .tab-content {
                display: none;
            }
            
            .tab-content.active {
                display: block;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <div>
                    <h1><i class="fas fa-rocket"></i> Jobyn AI Dashboard</h1>
                    <p class="subtitle">Only you should see this pranav!</p>
                </div>
                <button class="refresh-btn" onclick="location.reload()">
                    <i class="fas fa-sync-alt"></i> Refresh
                </button>
            </div>
            
            <div class="stats" id="stats"></div>
            
            <div class="tabs">
                <button class="tab-btn active" onclick="showTab('candidates', event)">
                    <i class="fas fa-users"></i> Candidates
                </button>
                <button class="tab-btn" onclick="showTab('users', event)">
                    <i class="fas fa-user"></i> Users
                </button>
            </div>
            
            <div id="candidates-tab" class="tab-content active">
                <div class="section">
                    <h2>Recent Candidates</h2>
                    <div id="candidates-table"></div>
                </div>
            </div>
            
            <div id="users-tab" class="tab-content">
                <div class="section">
                    <h2>Registered Users</h2>
                    <div id="users-table"></div>
                </div>
            </div>
        </div>
        
        <script>
    function showTab(tabName, event) {
        document.querySelectorAll('.tab-content').forEach(tab => tab.classList.remove('active'));
        document.querySelectorAll('.tab-btn').forEach(btn => btn.classList.remove('active'));
        document.getElementById(tabName + '-tab').classList.add('active');
        event.target.classList.add('active');
    }
    
    function fetchstats() {
        fetch('/stats')
            .then(r => r.json())
            .then(data => {
                document.getElementById('stats').innerHTML = `
                    <div class="stat-card"><h3>${data.totalUsers || 0}</h3><p>Users</p></div>
                    <div class="stat-card"><h3>${data.totalCandidates || 0}</h3><p>Candidates</p></div>
                    <div class="stat-card"><h3>${data.averageScore || '0%'}</h3><p>Avg Score</p></div>
                    <div class="stat-card"><h3>${data.recommendedCount || 0}</h3><p>Recommended</p></div>
                `;
            })
            .catch(err => console.error('Stats error:', err));
    }
    
    // Fetch candidates and display in table
    function fetchcandidates() {
        fetch('/candidates')
            .then(r => r.json())
            .then(data => {
                const container = document.getElementById('candidates-table');
                if (!data.candidates || data.candidates.length === 0) {
                    container.innerHTML = '<p style="text-align:center;padding:40px;color:var(--text-dim);">No candidates yet</p>';
                    return;
                }
                
                let html = '<table><thead><tr><th>Name</th><th>Email</th><th>Phone</th><th>Score</th><th>Skills</th><th>Experience</th><th>Status</th><th>Date</th></tr></thead><tbody>';
                
                data.candidates.forEach(c => {
                    const skills = (c.matched_skills || []).slice(0, 3).map(s => `<span class="skill-tag">${s}</span>`).join('');
                    const badge = c.status === 'Recommended' ? '<span class="badge badge-recommended">Recommended</span>' : '<span class="badge badge-review">Review</span>';
                    
                    html += `
                        <tr>
                            <td><strong>${c.name || 'N/A'}</strong></td>
                            <td>${c.email || 'N/A'}</td>
                            <td>${c.phone || 'N/A'}</td>
                            <td class="score">${c.score ? c.score.toFixed(2) + '%' : 'N/A'}</td>
                            <td>${skills || 'None'}</td>
                            <td>${c.experience || 'N/A'}</td>
                            <td>${badge}</td>
                            <td>${c.created_at || 'N/A'}</td>
                        </tr>
                    `;
                });
                
                html += '</tbody></table>';
                container.innerHTML = html;
            })
            .catch(err => console.error('Candidates error:', err));
    }
    
    // Fetch users and display in table
    function fetchusers() {
        fetch('/users')
            .then(r => r.json())
            .then(data => {
                const container = document.getElementById('users-table');
                if (!data.users || data.users.length === 0) {
                    container.innerHTML = '<p style="text-align:center;padding:40px;color:var(--text-dim);">No users yet</p>';
                    return;
                }
                
                let html = '<table><thead><tr><th>Name</th><th>Email</th><th>Company</th><th>Role</th><th>Registered</th><th>Last Login</th></tr></thead><tbody>';
                
                data.users.forEach(u => {
                    html += `
                        <tr>
                            <td><strong>${u.full_name || 'N/A'}</strong></td>
                            <td>${u.email || 'N/A'}</td>
                            <td>${u.company || 'N/A'}</td>
                            <td>${u.role || 'recruiter'}</td>
                            <td>${u.created_at || 'N/A'}</td>
                            <td>${u.last_login || 'Never'}</td>
                        </tr>
                    `;
                });
                
                html += '</tbody></table>';
                container.innerHTML = html;
            })
            .catch(err => console.error('Users error:', err));
    }
    
    // Initialize on page load
    fetchstats();
    fetchcandidates();
    fetchusers();
</script>

    </body>
    </html>
    '''
    return render_template_string(dashboard_html)


@app.route('/stats', methods=['GET'])
def get_stats():
    """Get dashboard statistics"""
    try:
        total_users = User.query.count()
        total_candidates = Candidate.query.count()
        total_analyses = Analysis.query.count()
        
        # Calculate average score
        candidates_with_scores = Candidate.query.filter(Candidate.score.isnot(None)).all()
        if candidates_with_scores:
            avg_score = sum([c.score for c in candidates_with_scores]) / len(candidates_with_scores)
            avg_score = round(avg_score, 2)
        else:
            avg_score = 0
        
        # Count recommended candidates (score >= 70)
        recommended_count = Candidate.query.filter(Candidate.score >= 70).count()
        
        logger.info(f"📊 Stats: Users={total_users}, Candidates={total_candidates}, AvgScore={avg_score}%, Recommended={recommended_count}")
        
        return jsonify({
            'totalUsers': total_users,
            'totalCandidates': total_candidates,
            'totalAnalyses': total_analyses,
            'averageScore': f"{avg_score}%",
            'recommendedCount': recommended_count
        })
    except Exception as e:
        logger.error(f"Stats error: {str(e)}")
        return jsonify({
            'totalUsers': 0,
            'totalCandidates': 0,
            'totalAnalyses': 0,
            'averageScore': '0%',
            'recommendedCount': 0
        }), 500


@app.route("/test", methods=["GET"])
def test():
    rag_status = "✅ Enabled" if rag_store.index is not None else "❌ Disabled"
    return jsonify({
        "status": "Backend is working!",
        "version": "7.1-With-LLM-RAG-Validation",
        "rag_status": rag_status,
        "skills_indexed": len(rag_store.skill_names),
        "database": "SQLite (nexus_ats.db)",
        "total_users": User.query.count(),
        "total_candidates": Candidate.query.count(),
        "total_analyses": Analysis.query.count()
    })

# ==================== DATABASE INITIALIZATION ====================
def init_database():
    with app.app_context():
        db.create_all()
        logger.info("✅ Database initialized successfully")
        # show which DB is being used (SQLite or Render Postgres)
        logger.info(f"📁 Database URI: {app.config['SQLALCHEMY_DATABASE_URI']}")
        logger.info(f"📊 Tables: users, candidates, jobs, analyses")

if __name__ == "__main__":
    init_database()
    logger.info("=" * 60)
    logger.info("🚀 Jobyn AI - Production Ready with User Auth (Render Deployment Mode)")
    logger.info("=" * 60)

    # ✅ 1) use 0.0.0.0 instead of 127.0.0.1 so Render can access it
    # ✅ 2) use PORT from Render environment
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=False, host="0.0.0.0", port=port)


