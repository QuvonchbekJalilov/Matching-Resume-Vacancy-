import os, re, hashlib, asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import List, Dict, Set, Optional
import torch
from fastapi import FastAPI
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer, util

# ==========================================
# ⚙️ CONFIGURATION
# ==========================================
MODEL_NAME = os.getenv("MODEL_NAME", "BAAI/bge-m3")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
torch.set_num_threads(os.cpu_count())

app = FastAPI(title="Resume-Vacancy Matcher API (v5 Ultra-Fast)")
executor = ThreadPoolExecutor(max_workers=min(8, os.cpu_count() or 4))

model: SentenceTransformer = None
VACANCY_CACHE: Dict[str, torch.Tensor] = {}
SKILL_PATTERNS: Dict[str, re.Pattern] = {}

# ==========================================
# 🔥 LOAD MODEL ON STARTUP
# ==========================================
@app.on_event("startup")
def load_model():
    global model, SKILL_PATTERNS
    model = SentenceTransformer(MODEL_NAME, device=DEVICE)
    print(f"✅ Model loaded: {MODEL_NAME} on {DEVICE} ({torch.get_num_threads()} threads)")
    # Compile regex patterns once
    for key, aliases in SKILL_ALIASES.items():
        SKILL_PATTERNS[key] = re.compile(r"\b(" + "|".join(re.escape(a) for a in aliases) + r")\b", re.IGNORECASE)


# ==========================================
# 📋 SKILLS MAP (multi-language)
# ==========================================
SKILL_ALIASES: Dict[str, Set[str]] = {
    # === IT & Programming ===
    "python": {
        "python", "питон", "пайтон", "python3", "py", "django", "flask", "fastapi",
        "pandas", "numpy", "scikit-learn", "tensorflow", "pytorch", "asyncio",
        "pytest", "scrapy", "selenium", "sqlalchemy", "matplotlib", "jupyter",
        "rest api", "backend development", "data analysis"
    },
    "django": {
        "django", "джанго", "django rest framework", "drf", "orm", "django admin",
        "models", "migrations", "templates", "views", "serializers", "auth",
        "middleware", "celery with django", "gunicorn", "nginx setup"
    },
    "fastapi": {
        "fastapi", "фастапи", "async python", "uvicorn", "pydantic",
        "restful api", "dependency injection", "swagger", "openapi", "asgi"
    },
    "flask": {
        "flask", "фласк", "jinja2", "werkzeug", "blueprints", "flask-restful",
        "flask-login", "flask-admin", "flask-sqlalchemy", "microservice api"
    },
    "postgresql": {
        "postgresql", "postgres", "psql", "постгрес", "pgadmin", "plpgsql",
        "database optimization", "indexing", "query tuning", "stored procedure"
    },
    "mysql": {
        "mysql", "майскл", "маискил", "mariadb", "database design",
        "joins", "foreign keys", "mysql workbench", "triggers", "sql queries"
    },
    "redis": {
        "redis", "редис", "in-memory cache", "pub/sub", "message broker",
        "redis cluster", "redis streams", "session storage"
    },
    "docker": {
        "docker", "докер", "dockerfile", "docker-compose", "containerization",
        "image build", "container registry", "volumes", "networking", "docker swarm"
    },
    "kubernetes": {
        "kubernetes", "кубернетес", "кубер", "k8s", "pods", "deployments",
        "helm", "ingress", "services", "namespaces", "minikube", "kubectl",
        "statefulsets", "autoscaling", "configmap", "secrets"
    },
    "aws": {
        "aws", "amazon web services", "амазон веб сервис", "ec2", "lambda", "s3",
        "rds", "cloudwatch", "route53", "cloudformation", "iam", "api gateway",
        "dynamodb", "elastic beanstalk", "ecs", "ecr", "sns", "sqs"
    },
    "gcp": {
        "gcp", "google cloud", "гугл клауд", "cloud storage", "bigquery",
        "cloud functions", "firebase", "app engine", "compute engine", "pubsub"
    },
    "azure": {
        "azure", "азур", "azure devops", "app services", "cosmos db",
        "azure pipelines", "azure blob storage", "aks"
    },
    "ci/cd": {
        "ci/cd", "cicd", "github actions", "gitlab ci", "jenkins", "travis ci",
        "circleci", "bitbucket pipelines", "continuous integration",
        "continuous delivery", "deployment automation", "build pipelines"
    },
    "graphql": {
        "graphql", "графкьюэл", "apollo server", "graphql query", "graphql mutation",
        "subscriptions", "graphql schema", "resolver", "graphql client"
    },
    "rest": {
        "rest api", "restful", "рест апи", "api development", "http methods",
        "swagger", "openapi", "endpoint", "authorization headers"
    },
    "rabbitmq": {
        "rabbitmq", "раббит", "раббитмк", "message broker", "queue system",
        "routing key", "exchange", "celery integration", "amqp"
    },
    "kafka": {
        "kafka", "кафка", "event streaming", "producer", "consumer", "topic",
        "kafka streams", "zookeeper", "data pipeline"
    },
    "celery": {"celery", "селери"},
    "microservices": {
        "microservice", "microservices", "микросервис", "микросервисы",
        "service architecture", "api gateway", "event driven architecture",
        "containerized services", "distributed system", "grpc", "service mesh"
    },
    "linux": {
        "linux", "линукс", "bash", "shell", "terminal", "ubuntu", "centos", "debian",
        "systemd", "cron jobs", "permissions", "networking", "ssh", "vim"
    },
    "windows": {
        "windows", "виндовс", "cmd", "powershell", "iis", "registry", "windows server"
    },
    "macos": {
        "macos", "мак", "mac", "brew", "terminal", "xcode"
    },
    "git": {
        "git", "гит", "гитхаб", "github", "gitlab", "bitbucket", "branch", "merge",
        "commit", "pull request", "rebase", "version control", "gitflow"
    },
    "devops": {"devops", "девопс"},
    "machine learning": {
        "machine learning", "машинное обучение", "mashinali o‘rganish", "ML",
        "supervised learning", "unsupervised learning", "regression", "classification",
        "clustering", "model training", "sklearn", "feature engineering", "cross-validation"
    },
    "deep learning": {
        "deep learning", "глубокое обучение", "chuqur o‘rganish", "neural networks",
        "cnn", "rnn", "lstm", "transformers", "pytorch", "tensorflow", "keras"
    },

    "nlp": {
        "nlp", "natural language processing", "обработка естественного языка",
        "tilni qayta ishlash", "text classification", "lemmatization", "tokenization",
        "word2vec", "bert", "huggingface", "transformer models"
    },
    "transformers": {"transformers", "huggingface", "трансформеры"},
    "data science": {
        "data science", "data scientist", "наука о данных", "ma’lumotlar fani",
        "data analytics", "eda", "pandas", "numpy", "matplotlib", "seaborn",
        "data wrangling", "predictive modeling", "data cleaning"
    },
    "sql": {"sql", "эс кью эл", "sequel"},
    "nosql": {"nosql", "no sql", "нон sql"},
    "frontend": {
        "frontend", "фронтенд", "front-end", "html", "css", "javascript",
        "responsive design", "webpack", "vite", "sass", "tailwind", "react",
        "vue", "angular", "typescript", "ui components", "cross-browser testing"
    },
    "backend": {
        "backend", "бэкенд", "back-end", "server side", "api", "database integration",
        "authentication", "authorization", "caching", "middleware", "orm", "load balancing"
    },
    "fullstack": {"fullstack", "фулстек", "фулл стек", "to‘liq stack"},
    "javascript": {
        "javascript", "js", "джаваскрипт", "es6", "nodejs", "async await",
        "dom manipulation", "fetch api", "event handling", "npm", "babel", "webpack"
    },
    "typescript": {"typescript", "тайпскрипт"},
    "react": {"react", "reactjs", "react.js", "реакт"},
    "vue": {"vue", "vuejs", "vue.js", "вью"},
    "angular": {"angular", "ангуляр"},
    "nodejs": {"nodejs", "node.js", "нод", "ноджс"},
    "php": {"php", "пхп"},
    "laravel": {
        "laravel", "ларавел", "eloquent", "artisan", "queue", "blade", "middleware",
        "tinker", "passport", "sanctum", "livewire", "inertia", "laravel sail"
    },
    "symfony": {"symfony", "симфони"},
    "golang": {"golang", "go", "го"},
    "rust": {"rust", "раст"},
    "java": {"java", "ява"},
    "spring": {"spring", "спринг"},
    "c#": {"c#", "си шарп", "c sharp"},
    "cpp": {"c++", "си плюс плюс"},
    "mobile": {"mobile", "мобил", "мобиль", "mobil dasturlash"},
    "android": {"android", "андроид"},
    "ios": {"ios", "айос", "iphone"},
    "flutter": {"flutter", "флаттер"},
    "react native": {"react native", "реакт нейтив"},
    "unity": {"unity", "юнити"},
    "unreal engine": {"unreal engine", "анреал"},
    "blockchain": {"blockchain", "блокчейн"},
    "ai": {"ai", "artificial intelligence", "искусственный интеллект", "sun’iy intellekt"},
    "cv": {"computer vision", "компьютерное зрение", "kompyuter ko‘rish"},

    # === Marketing / SMM / Sales ===
    "marketing": {
        "marketing", "маркетинг", "dizital marketing", "marketing strategiyasi", "digital marketing",
        "цифровой маркетинг", "маркетолог", "marketing campaign", "brand marketing", "performance marketing",
        "growth marketing", "strategic marketing", "email marketing", "event marketing", "affiliate marketing",
        "influencer marketing", "mobile marketing", "omnichannel marketing", "marketing automation",
        "marketing analytics", "content marketing", "b2b marketing", "b2c marketing", "lead nurturing",
        "retention marketing", "conversion optimization", "remarketing", "campaign management",
        "target audience", "customer journey", "marketing funnel", "roi", "romi"
    },

    "seo": {
        "seo", "search engine optimization", "поисковая оптимизация", "keyword research",
        "on-page seo", "off-page seo", "technical seo", "seo audit", "backlinks", "meta tags",
        "google search console", "sitemap", "robots.txt", "internal linking", "keyword ranking",
        "semrush", "ahrefs", "seranking", "moz", "seo copywriting", "content optimization",
        "organic traffic", "search visibility"
    },

    "smm": {
        "smm", "social media marketing", "social media", "соцсети", "инстаграм", "instagram",
        "tiktok", "telegram", "facebook", "linkedin", "youtube", "twitter", "threads",
        "community management", "social strategy", "social analytics", "engagement rate",
        "content calendar", "social listening", "targeting", "social campaign", "influencer outreach",
        "storytelling", "brand awareness", "reels", "ugc content", "social ads", "paid social"
    },

    "ppc": {
        "ppc", "contextual advertising", "контекстная реклама", "google ads", "yandex direct",
        "paid search", "search ads", "display ads", "remarketing", "cpc", "cpm", "ctr",
        "ad targeting", "ad copy", "bid strategy", "performance max", "ad extensions",
        "conversion tracking", "utm tags", "google analytics", "meta ads", "a/b testing"
    },

    "branding": {
        "branding", "брендинг", "brand identity", "бренд стратегия", "brand awareness",
        "brand voice", "brand tone", "brand consistency", "rebranding", "corporate branding",
        "visual identity", "brand guidelines", "brand book", "brand story", "employer branding",
        "brand positioning", "brand personality", "logo identity", "naming", "packaging design",
        "brand loyalty", "brand recognition"
    },

    "content": {
        "content", "контент", "mazmun", "kontent", "content plan", "content strategy",
        "content creation", "content marketing", "content writing", "content calendar",
        "visual content", "copywriting", "ugc", "storytelling", "content optimization",
        "post scheduling", "editorial plan", "branded content", "script writing", "reels content"
    },

    "copywriting": {
        "copywriting", "копирайтер", "copywriter", "yozuvchi", "sales copy", "ad copy",
        "landing page copy", "email copy", "storytelling", "creative writing",
        "text editing", "proofreading", "tone of voice", "seo copy", "headline writing",
        "slogan creation", "commercial writing", "tagline", "call to action"
    },

    "advertising": {
        "advertising", "реклама", "e’lon", "reklama", "media buying", "media planning",
        "programmatic ads", "display advertising", "native ads", "outdoor advertising",
        "tv advertising", "radio advertising", "online ads", "retargeting", "google display network",
        "social ads", "meta ads", "creative campaign", "banner ads", "sponsorship", "influencer ads"
    },

    "sales": {
        "sales", "продажи", "savdo", "sotuv", "sales funnel", "b2b sales", "b2c sales",
        "cold calling", "lead qualification", "sales pitch", "negotiation", "crm pipeline",
        "customer acquisition", "customer retention", "cross-selling", "upselling",
        "sales target", "account management", "client relations", "sales performance",
        "sales reporting", "key account management"
    },

    "crm": {
        "crm", "customer relationship management", "срм", "mijozlar bazasi",
        "hubspot", "salesforce", "bitrix24", "zoho crm", "amo crm", "client database",
        "deal pipeline", "lead tracking", "automation", "email sequence", "contact management",
        "sales dashboard", "follow-up management", "customer segmentation"
    },

    "lead generation": {
        "lead generation", "лидогенерация", "mijoz topish", "prospecting", "lead qualification",
        "cold outreach", "cold email", "lead magnet", "list building", "linkedin outreach",
        "lead capture", "inbound marketing", "outbound marketing", "b2b outreach",
        "lead scoring", "marketing automation", "crm integration"
    },

    "market research": {
        "market research", "рынок", "bozor tahlili", "competitive analysis",
        "survey", "focus group", "market segmentation", "consumer insights",
        "trend analysis", "swot analysis", "pricing analysis", "demand forecasting",
        "product positioning", "target audience analysis", "competitor benchmark"
    },

    "communication": {
        "communication", "коммуникация", "aloqa", "muloqot", "business communication",
        "presentation skills", "public speaking", "verbal communication",
        "non-verbal communication", "email etiquette", "negotiation", "client interaction",
        "team communication", "persuasion", "storytelling", "reporting", "brand messaging"
    },

    "negotiation": {
        "negotiation", "переговоры", "muzokara", "deal closing", "contract negotiation",
        "objection handling", "conflict resolution", "price negotiation",
        "win-win strategy", "sales negotiation", "stakeholder communication"
    },


    # === HR / Management / Business (expanded) ===
    "hr": {
        "hr", "human resources", "кадры", "kadrlar bo‘limi", "personnel management",
        "employee relations", "hr manager", "hr specialist", "hr generalist",
        "hr policies", "hr strategy", "hr administration", "employee onboarding",
        "offboarding", "performance review", "compensation", "benefits management",
        "employee engagement", "hr analytics", "hr documentation", "labor law",
        "organizational culture", "hr planning", "people operations"
    },

    "recruitment": {
        "recruitment", "recruiter", "рекрутинг", "ishga olish", "talent acquisition",
        "candidate sourcing", "job posting", "cv screening", "headhunting", "interviewing",
        "technical recruitment", "mass recruitment", "recruitment funnel",
        "onboarding", "ats", "applicant tracking system", "talent pool",
        "reference check", "offer negotiation", "candidate pipeline"
    },

    "headhunting": {
        "headhunting", "хэдхантинг", "talent search", "executive search",
        "passive candidate sourcing", "linkedin sourcing", "headhunter",
        "cold outreach", "networking", "cv hunting"
    },

    "training": {
        "training", "обучение", "o‘qitish", "staff training", "corporate training",
        "employee development", "mentoring", "coaching", "career development",
        "onboarding training", "skills improvement", "learning management system",
        "e-learning", "training plan", "workshop", "seminar", "hrd", "professional growth"
    },

    "management": {
        "management", "менеджмент", "boshqaruv", "business management",
        "people management", "strategic management", "operations management",
        "change management", "performance management", "process management",
        "managerial skills", "decision making", "delegation", "crisis management",
        "stakeholder management", "management system", "managerial reporting"
    },

    "project management": {
        "project management", "управление проектами", "loyiha boshqaruvi",
        "project planning", "project scheduling", "project coordination",
        "agile", "scrum", "kanban", "waterfall", "pmi", "pmp", "prince2",
        "jira", "trello", "asana", "risk management", "timeline tracking",
        "milestone", "deliverables", "project charter", "stakeholder management",
        "resource allocation", "project budget", "kpi tracking", "gantt chart"
    },

    "leadership": {
        "leadership", "лидерство", "rahbarlik", "team leadership", "executive leadership",
        "servant leadership", "emotional intelligence", "decision making",
        "inspiring others", "strategic vision", "mentorship", "situational leadership",
        "organizational leadership", "conflict management", "influence", "motivation"
    },

    "teamwork": {
        "teamwork", "командная работа", "jamoaviy ish", "collaboration", "team coordination",
        "cross-functional team", "agile team", "pair work", "team communication",
        "peer support", "group projects", "team goals", "collective decision making",
        "synergy", "remote teamwork"
    },

    "motivation": {
        "motivation", "мотивация", "rag‘batlantirish", "employee motivation",
        "self-motivation", "performance incentives", "intrinsic motivation",
        "extrinsic motivation", "team motivation", "motivation strategy",
        "bonus system", "recognition", "rewards", "gamification", "okrs"
    },

    "planning": {
        "planning", "планирование", "rejalashtirish", "strategic planning",
        "operational planning", "tactical planning", "business planning",
        "workforce planning", "capacity planning", "budget planning",
        "resource planning", "timeline planning", "goal setting", "forecasting",
        "roadmap", "action plan"
    },

    "organization": {
        "organization", "организация", "tashkil etish", "organizational structure",
        "organizational development", "workflow management", "business process",
        "organization culture", "department coordination", "corporate structure",
        "org chart", "company hierarchy", "administration", "planning process"
    },

    "finance": {
        "finance", "финансы", "moliya", "financial analysis", "budgeting",
        "forecasting", "cost control", "investment", "accounting principles",
        "financial reporting", "balance sheet", "cash flow", "roi", "profitability",
        "kpi", "financial planning", "business finance", "corporate finance",
        "capital management", "expenses", "revenue", "financial risk"
    },

    "accounting": {
        "accounting", "бухгалтерия", "hisob-kitob", "bookkeeping", "tax accounting",
        "audit", "balance sheet", "accounts payable", "accounts receivable",
        "financial statements", "invoice", "ledger", "journal entry", "vat",
        "payroll", "tax report", "cost accounting", "erp accounting", "1c", "sap fi"
    },

    "analytics": {
        "analytics", "аналитика", "tahlil", "data analysis", "business analytics",
        "financial analytics", "marketing analytics", "predictive analytics",
        "reporting", "dashboard", "excel analysis", "google sheets", "tableau",
        "power bi", "kpi analysis", "trend analysis", "data visualization",
        "data-driven decisions", "bi tools", "metric tracking"
    },


    # === Design / Creative / Media (expanded) ===
    "design": {
        "design", "дизайн", "dizayn", "visual design", "creative direction",
        "art direction", "color theory", "composition", "layout design",
        "typography", "grid systems", "mockups", "prototyping", "wireframes",
        "interface design", "user flow", "concept design", "storyboarding",
        "design systems", "design thinking", "branding design", "style guide"
    },

    "graphic design": {
        "graphic design", "графический дизайн", "poster design", "flyer design",
        "print design", "brochure design", "logo design", "social media design",
        "banner design", "corporate identity", "packaging design", "business card",
        "infographic", "layout composition", "visual branding", "color correction",
        "advertising design", "brand visuals", "marketing materials", "illustrative design"
    },

    "ui/ux": {
        "ui", "ux", "ui/ux", "user interface", "user experience", "дизайн интерфейсов",
        "usability", "wireframing", "prototyping", "user flow", "information architecture",
        "interaction design", "mobile ui", "responsive design", "user journey",
        "accessibility", "design system", "a/b testing", "heuristic evaluation",
        "persona mapping", "ux writing", "research", "visual hierarchy", "microinteractions"
    },

    "photoshop": {
        "photoshop", "фотошоп", "adobe photoshop", "photo retouching", "color correction",
        "photo manipulation", "layer masking", "image compositing", "digital painting",
        "photo montage", "mockup creation", "texture editing", "background removal",
        "filter effects", "poster creation", "visual editing"
    },

    "illustrator": {
        "illustrator", "иллюстратор", "adobe illustrator", "vector design",
        "logo creation", "icon design", "infographics", "line art",
        "illustration", "character design", "vector tracing", "typography design",
        "flat illustration", "digital sketching"
    },

    "figma": {
        "figma", "фигма", "prototype creation", "auto layout", "components",
        "design system", "ui kits", "figjam", "collaboration", "frame design",
        "responsive prototype", "interactive components", "design handoff"
    },

    "adobe": {
        "adobe", "адоб", "adobe creative suite", "adobe cc", "adobe xd", "adobe indesign",
        "adobe premiere", "adobe after effects", "adobe lightroom", "adobe audition",
        "creative cloud", "adobe animate"
    },

    "motion design": {
        "motion design", "моушен дизайн", "motion graphics", "animation", "2d animation",
        "3d animation", "kinetic typography", "title animation", "explainer video",
        "video intro", "transitions", "composition", "after effects", "motion templates",
        "visual storytelling", "frame-by-frame animation"
    },

    "video editing": {
        "video editing", "видеомонтаж", "video montaj", "video production", "cutting",
        "color grading", "sound design", "timeline editing", "transitions", "visual effects",
        "subtitles", "video stabilization", "premiere pro", "davinci resolve", "capcut",
        "final cut pro", "short video editing", "content editing", "montage"
    },

    "branding": {
        "branding", "брендинг", "brand identity", "logo design", "brand strategy",
        "visual identity", "brand guidelines", "rebranding", "brand tone", "brand colors",
        "brandbook", "corporate style", "positioning", "brand storytelling",
        "packaging identity", "brand consistency"
    },

    "photography": {
        "photography", "фотография", "fotografiya", "photo editing", "portrait photography",
        "product photography", "lighting setup", "studio photography", "dslr", "camera setup",
        "composition", "exposure", "white balance", "photo retouching", "color correction",
        "photo session", "image post-processing"
    },

    "3d modeling": {
        "3d", "3d modeling", "3д моделирование", "3d rendering", "3d visualization",
        "blender", "3ds max", "maya", "cinema 4d", "zbrush", "substance painter",
        "3d texturing", "low poly modeling", "high poly modeling", "rigging", "uv mapping",
        "lighting", "render setup", "environment design", "3d animation", "architectural visualization"
    },

    "animation": {
        "animation", "анимация", "2d animation", "3d animation", "frame-by-frame",
        "character animation", "motion capture", "rigging", "keyframing",
        "after effects animation", "toon boom", "spine animation", "animatic creation",
        "storyboard animation", "rendering", "timing and spacing", "stop motion"
    },


    # === Logistics / Others ===
    "logistics": {
        "logistics", "логистика", "transport management", "supply planning", "shipment", "shipment tracking",
        "delivery planning", "inventory", "inventory control", "inventory management", "warehouse management",
        "fulfillment", "order processing", "freight", "freight forwarding", "cargo", "dispatch", "eld",
        "route optimization", "truck scheduling", "carrier management", "3pl", "third party logistics",
        "logistics coordination", "supply chain", "supply planning", "fleet management", "transportation",
        "shipment update", "delivery update", "shipment documentation", "import/export", "customs clearance",
        "air freight", "sea freight", "rail logistics", "road freight", "dispatch control", "vehicle tracking",
        "gps monitoring", "tms", "wms", "scm", "stock management", "material flow", "distribution",
        "reverse logistics", "last mile delivery", "package tracking", "shipment scheduling",
        "transport dispatcher", "shipment coordinator", "export documentation", "delivery performance"
    },
    "supply chain": {
        "supply chain", "цепочка поставок", "procurement", "vendor management", "supplier relationship",
        "material planning", "demand forecasting", "production planning", "supply analysis", "supply optimization",
        "purchasing", "sourcing", "supplier audit", "contract negotiation", "lead time", "order fulfillment",
        "logistics analytics", "supply reporting", "demand planning", "inventory optimization", "ERP", "SAP SCM"
    },
    "warehouse": {
        "warehouse", "склад", "ombor", "inventory storage", "picking", "packing", "sorting", "labeling",
        "warehouse operations", "inventory accuracy", "stock control", "warehouse safety", "forklift operation",
        "warehouse automation", "barcode scanning", "WMS", "warehouse optimization", "warehouse layout",
        "cross-docking", "receiving", "dispatching", "stock rotation", "cycle counting"
    },
    "customer support": {"customer support", "поддержка клиентов", "mijozlarga xizmat"},
    "barista": {"barista", "бариста", "kofe tayyorlovchi"},
    "real estate": {"real estate", "недвижимость", "ko‘chmas mulk"},
    "driver": {
        "driver", "водитель", "haydovchi", "truck driver", "delivery driver", "courier", "taxi driver",
        "route planning", "gps navigation", "vehicle inspection", "eld system", "dot compliance", "cargo handling",
        "safe driving", "defensive driving", "fleet compliance", "trip reporting"
    },
    # === Education / Teaching / E-learning ===
    "education": {
        "education", "ta’lim", "образование", "teacher", "tutor", "lecturer", "professor",
        "training specialist", "school", "college", "university", "curriculum development",
        "lesson planning", "classroom management", "student engagement", "pedagogy",
        "distance learning", "online learning", "e-learning", "LMS", "Moodle", "Google Classroom",
        "teaching materials", "assessment", "grading", "student success", "academic advisor"
    },

    "e-learning": {
        "e-learning", "онлайн обучение", "masofaviy ta’lim", "digital education",
        "online courses", "learning platform", "MOOC", "video lessons", "Udemy",
        "Coursera", "interactive learning", "webinar", "education technology", "EdTech",
        "SCORM", "content authoring", "virtual classroom"
    },

    # === Healthcare / Medicine / Pharmacy ===
    "healthcare": {
        "healthcare", "здравоохранение", "sog‘liqni saqlash", "doctor", "nurse",
        "clinic", "hospital", "paramedic", "physician", "medical assistant",
        "telemedicine", "medical records", "health services", "patient care",
        "health insurance", "clinical practice", "public health", "health monitoring"
    },

    "pharmacy": {
        "pharmacy", "фармация", "dorixona", "pharmacist", "drug", "medication",
        "prescription", "pharmacology", "pharmaceutical", "inventory management",
        "pharmaceutical sales", "dosage", "side effects", "dispensing", "OTC drugs"
    },

    "medicine": {
        "medicine", "медицина", "tibbiyot", "diagnosis", "treatment", "therapy",
        "laboratory", "medical analysis", "clinical trials", "medical devices",
        "immunology", "surgery", "cardiology", "neurology", "pediatrics",
        "internal medicine", "general practitioner"
    },

    # === Engineering / Construction / Architecture ===
    "engineering": {
        "engineering", "инжиниринг", "muhandislik", "mechanical engineering",
        "electrical engineering", "civil engineering", "industrial engineering",
        "automation", "CAD", "SolidWorks", "AutoCAD", "engineering design",
        "technical drawing", "R&D", "product design", "engineering analysis",
        "thermal systems", "HVAC", "manufacturing process"
    },

    "construction": {
        "construction", "строительство", "qurilish", "architecture", "project site",
        "blueprint", "civil works", "construction management", "site supervision",
        "materials management", "BIM", "Revit", "cost estimation", "safety management",
        "infrastructure", "urban planning", "structural design", "engineering drawing"
    },

    "architecture": {
        "architecture", "архитектура", "arxitektura", "architectural design",
        "urban planning", "interior design", "landscape design", "3d visualization",
        "Revit", "AutoCAD Architecture", "floor plan", "rendering", "concept design",
        "construction documentation", "building codes"
    },

    # === Legal / Compliance / Government ===
    "legal": {
        "legal", "юридический", "huquqiy", "lawyer", "attorney", "paralegal",
        "legal counsel", "contract law", "labor law", "corporate law",
        "intellectual property", "litigation", "legal documentation", "GDPR",
        "compliance", "privacy policy", "court", "legal writing", "due diligence"
    },

    "compliance": {
        "compliance", "соответствие", "muvofiqlik", "risk management",
        "regulations", "corporate governance", "anti-money laundering",
        "data protection", "audit compliance", "SOX", "GDPR", "policy enforcement",
        "internal audit", "financial compliance", "ISO standards"
    },

    # === Cybersecurity / Networking ===
    "cybersecurity": {
        "cybersecurity", "кибербезопасность", "axborot xavfsizligi", "network security",
        "information security", "penetration testing", "ethical hacking",
        "vulnerability scanning", "firewall", "VPN", "IDS", "IPS", "encryption",
        "OWASP", "ISO 27001", "SOC", "incident response", "threat detection"
    },

    "networking": {
        "networking", "сетевые технологии", "tarmoq texnologiyalari", "TCP/IP",
        "DNS", "DHCP", "routing", "switching", "LAN", "WAN", "VPN", "firewall",
        "Cisco", "MikroTik", "network configuration", "Wi-Fi", "load balancing",
        "network monitoring", "NOC", "server administration"
    },

    # === Hospitality / Tourism / Restaurant ===
    "hospitality": {
        "hospitality", "гостеприимство", "mehmondo‘stlik", "hotel management",
        "guest relations", "front desk", "reservation", "concierge", "hospitality service",
        "restaurant", "catering", "menu planning", "food service", "barista",
        "bartender", "customer satisfaction", "HACCP", "event management"
    },

    "tourism": {
        "tourism", "туризм", "sayohat", "travel agency", "tour operator",
        "booking", "itinerary", "visa support", "tour guide", "holiday packages",
        "hotel booking", "aviation", "airport transfer", "tour planning",
        "ticketing", "travel insurance", "expedition"
    },

    # === E-commerce / Retail / Procurement ===
    "e-commerce": {
        "e-commerce", "электронная коммерция", "onlayn savdo", "shopify", "woocommerce",
        "magento", "marketplace", "amazon seller", "product listing", "order management",
        "customer support", "online store", "inventory sync", "e-commerce analytics",
        "dropshipping", "payment gateway", "refunds", "conversion rate", "UX optimization"
    },

    "retail": {
        "retail", "ритейл", "chakana savdo", "POS system", "store management",
        "sales associate", "cashier", "merchandising", "inventory control",
        "store operations", "visual merchandising", "product placement", "discount strategy",
        "customer service", "retail analytics"
    },

    "procurement": {
        "procurement", "закупки", "sotib olish", "purchasing", "supplier management",
        "contract negotiation", "price comparison", "RFQ", "RFP", "sourcing",
        "vendor management", "cost saving", "inventory planning", "procurement process",
        "ERP procurement", "purchase order"
    },

    # === Game Development / AR-VR ===
    "game development": {
        "game development", "разработка игр", "o‘yin ishlab chiqish", "unity", "unreal engine",
        "game design", "level design", "character animation", "2d games", "3d games",
        "game mechanics", "gameplay", "physics engine", "shader", "ai for games",
        "mobile games", "multiplayer", "VR", "AR", "metaverse", "gamification"
    },

    "ar-vr": {
        "ar", "vr", "ar/vr", "augmented reality", "virtual reality",
        "3d environment", "unity 3d", "oculus", "meta quest", "3d interaction",
        "immersive experience", "xr", "simulation", "rendering", "spatial design"
    },

    # === Banking / Fintech / Investment ===
    "banking": {
        "banking", "банковское дело", "bank ishi", "retail banking", "corporate banking",
        "credit analysis", "loan management", "risk assessment", "AML", "KYC",
        "compliance", "financial services", "branch operations", "payment processing",
        "core banking", "swift", "credit scoring"
    },

    "fintech": {
        "fintech", "финтех", "financial technology", "digital banking", "mobile payments",
        "blockchain finance", "cryptocurrency", "payment gateway", "wallet system",
        "api integration", "open banking", "transaction monitoring", "digital lending"
    },

    "investment": {
        "investment", "инвестиции", "investitsiya", "portfolio management", "asset management",
        "financial markets", "stocks", "bonds", "ETF", "mutual funds", "valuation",
        "risk analysis", "capital markets", "investment strategy", "IPO", "venture capital",
        "private equity", "fundraising"
    },

    # === Customer Support / Call Center ===
    "customer service": {
        "customer service", "обслуживание клиентов", "mijozlarga xizmat",
        "call center", "helpdesk", "support agent", "ticketing system",
        "crm", "inbound calls", "outbound calls", "customer satisfaction",
        "problem resolution", "escalation", "service quality", "chat support",
        "email support", "customer feedback", "technical support"
    },

    "technical support": {
        "technical support", "техническая поддержка", "texnik yordam",
        "helpdesk", "troubleshooting", "incident management", "it support",
        "hardware maintenance", "software support", "remote support",
        "system diagnostics", "user support", "service desk", "ticketing"
    }

}


# ==========================================
# 🧠 TEXT NORMALIZATION & SKILL EXTRACTION
# ==========================================
def normalize_text(t: str) -> str:
    return re.sub(r"\s+", " ", t.strip().lower())

def extract_skills(text: str) -> Set[str]:
    found = set()
    for key, pattern in SKILL_PATTERNS.items():
        if pattern.search(text):
            found.add(key)
    return found

def precompute_skills(texts: List[str]) -> List[Set[str]]:
    return [extract_skills(normalize_text(t)) for t in texts]

def is_related_title(title_a: str, title_b: str, sim: float, threshold: float = 0.55) -> bool:
    """
    Wide adaptive title filter with domain awareness and synonym expansion.
    Works for English + Russian and mixed-language titles.
    """
    if sim < threshold:
        return False

    a, b = title_a.lower(), title_b.lower()

    # --- Unified normalization ---
    def normalize(s: str) -> str:
        s = s.lower()
        s = s.replace("-", " ").replace("_", " ")
        s = re.sub(r"[^a-zа-яё0-9\\s]", "", s)
        return re.sub(r"\\s+", " ", s).strip()

    a, b = normalize(a), normalize(b)

    # --- Broadened tech stacks & key skill keywords ---
    tech_keywords = [
        "python", "php", "java", "javascript", "typescript", "c#", "dotnet", "react",
        "angular", "vue", "flutter", "node", "golang", "ruby", "laravel", "django",
        "fastapi", "spring", "kotlin", "swift", "ml", "ai", "data", "sql", "devops",
        "aws", "azure", "gcp", "docker", "kubernetes"
    ]
    tech_a = [t for t in tech_keywords if t in a]
    tech_b = [t for t in tech_keywords if t in b]
    if tech_a and tech_b and not any(t in tech_b for t in tech_a):
        return False

    # --- Domain clusters (expanded and multilingual) ---
    clusters = {
            "tech": [
                "developer", "engineer", "programmer", "backend", "frontend", "fullstack",
                "software", "it", "qa", "tester", "devops", "data scientist", "data analyst",
                "ml engineer", "ai", "аналитик", "разработчик", "инженер"
            ],
            "design": [
                "designer", "graphic", "ux", "ui", "product designer", "motion", "illustrator",
                "art", "visual", "video", "3d", "photoshop", "figma"
            ],
            "marketing": [
                "marketing", "advertising", "smm", "seo", "content", "copywriter", "brand",
                "communications", "targeting", "ppc", "performance", "digital", "pr", "manager"
            ],
            "logistics": [
                "logistics", "supply", "warehouse", "driver", "courier", "delivery", "dispatcher",
                "coordinator", "shipping", "fleet", "supply chain"
            ],
            "finance": [
                "accountant", "finance", "auditor", "bank", "economist", "cashier", "controller",
                "финансы", "бухгалтер", "аудитор"
            ],
            "medical": [
                "doctor", "nurse", "dentist", "pharma", "veterinarian", "clinician",
                "therapist", "surgeon", "врач", "медсестра", "медик"
            ],
            "education": [
                "teacher", "tutor", "lecturer", "trainer", "instructor", "professor", "educator",
                "преподаватель", "учитель", "репетитор", "тренер"
            ],
            "management": [
                "manager", "director", "lead", "head", "supervisor", "chief", "cto", "ceo",
                "product owner", "project manager", "scrum master", "team lead"
            ],
            "sales": [
                "sales", "account manager", "business development", "bdm", "b2b", "b2c",
                "client", "customer", "retail", "shop", "call center", "продаж"
            ],
            "support": [
                "support", "helpdesk", "customer service", "call center", "оператор", "поддержка"
            ],
        }
    def get_cluster(t: str) -> str:
        for k, kws in clusters.items():
            for w in kws:
                if w in t:
                    return k
        return "other"

    ca, cb = get_cluster(a), get_cluster(b)

    # --- Cross-domain allowances ---
    same_or_related = {
        ("tech", "management"),
        ("management", "tech"),
        ("marketing", "sales"),
        ("sales", "marketing"),
        ("education", "management"),
        ("tech", "data"),  # possible data/ML overlap
    }

    if ca == cb:
        return True
    if (ca, cb) in same_or_related or (cb, ca) in same_or_related:
        return True

    # Otherwise — clearly different domains
    if ca != cb and "other" not in (ca, cb):
        return False

    # Fallback: if titles share a significant word overlap
    set_a, set_b = set(a.split()), set(b.split())
    overlap = len(set_a & set_b) / max(1, len(set_a | set_b))
    if overlap > 0.25:
        return True

    return False




# ==========================================
# ⚡ ENCODING WITH CACHE
# ==========================================
def encode_texts(texts: List[str], mode: str) -> torch.Tensor:
    """Batch encode with caching support."""
    embeddings, to_encode, encode_map = [], [], []
    for idx, t in enumerate(texts):
        key = hashlib.md5(f"{mode}:{t}".encode()).hexdigest()
        if key in VACANCY_CACHE:
            embeddings.append(VACANCY_CACHE[key])
        else:
            to_encode.append(t)
            encode_map.append((idx, key))

    if to_encode:
        prefix = "query: " if "e5" in MODEL_NAME.lower() and mode == "query" else \
                 "passage: " if "e5" in MODEL_NAME.lower() and mode == "passage" else ""
        new_embs = model.encode(
            [prefix + t for t in to_encode],
            convert_to_tensor=True,
            normalize_embeddings=True,
            device=DEVICE,
            batch_size=32,
        )
        for (idx, key), emb in zip(encode_map, new_embs):
            VACANCY_CACHE[key] = emb
            embeddings.append(emb)

    if len(embeddings) != len(texts):
        # Sort back to correct order
        emb_dict = {hashlib.md5(f"{mode}:{t}".encode()).hexdigest(): VACANCY_CACHE[hashlib.md5(f"{mode}:{t}".encode()).hexdigest()] for t in texts}
        embeddings = [emb_dict[hashlib.md5(f"{mode}:{t}".encode()).hexdigest()] for t in texts]

    return torch.stack(embeddings)


# ==========================================
# 📦 MODELS
# ==========================================
class ResumeInput(BaseModel):
    title: str
    description: str

class VacancyInput(BaseModel):
    id: Optional[str] = None
    title: str
    text: str

class BulkMatchRequest(BaseModel):
    resumes: List[ResumeInput]
    vacancies: List[VacancyInput]
    top_k: int = Field(5, ge=1)
    min_score: float = Field(0.0, ge=0.0, le=100.0)
    weight_embed: float = Field(0.75, ge=0.0, le=1.0)
    weight_jaccard: float = Field(0.15, ge=0.0, le=1.0)
    weight_cov: float = Field(0.10, ge=0.0, le=1.0)
    title_threshold: float = Field(0.6, ge=0.0, le=1.0)

class BulkTopItem(BaseModel):
    vacancy_id: Optional[str]
    vacancy_index: int
    score: float
    embedding_score: float
    skills_jaccard: float
    keyword_coverage: float
    skill_matches: List[str]
    skill_missing: List[str]

class BulkMatchResponse(BaseModel):
    model: str
    device: str
    resumes: int
    vacancies: int
    top_k: int
    results: List[List[BulkTopItem]]


# ==========================================
# 🚀 CORE MATCH FUNCTION
# ==========================================
def do_bulk_match(req: BulkMatchRequest) -> dict:
    resumes = req.resumes
    vacancies = req.vacancies
    N, M = len(resumes), len(vacancies)
    assert N > 0 and M > 0, "Empty input"

    # --- Stage 1: Title similarity ---
    r_titles = [normalize_text(r.title) for r in resumes]
    v_titles = [normalize_text(v.title) for v in vacancies]

    R_titles = encode_texts(r_titles, "query")
    V_titles = encode_texts(v_titles, "passage")

    with torch.no_grad():
        title_sims = util.cos_sim(R_titles, V_titles)

    # --- Stage 2: Description match only if title similarity is high ---
    results = []
    top_k = min(req.top_k, M)

    for i in range(N):
        good_idx = []
        for j in range(M):
            sim = title_sims[i, j].item()
            if is_related_title(resumes[i].title, vacancies[j].title, sim, req.title_threshold):
                good_idx.append(j)

        r_desc = normalize_text(resumes[i].description)
        v_texts = [normalize_text(vacancies[j].text) for j in good_idx]

        r_sk = precompute_skills([r_desc])[0]
        v_sk = precompute_skills(v_texts)

        R = encode_texts([r_desc], "query")
        V = encode_texts(v_texts, "passage")

        with torch.no_grad():
            S = util.cos_sim(R, V)[0]

        top_vals, top_idx = torch.topk(S, min(top_k, len(v_texts)))
        items = []
        for val, rel_idx in zip(top_vals.tolist(), top_idx.tolist()):
            j = good_idx[rel_idx]
            emb01 = max(0.0, min(1.0, (val + 1.0) / 2.0))
            vsk = v_sk[rel_idx]
            inter = sorted(r_sk & vsk)
            missing = sorted(vsk - r_sk)
            union = r_sk | vsk
            jaccard = len(inter) / len(union) if union else 0.0
            coverage = len(inter) / len(vsk) if vsk else 0.0
            final01 = req.weight_embed * emb01 + req.weight_jaccard * jaccard + req.weight_cov * coverage
            score = round(final01 * 100, 2)
            if score >= req.min_score:
                items.append(BulkTopItem(
                    vacancy_id=vacancies[j].id,
                    vacancy_index=j,
                    score=score,
                    embedding_score=round(emb01 * 100, 2),
                    skills_jaccard=round(jaccard * 100, 2),
                    keyword_coverage=round(coverage * 100, 2),
                    skill_matches=inter,
                    skill_missing=missing,
                ))
        items.sort(key=lambda x: x.score, reverse=True)
        results.append(items)

    return {
        "model": MODEL_NAME,
        "device": DEVICE,
        "resumes": N,
        "vacancies": M,
        "top_k": top_k,
        "results": results,
    }


# ==========================================
# 🔄 API ENDPOINTS
# ==========================================
@app.post("/bulk-match-fast", response_model=BulkMatchResponse)
async def bulk_match(req: BulkMatchRequest):
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(executor, lambda: do_bulk_match(req))


@app.get("/")
def root():
    return {"ok": True, "message": "Resume-Vacancy Matcher API v5 Ultra-Fast", "endpoints": ["/bulk-match-fast (POST)"]}

@app.get("/health")
def health():
    return {"status": "ok", "model": MODEL_NAME, "device": DEVICE}
