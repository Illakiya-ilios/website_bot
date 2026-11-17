"""
Chat-driven website navigation.

When a visitor explicitly asks to open / go to a page ("take me to the case
studies page", "open careers"), find_navigation() returns the matching page so
the API can tell the website to open it.

Only paths listed in SITE_PAGES are ever returned (an allowlist), so the bot
can never send visitors to an arbitrary or external URL. Paths are relative
("/careers"), so they work on both ilios.digital and www.ilios.digital.

To add a page: add an entry to SITE_PAGES. The words in the path are matched
automatically; add extra phrases to "aliases" for names people actually use.
"""
import re

# path -> (label shown in the chat reply, extra aliases)
SITE_PAGES = {
    # Company
    "/": ("home", ["home", "homepage", "home page", "main page", "landing page"]),
    "/about": ("About Us", ["about", "about us", "about ilios", "who we are", "our story"]),
    "/leadership": ("Leadership", ["leadership", "leaders", "leadership team", "directors", "management team", "founders"]),
    "/careers": ("Careers", ["careers", "career", "jobs", "job openings", "openings", "open positions", "positions", "vacancies", "hiring", "work with you"]),
    "/contact": ("Contact Us", ["contact", "contact us", "get in touch", "reach you", "contact form"]),
    "/projects": ("Projects", ["projects", "our work", "portfolio"]),
    "/insights": ("Insights", ["insights", "blog", "articles"]),
    "/partners": ("Partners", ["partners", "partnerships", "partner"]),
    "/industries": ("Industries", ["industries", "industry", "sectors"]),

    # Industries
    "/industries/government-public-sector": ("Government & Public Sector", ["government", "public sector"]),
    "/industries/banking-financial-services-fintech": ("Banking & Financial Services", ["banking", "bfsi", "financial services", "fintech"]),
    "/industries/healthcare-life-sciences": ("Healthcare & Life Sciences", ["healthcare", "health care", "life sciences"]),
    "/industries/enterprise-retail-smart-cities": ("Enterprise, Retail & Smart Cities", ["retail", "smart cities", "smart city"]),

    # Partners
    "/partners/aws": ("AWS Partnership", ["aws", "amazon web services"]),
    "/partners/azure": ("Microsoft Azure Partnership", ["azure", "microsoft"]),
    "/partners/gcp": ("Google Cloud Partnership", ["gcp", "google cloud"]),
    "/partners/claude": ("Anthropic Claude Partnership", ["claude", "anthropic"]),

    # Case studies
    "/insights/case-studies": ("Case Studies", ["case studies", "case study", "success stories", "customer stories", "client stories"]),
    "/insights/case-studies/directorate-technical-education-tn": ("Directorate of Technical Education case study", ["directorate of technical education", "technical education", "dote"]),
    "/insights/case-studies/fiduccia-ventures-agentic-ai-recruitment": ("Fiduccia Ventures case study", ["fiduccia", "fiduccia ventures", "ai recruitment"]),
    "/insights/case-studies/gcc-agentic-ai-municipal-finance": ("GCC Municipal Finance case study", ["gcc", "greater chennai corporation", "municipal finance"]),
    "/insights/case-studies/hrce-production-workload-migration": ("HRCE case study", ["hrce", "hr ce"]),
    "/insights/case-studies/tn-finance-wamis-2": ("TN Finance WAMIS case study", ["wamis", "tn finance"]),
    "/insights/case-studies/tnega-whatsapp-platform-migration": ("TNeGA WhatsApp case study", ["tnega", "whatsapp"]),

    # Products
    "/products/cadlens": ("CADLens", ["cadlens", "cad lens"]),
    "/products/cloudcut": ("CloudCut", ["cloudcut", "cloud cut"]),
    "/products/dpdp-consultancy": ("DPDP Consultancy", ["dpdp", "consent cockpit", "privacy auto pilot", "privacy autopilot"]),
    "/products/flowlens": ("FlowLens", ["flowlens", "flow lens"]),
    "/products/genverse": ("GenVerse", ["genverse", "gen verse"]),
    "/products/managed-service": ("Managed Service product", ["managed service product"]),
    "/products/quantumdesk": ("Quantum Desk", ["quantumdesk", "quantum desk"]),
    "/products/sleuth": ("Sleuth", ["sleuth"]),
    "/products/sourcelens": ("SourceLens", ["sourcelens", "source lens"]),
    "/products/zenom": ("Zenom", ["zenom"]),

    # Services
    "/services/workspace-licensing": ("Workspace Licensing", ["workspace licensing", "licensing", "licenses", "licences"]),
    "/services/ai-solutions": ("AI Solutions", ["ai solutions", "ai services", "artificial intelligence"]),
    "/services/ai-solutions/agentic-ai-autonomous-workflows": ("Agentic AI & Autonomous Workflows", ["agentic ai", "agentic", "autonomous workflows", "ai agents"]),
    "/services/ai-solutions/enterprise-knowledge-copilots": ("Enterprise Knowledge Copilots", ["copilots", "copilot", "knowledge copilot"]),
    "/services/ai-solutions/genai-strategy-readiness": ("GenAI Strategy & Readiness", ["genai strategy", "ai strategy", "ai readiness"]),
    "/services/ai-solutions/llm-fine-tuning-rag": ("LLM Fine-Tuning & RAG", ["fine tuning", "finetuning", "rag", "llm"]),
    "/services/ai-solutions/responsible-ai-governance": ("Responsible AI Governance", ["responsible ai", "ai governance"]),
    "/services/cloud": ("Cloud & DevOps", ["cloud", "cloud services", "cloud and devops"]),
    "/services/cloud/devops-infrastructure-as-code": ("DevOps & Infrastructure as Code", ["devops", "infrastructure as code", "iac"]),
    "/services/cloud/finops-cloud-cost-optimization": ("FinOps & Cloud Cost Optimization", ["finops", "cloud cost", "cost optimization", "cost optimisation", "cost management"]),
    "/services/cloud/multi-cloud-architecture-migration": ("Multi-Cloud Architecture & Migration", ["multi cloud", "multicloud", "cloud migration"]),
    "/services/cloud/platform-engineering-containers": ("Platform Engineering & Containers", ["platform engineering", "containers", "kubernetes"]),
    "/services/cloud/site-reliability-engineering": ("Site Reliability Engineering", ["site reliability", "sre"]),
    "/services/cybersecurity": ("Cybersecurity", ["cybersecurity", "cyber security", "security services", "security"]),
    "/services/cybersecurity/cloud-network-security": ("Cloud & Network Security", ["network security", "cloud security"]),
    "/services/cybersecurity/governance-risk-compliance": ("Governance, Risk & Compliance", ["grc", "governance risk", "risk and compliance"]),
    "/services/cybersecurity/managed-soc-threat-response": ("Managed SOC & Threat Response", ["soc", "threat response", "managed soc"]),
    "/services/cybersecurity/vulnerability-management-pen-testing": ("Vulnerability Management & Pen Testing", ["pen testing", "penetration testing", "pentest", "vulnerability"]),
    "/services/cybersecurity/zero-trust-architecture-identity": ("Zero Trust & Identity", ["zero trust", "identity"]),
    "/services/managed-services": ("Managed Services", ["managed services", "managed service"]),
    "/services/managed-services/application-support-ams": ("Application Support (AMS)", ["application support", "ams"]),
    "/services/managed-services/backup-disaster-recovery": ("Backup & Disaster Recovery", ["backup", "disaster recovery"]),
    "/services/managed-services/managed-multi-cloud-hybrid-infra": ("Managed Multi-Cloud & Hybrid Infrastructure", ["hybrid infra", "hybrid infrastructure", "managed infrastructure"]),
    "/services/managed-services/service-desk-aiops-observability": ("Service Desk, AIOps & Observability", ["service desk", "aiops", "observability"]),
    "/services/software-development": ("Software Development", ["software development", "app development", "application development"]),
    "/services/software-development/api-engineering-integration": ("API Engineering & Integration", ["api engineering", "api integration"]),
    "/services/software-development/app-portfolio-assessment-strategy": ("App Portfolio Assessment", ["portfolio assessment", "app assessment"]),
    "/services/software-development/application-modernization-refactoring": ("Application Modernization", ["modernization", "modernisation", "refactoring"]),
    "/services/software-development/custom-enterprise-app-development": ("Custom Enterprise App Development", ["custom app", "custom enterprise app", "custom development"]),
    "/services/software-development/quality-engineering-test-automation": ("Quality Engineering & Test Automation", ["test automation", "quality engineering", "qa", "testing"]),
}

# The visitor must clearly ask to go somewhere. Plain questions such as
# "what is Zenom?" are answered in the chat and do not navigate.
_NAV_INTENT = re.compile(
    r"\b(open|take me|go to|goto|navigate|bring me|redirect|direct me|visit|"
    r"show me (?:the|your)|link (?:to|for)|where (?:can|do) i (?:find|see))\b"
    r"|\bpage\b",
    re.IGNORECASE,
)


def _normalize(text: str) -> str:
    """Lowercase and turn punctuation / hyphens into single spaces."""
    return " " + " ".join(re.findall(r"[a-z0-9]+", text.lower())) + " "


def _build_index():
    index = []  # (normalized phrase, path, label)
    for path, (label, aliases) in SITE_PAGES.items():
        phrases = set(aliases)
        phrases.add(label)
        slug = path.rstrip("/").rsplit("/", 1)[-1]
        if slug:
            phrases.add(slug.replace("-", " "))
        for p in phrases:
            norm = _normalize(p).strip()
            if norm:
                index.append((norm, path, label))
    return index


_INDEX = _build_index()


def find_navigation(message: str):
    """
    Return (path, label) if the message asks to open a known page, else None.

    Scoring:
    - Each matching page scores the length of its longest matched phrase.
    - A page gets a large bonus when its parent section was also mentioned,
      so "GCC case study" opens the GCC case study (parent: case studies)
      and "AWS partner page" opens /partners/aws.
    - A parent is dropped when one of its child pages also matched.
    """
    if not _NAV_INTENT.search(message):
        return None

    text = _normalize(message)
    matched = {}  # path -> (longest phrase length, label)
    for phrase, path, label in _INDEX:
        if f" {phrase} " in text and len(phrase) > matched.get(path, (0, ""))[0]:
            matched[path] = (len(phrase), label)
    if not matched:
        return None

    def is_ancestor(parent, child):
        return parent != "/" and child.startswith(parent + "/")

    best_path, best_score = None, -1
    for path, (length, _) in matched.items():
        if any(is_ancestor(path, other) for other in matched):
            continue  # a more specific child page matched
        score = length + (100 if any(is_ancestor(o, path) for o in matched) else 0)
        if score > best_score:
            best_path, best_score = path, score

    return best_path, matched[best_path][1]
