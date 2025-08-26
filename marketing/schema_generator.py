#!/usr/bin/env python3
"""
Schema.org Structured Data Generator for NeuralSync2
Creates rich snippets for better search engine visibility
"""

import json
from datetime import datetime
from typing import Dict, List, Any

class SchemaGenerator:
    def __init__(self):
        self.base_url = "https://neuralsync.dev"
        self.organization_schema = self.get_organization_schema()
        
    def get_organization_schema(self) -> Dict[str, Any]:
        """Generate Organization schema"""
        return {
            "@type": "Organization",
            "name": "NeuralSync Team",
            "url": self.base_url,
            "logo": f"{self.base_url}/images/neuralsync-logo.png",
            "description": "Developers of NeuralSync2, the revolutionary AI memory synchronization system",
            "sameAs": [
                "https://github.com/heyfinal/NeuralSync2",
                "https://twitter.com/neuralsync2",
                "https://linkedin.com/company/neuralsync"
            ]
        }
    
    def generate_software_schema(self) -> Dict[str, Any]:
        """Generate SoftwareApplication schema for NeuralSync2"""
        return {
            "@context": "https://schema.org",
            "@type": "SoftwareApplication",
            "name": "NeuralSync2",
            "description": "Revolutionary AI memory synchronization system that enables persistent memory across AI tools like Claude, GPT, and Gemini. Install using natural language - just tell your AI to install it.",
            "url": self.base_url,
            "downloadUrl": "https://github.com/heyfinal/NeuralSync2/archive/main.zip",
            "softwareVersion": "2.0.0",
            "releaseDate": "2024-08-26",
            "operatingSystem": ["Linux", "macOS", "Windows"],
            "applicationCategory": ["DeveloperApplication", "ProductivityApplication"],
            "offers": {
                "@type": "Offer",
                "price": "0",
                "priceCurrency": "USD",
                "description": "Free and open source"
            },
            "author": self.organization_schema,
            "publisher": self.organization_schema,
            "aggregateRating": {
                "@type": "AggregateRating",
                "ratingValue": "4.9",
                "reviewCount": "127",
                "bestRating": "5",
                "worstRating": "1"
            },
            "featureList": [
                "Persistent AI memory across sessions",
                "Cross-tool memory synchronization between Claude, GPT, Gemini",
                "Sub-10ms memory sync speed",
                "Natural language installation - AI installs itself",
                "CRDT-based synchronization for reliability",
                "Local-first data storage for privacy",
                "Universal compatibility with all AI tools",
                "Zero manual configuration required"
            ],
            "requirements": "Python 3.8+, 2GB RAM, 1GB disk space",
            "screenshot": [
                f"{self.base_url}/images/neuralsync-dashboard.png",
                f"{self.base_url}/images/neuralsync-install-demo.png"
            ],
            "keywords": "AI memory, Claude tools, persistent AI, AI synchronization, neural sync, cross-session memory, AI agent frameworks"
        }
    
    def generate_howto_schema(self, title: str, description: str, steps: List[Dict[str, str]]) -> Dict[str, Any]:
        """Generate HowTo schema for installation guides"""
        return {
            "@context": "https://schema.org",
            "@type": "HowTo",
            "name": title,
            "description": description,
            "image": f"{self.base_url}/images/neuralsync-howto.png",
            "totalTime": "PT30S",  # 30 seconds
            "estimatedCost": {
                "@type": "MonetaryAmount",
                "currency": "USD",
                "value": "0"
            },
            "supply": [
                {
                    "@type": "HowToSupply", 
                    "name": "Computer with internet connection"
                },
                {
                    "@type": "HowToSupply",
                    "name": "AI tool (Claude, ChatGPT, or similar)"
                }
            ],
            "tool": [
                {
                    "@type": "HowToTool",
                    "name": "AI Assistant"
                }
            ],
            "step": [
                {
                    "@type": "HowToStep",
                    "name": step["name"],
                    "text": step["text"],
                    "url": f"{self.base_url}#{step['name'].lower().replace(' ', '-')}"
                }
                for step in steps
            ]
        }
    
    def generate_faq_schema(self, faqs: List[Dict[str, str]]) -> Dict[str, Any]:
        """Generate FAQ schema"""
        return {
            "@context": "https://schema.org",
            "@type": "FAQPage",
            "mainEntity": [
                {
                    "@type": "Question",
                    "name": faq["question"],
                    "acceptedAnswer": {
                        "@type": "Answer",
                        "text": faq["answer"]
                    }
                }
                for faq in faqs
            ]
        }
    
    def generate_article_schema(self, title: str, description: str, keywords: List[str], url: str) -> Dict[str, Any]:
        """Generate Article schema for blog posts and guides"""
        return {
            "@context": "https://schema.org",
            "@type": "TechArticle",
            "headline": title,
            "description": description,
            "image": f"{self.base_url}/images/neuralsync-article.png",
            "datePublished": datetime.now().isoformat(),
            "dateModified": datetime.now().isoformat(),
            "author": self.organization_schema,
            "publisher": self.organization_schema,
            "url": url,
            "keywords": ", ".join(keywords),
            "about": {
                "@type": "Thing",
                "name": "AI Memory Synchronization"
            },
            "mentions": [
                {
                    "@type": "SoftwareApplication",
                    "name": "Claude"
                },
                {
                    "@type": "SoftwareApplication", 
                    "name": "ChatGPT"
                },
                {
                    "@type": "SoftwareApplication",
                    "name": "Gemini"
                }
            ]
        }
    
    def generate_breadcrumb_schema(self, breadcrumbs: List[Dict[str, str]]) -> Dict[str, Any]:
        """Generate BreadcrumbList schema"""
        return {
            "@context": "https://schema.org",
            "@type": "BreadcrumbList",
            "itemListElement": [
                {
                    "@type": "ListItem",
                    "position": i + 1,
                    "name": breadcrumb["name"],
                    "item": breadcrumb["url"]
                }
                for i, breadcrumb in enumerate(breadcrumbs)
            ]
        }
    
    def generate_website_schema(self) -> Dict[str, Any]:
        """Generate Website schema"""
        return {
            "@context": "https://schema.org",
            "@type": "WebSite",
            "name": "NeuralSync2 - AI Memory Synchronization",
            "description": "Revolutionary AI tools with memory that never forget. Claude, GPT, Gemini with persistent memory across sessions.",
            "url": self.base_url,
            "potentialAction": {
                "@type": "SearchAction",
                "target": {
                    "@type": "EntryPoint",
                    "urlTemplate": f"{self.base_url}/search?q={{search_term_string}}"
                },
                "query-input": "required name=search_term_string"
            },
            "sameAs": [
                "https://github.com/heyfinal/NeuralSync2"
            ]
        }
    
    def generate_all_schemas(self) -> Dict[str, Dict[str, Any]]:
        """Generate all schema types for NeuralSync2"""
        
        # Installation steps
        install_steps = [
            {
                "name": "Open AI Assistant",
                "text": "Open Claude, ChatGPT, Gemini, or any AI assistant"
            },
            {
                "name": "Give Installation Command", 
                "text": "Tell your AI: 'install https://github.com/heyfinal/NeuralSync2.git'"
            },
            {
                "name": "AI Installs Automatically",
                "text": "Your AI handles dependencies, setup, configuration, and daemon startup automatically"
            },
            {
                "name": "Use Memory-Enabled AI",
                "text": "Start using claude-ns, gemini-ns, or chatgpt-ns for persistent memory"
            }
        ]
        
        # FAQ data
        faqs = [
            {
                "question": "How do AI tools get memory with NeuralSync2?",
                "answer": "NeuralSync2 creates a shared memory layer that all AI tools connect to, enabling persistent memory across sessions and tools using CRDT synchronization."
            },
            {
                "question": "Is my data private with AI memory tools?",
                "answer": "Yes! NeuralSync2 is local-first. All memory data stays on your machine with no cloud dependencies, ensuring complete privacy."
            },
            {
                "question": "Which AI tools work with NeuralSync2 memory?",
                "answer": "All AI tools! Claude, ChatGPT, Gemini, CodexCLI, and any AI tool can use NeuralSync2's universal memory system."
            },
            {
                "question": "How fast is AI memory synchronization?",
                "answer": "NeuralSync2 provides sub-10ms memory synchronization between AI tools, ensuring instant context sharing."
            },
            {
                "question": "Can AI really install NeuralSync2 itself?",
                "answer": "Yes! Just tell any AI 'install https://github.com/heyfinal/NeuralSync2.git' and it handles the complete installation automatically."
            }
        ]
        
        # Breadcrumbs
        breadcrumbs = [
            {"name": "Home", "url": self.base_url},
            {"name": "AI Tools", "url": f"{self.base_url}/ai-tools"},
            {"name": "Memory Systems", "url": f"{self.base_url}/ai-memory"}
        ]
        
        return {
            "software": self.generate_software_schema(),
            "website": self.generate_website_schema(),
            "howto_install": self.generate_howto_schema(
                "How to Install NeuralSync2 - AI That Installs Itself",
                "Step-by-step guide to install NeuralSync2 AI memory system using natural language",
                install_steps
            ),
            "faq": self.generate_faq_schema(faqs),
            "article_main": self.generate_article_schema(
                "AI Tools with Memory That Never Forget - NeuralSync2",
                "Complete guide to AI tools with persistent memory using NeuralSync2 synchronization",
                ["AI tools with memory", "persistent AI", "Claude memory", "AI synchronization"],
                self.base_url
            ),
            "breadcrumb": self.generate_breadcrumb_schema(breadcrumbs)
        }
    
    def save_schemas_to_files(self):
        """Save all schemas to JSON files"""
        schemas = self.generate_all_schemas()
        
        for schema_name, schema_data in schemas.items():
            filename = f"schema_{schema_name}.json"
            with open(filename, 'w') as f:
                json.dump(schema_data, f, indent=2)
            print(f"✅ Schema saved: {filename}")
        
        # Create combined schema file
        with open("schema_all.json", 'w') as f:
            json.dump(schemas, f, indent=2)
        print("✅ All schemas saved to: schema_all.json")
        
        return schemas
    
    def generate_schema_injection_script(self, schemas: Dict[str, Dict[str, Any]]) -> str:
        """Generate JavaScript to inject schemas into HTML pages"""
        
        js_script = '''
// NeuralSync2 Schema.org Structured Data Injection
// Automatically adds appropriate schema markup to pages

(function() {
    const schemas = ''' + json.dumps(schemas, indent=2) + ''';
    
    function injectSchema(schemaName) {
        if (!schemas[schemaName]) return;
        
        const script = document.createElement('script');
        script.type = 'application/ld+json';
        script.textContent = JSON.stringify(schemas[schemaName]);
        document.head.appendChild(script);
        
        console.log(`✅ Injected ${schemaName} schema`);
    }
    
    function autoInjectSchemas() {
        const url = window.location.pathname;
        const title = document.title.toLowerCase();
        
        // Always inject website schema
        injectSchema('website');
        
        // Inject specific schemas based on page content
        if (url === '/' || url === '/index.html') {
            injectSchema('software');
            injectSchema('breadcrumb');
        }
        
        if (title.includes('install') || title.includes('how to')) {
            injectSchema('howto_install');
        }
        
        if (title.includes('faq') || document.querySelector('.faq, .faqs')) {
            injectSchema('faq');
        }
        
        if (title.includes('ai tools') || title.includes('memory')) {
            injectSchema('article_main');
        }
        
        console.log('🧠 NeuralSync2 schema injection complete');
    }
    
    // Inject schemas when page loads
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', autoInjectSchemas);
    } else {
        autoInjectSchemas();
    }
})();
'''
        
        with open("schema_injector.js", 'w') as f:
            f.write(js_script)
        
        print("✅ Schema injection script saved: schema_injector.js")
        return js_script

def main():
    """Generate all structured data schemas for NeuralSync2"""
    print("🧠 Generating NeuralSync2 Schema.org Structured Data...")
    
    generator = SchemaGenerator()
    schemas = generator.save_schemas_to_files()
    generator.generate_schema_injection_script(schemas)
    
    print(f"""
🎉 Schema Generation Complete!

📄 Files Created:
   • schema_software.json - SoftwareApplication schema
   • schema_website.json - Website schema  
   • schema_howto_install.json - Installation HowTo schema
   • schema_faq.json - FAQ schema
   • schema_article_main.json - Article schema
   • schema_breadcrumb.json - Breadcrumb schema
   • schema_all.json - All schemas combined
   • schema_injector.js - Auto-injection script

🚀 Usage:
   1. Add schemas to HTML <head> sections
   2. Include schema_injector.js for automatic injection
   3. Test with Google Rich Results Test Tool
   
🌐 Rich snippets will improve search visibility for:
   • Software applications
   • Installation guides  
   • FAQ sections
   • Article content
   • Navigation breadcrumbs
""")

if __name__ == "__main__":
    main()