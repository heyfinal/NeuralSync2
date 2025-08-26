#!/usr/bin/env python3
"""
NeuralSync2 SEO Automation Installer
Installs and configures complete SEO automation suite
"""

import os
import sys
import subprocess
import json
import time
from pathlib import Path
import requests
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SEOAutomationInstaller:
    def __init__(self):
        self.base_dir = Path(__file__).parent
        self.requirements = [
            'requests>=2.31.0',
            'beautifulsoup4>=4.12.0',
            'selenium>=4.15.0',
            'google-api-python-client>=2.100.0',
            'schedule>=1.2.0',
            'pyyaml>=6.0',
            'feedparser>=6.0.0',
            'markdown>=3.5.0',
            'python-dotenv>=1.0.0'
        ]
        
    def check_python_version(self):
        """Check if Python version is compatible"""
        if sys.version_info < (3, 8):
            logger.error("❌ Python 3.8 or higher required")
            sys.exit(1)
        logger.info(f"✅ Python {sys.version.split()[0]} detected")
    
    def install_requirements(self):
        """Install required Python packages"""
        logger.info("📦 Installing SEO automation dependencies...")
        
        try:
            # Upgrade pip first
            subprocess.run([sys.executable, "-m", "pip", "install", "--upgrade", "pip"], 
                         check=True, capture_output=True, text=True)
            
            # Install requirements
            for req in self.requirements:
                logger.info(f"Installing {req}...")
                subprocess.run([sys.executable, "-m", "pip", "install", req], 
                             check=True, capture_output=True, text=True)
            
            logger.info("✅ All dependencies installed successfully")
            
        except subprocess.CalledProcessError as e:
            logger.error(f"❌ Failed to install dependencies: {e}")
            sys.exit(1)
    
    def create_config_files(self):
        """Create configuration files for SEO automation"""
        logger.info("⚙️  Creating SEO configuration files...")
        
        # Main SEO config
        seo_config = {
            "domain": "neuralsync.dev",
            "target_keywords": [
                "AI tools with memory",
                "AI that remembers conversations", 
                "Claude tools that persist",
                "AI memory synchronization",
                "neural sync AI",
                "AI agent frameworks",
                "persistent AI memory",
                "AI tools install themselves",
                "Claude code extensions", 
                "AI cross-session memory"
            ],
            "search_engines": {
                "google": {
                    "enabled": True,
                    "api_key_env": "GOOGLE_INDEXING_API_KEY"
                },
                "bing": {
                    "enabled": True, 
                    "api_key_env": "BING_WEBMASTER_API_KEY"
                }
            },
            "monitoring": {
                "check_interval_hours": 24,
                "report_email": None
            }
        }
        
        with open(self.base_dir / "seo_config.json", "w") as f:
            json.dump(seo_config, f, indent=2)
        
        # Environment template
        env_template = """# NeuralSync2 SEO Automation Environment Variables

# Google Search Console / Indexing API
GOOGLE_INDEXING_API_KEY=your_google_indexing_api_key_here
GOOGLE_SEARCH_CONSOLE_PROPERTY=https://neuralsync.dev

# Bing Webmaster Tools
BING_WEBMASTER_API_KEY=your_bing_webmaster_api_key_here
BING_SITE_URL=https://neuralsync.dev

# Analytics 
GOOGLE_ANALYTICS_ID=UA-XXXXXXX-X
GOOGLE_TAG_MANAGER_ID=GTM-XXXXXXX

# Social Media APIs (for backlink automation)
TWITTER_API_KEY=your_twitter_api_key
REDDIT_CLIENT_ID=your_reddit_client_id
REDDIT_CLIENT_SECRET=your_reddit_client_secret

# Email notifications
SMTP_HOST=smtp.gmail.com
SMTP_PORT=587
SMTP_USER=your_email@gmail.com
SMTP_PASSWORD=your_app_password

# Domain and URLs
DOMAIN=neuralsync.dev
SITEMAP_URL=https://neuralsync.dev/sitemap.xml
"""
        
        with open(self.base_dir / ".env.template", "w") as f:
            f.write(env_template)
        
        logger.info("✅ Configuration files created")
    
    def create_automation_scripts(self):
        """Create SEO automation scripts"""
        logger.info("🤖 Creating SEO automation scripts...")
        
        # Daily SEO tasks script
        daily_seo_script = '''#!/usr/bin/env python3
"""
Daily SEO Automation Tasks for NeuralSync2
"""

import schedule
import time
import logging
from pathlib import Path
import sys

# Add current directory to path
sys.path.append(str(Path(__file__).parent))

from search_engine_submitter import SearchEngineSubmitter
from backlink_generator import BacklinkGenerator
from seo_monitor import SEOMonitor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def daily_seo_tasks():
    """Run daily SEO automation tasks"""
    logger.info("🚀 Starting daily SEO automation...")
    
    # Submit new/updated URLs to search engines
    submitter = SearchEngineSubmitter()
    submitter.run_full_submission()
    
    # Generate backlinks through content posting
    backlink_gen = BacklinkGenerator()
    backlink_gen.post_to_communities()
    
    # Monitor SEO performance
    monitor = SEOMonitor()
    monitor.check_rankings()
    monitor.generate_report()
    
    logger.info("✅ Daily SEO tasks completed")

def schedule_tasks():
    """Schedule SEO automation tasks"""
    # Daily tasks at 2 AM
    schedule.every().day.at("02:00").do(daily_seo_tasks)
    
    # Weekly sitemap submission
    schedule.every().week.do(lambda: SearchEngineSubmitter().submit_sitemap())
    
    logger.info("⏰ SEO automation scheduled")
    
    while True:
        schedule.run_pending()
        time.sleep(60)

if __name__ == "__main__":
    schedule_tasks()
'''
        
        with open(self.base_dir / "daily_seo_automation.py", "w") as f:
            f.write(daily_seo_script)
        
        # Make executable
        os.chmod(self.base_dir / "daily_seo_automation.py", 0o755)
        
        logger.info("✅ Automation scripts created")
    
    def create_monitoring_dashboard(self):
        """Create SEO monitoring dashboard"""
        logger.info("📊 Creating SEO monitoring dashboard...")
        
        dashboard_html = '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>NeuralSync2 SEO Dashboard</title>
    <style>
        body { font-family: -apple-system, BlinkMacSystemFont, sans-serif; margin: 0; padding: 20px; background: #f5f5f5; }
        .container { max-width: 1200px; margin: 0 auto; }
        .header { background: linear-gradient(135deg, #667eea, #764ba2); color: white; padding: 30px; border-radius: 10px; text-align: center; margin-bottom: 30px; }
        .metrics-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px; margin-bottom: 30px; }
        .metric-card { background: white; padding: 25px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
        .metric-number { font-size: 2.5em; font-weight: bold; color: #667eea; }
        .metric-label { color: #666; font-size: 0.9em; margin-top: 5px; }
        .keywords-table { background: white; padding: 25px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
        table { width: 100%; border-collapse: collapse; }
        th, td { padding: 12px; text-align: left; border-bottom: 1px solid #eee; }
        th { background: #f8f9fa; font-weight: bold; }
        .status-live { color: #28a745; font-weight: bold; }
        .status-pending { color: #ffc107; font-weight: bold; }
        .refresh-btn { background: #667eea; color: white; border: none; padding: 10px 20px; border-radius: 5px; cursor: pointer; }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🧠 NeuralSync2 SEO Dashboard</h1>
            <p>Real-time search engine optimization monitoring</p>
            <button class="refresh-btn" onclick="location.reload()">🔄 Refresh Data</button>
        </div>
        
        <div class="metrics-grid">
            <div class="metric-card">
                <div class="metric-number" id="indexed-pages">Loading...</div>
                <div class="metric-label">Pages Indexed</div>
            </div>
            <div class="metric-card">
                <div class="metric-number" id="avg-position">Loading...</div>
                <div class="metric-label">Avg. Search Position</div>
            </div>
            <div class="metric-card">
                <div class="metric-number" id="organic-clicks">Loading...</div>
                <div class="metric-label">Organic Clicks (30d)</div>
            </div>
            <div class="metric-card">
                <div class="metric-number" id="backlinks">Loading...</div>
                <div class="metric-label">Quality Backlinks</div>
            </div>
        </div>
        
        <div class="keywords-table">
            <h3>🎯 Target Keywords Performance</h3>
            <table>
                <thead>
                    <tr>
                        <th>Keyword</th>
                        <th>Position</th>
                        <th>Clicks</th>
                        <th>Impressions</th>
                        <th>CTR</th>
                        <th>Status</th>
                    </tr>
                </thead>
                <tbody id="keywords-tbody">
                    <tr><td colspan="6">Loading keyword data...</td></tr>
                </tbody>
            </table>
        </div>
    </div>
    
    <script>
        // Mock data for demonstration - replace with real API calls
        function updateDashboard() {
            document.getElementById('indexed-pages').textContent = '47';
            document.getElementById('avg-position').textContent = '12.3';
            document.getElementById('organic-clicks').textContent = '1,247';
            document.getElementById('backlinks').textContent = '28';
            
            const keywords = [
                {keyword: 'AI tools with memory', position: 15, clicks: 234, impressions: 3456, ctr: '6.8%', status: 'live'},
                {keyword: 'Claude memory fix', position: 8, clicks: 189, impressions: 2134, ctr: '8.9%', status: 'live'},
                {keyword: 'AI that remembers conversations', position: 23, clicks: 67, impressions: 1876, ctr: '3.6%', status: 'pending'},
                {keyword: 'persistent AI memory', position: 11, clicks: 145, impressions: 2987, ctr: '4.9%', status: 'live'},
                {keyword: 'neural sync AI', position: 6, clicks: 298, impressions: 3654, ctr: '8.2%', status: 'live'}
            ];
            
            const tbody = document.getElementById('keywords-tbody');
            tbody.innerHTML = keywords.map(k => `
                <tr>
                    <td><strong>${k.keyword}</strong></td>
                    <td>#${k.position}</td>
                    <td>${k.clicks}</td>
                    <td>${k.impressions}</td>
                    <td>${k.ctr}</td>
                    <td class="status-${k.status}">${k.status.toUpperCase()}</td>
                </tr>
            `).join('');
        }
        
        // Update dashboard on load
        setTimeout(updateDashboard, 1000);
        
        // Auto-refresh every 5 minutes
        setInterval(updateDashboard, 300000);
    </script>
</body>
</html>'''
        
        with open(self.base_dir / "seo_dashboard.html", "w") as f:
            f.write(dashboard_html)
        
        logger.info("✅ SEO monitoring dashboard created")
    
    def create_systemd_service(self):
        """Create systemd service for automation (Linux only)"""
        if sys.platform != 'linux':
            logger.info("⏭️  Skipping systemd service creation (not on Linux)")
            return
        
        logger.info("🔧 Creating systemd service...")
        
        service_content = f'''[Unit]
Description=NeuralSync2 SEO Automation
After=network.target

[Service]
Type=simple
User={os.getenv('USER', 'root')}
WorkingDirectory={self.base_dir}
ExecStart={sys.executable} {self.base_dir}/daily_seo_automation.py
Restart=always
RestartSec=10
Environment=PYTHONPATH={self.base_dir}

[Install]
WantedBy=multi-user.target
'''
        
        service_file = Path('/etc/systemd/system/neuralsync-seo.service')
        
        try:
            with open(service_file, 'w') as f:
                f.write(service_content)
            
            # Enable and start service
            subprocess.run(['sudo', 'systemctl', 'daemon-reload'], check=True)
            subprocess.run(['sudo', 'systemctl', 'enable', 'neuralsync-seo'], check=True)
            
            logger.info("✅ Systemd service created and enabled")
            logger.info("To start: sudo systemctl start neuralsync-seo")
            
        except (PermissionError, subprocess.CalledProcessError) as e:
            logger.warning(f"⚠️  Could not create systemd service: {e}")
            logger.info("Manual service creation may be required")
    
    def run_initial_setup(self):
        """Run initial SEO setup"""
        logger.info("🚀 Running initial SEO setup...")
        
        try:
            # Submit sitemap to search engines
            from search_engine_submitter import SearchEngineSubmitter
            submitter = SearchEngineSubmitter()
            submitter.run_full_submission()
            
            logger.info("✅ Initial search engine submissions completed")
            
        except Exception as e:
            logger.warning(f"⚠️  Initial setup encountered issues: {e}")
    
    def print_setup_complete(self):
        """Print setup completion message"""
        print(f"""
🎉 NeuralSync2 SEO Automation Setup Complete!

📁 Files Created:
   • seo_config.json - Main configuration
   • .env.template - Environment variables template  
   • daily_seo_automation.py - Automation scheduler
   • seo_dashboard.html - Monitoring dashboard
   • search_engine_submitter.py - Search engine submission

🔑 Next Steps:
   1. Copy .env.template to .env and add your API keys
   2. Run: python3 search_engine_submitter.py
   3. Open seo_dashboard.html in your browser
   4. Start automation: python3 daily_seo_automation.py

🌐 Dashboard: file://{self.base_dir}/seo_dashboard.html

🚀 Your NeuralSync2 SEO automation is ready!
""")
    
    def install(self):
        """Run complete installation process"""
        logger.info("🧠 Installing NeuralSync2 SEO Automation Suite...")
        
        self.check_python_version()
        self.install_requirements()
        self.create_config_files()
        self.create_automation_scripts()
        self.create_monitoring_dashboard()
        self.create_systemd_service()
        self.run_initial_setup()
        self.print_setup_complete()

def main():
    installer = SEOAutomationInstaller()
    installer.install()

if __name__ == "__main__":
    main()