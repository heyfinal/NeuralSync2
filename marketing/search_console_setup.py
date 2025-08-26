#!/usr/bin/env python3
"""
Google Search Console and Bing Webmaster Tools Setup
Generates configuration files and verification assets
"""

import json
import os
from datetime import datetime
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SearchConsoleSetup:
    def __init__(self):
        self.domain = "neuralsync.dev"
        self.base_url = f"https://{self.domain}"
        self.setup_date = datetime.now().isoformat()
        
    def generate_google_search_console_config(self):
        """Generate Google Search Console configuration"""
        logger.info("🔍 Generating Google Search Console configuration...")
        
        # Service account configuration
        gsc_config = {
            "project_id": "neuralsync2-seo",
            "site_url": self.base_url,
            "verification_method": "html_file",
            "sitemap_urls": [
                f"{self.base_url}/sitemap.xml",
                f"{self.base_url}/sitemap_index.xml"
            ],
            "target_countries": ["US", "CA", "GB", "AU", "DE"],
            "target_languages": ["en"],
            "setup_date": self.setup_date
        }
        
        with open("google_search_console_config.json", 'w') as f:
            json.dump(gsc_config, f, indent=2)
        
        # HTML verification file template
        html_verification = """<!DOCTYPE html>
<html>
<head>
    <title>Google Search Console Verification</title>
    <meta name="google-site-verification" content="YOUR_VERIFICATION_CODE_HERE" />
</head>
<body>
    <h1>NeuralSync2 - Google Search Console Verification</h1>
    <p>This file verifies ownership of neuralsync.dev for Google Search Console.</p>
    <p>Replace YOUR_VERIFICATION_CODE_HERE with your actual verification code.</p>
</body>
</html>"""
        
        with open("google-site-verification.html", 'w') as f:
            f.write(html_verification)
        
        # Google Analytics integration
        ga_config = {
            "measurement_id": "G-XXXXXXXXXX",  # Replace with actual ID
            "property_id": "XXXXXXXXXX",
            "integration_date": self.setup_date,
            "events_tracking": {
                "page_views": True,
                "scroll_tracking": True,
                "outbound_clicks": True,
                "file_downloads": True,
                "github_clicks": True
            },
            "goals": [
                {
                    "name": "GitHub Repository Visit",
                    "type": "destination",
                    "url": "/github"
                },
                {
                    "name": "Installation Command Copy",
                    "type": "event",
                    "category": "engagement",
                    "action": "copy_install_command"
                },
                {
                    "name": "Documentation View",
                    "type": "destination",
                    "url": "/docs"
                }
            ]
        }
        
        with open("google_analytics_config.json", 'w') as f:
            json.dump(ga_config, f, indent=2)
        
        logger.info("✅ Google Search Console configuration generated")
        return gsc_config
    
    def generate_bing_webmaster_config(self):
        """Generate Bing Webmaster Tools configuration"""
        logger.info("🔍 Generating Bing Webmaster Tools configuration...")
        
        bing_config = {
            "site_url": self.base_url,
            "verification_method": "xml_file",
            "api_key": "YOUR_BING_WEBMASTER_API_KEY",
            "sitemap_urls": [
                f"{self.base_url}/sitemap.xml"
            ],
            "crawl_settings": {
                "crawl_rate": "normal",
                "preferred_crawl_time": "02:00-06:00"
            },
            "geo_targeting": {
                "primary_country": "US",
                "additional_countries": ["CA", "GB", "AU"]
            },
            "setup_date": self.setup_date
        }
        
        with open("bing_webmaster_config.json", 'w') as f:
            json.dump(bing_config, f, indent=2)
        
        # XML verification file template
        xml_verification = '''<?xml version="1.0"?>
<users>
    <user>YOUR_BING_VERIFICATION_CODE_HERE</user>
</users>'''
        
        with open("BingSiteAuth.xml", 'w') as f:
            f.write(xml_verification)
        
        logger.info("✅ Bing Webmaster Tools configuration generated")
        return bing_config
    
    def generate_yandex_webmaster_config(self):
        """Generate Yandex Webmaster configuration (for global reach)"""
        logger.info("🔍 Generating Yandex Webmaster configuration...")
        
        yandex_config = {
            "site_url": self.base_url,
            "verification_method": "html_file",
            "sitemap_urls": [f"{self.base_url}/sitemap.xml"],
            "geo_targeting": "worldwide",
            "setup_date": self.setup_date
        }
        
        with open("yandex_webmaster_config.json", 'w') as f:
            json.dump(yandex_config, f, indent=2)
        
        # HTML verification file
        yandex_verification = """<!DOCTYPE html>
<html>
<head>
    <title>Yandex Webmaster Verification</title>
    <meta name="yandex-verification" content="YOUR_YANDEX_VERIFICATION_CODE_HERE" />
</head>
<body>
    <h1>NeuralSync2 - Yandex Webmaster Verification</h1>
    <p>This file verifies ownership for Yandex Webmaster.</p>
</body>
</html>"""
        
        with open("yandex-verification.html", 'w') as f:
            f.write(yandex_verification)
        
        logger.info("✅ Yandex Webmaster configuration generated")
        return yandex_config
    
    def create_search_console_automation_script(self):
        """Create automation script for search console management"""
        logger.info("🤖 Creating search console automation script...")
        
        automation_script = '''#!/usr/bin/env python3
"""
NeuralSync2 Search Console Automation
Automates sitemap submission and performance monitoring
"""

import requests
import json
import time
from google.oauth2.service_account import Credentials
from googleapiclient.discovery import build
import os

class SearchConsoleAutomation:
    def __init__(self):
        self.domain = "neuralsync.dev"
        self.base_url = f"https://{self.domain}"
        
        # Initialize Google Search Console API
        self.gsc_service = None
        self.setup_google_api()
    
    def setup_google_api(self):
        """Setup Google Search Console API client"""
        try:
            # Load service account credentials
            creds_file = "google_search_console_credentials.json"
            if os.path.exists(creds_file):
                credentials = Credentials.from_service_account_file(
                    creds_file,
                    scopes=['https://www.googleapis.com/auth/webmasters']
                )
                self.gsc_service = build('searchconsole', 'v1', credentials=credentials)
                print("✅ Google Search Console API initialized")
            else:
                print("⚠️  Google Search Console credentials not found")
        except Exception as e:
            print(f"❌ Error setting up Google API: {e}")
    
    def submit_sitemap_to_google(self):
        """Submit sitemap to Google Search Console"""
        if not self.gsc_service:
            print("❌ Google Search Console not configured")
            return
        
        try:
            sitemap_url = f"{self.base_url}/sitemap.xml"
            
            request = self.gsc_service.sitemaps().submit(
                siteUrl=self.base_url,
                feedpath=sitemap_url
            )
            
            request.execute()
            print(f"✅ Sitemap submitted to Google: {sitemap_url}")
            
        except Exception as e:
            print(f"❌ Error submitting sitemap to Google: {e}")
    
    def submit_sitemap_to_bing(self):
        """Submit sitemap to Bing Webmaster Tools"""
        api_key = os.getenv('BING_WEBMASTER_API_KEY')
        if not api_key:
            print("⚠️  Bing Webmaster API key not found")
            return
        
        try:
            url = "https://ssl.bing.com/webmaster/api.svc/json/SubmitUrl"
            params = {
                'apikey': api_key,
                'siteUrl': self.base_url,
                'url': f"{self.base_url}/sitemap.xml"
            }
            
            response = requests.post(url, params=params)
            
            if response.status_code == 200:
                print("✅ Sitemap submitted to Bing")
            else:
                print(f"❌ Bing submission failed: {response.status_code}")
                
        except Exception as e:
            print(f"❌ Error submitting to Bing: {e}")
    
    def get_search_performance(self, days=30):
        """Get search performance data from Google Search Console"""
        if not self.gsc_service:
            return None
        
        try:
            from datetime import datetime, timedelta
            
            end_date = datetime.now().date()
            start_date = end_date - timedelta(days=days)
            
            request = {
                'startDate': str(start_date),
                'endDate': str(end_date),
                'dimensions': ['query', 'page'],
                'rowLimit': 100
            }
            
            response = self.gsc_service.searchanalytics().query(
                siteUrl=self.base_url,
                body=request
            ).execute()
            
            rows = response.get('rows', [])
            print(f"✅ Retrieved {len(rows)} search performance records")
            return rows
            
        except Exception as e:
            print(f"❌ Error getting search performance: {e}")
            return None
    
    def monitor_indexing_status(self):
        """Monitor URL indexing status"""
        if not self.gsc_service:
            return
        
        important_urls = [
            f"{self.base_url}/",
            f"{self.base_url}/ai-tools-with-memory",
            f"{self.base_url}/claude-memory-fix",
            f"{self.base_url}/persistent-ai-memory"
        ]
        
        for url in important_urls:
            try:
                request = self.gsc_service.urlInspection().index().inspect(
                    body={'inspectionUrl': url, 'siteUrl': self.base_url}
                )
                
                response = request.execute()
                
                index_status = response.get('indexStatusResult', {})
                coverage_state = index_status.get('coverageState', 'Unknown')
                
                print(f"📊 {url}: {coverage_state}")
                time.sleep(1)  # Rate limiting
                
            except Exception as e:
                print(f"❌ Error checking {url}: {e}")
    
    def run_daily_tasks(self):
        """Run daily search console maintenance tasks"""
        print("🚀 Running daily search console tasks...")
        
        self.submit_sitemap_to_google()
        self.submit_sitemap_to_bing()
        
        performance_data = self.get_search_performance()
        if performance_data:
            # Save performance data
            with open(f"search_performance_{int(time.time())}.json", 'w') as f:
                json.dump(performance_data, f, indent=2, default=str)
        
        self.monitor_indexing_status()
        
        print("✅ Daily search console tasks completed")

if __name__ == "__main__":
    automation = SearchConsoleAutomation()
    automation.run_daily_tasks()
'''
        
        with open("search_console_automation.py", 'w') as f:
            f.write(automation_script)
        
        os.chmod("search_console_automation.py", 0o755)
        logger.info("✅ Search console automation script created")
    
    def create_analytics_tracking_code(self):
        """Create analytics tracking code for websites"""
        logger.info("📊 Creating analytics tracking code...")
        
        tracking_code = '''<!-- Google Analytics 4 -->
<script async src="https://www.googletagmanager.com/gtag/js?id=G-XXXXXXXXXX"></script>
<script>
  window.dataLayer = window.dataLayer || [];
  function gtag(){dataLayer.push(arguments);}
  gtag('js', new Date());
  gtag('config', 'G-XXXXXXXXXX', {
    page_title: document.title,
    page_location: window.location.href,
    custom_map: {
      'custom_parameter': 'neuralsync_version'
    }
  });

  // Custom events for NeuralSync2
  
  // Track installation command copies
  document.addEventListener('copy', function(e) {
    const selectedText = window.getSelection().toString();
    if (selectedText.includes('install') && selectedText.includes('neuralsync')) {
      gtag('event', 'copy_install_command', {
        'event_category': 'engagement',
        'event_label': 'installation',
        'value': 1
      });
    }
  });
  
  // Track GitHub repository visits
  document.querySelectorAll('a[href*="github.com/heyfinal/NeuralSync2"]').forEach(link => {
    link.addEventListener('click', function() {
      gtag('event', 'github_visit', {
        'event_category': 'outbound',
        'event_label': 'repository',
        'value': 1
      });
    });
  });
  
  // Track scroll depth
  let scrollDepth = 0;
  const maxScroll = document.body.scrollHeight - window.innerHeight;
  
  window.addEventListener('scroll', function() {
    const currentScroll = window.scrollY;
    const currentDepth = Math.round((currentScroll / maxScroll) * 100);
    
    if (currentDepth > scrollDepth && currentDepth % 25 === 0) {
      scrollDepth = currentDepth;
      gtag('event', 'scroll', {
        'event_category': 'engagement',
        'event_label': `${scrollDepth}%`,
        'value': scrollDepth
      });
    }
  });
  
  // Track feature interest
  document.querySelectorAll('.feature, .feature-card').forEach(feature => {
    feature.addEventListener('click', function() {
      const featureName = this.querySelector('h3, h4')?.textContent || 'unknown';
      gtag('event', 'feature_interest', {
        'event_category': 'engagement',
        'event_label': featureName.toLowerCase(),
        'value': 1
      });
    });
  });
</script>

<!-- Microsoft Clarity -->
<script type="text/javascript">
    (function(c,l,a,r,i,t,y){
        c[a]=c[a]||function(){(c[a].q=c[a].q||[]).push(arguments)};
        t=l.createElement(r);t.async=1;t.src="https://www.clarity.ms/tag/"+i;
        y=l.getElementsByTagName(r)[0];y.parentNode.insertBefore(t,y);
    })(window, document, "clarity", "script", "YOUR_CLARITY_PROJECT_ID");
</script>

<!-- Hotjar Tracking Code -->
<script>
    (function(h,o,t,j,a,r){
        h.hj=h.hj||function(){(h.hj.q=h.hj.q||[]).push(arguments)};
        h._hjSettings={hjid:YOUR_HOTJAR_ID,hjsv:6};
        a=o.getElementsByTagName('head')[0];
        r=o.createElement('script');r.async=1;
        r.src=t+h._hjSettings.hjid+j+h._hjSettings.hjsv;
        a.appendChild(r);
    })(window,document,'https://static.hotjar.com/c/hotjar-','.js?sv=');
</script>'''
        
        with open("analytics_tracking_code.html", 'w') as f:
            f.write(tracking_code)
        
        logger.info("✅ Analytics tracking code created")
    
    def create_search_console_documentation(self):
        """Create setup documentation"""
        logger.info("📚 Creating search console documentation...")
        
        documentation = f'''# Search Console Setup Guide for NeuralSync2

## Overview
This guide covers setup for Google Search Console, Bing Webmaster Tools, and other search engines for optimal SEO monitoring and indexing.

## Google Search Console Setup

### 1. Account Setup
1. Visit [Google Search Console](https://search.google.com/search-console)
2. Add property: `{self.base_url}`
3. Choose verification method: HTML file upload

### 2. Verification
1. Upload `google-site-verification.html` to your website root
2. Replace `YOUR_VERIFICATION_CODE_HERE` with actual verification code
3. Click "Verify" in Google Search Console

### 3. Sitemap Submission
1. Go to Sitemaps section in GSC
2. Submit: `{self.base_url}/sitemap.xml`
3. Monitor indexing status regularly

### 4. API Setup (Optional)
1. Create Google Cloud Project: `neuralsync2-seo`
2. Enable Search Console API
3. Create service account credentials
4. Download JSON credentials as `google_search_console_credentials.json`

## Bing Webmaster Tools Setup

### 1. Account Setup
1. Visit [Bing Webmaster Tools](https://www.bing.com/webmasters)
2. Add site: `{self.base_url}`
3. Choose XML file verification

### 2. Verification
1. Upload `BingSiteAuth.xml` to website root
2. Replace `YOUR_BING_VERIFICATION_CODE_HERE` with actual code
3. Verify in Bing Webmaster Tools

### 3. API Access
1. Generate API key in Bing Webmaster Tools
2. Set environment variable: `BING_WEBMASTER_API_KEY=your_key_here`

## Analytics Integration

### Google Analytics 4
1. Create GA4 property
2. Get Measurement ID (G-XXXXXXXXXX)
3. Update tracking code in `analytics_tracking_code.html`
4. Install tracking code on all pages

### Microsoft Clarity
1. Sign up at [Microsoft Clarity](https://clarity.microsoft.com/)
2. Create project for NeuralSync2
3. Update `YOUR_CLARITY_PROJECT_ID` in tracking code

### Hotjar (Optional)
1. Create Hotjar account
2. Get site ID
3. Update `YOUR_HOTJAR_ID` in tracking code

## Automation Setup

### Daily Tasks
The `search_console_automation.py` script handles:
- Sitemap submissions to Google and Bing
- Performance data collection
- Indexing status monitoring
- Error reporting

### Scheduled Execution
Add to crontab for daily execution:
```bash
0 2 * * * /usr/bin/python3 /path/to/search_console_automation.py
```

## Monitoring Checklist

### Weekly Tasks
- [ ] Check Google Search Console for crawl errors
- [ ] Monitor search performance trends  
- [ ] Review top-performing keywords
- [ ] Check mobile usability issues

### Monthly Tasks
- [ ] Analyze search traffic patterns
- [ ] Update target keywords based on performance
- [ ] Review and optimize underperforming pages
- [ ] Submit new content to search engines

## Key Metrics to Track

### Search Performance
- **Impressions**: How often pages appear in search
- **Clicks**: Actual visits from search results
- **CTR**: Click-through rate (clicks/impressions)
- **Average Position**: Average ranking in search results

### Technical Health
- **Coverage**: Pages successfully indexed
- **Core Web Vitals**: Page experience metrics
- **Mobile Usability**: Mobile-friendly status
- **Security Issues**: HTTPS and security problems

## Troubleshooting

### Common Issues
1. **Verification Failed**: Check file upload and code accuracy
2. **Sitemap Not Found**: Verify sitemap URL is accessible
3. **Indexing Issues**: Check robots.txt for blocking rules
4. **API Errors**: Verify credentials and permissions

### Support Resources
- Google Search Console Help: https://support.google.com/webmasters
- Bing Webmaster Help: https://www.bing.com/webmasters/help
- NeuralSync2 Repository: https://github.com/heyfinal/NeuralSync2

## Files Generated
- `google_search_console_config.json` - GSC configuration
- `google-site-verification.html` - Google verification file
- `bing_webmaster_config.json` - Bing configuration  
- `BingSiteAuth.xml` - Bing verification file
- `analytics_tracking_code.html` - Complete tracking code
- `search_console_automation.py` - Automation script

---
Generated: {self.setup_date}
Domain: {self.domain}
Base URL: {self.base_url}
'''
        
        with open("search_console_setup_guide.md", 'w') as f:
            f.write(documentation)
        
        logger.info("✅ Search console documentation created")
    
    def run_complete_setup(self):
        """Run complete search console setup"""
        logger.info("🚀 Running complete search console setup for NeuralSync2...")
        
        # Generate all configuration files
        self.generate_google_search_console_config()
        self.generate_bing_webmaster_config()
        self.generate_yandex_webmaster_config()
        
        # Create automation and tracking
        self.create_search_console_automation_script()
        self.create_analytics_tracking_code()
        self.create_search_console_documentation()
        
        logger.info("✅ Search console setup complete!")
        
        print(f"""
🎉 Search Console Setup Complete!

📁 Files Generated:
   • google_search_console_config.json - GSC configuration
   • google-site-verification.html - Google verification file  
   • bing_webmaster_config.json - Bing configuration
   • BingSiteAuth.xml - Bing verification file
   • yandex_webmaster_config.json - Yandex configuration
   • yandex-verification.html - Yandex verification file
   • analytics_tracking_code.html - Complete tracking code
   • search_console_automation.py - Daily automation script
   • search_console_setup_guide.md - Complete setup guide

🔑 Next Steps:
   1. Upload verification files to website root
   2. Add properties in Google Search Console and Bing Webmaster Tools
   3. Replace placeholder codes with actual verification codes
   4. Install analytics tracking code on all pages
   5. Run automation script: python3 search_console_automation.py

🌐 Domain: {self.domain}
📊 Ready for search engine optimization monitoring!
""")

def main():
    setup = SearchConsoleSetup()
    setup.run_complete_setup()

if __name__ == "__main__":
    main()