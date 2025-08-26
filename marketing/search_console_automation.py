#!/usr/bin/env python3
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
