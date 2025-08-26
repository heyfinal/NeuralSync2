#!/usr/bin/env python3
"""
NeuralSync2 Search Engine Submission Automation
Submits URLs to major search engines for indexing
"""

import requests
import json
import time
import os
from urllib.parse import urlencode, urlparse
import logging
from typing import List, Dict, Any
import xml.etree.ElementTree as ET

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SearchEngineSubmitter:
    def __init__(self):
        self.base_domain = "neuralsync.dev"
        self.urls = [
            f"https://{self.base_domain}/",
            f"https://{self.base_domain}/ai-tools-with-memory",
            f"https://{self.base_domain}/claude-memory-fix", 
            f"https://{self.base_domain}/ai-remembers-conversations",
            f"https://{self.base_domain}/persistent-ai-memory",
            f"https://{self.base_domain}/ai-memory-synchronization",
            f"https://{self.base_domain}/neural-sync-ai",
            f"https://{self.base_domain}/ai-agent-frameworks",
            f"https://{self.base_domain}/ai-tools-install-themselves",
            f"https://{self.base_domain}/claude-code-extensions",
            f"https://{self.base_domain}/ai-cross-session-memory",
            f"https://github.com/heyfinal/NeuralSync2"
        ]
        
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'NeuralSync2-SEO-Bot/1.0 (+https://neuralsync.dev/)'
        })

    def submit_to_google_indexing_api(self, api_key: str, urls: List[str]) -> Dict[str, Any]:
        """Submit URLs to Google Indexing API"""
        results = {'success': [], 'failed': []}
        
        endpoint = "https://indexing.googleapis.com/v3/urlNotifications:publish"
        headers = {
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {api_key}'
        }
        
        for url in urls:
            try:
                data = {
                    "url": url,
                    "type": "URL_UPDATED"
                }
                
                response = self.session.post(endpoint, headers=headers, json=data)
                
                if response.status_code == 200:
                    results['success'].append(url)
                    logger.info(f"✅ Successfully submitted to Google: {url}")
                else:
                    results['failed'].append({'url': url, 'error': response.text})
                    logger.error(f"❌ Failed Google submission for {url}: {response.text}")
                
                time.sleep(1)  # Rate limiting
                
            except Exception as e:
                results['failed'].append({'url': url, 'error': str(e)})
                logger.error(f"❌ Error submitting {url} to Google: {e}")
        
        return results

    def submit_to_bing_webmaster(self, api_key: str, site_url: str, urls: List[str]) -> Dict[str, Any]:
        """Submit URLs to Bing Webmaster Tools API"""
        results = {'success': [], 'failed': []}
        
        endpoint = f"https://ssl.bing.com/webmaster/api.svc/json/SubmitUrl"
        
        for url in urls:
            try:
                params = {
                    'apikey': api_key,
                    'siteUrl': site_url,
                    'url': url
                }
                
                response = self.session.post(endpoint, params=params)
                
                if response.status_code == 200:
                    results['success'].append(url)
                    logger.info(f"✅ Successfully submitted to Bing: {url}")
                else:
                    results['failed'].append({'url': url, 'error': response.text})
                    logger.error(f"❌ Failed Bing submission for {url}: {response.text}")
                
                time.sleep(2)  # Bing has stricter rate limits
                
            except Exception as e:
                results['failed'].append({'url': url, 'error': str(e)})
                logger.error(f"❌ Error submitting {url} to Bing: {e}")
        
        return results

    def submit_to_generic_search_engines(self, urls: List[str]) -> Dict[str, Any]:
        """Submit URLs to search engines that accept direct submissions"""
        results = {'success': [], 'failed': []}
        
        # Directory submission services and search engines
        submission_endpoints = [
            # DuckDuckGo uses different crawling methods - no direct submission
            # Yahoo uses Bing's index now
            "https://www.google.com/ping?sitemap=https://neuralsync.dev/sitemap.xml",
            "https://www.bing.com/ping?sitemap=https://neuralsync.dev/sitemap.xml",
        ]
        
        for endpoint in submission_endpoints:
            try:
                response = self.session.get(endpoint, timeout=30)
                
                if response.status_code == 200:
                    results['success'].append(endpoint)
                    logger.info(f"✅ Successfully pinged sitemap: {endpoint}")
                else:
                    results['failed'].append({'url': endpoint, 'error': response.text})
                    logger.error(f"❌ Failed sitemap ping: {endpoint}")
                
                time.sleep(3)
                
            except Exception as e:
                results['failed'].append({'url': endpoint, 'error': str(e)})
                logger.error(f"❌ Error pinging sitemap {endpoint}: {e}")
        
        return results

    def submit_to_directories(self, urls: List[str]) -> Dict[str, Any]:
        """Submit to web directories and aggregators"""
        results = {'success': [], 'failed': []}
        
        # Directory APIs and submission forms (would need individual implementation)
        directory_services = [
            {
                'name': 'DMOZ Alternative - CurryGuide',
                'url': 'https://curryguide.com/submit',
                'method': 'GET'  # Just ping for now
            },
            {
                'name': 'Best of the Web',
                'url': 'https://botw.org/submit',
                'method': 'GET'
            },
            {
                'name': 'JoeAnt Directory',
                'url': 'https://joeant.com/submit',
                'method': 'GET'
            }
        ]
        
        for directory in directory_services:
            try:
                response = self.session.get(directory['url'], timeout=10)
                if response.status_code == 200:
                    results['success'].append(directory['name'])
                    logger.info(f"✅ Successfully reached directory: {directory['name']}")
                else:
                    results['failed'].append({'directory': directory['name'], 'error': f"HTTP {response.status_code}"})
                
                time.sleep(5)
                
            except Exception as e:
                results['failed'].append({'directory': directory['name'], 'error': str(e)})
                logger.error(f"❌ Error contacting directory {directory['name']}: {e}")
        
        return results

    def create_submission_report(self, results: Dict[str, Any]) -> str:
        """Create a detailed submission report"""
        report = f"""
# NeuralSync2 Search Engine Submission Report
Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}

## Summary
- Total URLs: {len(self.urls)}
- Submission Services Attempted: {len(results)}

## Results by Service:
"""
        
        for service, result in results.items():
            successful = len(result.get('success', []))
            failed = len(result.get('failed', []))
            
            report += f"""
### {service}
- ✅ Successful: {successful}
- ❌ Failed: {failed}
"""
            
            if result.get('success'):
                report += "**Successful submissions:**\n"
                for success in result['success']:
                    report += f"- {success}\n"
            
            if result.get('failed'):
                report += "**Failed submissions:**\n"
                for failure in result['failed']:
                    if isinstance(failure, dict):
                        report += f"- {failure.get('url', failure.get('directory', 'Unknown'))}: {failure.get('error', 'Unknown error')}\n"
                    else:
                        report += f"- {failure}\n"
        
        return report

    def run_full_submission(self, google_api_key: str = None, bing_api_key: str = None):
        """Run complete search engine submission process"""
        logger.info("🚀 Starting NeuralSync2 search engine submission process...")
        
        all_results = {}
        
        # Submit sitemap pings
        logger.info("📍 Submitting sitemap pings...")
        all_results['sitemap_pings'] = self.submit_to_generic_search_engines(self.urls)
        
        # Submit to Google if API key provided
        if google_api_key:
            logger.info("🔍 Submitting to Google Indexing API...")
            all_results['google'] = self.submit_to_google_indexing_api(google_api_key, self.urls)
        else:
            logger.warning("⚠️  No Google API key provided, skipping Google Indexing API")
        
        # Submit to Bing if API key provided  
        if bing_api_key:
            logger.info("🔍 Submitting to Bing Webmaster Tools...")
            all_results['bing'] = self.submit_to_bing_webmaster(bing_api_key, f"https://{self.base_domain}", self.urls)
        else:
            logger.warning("⚠️  No Bing API key provided, skipping Bing Webmaster API")
        
        # Submit to directories
        logger.info("📁 Submitting to web directories...")
        all_results['directories'] = self.submit_to_directories(self.urls)
        
        # Generate report
        report = self.create_submission_report(all_results)
        
        # Save report
        report_file = f"submission_report_{int(time.time())}.md"
        with open(report_file, 'w') as f:
            f.write(report)
        
        logger.info(f"📊 Submission complete! Report saved to: {report_file}")
        print(report)
        
        return all_results

def main():
    """Main execution function"""
    submitter = SearchEngineSubmitter()
    
    # Get API keys from environment variables (if available)
    google_api_key = os.getenv('GOOGLE_INDEXING_API_KEY')
    bing_api_key = os.getenv('BING_WEBMASTER_API_KEY')
    
    if not google_api_key and not bing_api_key:
        print("""
🔑 API Keys Not Found
To use Google Indexing API: export GOOGLE_INDEXING_API_KEY=your_key_here
To use Bing Webmaster API: export BING_WEBMASTER_API_KEY=your_key_here

Running with sitemap pings and directory submissions only...
""")
    
    # Run submission process
    results = submitter.run_full_submission(google_api_key, bing_api_key)
    
    return results

if __name__ == "__main__":
    main()