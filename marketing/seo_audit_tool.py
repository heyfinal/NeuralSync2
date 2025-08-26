#!/usr/bin/env python3
"""
NeuralSync2 Technical SEO Audit and Implementation Tool
Comprehensive SEO analysis and automated optimization
"""

import requests
import json
import time
from urllib.parse import urljoin, urlparse
from bs4 import BeautifulSoup
import re
from typing import Dict, List, Any, Optional
import logging
from pathlib import Path
import xml.etree.ElementTree as ET

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SEOAuditTool:
    def __init__(self, base_url: str = "https://neuralsync.dev"):
        self.base_url = base_url.rstrip('/')
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'NeuralSync2-SEO-Audit/1.0 (+https://neuralsync.dev/)'
        })
        self.audit_results = {}
        self.recommendations = []
        
    def audit_page_speed(self, url: str) -> Dict[str, Any]:
        """Audit page loading speed and performance"""
        logger.info(f"🚀 Auditing page speed for {url}")
        
        try:
            start_time = time.time()
            response = self.session.get(url, timeout=30)
            load_time = time.time() - start_time
            
            results = {
                'url': url,
                'load_time_seconds': round(load_time, 3),
                'status_code': response.status_code,
                'content_size_kb': round(len(response.content) / 1024, 2),
                'response_headers': dict(response.headers)
            }
            
            # Performance scoring
            if load_time < 1.0:
                results['speed_score'] = 'Excellent'
                results['speed_rating'] = 5
            elif load_time < 2.0:
                results['speed_score'] = 'Good'
                results['speed_rating'] = 4
            elif load_time < 3.0:
                results['speed_score'] = 'Average'
                results['speed_rating'] = 3
            elif load_time < 5.0:
                results['speed_score'] = 'Poor'
                results['speed_rating'] = 2
            else:
                results['speed_score'] = 'Critical'
                results['speed_rating'] = 1
            
            return results
            
        except Exception as e:
            logger.error(f"❌ Error auditing page speed: {e}")
            return {'url': url, 'error': str(e)}
    
    def audit_meta_tags(self, url: str) -> Dict[str, Any]:
        """Audit HTML meta tags and SEO elements"""
        logger.info(f"🔍 Auditing meta tags for {url}")
        
        try:
            response = self.session.get(url, timeout=30)
            soup = BeautifulSoup(response.content, 'html.parser')
            
            results = {
                'url': url,
                'title': None,
                'meta_description': None,
                'meta_keywords': None,
                'canonical_url': None,
                'og_tags': {},
                'twitter_tags': {},
                'schema_markup': [],
                'h1_tags': [],
                'h2_tags': [],
                'images_without_alt': 0,
                'internal_links': 0,
                'external_links': 0
            }
            
            # Title tag
            title_tag = soup.find('title')
            if title_tag:
                results['title'] = title_tag.get_text().strip()
                results['title_length'] = len(results['title'])
            
            # Meta description
            meta_desc = soup.find('meta', attrs={'name': 'description'})
            if meta_desc:
                results['meta_description'] = meta_desc.get('content', '').strip()
                results['meta_description_length'] = len(results['meta_description'])
            
            # Meta keywords
            meta_keywords = soup.find('meta', attrs={'name': 'keywords'})
            if meta_keywords:
                results['meta_keywords'] = meta_keywords.get('content', '').strip()
            
            # Canonical URL
            canonical = soup.find('link', attrs={'rel': 'canonical'})
            if canonical:
                results['canonical_url'] = canonical.get('href')
            
            # Open Graph tags
            og_tags = soup.find_all('meta', attrs={'property': re.compile(r'^og:')})
            for og_tag in og_tags:
                prop = og_tag.get('property')
                content = og_tag.get('content')
                if prop and content:
                    results['og_tags'][prop] = content
            
            # Twitter tags
            twitter_tags = soup.find_all('meta', attrs={'name': re.compile(r'^twitter:')})
            for twitter_tag in twitter_tags:
                name = twitter_tag.get('name')
                content = twitter_tag.get('content')
                if name and content:
                    results['twitter_tags'][name] = content
            
            # Schema markup
            schema_scripts = soup.find_all('script', attrs={'type': 'application/ld+json'})
            for script in schema_scripts:
                try:
                    schema_data = json.loads(script.string)
                    results['schema_markup'].append(schema_data)
                except:
                    pass
            
            # Heading tags
            h1_tags = soup.find_all('h1')
            results['h1_tags'] = [tag.get_text().strip() for tag in h1_tags]
            
            h2_tags = soup.find_all('h2')
            results['h2_tags'] = [tag.get_text().strip() for tag in h2_tags]
            
            # Images without alt text
            images = soup.find_all('img')
            for img in images:
                if not img.get('alt'):
                    results['images_without_alt'] += 1
            
            # Links analysis
            links = soup.find_all('a', href=True)
            for link in links:
                href = link['href']
                if href.startswith(self.base_url) or href.startswith('/'):
                    results['internal_links'] += 1
                elif href.startswith('http'):
                    results['external_links'] += 1
            
            return results
            
        except Exception as e:
            logger.error(f"❌ Error auditing meta tags: {e}")
            return {'url': url, 'error': str(e)}
    
    def audit_technical_seo(self, url: str) -> Dict[str, Any]:
        """Audit technical SEO factors"""
        logger.info(f"⚙️ Auditing technical SEO for {url}")
        
        try:
            response = self.session.get(url, timeout=30)
            
            results = {
                'url': url,
                'https_enabled': url.startswith('https://'),
                'status_code': response.status_code,
                'redirects': len(response.history),
                'compression': 'gzip' in response.headers.get('content-encoding', ''),
                'caching_headers': {},
                'security_headers': {},
                'mobile_friendly': None,
                'structured_data_valid': None
            }
            
            # Check caching headers
            cache_headers = ['cache-control', 'expires', 'etag', 'last-modified']
            for header in cache_headers:
                if header in response.headers:
                    results['caching_headers'][header] = response.headers[header]
            
            # Check security headers
            security_headers = ['strict-transport-security', 'x-frame-options', 'x-content-type-options', 'x-xss-protection']
            for header in security_headers:
                if header in response.headers:
                    results['security_headers'][header] = response.headers[header]
            
            # Check for mobile viewport meta tag
            soup = BeautifulSoup(response.content, 'html.parser')
            viewport_meta = soup.find('meta', attrs={'name': 'viewport'})
            if viewport_meta:
                viewport_content = viewport_meta.get('content', '')
                results['mobile_friendly'] = 'width=device-width' in viewport_content
            else:
                results['mobile_friendly'] = False
            
            return results
            
        except Exception as e:
            logger.error(f"❌ Error auditing technical SEO: {e}")
            return {'url': url, 'error': str(e)}
    
    def check_robots_txt(self) -> Dict[str, Any]:
        """Check robots.txt configuration"""
        logger.info("🤖 Checking robots.txt")
        
        robots_url = f"{self.base_url}/robots.txt"
        
        try:
            response = self.session.get(robots_url, timeout=10)
            
            results = {
                'url': robots_url,
                'exists': response.status_code == 200,
                'content': response.text if response.status_code == 200 else None,
                'allows_crawling': True,
                'sitemap_declared': False,
                'issues': []
            }
            
            if response.status_code == 200:
                content = response.text.lower()
                
                # Check for sitemap declaration
                if 'sitemap:' in content:
                    results['sitemap_declared'] = True
                
                # Check for overly restrictive rules
                if 'disallow: /' in content and 'user-agent: *' in content:
                    results['allows_crawling'] = False
                    results['issues'].append('Robots.txt may be blocking all crawlers')
            else:
                results['issues'].append('Robots.txt not found - should be created')
            
            return results
            
        except Exception as e:
            logger.error(f"❌ Error checking robots.txt: {e}")
            return {'url': robots_url, 'error': str(e)}
    
    def check_sitemap_xml(self) -> Dict[str, Any]:
        """Check XML sitemap configuration"""
        logger.info("🗺️ Checking sitemap.xml")
        
        sitemap_url = f"{self.base_url}/sitemap.xml"
        
        try:
            response = self.session.get(sitemap_url, timeout=10)
            
            results = {
                'url': sitemap_url,
                'exists': response.status_code == 200,
                'valid_xml': False,
                'url_count': 0,
                'last_modified': None,
                'issues': []
            }
            
            if response.status_code == 200:
                try:
                    # Parse XML
                    root = ET.fromstring(response.content)
                    results['valid_xml'] = True
                    
                    # Count URLs
                    url_elements = root.findall('.//{http://www.sitemaps.org/schemas/sitemap/0.9}url')
                    results['url_count'] = len(url_elements)
                    
                    # Check for recent updates
                    lastmod_elements = root.findall('.//{http://www.sitemaps.org/schemas/sitemap/0.9}lastmod')
                    if lastmod_elements:
                        results['last_modified'] = lastmod_elements[0].text
                    
                    if results['url_count'] == 0:
                        results['issues'].append('Sitemap contains no URLs')
                        
                except ET.ParseError as e:
                    results['issues'].append(f'Invalid XML format: {e}')
            else:
                results['issues'].append('Sitemap.xml not found - should be created')
            
            return results
            
        except Exception as e:
            logger.error(f"❌ Error checking sitemap.xml: {e}")
            return {'url': sitemap_url, 'error': str(e)}
    
    def audit_keyword_optimization(self, url: str, target_keywords: List[str]) -> Dict[str, Any]:
        """Audit keyword optimization for target keywords"""
        logger.info(f"🎯 Auditing keyword optimization for {url}")
        
        try:
            response = self.session.get(url, timeout=30)
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Get page text content
            page_text = soup.get_text().lower()
            title_text = soup.find('title').get_text().lower() if soup.find('title') else ''
            meta_desc = soup.find('meta', attrs={'name': 'description'})
            meta_desc_text = meta_desc.get('content', '').lower() if meta_desc else ''
            
            results = {
                'url': url,
                'keyword_analysis': {},
                'keyword_density': {},
                'title_optimization': {},
                'meta_description_optimization': {},
                'heading_optimization': {},
                'overall_score': 0
            }
            
            h1_text = ' '.join([h1.get_text().lower() for h1 in soup.find_all('h1')])
            h2_text = ' '.join([h2.get_text().lower() for h2 in soup.find_all('h2')])
            
            total_words = len(page_text.split())
            
            for keyword in target_keywords:
                keyword_lower = keyword.lower()
                
                # Count keyword occurrences
                keyword_count = page_text.count(keyword_lower)
                density = (keyword_count / total_words * 100) if total_words > 0 else 0
                
                keyword_analysis = {
                    'keyword': keyword,
                    'count': keyword_count,
                    'density_percent': round(density, 2),
                    'in_title': keyword_lower in title_text,
                    'in_meta_description': keyword_lower in meta_desc_text,
                    'in_h1': keyword_lower in h1_text,
                    'in_h2': keyword_lower in h2_text,
                    'score': 0
                }
                
                # Calculate keyword score
                score = 0
                if keyword_analysis['in_title']:
                    score += 25
                if keyword_analysis['in_meta_description']:
                    score += 20
                if keyword_analysis['in_h1']:
                    score += 20
                if keyword_analysis['in_h2']:
                    score += 10
                if 0.5 <= density <= 3.0:  # Optimal keyword density
                    score += 15
                elif density > 0:
                    score += 5
                if keyword_count >= 3:
                    score += 10
                
                keyword_analysis['score'] = score
                results['keyword_analysis'][keyword] = keyword_analysis
            
            # Calculate overall score
            if results['keyword_analysis']:
                results['overall_score'] = round(
                    sum(kw['score'] for kw in results['keyword_analysis'].values()) / 
                    len(results['keyword_analysis'])
                )
            
            return results
            
        except Exception as e:
            logger.error(f"❌ Error auditing keyword optimization: {e}")
            return {'url': url, 'error': str(e)}
    
    def generate_recommendations(self, audit_results: Dict[str, Any]) -> List[str]:
        """Generate SEO improvement recommendations based on audit results"""
        recommendations = []
        
        # Page speed recommendations
        if 'page_speed' in audit_results:
            speed_data = audit_results['page_speed']
            if isinstance(speed_data, dict) and speed_data.get('speed_rating', 5) < 4:
                recommendations.append(f"🚀 CRITICAL: Improve page speed (current: {speed_data.get('load_time_seconds', 'N/A')}s)")
                recommendations.append("   • Enable GZIP compression")
                recommendations.append("   • Optimize images and assets")
                recommendations.append("   • Use CDN for static content")
        
        # Meta tags recommendations
        if 'meta_tags' in audit_results:
            meta_data = audit_results['meta_tags']
            if isinstance(meta_data, dict):
                if not meta_data.get('title'):
                    recommendations.append("📝 CRITICAL: Add HTML title tag")
                elif meta_data.get('title_length', 0) > 60:
                    recommendations.append("📝 WARNING: Title tag too long (>60 chars)")
                elif meta_data.get('title_length', 0) < 30:
                    recommendations.append("📝 WARNING: Title tag too short (<30 chars)")
                
                if not meta_data.get('meta_description'):
                    recommendations.append("📝 CRITICAL: Add meta description")
                elif meta_data.get('meta_description_length', 0) > 160:
                    recommendations.append("📝 WARNING: Meta description too long (>160 chars)")
                
                if not meta_data.get('canonical_url'):
                    recommendations.append("🔗 WARNING: Add canonical URL")
                
                if len(meta_data.get('h1_tags', [])) != 1:
                    recommendations.append("📊 WARNING: Use exactly one H1 tag per page")
                
                if meta_data.get('images_without_alt', 0) > 0:
                    recommendations.append(f"🖼️ WARNING: {meta_data['images_without_alt']} images missing alt text")
        
        # Technical SEO recommendations
        if 'technical_seo' in audit_results:
            tech_data = audit_results['technical_seo']
            if isinstance(tech_data, dict):
                if not tech_data.get('https_enabled'):
                    recommendations.append("🔒 CRITICAL: Enable HTTPS")
                
                if not tech_data.get('mobile_friendly'):
                    recommendations.append("📱 CRITICAL: Add mobile viewport meta tag")
                
                if not tech_data.get('compression'):
                    recommendations.append("⚡ IMPORTANT: Enable GZIP compression")
                
                if 'cache-control' not in tech_data.get('caching_headers', {}):
                    recommendations.append("⏰ IMPORTANT: Add cache-control headers")
        
        # Robots.txt recommendations
        if 'robots_txt' in audit_results:
            robots_data = audit_results['robots_txt']
            if isinstance(robots_data, dict):
                if not robots_data.get('exists'):
                    recommendations.append("🤖 IMPORTANT: Create robots.txt file")
                elif not robots_data.get('sitemap_declared'):
                    recommendations.append("🗺️ IMPORTANT: Add sitemap declaration to robots.txt")
        
        # Sitemap recommendations
        if 'sitemap_xml' in audit_results:
            sitemap_data = audit_results['sitemap_xml']
            if isinstance(sitemap_data, dict):
                if not sitemap_data.get('exists'):
                    recommendations.append("🗺️ CRITICAL: Create XML sitemap")
                elif sitemap_data.get('url_count', 0) == 0:
                    recommendations.append("🗺️ CRITICAL: Add URLs to sitemap")
        
        # Keyword optimization recommendations
        if 'keyword_optimization' in audit_results:
            kw_data = audit_results['keyword_optimization']
            if isinstance(kw_data, dict) and kw_data.get('overall_score', 0) < 60:
                recommendations.append("🎯 IMPORTANT: Improve keyword optimization")
                
                for keyword, analysis in kw_data.get('keyword_analysis', {}).items():
                    if analysis['score'] < 50:
                        suggestions = []
                        if not analysis['in_title']:
                            suggestions.append("add to title")
                        if not analysis['in_meta_description']:
                            suggestions.append("add to meta description")
                        if not analysis['in_h1']:
                            suggestions.append("add to H1")
                        if analysis['density_percent'] < 0.5:
                            suggestions.append("increase keyword density")
                        
                        if suggestions:
                            recommendations.append(f"   • '{keyword}': {', '.join(suggestions)}")
        
        return recommendations
    
    def run_comprehensive_audit(self, target_urls: List[str], target_keywords: List[str]) -> Dict[str, Any]:
        """Run comprehensive SEO audit"""
        logger.info("🔍 Starting comprehensive SEO audit for NeuralSync2...")
        
        audit_results = {
            'audit_timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'base_url': self.base_url,
            'pages_audited': [],
            'robots_txt': self.check_robots_txt(),
            'sitemap_xml': self.check_sitemap_xml(),
            'recommendations': [],
            'overall_score': 0
        }
        
        # Audit each target URL
        for url in target_urls:
            logger.info(f"Auditing URL: {url}")
            
            page_audit = {
                'url': url,
                'page_speed': self.audit_page_speed(url),
                'meta_tags': self.audit_meta_tags(url),
                'technical_seo': self.audit_technical_seo(url),
                'keyword_optimization': self.audit_keyword_optimization(url, target_keywords)
            }
            
            audit_results['pages_audited'].append(page_audit)
            time.sleep(1)  # Rate limiting
        
        # Generate recommendations
        audit_results['recommendations'] = self.generate_recommendations(audit_results)
        
        # Calculate overall score
        page_scores = []
        for page in audit_results['pages_audited']:
            kw_score = page.get('keyword_optimization', {}).get('overall_score', 0)
            speed_score = page.get('page_speed', {}).get('speed_rating', 3) * 20
            
            # Technical factors scoring
            tech_score = 0
            tech_data = page.get('technical_seo', {})
            if tech_data.get('https_enabled'):
                tech_score += 20
            if tech_data.get('mobile_friendly'):
                tech_score += 15
            if tech_data.get('compression'):
                tech_score += 10
            
            # Meta tags scoring
            meta_score = 0
            meta_data = page.get('meta_tags', {})
            if meta_data.get('title') and 30 <= meta_data.get('title_length', 0) <= 60:
                meta_score += 20
            if meta_data.get('meta_description') and 120 <= meta_data.get('meta_description_length', 0) <= 160:
                meta_score += 15
            if len(meta_data.get('h1_tags', [])) == 1:
                meta_score += 10
            
            page_score = (kw_score + speed_score + tech_score + meta_score) / 4
            page_scores.append(page_score)
        
        if page_scores:
            audit_results['overall_score'] = round(sum(page_scores) / len(page_scores))
        
        return audit_results
    
    def create_audit_report(self, audit_results: Dict[str, Any]) -> str:
        """Create detailed SEO audit report"""
        
        report = f"""# NeuralSync2 SEO Audit Report
Generated: {audit_results.get('audit_timestamp', 'N/A')}
Overall SEO Score: {audit_results.get('overall_score', 0)}/100

## Executive Summary
This comprehensive SEO audit analyzed {len(audit_results.get('pages_audited', []))} pages for technical SEO, on-page optimization, and keyword targeting.

## Overall Performance
- **SEO Score**: {audit_results.get('overall_score', 0)}/100
- **Pages Analyzed**: {len(audit_results.get('pages_audited', []))}
- **Critical Issues**: {len([r for r in audit_results.get('recommendations', []) if 'CRITICAL' in r])}
- **Warnings**: {len([r for r in audit_results.get('recommendations', []) if 'WARNING' in r])}

## Technical SEO Status

### Robots.txt
- **Status**: {'✅ Found' if audit_results.get('robots_txt', {}).get('exists') else '❌ Missing'}
- **Sitemap Declared**: {'✅ Yes' if audit_results.get('robots_txt', {}).get('sitemap_declared') else '❌ No'}

### XML Sitemap  
- **Status**: {'✅ Found' if audit_results.get('sitemap_xml', {}).get('exists') else '❌ Missing'}
- **URLs Count**: {audit_results.get('sitemap_xml', {}).get('url_count', 0)}
- **Valid XML**: {'✅ Yes' if audit_results.get('sitemap_xml', {}).get('valid_xml') else '❌ No'}

## Page-by-Page Analysis
"""

        # Add page-specific analysis
        for i, page in enumerate(audit_results.get('pages_audited', []), 1):
            url = page.get('url', 'Unknown')
            speed_data = page.get('page_speed', {})
            meta_data = page.get('meta_tags', {})
            kw_data = page.get('keyword_optimization', {})
            
            report += f"""
### Page {i}: {url}

**Performance**
- Load Time: {speed_data.get('load_time_seconds', 'N/A')}s
- Speed Score: {speed_data.get('speed_score', 'N/A')}
- Content Size: {speed_data.get('content_size_kb', 'N/A')} KB

**Meta Tags**
- Title: {meta_data.get('title', 'Missing')[:60]}{'...' if len(str(meta_data.get('title', ''))) > 60 else ''}
- Title Length: {meta_data.get('title_length', 0)} chars
- Meta Description: {'Present' if meta_data.get('meta_description') else 'Missing'}
- Description Length: {meta_data.get('meta_description_length', 0)} chars
- H1 Tags: {len(meta_data.get('h1_tags', []))}
- Schema Markup: {len(meta_data.get('schema_markup', []))} items

**Keyword Optimization**
- Overall Score: {kw_data.get('overall_score', 0)}/100
"""

            # Add keyword-specific analysis
            for keyword, analysis in kw_data.get('keyword_analysis', {}).items():
                report += f"""
- **{keyword}**: {analysis.get('score', 0)}/100
  - Density: {analysis.get('density_percent', 0)}%
  - In Title: {'✅' if analysis.get('in_title') else '❌'}
  - In Meta: {'✅' if analysis.get('in_meta_description') else '❌'}
  - In H1: {'✅' if analysis.get('in_h1') else '❌'}"""

        # Add recommendations section
        report += """

## Recommendations

### High Priority (Critical Issues)
"""
        critical_recommendations = [r for r in audit_results.get('recommendations', []) if 'CRITICAL' in r]
        for rec in critical_recommendations:
            report += f"- {rec}\n"

        report += """
### Medium Priority (Important Issues)
"""
        important_recommendations = [r for r in audit_results.get('recommendations', []) if 'IMPORTANT' in r]
        for rec in important_recommendations:
            report += f"- {rec}\n"

        report += """
### Low Priority (Warnings)
"""
        warning_recommendations = [r for r in audit_results.get('recommendations', []) if 'WARNING' in r]
        for rec in warning_recommendations:
            report += f"- {rec}\n"

        report += """

## Implementation Checklist

### Immediate Actions (This Week)
- [ ] Fix all CRITICAL issues identified
- [ ] Implement missing robots.txt and sitemap.xml
- [ ] Optimize page load speeds
- [ ] Add missing meta descriptions and title tags

### Short Term (Next Month)  
- [ ] Improve keyword optimization scores
- [ ] Add missing alt text to images
- [ ] Implement proper heading structure
- [ ] Enable technical SEO improvements

### Long Term (Ongoing)
- [ ] Monitor SEO performance metrics
- [ ] Regular content optimization
- [ ] Backlink building campaigns
- [ ] Technical infrastructure improvements

## SEO Tools Integration

### Google Search Console
- Submit sitemap: {audit_results.get('base_url', '')}/sitemap.xml
- Monitor search performance
- Track indexing status

### Bing Webmaster Tools
- Submit sitemap for Bing indexing
- Monitor Bing search visibility

---

*Report generated by NeuralSync2 SEO Audit Tool*
*For support: https://github.com/heyfinal/NeuralSync2*
"""

        return report

def main():
    """Execute comprehensive SEO audit"""
    print("🧠 NeuralSync2 SEO Audit Tool")
    
    # Target URLs to audit
    target_urls = [
        "https://neuralsync.dev/",
        "https://neuralsync.dev/ai-tools-with-memory",
        "https://neuralsync.dev/claude-memory-fix",
        "https://neuralsync.dev/persistent-ai-memory",
        "https://neuralsync.dev/ai-agent-frameworks"
    ]
    
    # Target keywords for optimization
    target_keywords = [
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
    ]
    
    # Run audit
    auditor = SEOAuditTool()
    audit_results = auditor.run_comprehensive_audit(target_urls, target_keywords)
    
    # Generate and save report
    report = auditor.create_audit_report(audit_results)
    
    report_filename = f"seo_audit_report_{int(time.time())}.md"
    with open(report_filename, 'w') as f:
        f.write(report)
    
    # Save detailed audit data
    audit_data_filename = f"seo_audit_data_{int(time.time())}.json"
    with open(audit_data_filename, 'w') as f:
        json.dump(audit_results, f, indent=2, default=str)
    
    logger.info(f"✅ SEO audit complete!")
    logger.info(f"📊 Report saved: {report_filename}")
    logger.info(f"📋 Data saved: {audit_data_filename}")
    
    print(f"\n📊 SEO AUDIT RESULTS")
    print(f"Overall Score: {audit_results.get('overall_score', 0)}/100")
    print(f"Critical Issues: {len([r for r in audit_results.get('recommendations', []) if 'CRITICAL' in r])}")
    print(f"Report: {report_filename}")
    
    return audit_results

if __name__ == "__main__":
    main()