#!/usr/bin/env python3
"""
Discovery Amplifier - Maximizes organic discoverability through SEO optimization
Creates discoverable content that ranks highly for relevant searches
"""

import asyncio
import json
import re
import time
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from urllib.parse import urljoin, urlparse
import hashlib

import requests
from bs4 import BeautifulSoup
import yake
import pandas as pd


@dataclass
class SEOMetadata:
    """SEO metadata for a piece of content"""
    title: str
    description: str
    keywords: List[str]
    canonical_url: str
    og_title: str
    og_description: str
    og_image: str
    twitter_title: str
    twitter_description: str
    schema_markup: Dict[str, Any]


@dataclass
class DiscoveryTarget:
    """Target platform/location for content discovery"""
    platform: str
    url: str
    audience_type: str
    content_format: str
    ranking_factors: List[str]
    placement_strategy: str


class DiscoveryAmplifier:
    """
    Optimizes content for maximum organic discoverability
    
    Uses advanced SEO techniques, keyword research, and strategic
    placement to ensure content is found by target audiences.
    """
    
    def __init__(self, core):
        self.core = core
        self.keyword_extractor = yake.KeywordExtractor(
            lan="en",
            n=3,
            dedupLim=0.7,
            top=20,
            features=None
        )
        
        # Discovery targets where AI developers search
        self.discovery_targets = self._initialize_discovery_targets()
        
        # SEO optimization patterns
        self.seo_patterns = {
            "title_optimization": {
                "max_length": 60,
                "include_primary_keyword": True,
                "emotional_triggers": [
                    "Revolutionary", "Breakthrough", "Game-Changing",
                    "Ultimate", "Perfect", "Effortless", "Instant"
                ]
            },
            "meta_description": {
                "max_length": 160,
                "include_cta": True,
                "answer_user_intent": True
            },
            "content_structure": {
                "h1_count": 1,
                "h2_min": 3,
                "paragraph_max_length": 150,
                "keyword_density": {"min": 0.5, "max": 2.5}
            }
        }
        
    def _initialize_discovery_targets(self) -> List[DiscoveryTarget]:
        """Initialize target platforms for content discovery"""
        return [
            DiscoveryTarget(
                platform="GitHub",
                url="https://github.com",
                audience_type="developers",
                content_format="repository_readme",
                ranking_factors=["stars", "forks", "recent_activity", "keywords_in_readme"],
                placement_strategy="awesome_lists"
            ),
            DiscoveryTarget(
                platform="Reddit",
                url="https://reddit.com",
                audience_type="technical",
                content_format="discussion_post",
                ranking_factors=["upvotes", "comments", "engagement_time"],
                placement_strategy="relevant_subreddits"
            ),
            DiscoveryTarget(
                platform="Hacker News",
                url="https://news.ycombinator.com",
                audience_type="startup_tech",
                content_format="link_submission",
                ranking_factors=["points", "comments", "velocity"],
                placement_strategy="trending_topics"
            ),
            DiscoveryTarget(
                platform="Dev.to",
                url="https://dev.to",
                audience_type="web_developers",
                content_format="technical_article",
                ranking_factors=["reactions", "reading_time", "tags"],
                placement_strategy="trending_tags"
            ),
            DiscoveryTarget(
                platform="Stack Overflow",
                url="https://stackoverflow.com",
                audience_type="problem_solvers",
                content_format="question_answer",
                ranking_factors=["votes", "views", "accepted_answer"],
                placement_strategy="high_traffic_questions"
            ),
            DiscoveryTarget(
                platform="Google Search",
                url="https://google.com",
                audience_type="searchers",
                content_format="web_page",
                ranking_factors=["relevance", "authority", "freshness", "user_intent"],
                placement_strategy="long_tail_keywords"
            )
        ]
        
    async def optimize_all_content(self) -> int:
        """Optimize all cached content for discovery"""
        optimized_count = 0
        
        try:
            # Get all content from cache
            for content_id, content in self.core.content_cache.items():
                try:
                    # Optimize content for discovery
                    optimized_content = await self.optimize_content_for_discovery(content)
                    
                    # Update in cache and database
                    self.core.content_cache[content_id] = optimized_content
                    self.core._store_content(optimized_content)
                    
                    optimized_count += 1
                    
                except Exception as e:
                    self.core.logger.error(f"Content optimization error for {content_id}: {e}")
                    
            return optimized_count
            
        except Exception as e:
            self.core.logger.error(f"Batch optimization error: {e}")
            return 0
            
    async def optimize_content_for_discovery(self, content: 'ViralContent') -> 'ViralContent':
        """Optimize a single piece of content for maximum discoverability"""
        try:
            # Extract and enhance keywords
            enhanced_keywords = await self._extract_and_enhance_keywords(content.content)
            
            # Generate SEO metadata
            seo_metadata = await self._generate_seo_metadata(content, enhanced_keywords)
            
            # Optimize content structure
            optimized_content_text = await self._optimize_content_structure(content.content, enhanced_keywords)
            
            # Calculate new discovery score
            new_discovery_score = await self._calculate_enhanced_discovery_score(
                optimized_content_text, enhanced_keywords, seo_metadata
            )
            
            # Create optimized content copy
            optimized_content = content
            optimized_content.content = optimized_content_text
            optimized_content.seo_keywords = enhanced_keywords
            optimized_content.discovery_score = new_discovery_score
            optimized_content.updated_at = datetime.now()
            
            # Add SEO metadata to performance metrics
            optimized_content.performance_metrics.update({
                "seo_metadata": seo_metadata.__dict__ if seo_metadata else {},
                "optimization_timestamp": datetime.now().isoformat(),
                "keyword_count": len(enhanced_keywords)
            })
            
            return optimized_content
            
        except Exception as e:
            self.core.logger.error(f"Content optimization error: {e}")
            return content
            
    async def _extract_and_enhance_keywords(self, content: str) -> List[str]:
        """Extract and enhance keywords for better discoverability"""
        try:
            # Extract keywords using YAKE
            extracted_keywords = self.keyword_extractor.extract_keywords(content)
            base_keywords = [kw[1] for kw in extracted_keywords[:15]]  # Top 15 keywords
            
            # Add NeuralSync2-specific keywords
            neuralsync_keywords = [
                "AI synchronization", "neural sync", "AI memory persistence",
                "AI tool integration", "Claude Code", "AI development tools",
                "cross-platform AI", "AI agent frameworks", "perfect AI memory",
                "sub-10ms sync", "CRDT AI", "temporal knowledge graphs"
            ]
            
            # Add trending AI keywords (would be updated regularly)
            trending_keywords = await self._get_trending_ai_keywords()
            
            # Combine and deduplicate
            all_keywords = list(set(base_keywords + neuralsync_keywords + trending_keywords))
            
            # Sort by relevance to content
            relevant_keywords = [kw for kw in all_keywords 
                               if any(word.lower() in content.lower() 
                                     for word in kw.split())]
            
            return relevant_keywords[:20]  # Top 20 most relevant
            
        except Exception as e:
            self.core.logger.error(f"Keyword extraction error: {e}")
            return []
            
    async def _get_trending_ai_keywords(self) -> List[str]:
        """Get trending AI-related keywords"""
        # In a real implementation, this would fetch from trend APIs
        # For now, return manually curated trending keywords
        return [
            "generative AI", "large language models", "AI agents",
            "artificial intelligence tools", "machine learning ops",
            "AI workflow automation", "intelligent assistants",
            "AI-powered development", "neural networks", "deep learning"
        ]
        
    async def _generate_seo_metadata(self, content: 'ViralContent', keywords: List[str]) -> SEOMetadata:
        """Generate comprehensive SEO metadata"""
        try:
            primary_keyword = keywords[0] if keywords else "AI tools"
            
            # Optimize title
            optimized_title = await self._optimize_title(content.title, primary_keyword)
            
            # Generate meta description
            meta_description = await self._generate_meta_description(content, primary_keyword)
            
            # Create social media metadata
            og_title = f"{optimized_title} | NeuralSync2"
            og_description = meta_description
            
            # Generate schema markup
            schema_markup = await self._generate_schema_markup(content, keywords)
            
            return SEOMetadata(
                title=optimized_title,
                description=meta_description,
                keywords=keywords,
                canonical_url=f"https://neuralsync2.dev/content/{content.id}",
                og_title=og_title,
                og_description=og_description,
                og_image="https://neuralsync2.dev/images/og-default.png",
                twitter_title=optimized_title,
                twitter_description=meta_description,
                schema_markup=schema_markup
            )
            
        except Exception as e:
            self.core.logger.error(f"SEO metadata generation error: {e}")
            return None
            
    async def _optimize_title(self, original_title: str, primary_keyword: str) -> str:
        """Optimize title for SEO while maintaining viral appeal"""
        try:
            # Ensure primary keyword is included
            if primary_keyword.lower() not in original_title.lower():
                if len(original_title) < 50:
                    optimized_title = f"{original_title} - {primary_keyword.title()}"
                else:
                    optimized_title = f"{primary_keyword.title()}: {original_title}"
            else:
                optimized_title = original_title
                
            # Ensure title length is optimal
            if len(optimized_title) > 60:
                # Truncate while keeping primary keyword
                if primary_keyword.lower() in optimized_title[:60].lower():
                    optimized_title = optimized_title[:57] + "..."
                else:
                    # Rebuild shorter title with primary keyword
                    base_title = original_title[:40]
                    optimized_title = f"{base_title} - {primary_keyword.title()}"
                    
            return optimized_title
            
        except Exception as e:
            self.core.logger.error(f"Title optimization error: {e}")
            return original_title
            
    async def _generate_meta_description(self, content: 'ViralContent', primary_keyword: str) -> str:
        """Generate compelling meta description with primary keyword"""
        try:
            # Extract first meaningful paragraph
            paragraphs = [p.strip() for p in content.content.split('\n\n') if len(p.strip()) > 50]
            base_description = paragraphs[0] if paragraphs else content.content[:100]
            
            # Clean up markdown formatting
            clean_description = re.sub(r'[#*`]', '', base_description)
            clean_description = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', clean_description)  # Remove links
            
            # Ensure primary keyword is included
            if primary_keyword.lower() not in clean_description.lower():
                clean_description = f"{primary_keyword}: {clean_description}"
                
            # Add call to action
            cta_phrases = [
                "Discover how", "Learn more", "Get started", "Try it now", 
                "See the demo", "Download free"
            ]
            
            if len(clean_description) < 120:
                cta = f" {cta_phrases[hash(content.id) % len(cta_phrases)]} →"
                clean_description += cta
                
            # Ensure optimal length
            if len(clean_description) > 160:
                clean_description = clean_description[:157] + "..."
                
            return clean_description
            
        except Exception as e:
            self.core.logger.error(f"Meta description generation error: {e}")
            return f"Discover {primary_keyword} with NeuralSync2 - revolutionary AI tool integration."
            
    async def _generate_schema_markup(self, content: 'ViralContent', keywords: List[str]) -> Dict[str, Any]:
        """Generate structured data markup for better search visibility"""
        try:
            schema = {
                "@context": "https://schema.org/",
                "@type": "Article",
                "headline": content.title,
                "description": content.content[:200],
                "author": {
                    "@type": "Organization",
                    "name": "NeuralSync2",
                    "url": "https://github.com/heyfinal/NeuralSync2"
                },
                "publisher": {
                    "@type": "Organization",
                    "name": "NeuralSync2",
                    "logo": {
                        "@type": "ImageObject",
                        "url": "https://neuralsync2.dev/logo.png"
                    }
                },
                "datePublished": content.created_at.isoformat(),
                "dateModified": content.updated_at.isoformat(),
                "keywords": ", ".join(keywords),
                "articleSection": "Technology",
                "about": {
                    "@type": "Thing",
                    "name": "AI Tool Integration"
                }
            }
            
            # Add specific schema based on content type
            if content.content_type == "demo":
                schema["@type"] = "HowTo"
                schema["name"] = content.title
                
            elif content.content_type == "showcase":
                schema["@type"] = "TechArticle"
                schema["proficiencyLevel"] = "Expert"
                
            elif content.content_type == "challenge":
                schema["@type"] = "Event"
                schema["eventStatus"] = "EventScheduled"
                
            return schema
            
        except Exception as e:
            self.core.logger.error(f"Schema markup generation error: {e}")
            return {}
            
    async def _optimize_content_structure(self, content: str, keywords: List[str]) -> str:
        """Optimize content structure for SEO and readability"""
        try:
            lines = content.split('\n')
            optimized_lines = []
            primary_keyword = keywords[0] if keywords else "AI tools"
            
            h1_added = False
            h2_count = 0
            
            for line in lines:
                line = line.strip()
                
                if not line:
                    optimized_lines.append('')
                    continue
                    
                # Optimize headers
                if line.startswith('# ') and not h1_added:
                    # Ensure primary keyword in H1
                    h1_text = line[2:].strip()
                    if primary_keyword.lower() not in h1_text.lower():
                        h1_text = f"{h1_text} - {primary_keyword.title()}"
                    optimized_lines.append(f"# {h1_text}")
                    h1_added = True
                    
                elif line.startswith('## '):
                    h2_count += 1
                    optimized_lines.append(line)
                    
                elif line.startswith('### '):
                    optimized_lines.append(line)
                    
                # Optimize regular content
                else:
                    # Add keyword variations naturally
                    optimized_line = await self._naturally_integrate_keywords(line, keywords)
                    optimized_lines.append(optimized_line)
                    
            # Ensure minimum H2 count
            if h2_count < 3:
                # Add relevant H2 sections
                additional_sections = [
                    "## Key Benefits",
                    "## How It Works", 
                    "## Getting Started"
                ]
                
                for section in additional_sections[:3-h2_count]:
                    optimized_lines.insert(-5, section)
                    optimized_lines.insert(-5, f"Learn more about {primary_keyword} integration.")
                    optimized_lines.insert(-5, "")
                    
            return '\n'.join(optimized_lines)
            
        except Exception as e:
            self.core.logger.error(f"Content structure optimization error: {e}")
            return content
            
    async def _naturally_integrate_keywords(self, line: str, keywords: List[str]) -> str:
        """Naturally integrate keywords into content without over-optimization"""
        try:
            if len(line) < 20 or line.startswith('```') or line.startswith('|'):
                return line  # Skip short lines, code blocks, and tables
                
            # Randomly select a keyword to integrate (low probability)
            if len(keywords) > 0 and hash(line) % 10 == 0:  # 10% chance
                keyword = keywords[hash(line) % len(keywords)]
                
                # Find natural integration points
                integration_patterns = [
                    ("with", f"with {keyword}"),
                    ("using", f"using {keyword}"), 
                    ("through", f"through {keyword}"),
                    ("by", f"by leveraging {keyword}")
                ]
                
                for pattern, replacement in integration_patterns:
                    if f" {pattern} " in line.lower() and keyword.lower() not in line.lower():
                        return line.replace(f" {pattern} ", f" {replacement} ", 1)
                        
            return line
            
        except Exception as e:
            self.core.logger.error(f"Keyword integration error: {e}")
            return line
            
    async def _calculate_enhanced_discovery_score(self, content: str, keywords: List[str], seo_metadata: SEOMetadata) -> float:
        """Calculate enhanced discovery score after optimization"""
        try:
            base_score = 0.6  # Starting with higher base due to optimization
            
            # Keyword optimization score
            keyword_density = self._calculate_keyword_density(content, keywords)
            if 0.5 <= keyword_density <= 2.5:  # Optimal range
                base_score += 0.15
            elif keyword_density > 0:
                base_score += 0.08
                
            # Title optimization score
            if seo_metadata and len(seo_metadata.title) <= 60 and keywords[0].lower() in seo_metadata.title.lower():
                base_score += 0.1
                
            # Meta description score
            if seo_metadata and 120 <= len(seo_metadata.description) <= 160:
                base_score += 0.08
                
            # Content structure score
            h1_count = len(re.findall(r'^# ', content, re.MULTILINE))
            h2_count = len(re.findall(r'^## ', content, re.MULTILINE))
            if h1_count == 1 and h2_count >= 3:
                base_score += 0.12
                
            # Schema markup score
            if seo_metadata and seo_metadata.schema_markup:
                base_score += 0.05
                
            return min(1.0, base_score)
            
        except Exception as e:
            self.core.logger.error(f"Discovery score calculation error: {e}")
            return 0.7  # Default optimized score
            
    def _calculate_keyword_density(self, content: str, keywords: List[str]) -> float:
        """Calculate keyword density in content"""
        try:
            if not keywords:
                return 0.0
                
            word_count = len(content.split())
            if word_count == 0:
                return 0.0
                
            keyword_occurrences = 0
            content_lower = content.lower()
            
            for keyword in keywords[:5]:  # Check top 5 keywords
                keyword_occurrences += content_lower.count(keyword.lower())
                
            density = (keyword_occurrences / word_count) * 100
            return density
            
        except Exception as e:
            self.core.logger.error(f"Keyword density calculation error: {e}")
            return 0.0
            
    async def create_discovery_sitemap(self) -> str:
        """Create XML sitemap for all optimized content"""
        try:
            sitemap_header = '''<?xml version="1.0" encoding="UTF-8"?>
<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">'''
            
            sitemap_urls = []
            
            for content_id, content in self.core.content_cache.items():
                if content.discovery_score > 0.7:  # Only include high-quality content
                    url_entry = f'''
    <url>
        <loc>https://neuralsync2.dev/content/{content_id}</loc>
        <lastmod>{content.updated_at.strftime('%Y-%m-%d')}</lastmod>
        <changefreq>weekly</changefreq>
        <priority>{min(1.0, content.discovery_score):.1f}</priority>
    </url>'''
                    sitemap_urls.append(url_entry)
                    
            sitemap_footer = '\n</urlset>'
            
            full_sitemap = sitemap_header + ''.join(sitemap_urls) + sitemap_footer
            
            # Save sitemap
            sitemap_path = Path(self.core.config["output_directory"]) / "sitemap.xml"
            sitemap_path.parent.mkdir(exist_ok=True)
            with open(sitemap_path, 'w') as f:
                f.write(full_sitemap)
                
            return str(sitemap_path)
            
        except Exception as e:
            self.core.logger.error(f"Sitemap creation error: {e}")
            return ""
            
    async def generate_discovery_report(self) -> Dict[str, Any]:
        """Generate comprehensive discovery optimization report"""
        try:
            total_content = len(self.core.content_cache)
            optimized_content = sum(1 for c in self.core.content_cache.values() 
                                  if c.discovery_score > 0.7)
            
            avg_discovery_score = sum(c.discovery_score for c in self.core.content_cache.values()) / max(1, total_content)
            
            keyword_analysis = {}
            all_keywords = []
            for content in self.core.content_cache.values():
                all_keywords.extend(content.seo_keywords)
                
            if all_keywords:
                keyword_counts = Counter(all_keywords)
                keyword_analysis = {
                    "total_unique_keywords": len(set(all_keywords)),
                    "avg_keywords_per_content": len(all_keywords) / max(1, total_content),
                    "top_keywords": keyword_counts.most_common(10)
                }
                
            return {
                "timestamp": datetime.now().isoformat(),
                "content_stats": {
                    "total_content": total_content,
                    "optimized_content": optimized_content,
                    "optimization_rate": (optimized_content / max(1, total_content)) * 100,
                    "avg_discovery_score": avg_discovery_score
                },
                "keyword_analysis": keyword_analysis,
                "discovery_targets": len(self.discovery_targets),
                "sitemap_generated": True,
                "recommendations": await self._generate_optimization_recommendations()
            }
            
        except Exception as e:
            self.core.logger.error(f"Discovery report generation error: {e}")
            return {}
            
    async def _generate_optimization_recommendations(self) -> List[str]:
        """Generate recommendations for further optimization"""
        recommendations = []
        
        try:
            # Analyze current performance
            low_score_content = [c for c in self.core.content_cache.values() 
                               if c.discovery_score < 0.6]
            
            if len(low_score_content) > 0:
                recommendations.append(f"Optimize {len(low_score_content)} pieces of low-scoring content")
                
            # Check keyword diversity
            all_keywords = []
            for content in self.core.content_cache.values():
                all_keywords.extend(content.seo_keywords)
                
            if len(set(all_keywords)) < 50:
                recommendations.append("Expand keyword diversity to target more search queries")
                
            # Check content types
            content_types = [c.content_type for c in self.core.content_cache.values()]
            type_counts = Counter(content_types)
            
            if "challenge" not in type_counts:
                recommendations.append("Add challenge-type content to increase engagement")
                
            if not recommendations:
                recommendations.append("Continue monitoring and adapting based on performance metrics")
                
            return recommendations
            
        except Exception as e:
            self.core.logger.error(f"Recommendation generation error: {e}")
            return ["Monitor content performance and adjust strategy as needed"]


# Usage example and testing
if __name__ == "__main__":
    import sys
    sys.path.append('.')
    from viralforge_core import ViralForgeCore, ViralContent
    
    async def test_discovery_amplifier():
        """Test the discovery amplifier"""
        core = ViralForgeCore()
        amplifier = DiscoveryAmplifier(core)
        
        # Create test content
        test_content = ViralContent(
            id="test_001",
            title="NeuralSync2 Demo",
            content="# NeuralSync2 Demo\n\nThis is a test of AI tool integration...",
            content_type="demo",
            target_audience="developers",
            viral_hooks=["Perfect AI memory", "Sub-10ms sync"],
            seo_keywords=["AI tools", "synchronization"],
            discovery_score=0.5,
            created_at=datetime.now(),
            updated_at=datetime.now(),
            performance_metrics={}
        )
        
        core.content_cache["test_001"] = test_content
        
        print("Testing discovery amplifier...")
        optimized_count = await amplifier.optimize_all_content()
        print(f"Optimized {optimized_count} pieces of content")
        
        # Generate report
        report = await amplifier.generate_discovery_report()
        print(f"Discovery Report: {json.dumps(report, indent=2)}")
        
    # Run test
    asyncio.run(test_discovery_amplifier())