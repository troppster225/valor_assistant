from typing import Dict, List, Optional, Union, Any, Tuple
import torch
from google.cloud import storage
import os
import sys
from pathlib import Path
import numpy as np
import re

# Add the project root to the Python path
current_file = Path(__file__).resolve()
project_root = current_file.parent.parent.parent
sys.path.insert(0, str(project_root))

# Import industry classification
from src.models.industry_classification import IndustryClassification, IndustryEncoder
from src.utils.similarity import (
    calculate_industry_similarity,
    calculate_hierarchical_similarity
)

class BusinessMatcher:
    def __init__(self):
        self.industry_encoder = IndustryEncoder()
        self.classifier = IndustryClassification()

        
        # Define common patterns for financial extraction
        self.financial_patterns = {
            "revenue": [
                # Basic patterns
                r"(?i)(?:revenue|sales)(?:\s+of)?\s*(?:\$|usd)?\s*(\d+(?:\.\d+)?)\s*(?:million|m|mm|M)",
                r"(?i)(?:\$|usd)?(\d+(?:\.\d+)?)\s*(?:million|m|mm|M)\s*(?:revenue|in revenue|annual revenue)",
                # Range patterns
                r"(?i)(?:revenue|sales)(?:\s+of)?\s*(?:\$|usd)?\s*(\d+(?:\.\d+)?)\s*-\s*\$?(\d+(?:\.\d+)?)\s*(?:million|m|mm|M)",
                # LTM patterns
                r"(?i)ltm\s+(?:revenue|sales)(?:\s+of)?\s*(?:\$|usd)?\s*(\d+(?:\.\d+)?)\s*(?:million|m|mm|M)",
                # Specific formats from examples
                r"(?i)ltm\s+revenue\s+of\s+at\s+least\s*(?:\$|usd)?\s*(\d+(?:\.\d+)?)\s*(?:million|m|mm|M)",
                r"(?i)revenue\s+of\s+at\s+least\s*(?:\$|usd)?\s*(\d+(?:\.\d+)?)\s*(?:million|m|mm|M)",
                r"(?i)(?:ltm\s+)?revenue\s+of\s+at\s+least\s*\$?\s*(\d+(?:\.\d+)?)\s*(?:million|m|mm|M)",
                r"(?i)revenue\s*(?:of|:|\s+)?\s*\$?\s*(\d+(?:\.\d+)?)\s*(?:million|m|mm|M)",
                r"(?i)revenue\s*>\s*\$?\s*(\d+(?:\.\d+)?)\s*(?:million|m|mm|M)",
                r"(?i)minimum\s+revenue\s*(?:of)?\s*\$?\s*(\d+(?:\.\d+)?)\s*(?:million|m|mm|M)",
                r"(?i)revenues?\s*(?:of)?\s*\$?\s*(\d+(?:\.\d+)?)\s*(?:million|m|mm|M)",
            ],
            "ebitda": [
                # Basic patterns
                r"(?i)(?:ebitda)(?:\s+of)?\s*(?:\$|usd)?\s*(\d+(?:\.\d+)?)\s*(?:million|m|mm|M)",
                r"(?i)(?:\$|usd)?(\d+(?:\.\d+)?)\s*(?:million|m|mm|M)\s*(?:ebitda|in ebitda)",
                # Range patterns
                r"(?i)(?:ebitda)(?:\s+of)?\s*(?:\$|usd)?\s*(\d+(?:\.\d+)?)\s*-\s*\$?(\d+(?:\.\d+)?)\s*(?:million|m|mm|M)",
                # Specific formats from examples
                r"(?i)ebitda\s+of\s*(?:\$|usd)?\s*(\d+(?:\.\d+)?)\s*-\s*\$?(\d+(?:\.\d+)?)\s*(?:million|m|mm|M)",
                r"(?i)(?:ebitda)\s+(?:\$|usd)?(\d+(?:\.\d+)?)\s*(?:million|m|mm|M)",
                r"(?i)ebitda\s*(?:of)?\s*\$?\s*(\d+(?:\.\d+)?)\s*-\s*\$?\s*(\d+(?:\.\d+)?)\s*(?:million|m|mm|M)",
                r"(?i)ebitda\s*:\s*\$?\s*(\d+(?:\.\d+)?)\s*-\s*\$?\s*(\d+(?:\.\d+)?)\s*(?:million|m|mm|M)",
                r"(?i)ebitda\s*(?:of)?\s*\$?\s*(\d+(?:\.\d+)?)\s*(?:million|m|mm|M)",
                r"(?i)\$?\s*(\d+(?:\.\d+)?)\s*-\s*\$?\s*\d+\+?\s*(?:million|m|mm|M)\s+ebitda",
                r"(?i)ebitda\s+(?:of\s+)?\$?\s*(\d+(?:\.\d+)?)\s*-\s*\$?\s*\d+\+?\s*(?:million|m|mm|M)",
                r"(?i)\$?\s*(\d+(?:\.\d+)?)\s*(?:million|m|mm|M)\s+ebitda\s+for\s+new\s+platform",
            ],
            "enterprise_value": [
                # Transaction size patterns (often indicates enterprise value)
                r"(?i)(?:transaction|deal)\s+size(?:\s+of)?\s*(?:\$|usd)?\s*(\d+(?:\.\d+)?)\s*-\s*\$?(\d+(?:\.\d+)?)\s*(?:million|m|mm|M)",
                r"(?i)(?:\$|usd)?(\d+(?:\.\d+)?)\s*-\s*\$?(\d+(?:\.\d+)?)\s*(?:million|m|mm|M)\s*(?:transaction|deal)\s+size",
                # Investment size patterns
                r"(?i)(?:investment|equity)\s+size(?:\s+of)?\s*(?:\$|usd)?\s*(\d+(?:\.\d+)?)\s*-\s*\$?(\d+(?:\.\d+)?)\s*(?:million|m|mm|M)",
                # Total transaction value
                r"(?i)(?:total)\s+(?:transaction|deal)\s+(?:size|value)(?:\s+of)?\s*(?:\$|usd)?\s*(\d+(?:\.\d+)?)\s*-\s*\$?(\d+(?:\.\d+)?)\s*(?:million|m|mm|M)",
                r"(?i)transaction\s+size\s*(?:of)?\s*\$?\s*(\d+(?:\.\d+)?)\s*-\s*\$?\s*(\d+(?:\.\d+)?)\s*(?:million|m|mm|M)",
                r"(?i)total\s+transaction\s+size\s*(?:of)?\s*\$?\s*(\d+(?:\.\d+)?)\s*-\s*\$?\s*(\d+(?:\.\d+)?)\s*(?:million|m|mm|M)",
                r"(?i)investment\s+size\s*(?:of)?\s*\$?\s*(\d+(?:\.\d+)?)\s*-\s*\$?\s*(\d+(?:\.\d+)?)\s*(?:million|m|mm|M)",
                r"(?i)company\s+size\s*(?:of\s+)?\$?\s*(\d+(?:\.\d+)?)\s*-\s*\$?\s*\d+\+?\s*(?:million|m|mm|M)",
                r"(?i)investment\s+size\s*(?:of\s+)?\$?\s*(\d+(?:\.\d+)?)\s*-\s*\$?\s*\d+\+?\s*(?:million|m|mm|M)",
            ],
            "revenue_growth": [
                r"(?i)(?:revenue|sales)\s+growth(?:\s+of)?\s*(\d+(?:\.\d+)?)\s*%",
                r"(?i)growing(?:\sat|\s+)(\d+(?:\.\d+)?)\s*%",
                r"(?i)growth\s+rate(?:\s+of)?\s*(\d+(?:\.\d+)?)\s*%",
                r"(?i)(\d+(?:\.\d+)?)\s*%\s*(?:revenue|sales)?\s*growth\s+rate",
                r"(?i)(\d+(?:\.\d+)?)\s*%\s*(?:y-?o-?y|year(?:-|\s+)over(?:-|\s+)year)",
            ],
            "ebitda_margin": [
                r"(?i)(?:ebitda)\s+margin(?:\s+of)?\s*(\d+(?:\.\d+)?)\s*%",
                r"(?i)margin(?:\s+of)?\s*(\d+(?:\.\d+)?)\s*%",
                r"(?i)(\d+(?:\.\d+)?)\s*%\s*(?:ebitda)?\s*margin",
            ],
            "aum": [
                r"(?i)(?:aum|assets\s+under\s+management)(?:\s+of)?\s*(?:\$|usd)?\s*(\d+(?:\.\d+)?)\s*(?:billion|b|B)",
                r"(?i)(?:\$|usd)?(\d+(?:\.\d+)?)\s*(?:billion|b|B)(?:\s+of)?\s*(?:aum|assets\s+under\s+management)",
                r"(?i)(?:aum|assets\s+under\s+management)(?:\s+of)?\s*(?:over|approximately)?\s*(?:\$|usd)?\s*(\d+(?:\.\d+)?)\s*(?:billion|b|B)",
            ]
        }
            

        # Define industry keywords
        self.industry_keywords = {
        "Technology & Software": [
            "software", "saas", "technology", "it services", "digital", "cloud",
            "cybersecurity", "artificial intelligence", "ai", "machine learning",
            "platform", "application", "data", "analytics", "automation",
            "enterprise software", "tech stack", "api", "infrastructure",
            "governance", "compliance", "risk management", "digital transformation"
        ],
        
        "Healthcare & Life Sciences": [
            "healthcare", "medical", "biotech", "pharmaceutical", "life sciences",
            "health", "clinical", "hospital", "diagnostic", "patient care",
            "therapeutic", "medicine", "biotechnology", "ophthalmology", "urology",
            "behavioral health", "ehr", "clinical research", "pharma services",
            "healthcare it", "medical equipment", "patient", "physician", "clinic",
            "health system", "medical practice", "care delivery", "provider"
        ],
        
        "Industrial & Manufacturing": [
            "manufacturing", "industrial", "equipment", "machinery", "production",
            "assembly", "fabrication", "engineering", "automation", "processing",
            "factory", "plant", "industrial technology", "supply chain",
            "quality control", "maintenance", "repair", "operations", "oem",
            "materials", "process control", "robotics"
        ],
        
        "Consumer & Retail": [
            "retail", "consumer", "e-commerce", "brand", "fashion", "food",
            "beverage", "apparel", "lifestyle", "direct-to-consumer", "d2c",
            "consumer products", "store", "shop", "restaurant", "franchisee",
            "beauty products", "retail chain", "omnichannel", "point of sale",
            "wholesale", "distribution", "cpg", "quick-service", "casual dining"
        ],
        
        "Business Services": [
            "business services", "consulting", "professional services",
            "outsourcing", "staffing", "hr services", "marketing services",
            "advisory", "management consulting", "business solutions",
            "service provider", "managed services", "business operations",
            "process improvement", "workflow", "business strategy"
        ],
        
        "Financial Services": [
            "financial", "banking", "insurance", "investment", "fintech",
            "wealth management", "payments", "lending", "capital markets",
            "asset management", "financial technology", "risk management",
            "portfolio", "trading", "investment banking", "financial planning",
            "mortgage", "credit", "financial advisory"
        ],
        
        "Energy & Resources": [
            "energy", "oil", "gas", "renewable", "utilities", "power",
            "solar", "wind", "natural resources", "mining", "clean energy",
            "sustainable energy", "energy efficiency", "power distribution",
            "energy storage", "green energy", "environmental services",
            "sustainability", "carbon", "electricity"
        ],
        
        "Media & Telecommunications": [
            "media", "telecom", "communications", "entertainment",
            "broadcasting", "publishing", "advertising", "content",
            "digital media", "telecommunications", "streaming", "network",
            "bandwidth", "connectivity", "mobile", "wireless", "broadcast",
            "content management"
        ]
    }
        
        # Define regions and countries
        self.regions = {
            'north america': {
                'countries': ['usa', 'united states', 'canada', 'mexico'],
                'regions': ['northeast', 'midwest', 'south', 'west', 'southwest', 'southeast']
            },
            'europe': {
                'countries': ['uk', 'france', 'germany', 'italy', 'spain', 'netherlands'],
                'regions': ['western europe', 'eastern europe', 'northern europe', 'southern europe']
            },
            'asia pacific': {
                'countries': ['china', 'japan', 'korea', 'india', 'australia', 'singapore'],
                'regions': ['southeast asia', 'east asia', 'south asia', 'oceania']
            },
            'latin america': {
                'countries': ['brazil', 'mexico', 'argentina', 'colombia', 'chile'],
                'regions': ['south america', 'central america', 'caribbean']
            }
        }

        self.industry_relationships = {
        "Technology & Software": {
            "primary": ["Technology & Software"],
            "related": ["Industrial & Manufacturing", "Healthcare & Life Sciences", 
                       "Financial Services", "Business Services", "Media & Telecommunications"]
        },
        "Industrial & Manufacturing": {
            "primary": ["Industrial & Manufacturing"],
            "related": ["Technology & Software", "Business Services", "Energy & Resources"]
        },
        "Healthcare & Life Sciences": {
            "primary": ["Healthcare & Life Sciences"],
            "related": ["Technology & Software", "Business Services", "Consumer & Retail"]
        },
        "Financial Services": {
            "primary": ["Financial Services"],
            "related": ["Technology & Software", "Business Services"]
        },
        "Consumer & Retail": {
            "primary": ["Consumer & Retail"],
            "related": ["Technology & Software", "Business Services", "Media & Telecommunications"]
        },
        "Business Services": {
            "primary": ["Business Services"],
            "related": ["Technology & Software", "Financial Services", "Industrial & Manufacturing", 
                       "Healthcare & Life Sciences", "Consumer & Retail"]
        },
        "Media & Telecommunications": {
            "primary": ["Media & Telecommunications"],
            "related": ["Technology & Software", "Consumer & Retail", "Business Services"]
        },
        "Energy & Resources": {
            "primary": ["Energy & Resources"],
            "related": ["Industrial & Manufacturing", "Technology & Software", "Business Services"]
        }
    }
        
    def read_document(self, blob) -> str:
        """Read and validate document content"""
        try:
            content = blob.download_as_text()
            print(f"\nReading file: {blob.name}")
            print(f"Raw content length: {len(content)}")
            print("First 100 chars:", repr(content[:100]))
            
            if len(content) <= 4:
                print(f"WARNING: Very short content detected in {blob.name}!")
                print("Full content:", repr(content))
                return ""
                
            return content
        
        except Exception as e:
            print(f"Error reading {blob.name}: {str(e)}")
            import traceback
            print(traceback.format_exc())
            return ""
    def clean_text_for_analysis(self, text: str) -> str:
        """Clean and normalize text for analysis"""
        if not text:
            return ""
            
        # Print input text length
        print(f"Cleaning text of length: {len(text)}")
        
        # Remove special characters but preserve important ones
        text = re.sub(r'[^\w\s\.\-\•\:\n]', ' ', text)
        
        # Normalize whitespace while preserving structure
        lines = text.split('\n')
        cleaned_lines = []
        for line in lines:
            line = ' '.join(line.split())
            if line:
                cleaned_lines.append(line)
        
        text = '\n'.join(cleaned_lines)
        
        # Preserve important markers
        text = re.sub(r'([•\-]\s*)', r'\n\1', text)  # Bullets on new lines
        text = re.sub(r'([A-Z][A-Za-z\s]{10,}:)', r'\n\1', text)  # Section headers
        
        result = text.lower().strip()
        print(f"Cleaned text length: {len(result)}")
        return result

    def _extract_industry(self, text: str) -> Union[str, Dict[str, Any]]:
        """Extract industry information using comprehensive pattern matching"""
        if not text or len(text.strip()) < 10:
            print("Error: Input text is too short or empty")
            return "Other"

        try:
            # Clean and normalize text
            text = self.clean_text_for_analysis(text)
            
            print(f"\nAnalyzing text length: {len(text)} characters")
            print("First 200 characters of cleaned text:")
            print(text[:200])
            
            print("\nLooking for industry indicators:")
            
            # Clean the text more aggressively for special characters
            # Clean the text more aggressively for special characters
            text = re.sub(r'[$\u0024]', 's', text)  # Replace $ with 's'
            
            # Comprehensive explicit mentions for all industries
            explicit_mentions = {
                # Technology & Software
                "software": "Technology & Software",
                "saas": "Technology & Software",
                "tech-enabled": "Technology & Software",
                "technology": "Technology & Software",
                "digital solutions": "Technology & Software",
                "enterprise software": "Technology & Software",
                
                # Industrial & Manufacturing
                "manufacturing": "Industrial & Manufacturing",
                "industrial": "Industrial & Manufacturing",
                "production": "Industrial & Manufacturing",
                "fabrication": "Industrial & Manufacturing",
                "assembly": "Industrial & Manufacturing",
                "industrial services": "Industrial & Manufacturing",
                "industrial & manufacturing": "Industrial & Manufacturing",
                "industrial and manufacturing": "Industrial & Manufacturing",
                "industrial&manufacturing": "Industrial & Manufacturing",
                
                # Healthcare & Life Sciences
                "healthcare": "Healthcare & Life Sciences",
                "medical": "Healthcare & Life Sciences",
                "life sciences": "Healthcare & Life Sciences",
                "pharma": "Healthcare & Life Sciences",
                "clinical": "Healthcare & Life Sciences",
                "health services": "Healthcare & Life Sciences",
                
                # Consumer & Retail
                "consumer": "Consumer & Retail",
                "retail": "Consumer & Retail",
                "e-commerce": "Consumer & Retail",
                "direct-to-consumer": "Consumer & Retail",
                "consumer products": "Consumer & Retail",
                "food and beverage": "Consumer & Retail",
                
                # Business Services
                "business services": "Business Services",
                "professional services": "Business Services",
                "consulting": "Business Services",
                "advisory": "Business Services",
                "management services": "Business Services",
                "outsourcing": "Business Services",
                
                # Financial Services
                "financial services": "Financial Services",
                "banking": "Financial Services",
                "insurance": "Financial Services",
                "fintech": "Financial Services",
                "wealth management": "Financial Services",
                
                # Energy & Resources
                "energy": "Energy & Resources",
                "utilities": "Energy & Resources",
                "oil": "Energy & Resources",
                "gas": "Energy & Resources",
                "power": "Energy & Resources",
                "renewable": "Energy & Resources",
                
                # Media & Telecommunications
                "media": "Media & Telecommunications",
                "telecommunications": "Media & Telecommunications",
                "telecom": "Media & Telecommunications",
                "broadcasting": "Media & Telecommunications",
                "content": "Media & Telecommunications",
                "communications": "Media & Telecommunications"
            }
            
            # Look for explicit mentions in bullet points or section headers
            # Look for explicit mentions in bullet points or section headers
            lines = text.split('\n')
            industry_counts = {}  # Dictionary to store frequency counts
            industry_weights = {}  # Dictionary to store weighted scores
            
            # First pass: count mentions and their context
            for line in lines:
                line_lower = line.lower().strip()
                print(f"Processing line: '{line_lower}'")
                
                # Process all industry terms uniformly
                for keyword, industry in explicit_mentions.items():
                    # Skip investment terms unless clearly about financial services
                    if keyword == "investment" and not any(fin_term in line_lower for fin_term in ["financial services", "banking", "insurance"]):
                        continue
                        
                    if keyword in line_lower:
                        print(f"Found match: '{keyword}' in '{line_lower}'")
                        if industry not in industry_counts:
                            industry_counts[industry] = 0
                            industry_weights[industry] = 0
                        
                        weight = 1  # Base weight
                        
                        # Structural weights
                        if line_lower.startswith(('•', '-', '*')):
                            weight += 2  # Bullet points
                        if line_lower.endswith(':'):
                            weight += 2  # Section headers
                        
                        # Context weights
                        if any(core_indicator in line_lower for core_indicator in ["core", "target industries"]):
                            weight += 3  # Core/target industry listing
                        if keyword in line_lower.split():
                            weight += 1  # Whole word match
                            
                        industry_counts[industry] += 1
                        industry_weights[industry] += weight
                        print(f"  Final weight for this mention: {weight}")

            # Calculate similarity scores
            total_mentions = sum(industry_counts.values())
            total_weight = sum(industry_weights.values())
            
            if total_weight > 0:
                industry_similarities = {
                    industry: (count / total_mentions * 0.3 + 
                            weight / total_weight * 0.7)  # Balance between mentions and weights
                    for industry, (count, weight) in 
                    zip(industry_counts.keys(), zip(industry_counts.values(), industry_weights.values()))
                }
                
                # Print analysis
                print("\nIndustry Analysis:")
                for industry in industry_similarities:
                    print(f"{industry}:")
                    print(f"  Mentions: {industry_counts[industry]}")
                    print(f"  Weighted Score: {industry_weights[industry]}")
                    print(f"  Similarity: {industry_similarities[industry]:.2%}")
                
                # Determine primary industry and additional industries
                sorted_industries = sorted(industry_similarities.items(), 
                                        key=lambda x: x[1], 
                                        reverse=True)
                
                # If the highest scoring industry is significantly higher than others
                if (len(sorted_industries) > 1 and 
                    sorted_industries[0][1] > sorted_industries[1][1] * 1.5):  # 50% higher than next
                    return sorted_industries[0][0]  # Return primary industry
                
                # Identify significant industries (those with meaningful similarity scores)
                significant_industries = [
                    industry for industry, score in sorted_industries 
                    if score > max(0.2, sorted_industries[0][1] * 0.3)  # Dynamic threshold
                ]
                
                if significant_industries:
                    if len(significant_industries) == 1:
                        return significant_industries[0]
                    else:
                        return {
                            'type': 'Multiple Industries',
                            'industries': significant_industries,
                            'primary': sorted_industries[0][0],
                            'similarities': {k: f"{v:.2%}" for k, v in industry_similarities.items() 
                                        if k in significant_industries}
                        }
            
            return "Other"  # Fallback if no significant industries found

        except Exception as e:
            print(f"Error in industry extraction: {str(e)}")
            import traceback
            print(traceback.format_exc())
            return "Other"

    def _extract_financials(self, text: str) -> Dict:
        """Extract financial metrics using regex patterns"""
        text = text.lower()
        results = {}
        
        for metric, patterns in self.financial_patterns.items():
            for pattern in patterns:
                matches = re.findall(pattern, text)
                if matches:
                    try:
                        match = matches[0]
                        
                        # Handle different match types
                        if isinstance(match, tuple):
                            # If it's a range (tuple), take the lower value
                            value = float(match[0])
                        elif isinstance(match, str):
                            # If it's a single value (string), convert directly
                            value = float(match)
                        else:
                            continue
                            
                        results[metric] = value
                        break
                    except (ValueError, IndexError):
                        continue
        
        # Set default values for missing metrics
        default_metrics = [
            'revenue', 'ebitda', 'enterprise_value', 
            'revenue_growth', 'ebitda_margin', 'gross_margin'
        ]
        for metric in default_metrics:
            if metric not in results:
                results[metric] = 0.0
        
        return results

    def _extract_geography(self, text: str) -> Dict:
        """Extract geographic information using defined regions"""
        text = text.lower()
        results = {
            'primary_region': 'global',
            'countries': [],
            'specific_regions': []
        }
        
        # Check for global indicators
        if any(word in text for word in ['global', 'worldwide', 'international']):
            return results
        
        # Check each region
        for region, data in self.regions.items():
            # Check countries
            for country in data['countries']:
                if country in text:
                    results['countries'].append(country)
                    results['primary_region'] = region
            
            # Check specific regions
            for specific_region in data['regions']:
                if specific_region in text:
                    results['specific_regions'].append(specific_region)
                    results['primary_region'] = region
        
        return results

    def analyze_match(self, company_data: Dict, criteria: Dict) -> Dict:
        source_industry = company_data['industry']
        target_industry = criteria['target_industries']
        
        # Calculate industry similarity with relationships
        if isinstance(source_industry, dict) and source_industry['type'] == 'Multiple Industries':
            # Check primary and related industries
            industry_similarity = 0.0
            for ind in source_industry['industries']:
                if target_industry == ind:
                    industry_similarity = max(industry_similarity, 1.0)
                elif target_industry in self.industry_relationships.get(ind, {}).get('related', []):
                    industry_similarity = max(industry_similarity, 0.5)
            print(f"Multiple industries match score: {industry_similarity}")
        else:
            # Single industry matching with relationships
            if source_industry == target_industry:
                industry_similarity = 1.0
            elif target_industry in self.industry_relationships.get(source_industry, {}).get('related', []):
                industry_similarity = 0.5
            else:
                industry_similarity = 0.0
            print(f"Single industry match score: {industry_similarity} between {source_industry} and {target_industry}")
        
        # Rest of the method remains the same
        financial_similarity = self.compare_financials(
            company_data.get('financials', {}),
            criteria.get('financial_criteria', {})
        )
        
        geo_similarity = self.compare_geography(
            company_data.get('geography', {}).get('primary_region', 'global'),
            criteria.get('target_geography', '')
        )
        
        weights = {
            'industry': 0.5,
            'financial': 0.3,
            'geography': 0.2
        }
        
        weighted_score = (
            industry_similarity * weights['industry'] +
            financial_similarity * weights['financial'] +
            geo_similarity * weights['geography']
        )
        
        # Print detailed matching information
        print(f"\nMatching Details:")
        print(f"Industry Similarity: {industry_similarity:.2f}")
        print(f"Financial Similarity: {financial_similarity:.2f}")
        print(f"Geographic Similarity: {geo_similarity:.2f}")
        print(f"Weighted Score: {weighted_score:.2f}")
        
        return {
            'total_score': max(min(weighted_score, 1.0), 0.0),
            'components': {
                'industry': industry_similarity,
                'financial': financial_similarity,
                'geography': geo_similarity
            }
        }

    def compare_financials(self, company_financials: Dict, criteria_financials: Dict) -> float:
        """Compare financial metrics with range handling"""
        if not company_financials or not criteria_financials:
            return 0.0
            
        metrics = ['ebitda', 'enterprise_value']  # Focus on key metrics
        scores = []
        
        for metric in metrics:
            company_value = company_financials.get(metric, 0)
            target_value = criteria_financials.get(metric, 0)
            
            if company_value and target_value:
                # Handle ranges (if value ends with +, treat it as a minimum)
                company_min = float(str(company_value).replace('+', ''))
                target_min = float(str(target_value).replace('+', ''))
                
                # Calculate similarity based on ranges
                if str(company_value).endswith('+') or str(target_value).endswith('+'):
                    if company_min >= target_min:
                        scores.append(1.0)
                    else:
                        ratio = company_min / target_min
                        scores.append(max(0.0, min(1.0, ratio)))
                else:
                    # Standard comparison for exact values
                    ratio = min(company_value, target_value) / max(company_value, target_value)
                    scores.append(max(0.0, min(1.0, ratio)))
    
        return sum(scores) / len(metrics) if scores else 0.0

    def compare_geography(self, company_geo: str, target_geo: str) -> float:
        """Compare geographic locations"""
        if not company_geo or not target_geo:
            return 0.0
            
        company_geo = company_geo.lower()
        target_geo = target_geo.lower()
        
        # Direct match
        if company_geo == target_geo:
            return 1.0
            
        # Region matching
        if company_geo in self.regions and target_geo in self.regions[company_geo]['countries']:
            return 0.75
        
        # Global matching
        if company_geo == 'global' or target_geo == 'global':
            return 0.5
        
        return 0.0

    def match_business_parameters(self, llm: Any, parameters: Dict, bucket_name: str) -> Dict:
        storage_client = storage.Client()
        bucket = storage_client.bucket(bucket_name)
        
        results = {}
        source_industry = parameters['industry']
        
        print("\n=== DETAILED DOCUMENT PROCESSING LOG ===")
        
        # 1. First, let's explicitly list ALL files in the bucket
        all_files = list(bucket.list_blobs())
        print("\nAll files in bucket:")
        for file in all_files:
            print(f"- {file.name} (size: {file.size} bytes)")
        
        # 2. Specifically look for the one-pager
        one_pager = [blob for blob in all_files if "Pharos" in blob.name]
        print("\nSearching for Pharos document:")
        for doc in one_pager:
            print(f"Found Pharos document: {doc.name}")
        
        # 3. Check parsed directory
        parsed_files = [blob for blob in all_files if blob.name.startswith('parsed/')]
        print("\nParsed files:")
        for file in parsed_files:
            print(f"- {file.name}")
            try:
                # Try to read each parsed file
                content = file.download_as_text()
                print(f"  ✓ Successfully read content ({len(content)} characters)")
                print(f"  ✓ First 100 chars: {content[:100]}")
            except Exception as e:
                print(f"  ✗ Error reading file: {str(e)}")
        
        # 4. Process each document with detailed logging
        print("\n=== PROCESSING EACH DOCUMENT ===")
        
        for blob in parsed_files:
            print(f"\nProcessing: {blob.name}")
            try:
                # Download and read the document
                document_text = blob.download_as_text()
                print(f"1. Downloaded text: {len(document_text)} characters")
                
                # Extract industry
                industry = self._extract_industry(document_text)
                print(f"2. Extracted industry: {industry}")
                
                # Extract financials with detailed logging
                print("3. Extracting financials...")
                financials = self._extract_financials(document_text)
                print(f"   Found financials: {financials}")
                
                # Extract geography
                geography = self._extract_geography(document_text)
                print(f"4. Extracted geography: {geography}")
                
                # Compile company data
                company_data = {
                    'industry': industry,
                    'financials': financials,
                    'geography': geography
                }
                
                # Calculate match scores
                print("5. Calculating match scores...")
                match_results = self.analyze_match(
                    company_data,
                    {
                        'target_industries': source_industry,
                        'financial_criteria': {
                            'revenue': parameters.get('revenue', 0),
                            'ebitda': parameters.get('ebitda', 0),
                            'enterprise_value': parameters.get('enterprise_value', 0),
                            'revenue_growth': parameters.get('revenue_growth', 0),
                            'ebitda_margin': parameters.get('ebitda_margin', 0),
                            'gross_margin': parameters.get('gross_margin', 0)
                        },
                        'target_geography': parameters.get('geography', '')
                    }
                )
                
                results[blob.name] = {
                    'match_score': match_results['total_score'],
                    'industry_similarity': match_results['components']['industry'],
                    'company_data': company_data
                }
                
                print(f"6. Final match score: {match_results['total_score']:.2%}")
                
            except Exception as e:
                print(f"ERROR processing {blob.name}: {str(e)}")
                import traceback
                print(f"Traceback: {traceback.format_exc()}")
                continue
        
        # 5. Final summary
        print("\n=== PROCESSING SUMMARY ===")
        print(f"Total files in bucket: {len(all_files)}")
        print(f"Total parsed files: {len(parsed_files)}")
        print(f"Documents successfully processed: {len(results)}")
        if len(one_pager) > 0:
            print(f"Pharos document status: {'Processed' if any('Pharos' in k for k in results.keys()) else 'Not Processed'}")
        
        return results