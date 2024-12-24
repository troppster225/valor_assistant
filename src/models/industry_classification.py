from typing import Dict, List, Tuple
import torch
import torch.nn as nn

class IndustryClassification:
    INDUSTRY_HIERARCHY = {
    "Technology & Software": {
        "Enterprise Software": {
            "SaaS Applications": ["ERP", "CRM", "HCM"],
            "Security Software": ["Cybersecurity", "Access Control", "GRC"],
            "Development Tools": ["DevOps", "Testing", "Cloud Infrastructure"]
        },
        "Software Services": {
            "Custom Development": ["Application Development", "System Integration"],
            "IT Consulting": ["Digital Transformation", "Tech Strategy"],
            "Managed Services": ["Cloud Management", "IT Support"]
        }
    },
    "Healthcare & Life Sciences": {
        "Healthcare Providers": {
            "Clinical Practices": ["Ophthalmology", "Dental", "Urology"],
            "Facilities": ["Surgery Centers", "Clinics", "Laboratories"],
            "Home Health": ["Home Care", "Hospice", "Medical Equipment"]
        },
        "Life Sciences": {
            "Research Services": ["Clinical Trials", "Lab Services"],
            "Medical Devices": ["Diagnostic", "Therapeutic", "Monitoring"],
            "Pharmaceuticals": ["Drug Development", "Manufacturing", "Distribution"]
        }
    },
    "Industrial & Manufacturing": {
        "Manufacturing": {
            "Process Manufacturing": ["Chemical", "Food Processing", "Materials"],
            "Discrete Manufacturing": ["Machinery", "Equipment", "Components"],
            "Industrial Products": ["Tools", "Safety Equipment", "Supplies"]
        },
        "Industrial Services": {
            "Maintenance": ["Equipment Service", "Facility Maintenance"],
            "Engineering": ["Design", "Testing", "Inspection"],
            "Industrial Technology": ["Automation", "Control Systems"]
        }
    },
    "Consumer & Retail": {
        "Consumer Products": {
            "Personal Care": ["Beauty", "Health Products", "Wellness"],
            "Home Products": ["Furnishings", "Appliances", "Decor"],
            "Recreation": ["Sports Equipment", "Outdoor Gear"]
        },
        "Retail": {
            "Specialty Retail": ["Apparel", "Electronics", "Luxury"],
            "Food & Beverage": ["Restaurants", "Specialty Foods"],
            "E-commerce": ["Direct-to-Consumer", "Marketplaces"]
        }
    },
    "Business Services": {
        "Professional Services": {
            "Consulting": ["Management", "Operations", "Strategy"],
            "Business Process": ["HR Services", "Training", "Administrative"],
            "Marketing Services": ["Digital Marketing", "Advertising", "PR"]
        },
        "Facility Services": {
            "Building Services": ["Maintenance", "Security", "Cleaning"],
            "Construction Services": ["General Contracting", "Specialty Trade"],
            "Environmental Services": ["Waste Management", "Remediation"]
        }
    },
    "Financial Services": {
        "Banking & Lending": {
            "Commercial Banking": ["Business Banking", "Treasury Services"],
            "Consumer Finance": ["Personal Banking", "Consumer Lending"],
            "Specialty Finance": ["Equipment Finance", "Asset-based Lending"]
        },
        "Investment Services": {
            "Asset Management": ["Wealth Management", "Fund Management"],
            "Financial Technology": ["Payments", "Trading Platforms"],
            "Insurance": ["Property & Casualty", "Life & Health"]
        }
    },
    "Energy & Resources": {
        "Energy": {
            "Traditional Energy": ["Oil & Gas", "Power Generation"],
            "Renewable Energy": ["Solar", "Wind", "Battery Storage"],
            "Energy Services": ["Maintenance", "Engineering"]
        },
        "Natural Resources": {
            "Mining": ["Metals", "Minerals"],
            "Agriculture": ["Farming", "Processing"],
            "Water": ["Treatment", "Distribution"]
        }
    },
    "Media & Telecommunications": {
        "Media": {
            "Content Production": ["Film", "Television", "Digital Media"],
            "Publishing": ["Digital Publishing", "News Media"],
            "Entertainment": ["Gaming", "Interactive Media"]
        },
        "Telecommunications": {
            "Infrastructure": ["Network Equipment", "Fiber"],
            "Services": ["Wireless", "Broadband"],
            "Communications Tech": ["Unified Communications", "Collaboration"]
        }
    }
}
    def __init__(self):
        self.industry_map = self._build_industry_map()
        
    def _build_industry_map(self) -> Dict[str, List[str]]:
        """Build flattened industry map with variations"""
        industry_map = {}
        for category, subcats in self.INDUSTRY_HIERARCHY.items():
            variations = self._get_variations(category, subcats)
            industry_map[category] = variations
        return industry_map
        
    def _get_variations(self, category: str, data: Dict) -> List[str]:
        """Recursively get all variations of industry names"""
        variations = [category.lower()]
        if isinstance(data, dict):
            for subcat, subdata in data.items():
                variations.extend(self._get_variations(subcat, subdata))
        elif isinstance(data, list):
            variations.extend([item.lower() for item in data])
        return variations

    def get_mutually_exclusive_category(self, industry_text: str) -> Tuple[str, float]:
        """
        Map raw industry text to the most relevant top-level category
        Returns tuple of (category, confidence_score)
        """
        industry_text = industry_text.lower()
        best_match = None
        highest_score = 0.0
        
        for category, variations in self.industry_map.items():
            # Direct match with category
            if industry_text == category.lower():
                return category, 1.0
                
            # Check variations
            for variation in variations:
                if variation in industry_text or industry_text in variation:
                    score = len(set(industry_text.split()) & set(variation.split())) / max(
                        len(industry_text.split()), len(variation.split())
                    )
                    if score > highest_score:
                        highest_score = score
                        best_match = category
        
        if best_match:
            return best_match, highest_score
        
        # Return closest category based on word overlap if no direct match
        for category in self.INDUSTRY_HIERARCHY.keys():
            score = len(set(industry_text.split()) & set(category.lower().split())) / max(
                len(industry_text.split()), len(category.lower().split())
            )
            if score > highest_score:
                highest_score = score
                best_match = category
                
        return best_match or "Other", highest_score or 0.0

    def _clean_industry_text(self, text: str) -> str:
        """Clean and standardize industry text"""
        return text.lower().strip()

class IndustryEncoder(nn.Module):
    def __init__(self, embedding_dim: int = 128):
        super().__init__()
        self.classifier = IndustryClassification()
        num_industries = len(self.classifier.INDUSTRY_HIERARCHY)
        self.industry_embeddings = nn.Embedding(num_industries, embedding_dim)
        
    def _clean_industry_text(self, text: str) -> str:
        """Clean and standardize industry text"""
        return text.lower().strip()
        
    def standardize_industry(self, industry_text: str) -> Tuple[str, float]:
        """Map raw industry text to standardized category"""
        cleaned_text = self._clean_industry_text(industry_text)
        return self.classifier.get_mutually_exclusive_category(cleaned_text)

    def get_hierarchical_embedding(self, industry: str) -> torch.Tensor:
        """Get embedding for standardized industry"""
        industry_idx = list(self.classifier.INDUSTRY_HIERARCHY.keys()).index(industry)
        return self.industry_embeddings(torch.tensor([industry_idx]))

    def forward(self, industry_text: str) -> torch.Tensor:
        standardized, _ = self.standardize_industry(industry_text)
        return self.get_hierarchical_embedding(standardized)