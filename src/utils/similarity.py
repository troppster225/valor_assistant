# src/utils/similarity.py
from typing import Dict, List, Optional
from ..models.industry_classification import IndustryClassification

def get_industry_path(industry: str, hierarchy: Dict) -> Optional[List[str]]:
    """
    Helper function to get the full path of an industry in the hierarchy
    Returns a list representing the path from root to the target industry
    """
    def search_hierarchy(current_hierarchy: Dict, target: str, path: List[str] = []) -> Optional[List[str]]:
        for key, value in current_hierarchy.items():
            current_path = path + [key]
            if key == target:
                return current_path
            if isinstance(value, dict):
                result = search_hierarchy(value, target, current_path)
                if result:
                    return result
            elif isinstance(value, list) and target in value:
                return current_path + [target]
        return None

    return search_hierarchy(hierarchy, industry)

def calculate_industry_similarity(source_industry: str, target_industries: List[str]) -> float:
    """
    Calculate similarity between source industry and a list of target industries
    Returns the highest similarity score found
    """
    classifier = IndustryClassification()
    
    source_category = classifier.get_mutually_exclusive_category(source_industry)
    if not source_category:
        return 0.0
    
    max_similarity = 0.0
    for target in target_industries:
        target_category = classifier.get_mutually_exclusive_category(target)
        if not target_category:
            continue
            
        # Calculate similarity based on hierarchy level
        similarity = calculate_hierarchical_similarity(
            source_category,
            target_category,
            classifier.INDUSTRY_HIERARCHY
        )
        max_similarity = max(max_similarity, similarity)
    
    return max_similarity

def calculate_hierarchical_similarity(source: str, target: str, hierarchy: Dict) -> float:
    """
    Calculate similarity based on hierarchy level match
    Returns a float between 0 and 1 representing similarity
    """
    source_path = get_industry_path(source, hierarchy)
    target_path = get_industry_path(target, hierarchy)
    
    if not source_path or not target_path:
        return 0.0
    
    common_path_length = 0
    for s, t in zip(source_path, target_path):
        if s == t:
            common_path_length += 1
        else:
            break
    
    max_path_length = max(len(source_path), len(target_path))
    return common_path_length / max_path_length

def calculate_financial_similarity(firm_criteria: Dict, company_data: Dict) -> Optional[float]:
    """
    Calculate financial similarity score accounting for N/A values
    Returns None if no valid comparisons can be made
    """
    metrics = ['revenue', 'ebitda', 'enterprise_value']
    valid_comparisons = 0
    total_score = 0
    
    for metric in metrics:
        firm_value = firm_criteria.get(metric)
        company_value = company_data.get(metric)
        
        # Skip if either value is None, 0, or not provided
        if not firm_value or not company_value:
            continue
            
        valid_comparisons += 1
        # Calculate similarity for this metric
        # Allow for a 20% deviation while maintaining a high score
        ratio = min(company_value, firm_value) / max(company_value, firm_value)
        metric_score = min(1.0, ratio + 0.2)  # Add 20% tolerance
        total_score += metric_score
    
    # If no valid comparisons, return None
    if valid_comparisons == 0:
        return None
    
    return total_score / valid_comparisons