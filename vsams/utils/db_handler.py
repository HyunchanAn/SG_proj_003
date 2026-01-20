import json
import os

DB_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'database.json')

def load_db():
    if not os.path.exists(DB_PATH):
        return []
    with open(DB_PATH, 'r', encoding='utf-8') as f:
        return json.load(f)

def save_db(data):
    """
    Saves the list of products back to the JSON file.
    """
    with open(DB_PATH, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def query_recommendation(ai_material, ai_finish):
    """
    Returns a list of recommended products based on AI prediction.
    """
    db = load_db()
    recommendations = []
    
    for product in db:
        conditions = product.get('target_condition', {})
        target_materials = conditions.get('material_category', [])
        target_finishes = conditions.get('finish_type', [])
        
        # Simple logical matching: if material matches AND finish matches, it's a candidate
        # We can implement fuzzy matching or fallback logic here
        
        # Material matching (Case insensitive)
        mat_match = any(m.lower() == ai_material.lower() for m in target_materials)
        
        # Finish matching (Case insensitive)
        fin_match = any(f.lower() == ai_finish.lower() for f in target_finishes)
        
        if mat_match and fin_match:
            recommendations.append(product)
            
    # Fallback: if no exact match, return at least one default product
    if not recommendations and db:
         # Return a generic one (e.g., the first one) but mark it as 'Generic Recommendation'
        return [db[0]]
        
    return recommendations
