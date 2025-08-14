#!/usr/bin/env python3
"""Test the guardrail function"""

def is_medical_query(query: str) -> bool:
    """Check if the query is medical-related and appropriate for MediAid AI"""
    query_lower = query.lower()
    
    # Medical keywords that indicate legitimate medical queries
    medical_keywords = [
        'symptom', 'symptoms', 'disease', 'condition', 'diabetes', 'treatment',
        'medication', 'medicine', 'doctor', 'health', 'medical', 'pain', 'fever',
        'chest', 'heart', 'blood', 'pressure', 'pregnant', 'pregnancy'
    ]
    
    # Non-medical keywords
    non_medical_keywords = [
        'movie', 'film', 'music', 'song', 'game', 'programming', 'code',
        'business', 'investment', 'homework', 'essay', 'politics', 'recipe',
        'cooking', 'weather', 'news', 'celebrity', 'sport'
    ]
    
    # Check for explicit non-medical patterns
    non_medical_patterns = [
        'how to make', 'recipe for', 'best restaurant', 'movie recommendation',
        'programming tutorial', 'investment advice', 'weather forecast',
        'game walkthrough', 'homework help', 'latest news'
    ]
    
    if any(pattern in query_lower for pattern in non_medical_patterns):
        return False
    
    # Count medical vs non-medical keywords
    medical_count = sum(1 for keyword in medical_keywords if keyword in query_lower)
    non_medical_count = sum(1 for keyword in non_medical_keywords if keyword in query_lower)
    
    # Strong indicators of non-medical content
    if non_medical_count >= 2 and medical_count == 0:
        return False
    
    if non_medical_count > medical_count and non_medical_count >= 1:
        return False
    
    # Medical question patterns
    medical_question_patterns = [
        'what is', 'what are', 'how to treat', 'symptoms of', 'treatment for',
        'medication for', 'side effects', 'is it safe', 'can i take'
    ]
    
    has_medical_pattern = any(pattern in query_lower for pattern in medical_question_patterns)
    
    # Allow if it clearly has medical context
    if medical_count > 0 or has_medical_pattern:
        return True
    
    # For very short queries, be more lenient but check for obvious non-medical terms
    if len(query.split()) <= 3:
        obvious_non_medical = ['pizza', 'movie', 'game', 'music', 'weather', 'programming']
        if any(term in query_lower for term in obvious_non_medical):
            return False
        return True
    
    # Default: reject if we can't identify it as medical
    return False

def test_guardrail():
    """Test the guardrail function"""
    test_cases = [
        # Medical queries (should return True)
        ('What are diabetes symptoms?', True),
        ('How to treat high blood pressure?', True),
        ('Side effects of aspirin', True),
        ('Is chest pain serious?', True),
        ('Treatment for headache', True),
        ('pregnancy safe medications', True),
        
        # Non-medical queries (should return False)
        ('How to make pizza?', False),
        ('Best movie recommendations', False),
        ('Programming tutorial', False),
        ('Investment advice', False),
        ('Weather forecast', False),
        ('Latest news today', False),
        ('Recipe for cake', False),
        ('Homework help math', False)
    ]
    
    print('🔒 Guardrail Test Results:')
    print('=' * 60)
    correct = 0
    total = len(test_cases)
    
    for query, expected in test_cases:
        result = is_medical_query(query)
        status = '✅' if result == expected else '❌'
        print(f'{status} "{query}" -> {result} (expected: {expected})')
        if result == expected:
            correct += 1
    
    print('=' * 60)
    print(f'Accuracy: {correct}/{total} ({correct/total*100:.1f}%)')
    
    if correct == total:
        print('🎉 All tests passed! Guardrail is working correctly.')
    else:
        print(f'⚠️  {total-correct} tests failed. Review the logic.')

if __name__ == "__main__":
    test_guardrail()
