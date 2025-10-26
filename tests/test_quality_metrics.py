import pytest
from main import QualityMetrics

class TestQualityMetrics:
    """Unit tests for quality metrics calculation"""
    
    def test_completeness_perfect_sentence(self):
        """Test completeness calculation with proper sentences"""
        text = "This is a complete sentence with proper structure. It has subject and verb."
        result = QualityMetrics.calculate_completeness(text)
        assert 0 <= result <= 1
        assert result > 0.5  # Should recognize complete sentences
        
    def test_completeness_incomplete(self):
        """Test completeness with incomplete fragments"""
        text = "Incomplete fragment"
        result = QualityMetrics.calculate_completeness(text)
        assert 0 <= result <= 1
        
    def test_completeness_empty(self):
        """Test completeness with empty text"""
        text = ""
        result = QualityMetrics.calculate_completeness(text)
        assert result == 0.0
        
    def test_coherence_structured(self):
        """Test coherence with well-structured text"""
        text = "First sentence is clear. Second sentence follows logically. Third sentence concludes."
        result = QualityMetrics.calculate_coherence(text)
        assert 0 <= result <= 1
        assert result > 0.5  # Should show good coherence
        
    def test_coherence_single_sentence(self):
        """Test coherence with single sentence"""
        text = "Single sentence."
        result = QualityMetrics.calculate_coherence(text)
        assert result == 1.0  # Single sentence should be 100% coherent
        
    def test_coherence_fragmented(self):
        """Test coherence with random sentences"""
        text = "Apple. Elephant. Computer. Jump."
        result = QualityMetrics.calculate_coherence(text)
        assert 0 <= result <= 1
        assert result < 0.5  # Should show low coherence
        
    def test_creativity_diverse_vocabulary(self):
        """Test creativity with diverse vocabulary"""
        text = "innovation creativity diversity exploration discovery adventure"
        result = QualityMetrics.calculate_creativity(text)
        assert 0 <= result <= 1
        
    def test_creativity_repetitive(self):
        """Test creativity with repetitive text"""
        text = "the the the the the the"
        result = QualityMetrics.calculate_creativity(text)
        assert 0 <= result <= 1
        assert result < 0.5  # Should show low creativity
        
    def test_relevance_keyword_match(self):
        """Test relevance with keyword matching"""
        text = "quantum computing is revolutionary technology"
        prompt = "quantum computing"
        result = QualityMetrics.calculate_relevance(text, prompt)
        assert 0 <= result <= 1
        assert result > 0.5  # Should show high relevance
        
    def test_relevance_no_match(self):
        """Test relevance with no keyword match"""
        text = "completely different topic about cooking"
        prompt = "quantum computing"
        result = QualityMetrics.calculate_relevance(text, prompt)
        assert 0 <= result <= 1
        assert result < 0.5  # Should show low relevance
        
    def test_relevance_empty_prompt(self):
        """Test relevance with empty prompt"""
        text = "Some text"
        prompt = ""
        result = QualityMetrics.calculate_relevance(text, prompt)
        assert result == 1.0  # Empty prompt should return 1.0
        
    def test_overall_score_balanced(self):
        """Test overall score calculation"""
        text = "This is a complete, coherent, creative, and relevant response."
        prompt = "response"
        result = QualityMetrics.calculate_overall_score(text, prompt)
        
        assert 0 <= result['overall'] <= 1
        assert 'completeness' in result
        assert 'coherence' in result
        assert 'creativity' in result
        assert 'relevance' in result
        
        # Check all metrics are between 0 and 1
        for key, value in result.items():
            assert 0 <= value <= 1
            
    def test_overall_score_structure(self):
        """Test overall score has correct structure"""
        text = "Test text"
        prompt = "test"
        result = QualityMetrics.calculate_overall_score(text, prompt)
        
        expected_keys = ['completeness', 'coherence', 'creativity', 'relevance', 'overall']
        assert all(key in result for key in expected_keys)
        
    def test_coherence_edge_cases(self):
        """Test coherence with edge cases"""
        # Very long text
        long_text = ". ".join(["Sentence"] * 100)
        result = QualityMetrics.calculate_coherence(long_text)
        assert 0 <= result <= 1
        
        # Single word
        result = QualityMetrics.calculate_coherence("Word")
        assert 0 <= result <= 1
        
    def test_creativity_alliteration(self):
        """Test creativity detects alliteration"""
        text = "big blue birds bounce beautifully"
        result = QualityMetrics.calculate_creativity(text)
        assert 0 <= result <= 1
        
    def test_relevance_question_prompt(self):
        """Test relevance with question prompt"""
        text = "This is an answer to your question."
        prompt = "What is the answer?"
        result = QualityMetrics.calculate_relevance(text, prompt)
        assert 0 <= result <= 1
        
if __name__ == "__main__":
    pytest.main([__file__, "-v"])