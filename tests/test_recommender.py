"""
Core functionality tests for the movie recommendation engine.
"""

import pytest
from src.recommender import recommend

@pytest.mark.core
def test_single_movie_returns_results(valid_single_movie):
    """
    Validates that recommending for a single movie returns exactly 10 
    valid results without including the input movie itself.
    """
    results = recommend(valid_single_movie)
    
    assert isinstance(results, list), "Result should be a list"
    assert len(results) == 10, f"Expected 10 results, got {len(results)}"
    
    input_title = valid_single_movie[0]
    for result in results:
        # Check structure
        assert all(k in result for k in ("rank", "title", "score", "reason")), "Missing keys in result"
        # Check exclusion
        assert result["title"] != input_title, f"Input movie {input_title} found in recommendations"
        # Check threshold (0.05 is default)
        assert result["score"] > 0.05, f"Score {result['score']} below expected threshold of 0.05"

@pytest.mark.core
def test_two_movies_returns_results(valid_two_movies):
    """
    Validates recommendations for two movies, ensuring they are sorted by score
    descending and exclude both input movies.
    """
    results = recommend(valid_two_movies)
    
    assert isinstance(results, list)
    assert len(results) <= 10
    
    for result in results:
        assert result["title"] not in valid_two_movies, f"Input movie {result['title']} found in recommendations"

    # Check descending sort
    scores = [r["score"] for r in results]
    assert all(scores[i] >= scores[i+1] for i in range(len(scores)-1)), "Results not sorted by score descending"

@pytest.mark.core
def test_three_movies_returns_results(valid_three_movies):
    """
    Validates recommendations for three movies, checking ranking sequence 
    and ensuring all recommendations are relevant (score > 0).
    """
    results = recommend(valid_three_movies)
    
    assert isinstance(results, list)
    assert len(results) <= 10
    
    if results:
        assert all(r["score"] > 0.0 for r in results), "Found recommendation with non-positive score"
        
        # Check sequential ranks
        ranks = [r["rank"] for r in results]
        expected_ranks = list(range(1, len(results) + 1))
        assert ranks == expected_ranks, f"Expected ranks {expected_ranks}, got {ranks}"
        assert len(results) == results[-1]["rank"], "Last item rank does not match list length"

@pytest.mark.core
def test_five_movies_returns_results(valid_five_movies):
    """
    Validates recommendations for the maximum allowed limit of 5 movies,
    checking for data integrity and presence of explanations (reasons).
    """
    results = recommend(valid_five_movies)
    
    assert isinstance(results, list)
    assert len(results) <= 10
    
    for result in results:
        assert result["title"] not in valid_five_movies, f"Input movie {result['title']} found in recommendations"
        assert result["reason"].strip() != "", "Reason field is empty"
        assert isinstance(result["score"], float), f"Score {result['score']} is not a float"

@pytest.mark.invalid
def test_invalid_movie_title_skipped():
    """Validates graceful handling of completely unknown titles"""
    results = recommend(["This Movie Does Not Exist (9999)"])
    assert results == [], "Should return an empty list for unknown titles"

@pytest.mark.invalid
def test_mixed_valid_and_invalid_titles():
    """Validates that valid titles still work when mixed with invalid ones"""
    results = recommend(["Star Wars (1977)", "Fake Movie (0000)"])
    assert isinstance(results, list)
    assert len(results) > 0, "Should still return results for the valid portion of the input"
    for result in results:
        assert result["title"] != "Fake Movie (0000)", "Invalid movie title should not appear in results"

@pytest.mark.invalid
def test_invalid_title_warning_printed(capsys):
    """Validates that invalid titles produce visible feedback"""
    recommend(["Completely Fake Title"])
    captured = capsys.readouterr()
    assert "Warning: Movie 'Completely Fake Title' not found" in captured.out

@pytest.mark.edge
def test_empty_input_raises_error():
    """Validates that empty input never silently returns empty list"""
    with pytest.raises(ValueError):
        recommend([])

@pytest.mark.edge
def test_duplicate_titles_handled():
    """Validates duplicate input titles are handled gracefully"""
    results = recommend(["Star Wars (1977)", "Star Wars (1977)"])
    assert isinstance(results, list)
    assert len(results) > 0
    # Check no duplicates in results
    titles = [r["title"] for r in results]
    assert len(titles) == len(set(titles)), "Found duplicate titles in recommendations"

@pytest.mark.edge
def test_partial_title_not_matched():
    """Validates partial titles without year are not matched"""
    results = recommend(["Star Wars"])
    assert results == []

@pytest.mark.edge
def test_top_n_respected(valid_single_movie):
    """Validates top_n parameter limits result count correctly"""
    results = recommend(valid_single_movie, top_n=5)
    assert len(results) <= 5

@pytest.mark.edge
def test_top_n_max_is_ten(valid_single_movie):
    """Validates results never exceed 10 as per assignment requirement"""
    results = recommend(valid_single_movie, top_n=100)
    assert len(results) <= 10

@pytest.mark.edge
def test_reason_field_populated(valid_three_movies):
    """Validates bonus requirement — every recommendation has an explanation"""
    results = recommend(valid_three_movies)
    for result in results:
        reason = result["reason"]
        assert reason.startswith("Shares genres") or reason == "Highly rated by users who liked your input movies"

@pytest.mark.assignment
def test_assignment_max_input_is_five():
    """Assignment requires input of UP TO 5 movies — validates this boundary"""
    titles = [
        "Star Wars (1977)", "Fargo (1996)", "Toy Story (1995)",
        "Schindler's List (1993)", "Pulp Fiction (1994)", "Casablanca (1942)"
    ]
    with pytest.raises(ValueError, match="Maximum 5 input titles allowed"):
        recommend(titles)

@pytest.mark.assignment
def test_assignment_max_output_is_ten(valid_five_movies):
    """Assignment requires UP TO 10 recommendations — validates this boundary"""
    results = recommend(valid_five_movies, top_n=20)
    assert 0 < len(results) <= 10, f"Expected between 1 and 10 results, got {len(results)}"

@pytest.mark.assignment
def test_assignment_bonus_reason_exists(valid_three_movies):
    """Validates bonus requirement — every recommendation explains why it was suggested"""
    results = recommend(valid_three_movies)
    assert len(results) > 0
    for result in results:
        reason = result.get("reason")
        assert reason is not None, "Reason field is missing"
        assert isinstance(reason, str), "Reason field should be a string"
        assert reason.strip() != "", "Reason field is empty"

@pytest.mark.assignment
def test_assignment_ranks_are_valid(valid_single_movie):
    """Validates results are properly ranked from 1 to N"""
    results = recommend(valid_single_movie)
    assert results[0]["rank"] == 1
    assert results[-1]["rank"] == len(results)
    
    ranks = [r["rank"] for r in results]
    assert ranks == list(range(1, len(results) + 1)), "Ranks are not consecutive"

@pytest.mark.assignment
def test_assignment_scores_are_meaningful(valid_two_movies):
    """Validates similarity scores are meaningful and properly ordered"""
    results = recommend(valid_two_movies)
    if len(results) > 1:
        assert results[0]["score"] > results[-1]["score"], "Top result score should be higher than bottom"
        
        # Check adjacent scores are not identical to ensure meaningful differentiation
        # We check the top pair specifically
        assert results[0]["score"] != results[1]["score"], "Top two results have identical scores"

    for r in results:
        assert 0.0 <= r["score"] <= 1.0, f"Score {r['score']} out of range [0, 1]"

@pytest.mark.assignment
def test_assignment_no_self_recommendation(valid_five_movies):
    """Validates the engine never recommends an input movie back to the user"""
    results = recommend(valid_five_movies)
    result_titles = [r["title"] for r in results]
    for input_title in valid_five_movies:
        assert input_title not in result_titles, f"Input movie {input_title} was recommended"
