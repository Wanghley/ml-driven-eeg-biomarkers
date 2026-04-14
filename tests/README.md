# Tests

Unit tests and integration tests for the ML-Driven EEG Biomarkers project.

## Directory Structure

```
tests/
├── __init__.py
├── test_filters.py          # Unit tests for filter implementations
├── test_preprocessing.py    # Unit tests for preprocessing pipeline
└── README.md (this file)
```

## Running Tests

### Run All Tests
```bash
python -m pytest tests/ -v
```

### Run Specific Test File
```bash
python -m pytest tests/test_filters.py -v
```

### Run Specific Test
```bash
python -m pytest tests/test_filters.py::TestHampelFilter -v
```

### Run with Coverage
```bash
python -m pytest tests/ --cov=src --cov-report=html
```

## Test Files

### `test_filters.py`
Tests for low-level filter implementations:
- `TimeHampelFilter` - Time-domain Hampel filtering
- `FrequencyHampelFilter` - Frequency-domain Hampel filtering
- `SpectralInterpolationFilter` - Spectrum-fit harmonic removal
- `ZaplineFilter` - Zapline+ method
- `BaselineReferencedFilter` - Wiener with baseline reference
- `ArtifactFilterFactory` - Method routing

Tests check:
- Correct output shapes
- Parameter validation
- Numerical correctness
- Edge cases

### `test_preprocessing.py`
Tests for high-level preprocessing API:
- `EEGPreprocessor` class
- Data loading and preprocessing
- Artifact removal convenience methods
- Feature extraction
- ICA application

Tests check:
- Pipeline consistency
- Auto-detection from filenames
- Method routing
- Result validity

## Writing New Tests

### Test Structure
```python
import pytest
import mne
from src.preprocessing import EEGPreprocessor

class TestNewFeature:
    def setup_method(self):
        # Setup code here
        pass
    
    def test_something(self):
        # Test code
        assert result == expected
    
    def test_edge_case(self):
        # Edge case test
        with pytest.raises(ValueError):
            bad_function_call()
```

### Fixtures
Common fixtures available:
```python
@pytest.fixture
def sample_raw():
    """Return a sample MNE Raw object"""
    return mne.io.read_raw_edf("data/raw/test_sample.edf")
```

### Mocking
Use `unittest.mock` for isolated tests:
```python
from unittest.mock import Mock, patch

def test_with_mock():
    with patch('src.filters.read_raw_edf') as mock_read:
        mock_read.return_value = Mock()
        # Test code
```

## Coverage Goals

Target coverage:
- `src/filters.py`: > 90%
- `src/preprocessing.py`: > 85%
- Overall: > 80%

Check coverage:
```bash
pytest --cov=src --cov-report=term-missing
```

## CI/CD Integration

Tests should run on:
- Every commit (pre-commit hook)
- Pull requests (GitHub Actions)
- Before releases

## Common Issues

### Issue: Test fails with "No such file"
**Solution**: Ensure test data exists at `data/test_samples/`

### Issue: MNE import errors
**Solution**: Install dev dependencies
```bash
pip install -r requirements.txt
pip install pytest pytest-cov
```

### Issue: Tests run slowly
**Solution**: Use mocking or smaller sample data

## Test Data

Place minimal test data in:
```
data/test_samples/
├── sample_awake.edf    # Small test EDF file
└── sample_sleep.edf
```

Keep files small (< 10 MB) for fast tests.

## Contributing Tests

When adding new features:
1. Write tests first (TDD approach recommended)
2. Ensure all tests pass
3. Check coverage
4. Document test purpose in docstring

## Continuous Testing

Monitor test health:
```bash
# Quick smoke test
pytest tests/ -x  # Stop on first failure

# Full run with verbose output
pytest tests/ -vv

# Generate HTML report
pytest tests/ --html=report.html
```

## Notes

- Tests should be independent and isolated
- Use fixtures for shared setup
- Mock external dependencies
- Keep tests fast (< 100ms each)
- Document complex test logic

## Resources

- [pytest documentation](https://docs.pytest.org/)
- [MNE testing guide](https://mne.tools/stable/development/testing.html)
- [unittest.mock documentation](https://docs.python.org/3/library/unittest.mock.html)
