CONTRIBUTING.md
text
# Contributing to OSINTropy

Thanks for your interest in improving **OSINTropy** — the entropy-based OSINT intelligence platform!

We welcome contributions from security researchers, developers, and OSINT practitioners.

---

## 🎯 Code of Conduct

By participating in this project, you agree to:
- **Be respectful** of other contributors and maintainers
- **Follow ethical OSINT practices** — no illegal use
- **Respect privacy** — handle data responsibly
- **Collaborate constructively** — focus on improving the tool

---

## 🛠️ Code Style

### Python Standards
- Follow **PEP 8** style guidelines
- Use **type hints** where appropriate (Python 3.8+)
- Run `black` formatter before commits:
pip install black
black .

text
- Ensure all warnings are resolved:
flake8 --max-line-length=100

text
- Keep dependencies minimal — justify any new requirements

### Documentation
- Use **Google-style docstrings**:
def search_person(first_name: str, last_name: str) -> Dict[str, Any]:
"""
Search for a person across all configured sources.

text
  Args:
      first_name: Person's first name
      last_name: Person's last name
      
  Returns:
      Dictionary containing aggregated results
      
  Raises:
      ValueError: If names are invalid
  """
text

---

## 🔄 Development Workflow

### 1. Fork the Repository
Click "Fork" on GitHub to create your own copy.

### 2. Clone Your Fork
git clone https://github.com/YOUR-USERNAME/osintropy.git
cd osintropy/src

text

### 3. Create a Feature Branch
git checkout -b feature/your-feature-name

Examples:
feature/linkedin-scraper
fix/proxy-rotation-bug
docs/api-reference
text

### 4. Set Up Development Environment
Create virtual environment
python -m venv .venv
source .venv/bin/activate # Windows: .venv\Scripts\activate

Install dependencies + dev tools
pip install -r requirements.txt
pip install -r requirements-dev.txt # pytest, black, flake8, etc.

text

### 5. Make Your Changes
- Write clean, documented code
- Add **unit tests** for new features
- Update documentation (README, docstrings)
- Keep commits atomic and focused

### 6. Commit Changes
Use **conventional commit messages**:
Format: <type>(<scope>): <subject>
Examples:
git commit -m "feat(scrapers): Add LinkedIn profile scraper"
git commit -m "fix(proxy): Handle connection timeout gracefully"
git commit -m "docs(readme): Add visualization tutorial"
git commit -m "test(network): Add graph clustering tests"
git commit -m "refactor(entropy): Optimize Shannon calculation"

text

**Commit Types:**
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code formatting (no logic changes)
- `refactor`: Code restructuring (no behavior change)
- `test`: Adding/updating tests
- `chore`: Maintenance tasks

### 7. Push to Your Fork
git push origin feature/your-feature-name

text

### 8. Open a Pull Request
- Go to the [original repository](https://github.com/whisprer-specops/osintropy)
- Click "New Pull Request"
- Select your branch
- Fill out the PR template (see below)

---

## 🧪 Testing

### Run All Tests
From src/ directory
python tests/run_tests.py

text

### Run Specific Test Modules
python -m unittest tests.test_scrapers
python -m unittest tests.test_aggregation
python -m unittest tests.test_analysis
python -m unittest tests.test_utils

text

### Run with Coverage
pip install pytest pytest-cov
pytest --cov=. --cov-report=html tests/

View coverage report
Open htmlcov/index.html in browser
text

### Test Requirements
- **All new features** must have unit tests
- Maintain **>80% code coverage**
- Tests must pass before PR is merged
- Include edge cases and error conditions

### Manual Verification
For scraper changes:
1. Test against real websites (respectfully!)
2. Verify rate limiting works
3. Check proxy rotation
4. Test with invalid inputs
5. Verify error handling

---

## 📝 Documentation

### When to Update Docs

**You MUST update documentation if you:**
- Add a new feature or module
- Change API behavior
- Add/modify configuration options
- Fix a significant bug
- Add new scrapers or exporters

### What to Update
- **README.md** — High-level features and usage
- **Docstrings** — All public functions/classes
- **CHANGELOG.md** — Version history (see below)
- **API docs** (if applicable)

### Writing Good Documentation
✅ GOOD
def calculate_entropy(data: List[str], normalize: bool = True) -> float:
"""
Calculate Shannon entropy of a data distribution.

text
Shannon entropy measures the average information content in bits.
Higher values indicate more randomness/uncertainty.

Args:
    data: List of data values to analyze
    normalize: If True, return value between 0-1 instead of bits
    
Returns:
    Entropy score (0.0-1.0 if normalized, else in bits)
    
Example:
    >>> calculate_entropy(['A', 'A', 'B', 'C'])
    0.565  # Moderately diverse
"""
❌ BAD
def calculate_entropy(data, normalize=True):
# calculates entropy
pass

text

---

## 🚀 Pull Request Process

### PR Checklist
Before submitting, ensure:
- [ ] All tests pass (`python tests/run_tests.py`)
- [ ] Code follows PEP 8 style (`black .`)
- [ ] Added tests for new features
- [ ] Updated documentation
- [ ] Updated CHANGELOG.md (if significant change)
- [ ] No new warnings (`flake8`)
- [ ] Commits follow conventional format
- [ ] PR description clearly explains changes

### PR Template
When opening a PR, include:

Description
Brief summary of changes (1-2 sentences)

Motivation
Why is this change needed? What problem does it solve?

Changes Made
Added X feature

Fixed Y bug

Refactored Z module

Testing
How was this tested?

 Unit tests added/updated

 Manual testing performed

 All tests passing

Screenshots (if applicable)
Add screenshots for UI/visualization changes

Related Issues
Fixes #123
Relates to #456

Checklist
 Tests pass

 Documentation updated

 CHANGELOG updated

 Code formatted with black

text

### Review Process
1. **Automated checks** run (tests, linting)
2. **Maintainer review** (may request changes)
3. **Discussion** if needed
4. **Approval** and merge
5. **Thank you!** 🎉

---

## 🐛 Reporting Bugs

### Before Reporting
1. **Search existing issues** — it may already be reported
2. **Try latest version** — bug may be fixed
3. **Check documentation** — might be expected behavior

### Bug Report Template
Describe the bug
Clear description of what's wrong

To Reproduce
Steps to reproduce:

Run command X

With parameters Y

See error Z

Expected behavior
What should happen?

Actual behavior
What actually happened?

Environment

OS: [e.g., Windows 11, Ubuntu 22.04]

Python version: [e.g., 3.11.2]

OSINTropy version: [e.g., 2.0.0]

Error logs

python
Paste relevant error messages/stack traces
Additional context
Any other relevant information

text

---

## 💡 Feature Requests

We love new ideas! Open an issue with:

Feature Description
What do you want to add/change?

Use Case
Why is this useful? Who benefits?

Proposed Solution
How might this be implemented?

Alternatives Considered
Other ways to achieve this?

Additional Context
Screenshots, mockups, examples

text

---

## 🎓 First-Time Contributors

New to open source? Welcome! Here's how to get started:

### Good First Issues
Look for issues tagged:
- `good-first-issue` — Easy entry points
- `help-wanted` — Community assistance needed
- `documentation` — Doc improvements

### Getting Help
- Ask in the issue comments
- Join discussions
- Read existing PRs for examples
- Don't hesitate to ask questions!

### Learning Resources
- [How to Contribute to Open Source](https://opensource.guide/how-to-contribute/)
- [PEP 8 Style Guide](https://peps.python.org/pep-0008/)
- [Git Branching](https://learngitbranching.js.org/)
- [Python Type Hints](https://docs.python.org/3/library/typing.html)

---

## 📞 Communication

### GitHub Issues
- **Bug reports** — Something broken?
- **Feature requests** — New ideas?
- **Questions** — Need help?
- **Security issues** — See [SECURITY.md](SECURITY.md)

### Discussions
Use [GitHub Discussions](https://github.com/whisprer-specops/osintropy/discussions) for:
- General questions
- Architecture discussions
- Showing off your use cases
- Community chat

### Response Times
- **Critical bugs**: 24-48 hours
- **Regular issues**: 3-7 days
- **PRs**: 7-14 days
- **Questions**: Best effort

---

## 🏆 Recognition

Contributors are recognized in:
- **CHANGELOG.md** — Listed in version credits
- **README.md** — Contributors section
- **GitHub contributors graph**

Top contributors may be invited to become maintainers!

---

## ⚖️ Legal

By contributing, you agree that:
- Your contributions are your original work
- You license contributions under MIT License
- You have rights to submit the contribution
- You follow ethical OSINT practices

---

## 🙏 Thank You!

Every contribution helps make OSINTropy better for the security research community.

**Thanks for helping build the future of OSINT! 🔍✨**

---

*Questions? Open an issue or start a discussion!*

**Built with 🧠 by the OSINT community**