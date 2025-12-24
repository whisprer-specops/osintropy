SECURITY.md
text
# Security Policy

## 🛡️ Supported Versions

We actively maintain and provide security updates for the following versions:

| Version | Status | Support Ends |
|---------|--------|--------------|
| 2.0.x   | ✅ **Active Support** | TBD |
| 1.0.x   | ⚠️ **Security Only** | June 30, 2026 |
| 0.x.x   | ❌ **Unsupported** | December 31, 2025 |

### Support Levels
- **✅ Active Support**: Full bug fixes, security patches, and feature updates
- **⚠️ Security Only**: Critical security vulnerabilities only
- **❌ Unsupported**: No updates provided, use at your own risk

**Recommendation**: Always use the latest 2.x release for maximum security.

---

## 🚨 Reporting a Vulnerability

**We take security seriously.** If you discover a potential security vulnerability in OSINTropy, please follow responsible disclosure practices:

### ⚠️ DO NOT:
- ❌ Open a public GitHub issue for security vulnerabilities
- ❌ Share vulnerability details on social media or forums
- ❌ Exploit the vulnerability maliciously
- ❌ Sell vulnerability information to third parties

### ✅ DO:
1. **Report privately** using one of these methods:
   - **Preferred**: Use GitHub's ["Report a vulnerability"](https://github.com/whisprer-specops/osintropy/security/advisories/new) feature
   - **Email**: security@whispr.dev (PGP key available on request)
   - **Alternative**: Direct message to [@whisprer-specops](https://github.com/whisprer-specops)

2. **Include in your report**:
   - Detailed description of the vulnerability
   - Steps to reproduce
   - Potential impact assessment
   - Suggested fix (if available)
   - Your contact information
   - Whether you want public credit

3. **Allow time for response**:
   - We aim to acknowledge within **24-72 hours**
   - Initial assessment within **7 days**
   - Fix timeline communicated within **14 days**

### Our Commitment
- We will investigate all legitimate reports
- We will keep you informed of progress
- We will credit you publicly (if desired) after fix is released
- We will not pursue legal action against good-faith researchers

---

## 🔐 Security Scope

### What This Tool Secures

OSINTropy focuses on **secure, ethical, and responsible OSINT data gathering**:

✅ **In Scope:**
- Data handling and storage security
- Network communication security (HTTPS, proxy handling)
- Input validation and injection prevention
- Authentication for future API features
- Secure credential storage (if applicable)
- Privacy-preserving data processing
- Rate limiting and anti-abuse measures

### What This Tool Does NOT Guarantee

❌ **Out of Scope:**
- **Physical security**: Cannot prevent hardware-level attacks
- **Target website security**: We are not responsible for third-party site vulnerabilities
- **Legal compliance**: Users must ensure compliance with local laws
- **Data accuracy**: Scraped data quality depends on sources
- **Anonymity**: Not designed as an anonymization tool (use Tor separately)
- **DDoS protection**: Not designed to withstand targeted attacks

---

## ⚠️ Known Limitations

### Current Security Considerations

1. **Scraping Detection**
   - **Risk**: Target websites may block or rate-limit requests
   - **Mitigation**: Built-in rate limiting, proxy rotation available
   - **Status**: Ongoing development

2. **Data Storage**
   - **Risk**: SQLite database stored unencrypted by default
   - **Mitigation**: Use OS-level encryption (BitLocker, LUKS, FileVault)
   - **Future**: Encrypted database option (planned for 2.1.0)

3. **Proxy Security**
   - **Risk**: Proxy credentials stored in plaintext in `proxies.txt`
   - **Mitigation**: Use environment variables or secure credential stores
   - **Future**: Encrypted credential storage (planned for 2.2.0)

4. **Network Traffic**
   - **Risk**: HTTP traffic visible to network observers
   - **Mitigation**: Uses HTTPS where available, supports SOCKS5 proxies
   - **Recommendation**: Use VPN for sensitive investigations

5. **Dependency Vulnerabilities**
   - **Risk**: Third-party libraries may have security issues
   - **Mitigation**: Regular `pip` dependency updates
   - **Status**: Automated security scanning (Dependabot enabled)

---

## 🔒 Security Best Practices

### For Users

#### Data Handling
✅ DO: Use secure database paths
aggregator = OSINTAggregator(db_path='/secure/encrypted/volume/osint.db')

❌ DON'T: Store in publicly accessible locations
aggregator = OSINTAggregator(db_path='C:/Users/Public/osint.db')

text

#### Credential Management
✅ DO: Use environment variables
import os
proxy = os.environ.get('OSINT_PROXY')

❌ DON'T: Hardcode credentials
proxy = 'http://user:password123@proxy.com:8080'

text

#### Input Validation
✅ DO: Validate user input
import re
if not re.match(r'^[a-zA-Z\s-]+$', first_name):
raise ValueError("Invalid name format")

❌ DON'T: Trust all input
result = aggregator.search_person(user_input, user_input2) # Unsafe!

text

### For Developers

#### Code Review Checklist
- [ ] All user input is validated and sanitized
- [ ] SQL queries use parameterized statements (no string concatenation)
- [ ] Secrets are not committed to version control
- [ ] HTTPS is used for all external requests
- [ ] Error messages don't leak sensitive information
- [ ] Dependencies are up to date
- [ ] Security tests are included

#### Secure Coding Examples
✅ GOOD: Parameterized queries
cursor.execute("SELECT * FROM persons WHERE name = ?", (name,))

❌ BAD: String concatenation (SQL injection risk)
cursor.execute(f"SELECT * FROM persons WHERE name = '{name}'")

✅ GOOD: Secure random generation
import secrets
token = secrets.token_urlsafe(32)

❌ BAD: Predictable randomness
import random
token = str(random.randint(0, 999999)) # Don't use for security!

text

---

## 🔍 Security Audits

### Self-Assessment
You are encouraged to:
- **Inspect the source code** before using in sensitive environments
- **Run security scanners** (Bandit, Safety, etc.)
- **Audit dependencies** regularly
- **Test in isolated environments** first

### Automated Scanning
Install security tools
pip install bandit safety

Run static analysis
bandit -r . -ll # Find Python security issues

Check dependencies
safety check # Find known vulnerable packages

text

### Professional Audits
- No formal security audits have been conducted yet
- Contributions for security review are welcome
- Enterprise users should conduct their own audits

---

## 📋 Vulnerability Disclosure Timeline

### Our Process
1. **Report Received** (Day 0)
   - Acknowledge within 24-72 hours
   
2. **Initial Assessment** (Day 1-7)
   - Validate vulnerability
   - Determine severity (using CVSS scoring)
   - Assign priority
   
3. **Fix Development** (Day 7-30)
   - Develop and test patch
   - Create security advisory (private)
   - Prepare release notes
   
4. **Coordinated Disclosure** (Day 30-90)
   - Release patched version
   - Publish security advisory (public)
   - Credit reporter (if desired)
   - Notify affected users

### Severity Levels
| Severity | CVSS Score | Response Time | Example |
|----------|------------|---------------|---------|
| **Critical** | 9.0-10.0 | 24-48 hours | Remote code execution |
| **High** | 7.0-8.9 | 7 days | Authentication bypass |
| **Medium** | 4.0-6.9 | 30 days | Information disclosure |
| **Low** | 0.1-3.9 | 90 days | Minor info leak |

---

## 🛠️ Security Tools & Resources

### Recommended Tools
- **Bandit**: Python AST-based static analyzer
- **Safety**: Dependency vulnerability scanner
- **pip-audit**: Audit Python packages for vulnerabilities
- **Snyk**: Automated vulnerability scanning
- **OWASP ZAP**: Web application security testing

### Security Resources
- [OWASP Top 10](https://owasp.org/www-project-top-ten/)
- [Python Security Best Practices](https://python.readthedocs.io/en/stable/library/security_warnings.html)
- [CWE/SANS Top 25](https://cwe.mitre.org/top25/)
- [NIST Cybersecurity Framework](https://www.nist.gov/cyberframework)

---

## 🏛️ Legal & Ethical Considerations

### Responsible Use
OSINTropy is designed for **authorized security research and legitimate investigations only**.

#### You MUST:
- ✅ Obtain proper authorization before conducting investigations
- ✅ Respect privacy laws (GDPR, CCPA, etc.)
- ✅ Follow responsible disclosure practices
- ✅ Use rate limiting and respect `robots.txt`
- ✅ Secure collected data appropriately
- ✅ Delete data when no longer needed

#### You MUST NOT:
- ❌ Use for illegal surveillance or stalking
- ❌ Violate computer fraud laws (CFAA, Computer Misuse Act)
- ❌ Sell or distribute personal information without consent
- ❌ Bypass security measures or access controls
- ❌ Harass or intimidate individuals

### Liability
**THE AUTHORS AND CONTRIBUTORS ASSUME NO LIABILITY FOR MISUSE OF THIS TOOL.**

Users are solely responsible for:
- Legal compliance in their jurisdiction
- Ethical use of collected data
- Security of their own systems
- Damages resulting from misuse

---

## 📞 Contact

### Security Team
- **Email**: security@whispr.dev
- **PGP Key**: Available on request
- **GitHub**: [@whisprer-secops](https://github.com/whisprer-secops)

### Bug Bounty
We currently do not offer a formal bug bounty program, but we deeply appreciate responsible disclosure and will:
- Publicly credit researchers (if desired)
- Provide swag for significant findings (stickers, t-shirts)
- Consider bounties for critical vulnerabilities on a case-by-case basis

---

## 🔄 Version History

| Version | Release Date | Security Notes |
|---------|--------------|----------------|
| 2.0.0   | 2025-12-19 | Enhanced proxy security, input validation |
| 1.0.0   | 2024-10-15 | Initial release |

---

## ✅ Verification

### Source Code Integrity
Before using OSINTropy in production:

1. **Verify the source**:
git clone https://github.com/whisprer-specops/osintropy.git
cd osintropy
git log --show-signature # Check commit signatures

text

2. **Inspect the code**:
- Review all modules before running
- Check for suspicious network activity
- Audit database queries

3. **Build from source**:
Use virtual environment
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

Run tests
python tests/run_tests.py

text

4. **Monitor runtime**:
- Use network monitoring tools
- Check outbound connections
- Review log files

---

## 📚 Additional Resources

- **Security Policy**: This document
- **Contribution Guidelines**: [CONTRIBUTING.md](CONTRIBUTING.md)
- **License**: [LICENSE](LICENSE) (MIT)
- **Changelog**: [CHANGELOG.md](CHANGELOG.md)

---

**Last Updated**: December 21, 2025  
**Version**: 2.0.0  
**Status**: Active

**Stay safe, stay ethical, stay legal! 🔒**

---

*Questions about security? Open a [security advisory](https://github.com/whisprer-specops/osintropy/security/advisories/new) or contact security@whispr.dev*