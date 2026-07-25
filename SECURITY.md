# Security Policy

## Supported Versions

| Version | Supported          |
| ------- | ------------------ |
| 1.0.x   | ✅ Active support  |
| < 1.0   | ❌ No support      |

## Reporting a Vulnerability

If you discover a security vulnerability, please **do NOT** open a public issue.

Instead, please report it via [GitHub Security Advisories](https://github.com/Fengrru/kimi-attention/security/advisories/new) or email to the project maintainers. We take all
security issues seriously and will respond promptly.

### What to Include

- A clear description of the vulnerability
- Steps to reproduce or proof-of-concept code
- Potential impact of the vulnerability
- Any suggestions for mitigation (if available)

### Response Timeline

- **Acknowledgment**: Within 48 hours
- **Assessment**: Within 5 business days
- **Fix & Release**: Depends on severity; critical issues prioritized

## Security Best Practices for Users

- Always use the latest stable release
- Verify model checkpoint integrity before loading
- Do not load untrusted checkpoint files (`torch.load` with `weights_only=True`)
- Keep PyTorch and all dependencies updated to their latest secure versions

## Disclosure Policy

We follow a coordinated disclosure process:

1. Reporter submits vulnerability privately
2. We acknowledge and assess within 48 hours
3. We develop and test a fix
4. We release a patch and publish an advisory
5. We credit the reporter (if desired)
