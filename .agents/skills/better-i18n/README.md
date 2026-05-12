```
  ╔══════════════════════════════════════════╗
  ║   _  _  ___        _  _   _              ║
  ║  (_)/ |( _ ) _ __ | || | | |             ║
  ║  | || |/ _ \| '_ \| || |_| |             ║
  ║  | || | (_) | | | |__   _|_|             ║
  ║  |_||_|\___/|_| |_|  |_| (_)             ║
  ║                                          ║
  ║           Best Practices                 ║
  ╚══════════════════════════════════════════╝
```

# i18n Best Practices Skill

A comprehensive agent skill for building production-ready internationalization systems with Better i18n platform.

## Installation

```bash
npx skills add better-i18n/skills
```

## What This Skill Covers

### Getting Started
- Project configuration (`i18n.config.ts`)
- Source and target languages setup
- CDN-first vs GitHub-connected workflows

### CLI Usage
- `better-i18n scan` - Discover i18n keys in codebase
- `better-i18n check` - Validate translations
- `better-i18n sync` - Sync with platform

### Key Management
- Naming conventions and namespaces
- Translation statuses and workflows
- Bulk operations and search patterns

### Translation
- AI-assisted translation with glossary context
- ICU MessageFormat (plurals, dates, numbers)
- Quality review and approval workflows

### Integration
- GitHub sync with AST-based key discovery
- CDN delivery (`cdn.better-i18n.com`)
- MCP tools for AI coding assistants
- SDK integration (Next.js, React, TanStack Start)

### Best Practices
- RTL language support
- Accessibility considerations
- Performance optimization

## Structure

```
best-practices/
├── SKILL.md                    # Routing table
└── resources/
    ├── getting-started.md      # Project setup
    ├── cli-usage.md            # CLI commands
    ├── key-management.md       # Keys and namespaces
    ├── ai-translation.md       # AI translation
    ├── github-sync.md          # GitHub integration
    ├── cdn-delivery.md         # CDN setup
    ├── mcp-integration.md      # MCP tools
    ├── sdk-integration.md      # Framework SDKs
    └── best-practices.md       # Formatting, RTL
```

## Quick Start

Open [SKILL.md](./SKILL.md) - routes to the right resource based on your task.

## Links

- [Documentation](https://docs.better-i18n.com)
- [Dashboard](https://better-i18n.com)
- [CLI](https://npmjs.com/package/better-i18n)
- [MCP Server](https://npmjs.com/package/@better-i18n/mcp-server)

## License

MIT
