# DESIGN.md

## Design System Direction
Lumen Workspace uses a product register optimized for enterprise operations: dense information layout, low visual noise, clear focus states, and strict tokenized theming.

## Visual Strategy
- Pattern: Enterprise gateway + data-dense dashboard.
- Information hierarchy: utility first, decoration second.
- Interaction language: explicit states (`ready`, `blocked`, `degraded`, `error`) with semantic color tokens.

## Color System
- Tokens only via `lumen/channels/templates/_partials/tokens.html`.
- No raw color literals in page-level styles unless temporary migration.
- Theme parity required:
  - Light: high readability on neutral paper.
  - Dark: high readability without glow-heavy effects.

## Typography
- Primary UI font: `var(--lumen-font)`.
- Mono for technical labels and identifiers: `var(--lumen-font-mono)`.
- Body text minimum: 14px desktop, 16px equivalent touch readability on mobile UI sections.

## Components
- Sidebar navigation: active by background + color only (no thick side stripe accents).
- Cards: single-level surfaces, avoid nested-card overload.
- Tables: compact spacing with preserved readability.
- Buttons and inputs: visible focus ring using tokenized focus styles.

## Accessibility Baseline
- Contrast target: WCAG AA minimum.
- Keyboard navigation required for all interactive controls.
- Icon-only controls must expose accessible labels.
- Motion must respect `prefers-reduced-motion` where animation is non-essential.

## Performance Baseline
- No repeated page-level font imports when already provided by shared tokens.
- Avoid expensive layout-driven animations.
- Keep transitions short and purposeful (150-300ms).

## Responsive Baseline
- No horizontal overflow at 375px.
- Touch targets at least 44x44 where interaction is primary.
- Sidebar behavior must degrade cleanly on narrow viewports.

## Governance UI Rules
- Workspace governance view must use the same settings shell and theme model.
- User data rendered in DOM must be escaped by default.
- Role-gated screens must fail closed (`403`) when not authorized.

## Anti-Patterns to Avoid
- Emoji icons as primary UI icons.
- Gradient text headlines.
- Decorative glassmorphism as default container style.
- Hardcoded one-off colors that bypass the token system.
