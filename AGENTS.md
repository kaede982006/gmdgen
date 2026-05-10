## AGENTS.md

This file contains critical guidance for OpenCode agents to avoid mistakes and ramp up quickly in this repository.

### High-Signal Repo-Specific Guidance
- **Build/Test Order**: Always run `lint` → `typecheck` → `test` to ensure consistency.
- **Monorepo Structure**: `apps/` contains entrypoints; `packages/` holds reusable libraries.
- **Toolchain Quirks**: Exclude `src-gen/` from linting/formatting (generated code).
- **Testing**: Integration tests require a running database; use `npm run db:setup` first.
- **Environment**: Use `.env.local` for dev variables (not committed to version control).
- **Commands**: Use `npm run test -- --test-name-pattern=SomeTest` to run individual tests.
- **Existing Sources**: Verify against `opencode.json` for instruction file references.