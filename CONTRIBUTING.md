# Contributing to InferPlan

Thanks for your interest in improving InferPlan! Contributions of all
sizes are welcome — a tightened formula, a new model preset, a bug fix, or
a docs typo. This guide gets you set up and explains the few conventions
that keep the project auditable.

By contributing, you agree that your contributions are licensed under the
project's [Apache-2.0](LICENSE) license.

## Getting started

Requires Node 20+ (CI builds on Node 24).

```sh
npm install
npm run dev     # local dev server at http://localhost:5173/inferplan/
npm test        # engine + URL round-trip tests (vitest)
npm run lint    # oxlint
npm run build   # static production build in dist/
```

The stack is Vite + React + TypeScript + Tailwind. There is no backend —
every calculation runs client-side.

## How the codebase is laid out

| Path | What lives there |
|------|------------------|
| `src/engine/` | The calculation engine — **pure TypeScript, no React imports.** Unit-tested and reusable independently of the UI. |
| `src/engine/engine.test.ts` | The formula tests. |
| `src/data/models.ts` | Model presets. |
| `src/data/gpus.ts` | GPU database. |
| `src/state/url.ts` | The shareable-URL encoding (a compatibility surface). |
| `src/components/` | React UI. |
| `docs/FORMULAS.md` | Every formula, with citations. |

## The conventions that matter

These are what keep InferPlan a trustworthy calculator rather than a box of
magic numbers. PRs are reviewed against them:

1. **The engine stays pure.** Nothing in `src/engine/` may import React or
   touch the DOM. It is meant to be extractable as a standalone package.

2. **Every formula change gets a test.** Add or update a case in
   `src/engine/engine.test.ts`, and pin it to a published or measured
   number wherever one exists (a paper, a vendor benchmark, a datasheet).
   `docs/FORMULAS.md` is the citation index — update it in the same PR.

3. **Old share links must keep working.** URL parameters are a
   compatibility surface. If you add or rename one, keep existing links
   parsing and add a round-trip case to `src/state/url.test.ts`.

4. **UI changes are checked in both themes.** InferPlan ships a light and a
   dark theme; verify your change in both before opening the PR.

## Adding a model or GPU

Often the best first contribution. Any Hugging Face model can already be
imported at runtime, so a preset is only for convenience or for
architectures the importer can't infer.

- **Model:** add an entry to `src/data/models.ts`. Cite the Hugging Face
  repo so reviewers can check parameter counts against the safetensors
  index.
- **GPU:** add an entry to `src/data/gpus.ts` with VRAM, memory bandwidth
  (GB/s), and dense bf16 TFLOPS, and cite the datasheet.

## Submitting a pull request

1. Fork and branch from `main`.
2. Make your change, with tests and docs as above.
3. Run `npm test` and `npm run lint`.
4. Open the PR and fill in the template. Keep it focused — one logical
   change per PR is easiest to review.

Pushing to `main` deploys to GitHub Pages, and tests gate the deploy, so a
green test suite is required to merge.

## Reporting bugs and requesting features

Use the [issue templates](https://github.com/rabrooks/inferplan/issues/new/choose).
For bugs, the single most useful thing you can include is a **share-config
link** (the app encodes the whole configuration in the URL) so we can
reproduce your exact numbers.

## Code of conduct

This project follows the [Contributor Covenant](CODE_OF_CONDUCT.md). By
participating you are expected to uphold it.
