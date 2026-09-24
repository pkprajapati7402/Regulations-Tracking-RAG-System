# CI

`github-actions-ci.yml` is the GitHub Actions workflow for this project. It is kept here
rather than in `.github/workflows/` because the bot that created this branch does not hold
the `workflows` permission. To enable CI, move it into place and push from your own account:

```bash
mkdir -p .github/workflows
git mv ci/github-actions-ci.yml .github/workflows/ci.yml
git commit -m "Enable CI" && git push
```

It runs, on every push and PR: the test-suite, an index build from the seed corpus,
the retrieval evaluation track, and the amendment case study.
