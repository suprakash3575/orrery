# Pull requests and reviews

Read this before branching, pushing, writing a PR summary, or reviewing code.
It extends [AGENTS.md](../../AGENTS.md).

## Pull request process

One PR at a time, each requiring explicit approval before the next is started.

1. **Branch from a freshly fetched `main`,** not from the previous PR's
   branch. Earlier PRs merge into `main` first, and a local `origin/main` can
   be stale — `git fetch` before branching. Commit or stash unrelated
   working-tree changes first; resetting onto a new base discards them
   silently.
2. **Implement and commit** as in AGENTS.md's "Required working practices",
   keeping the branch to the scope of one PR from any plan. Unrelated changes
   that appear on the branch belong on their own branch — call them out rather
   than letting them ride along in the diff.
3. **Run the full verification block** (see "Verification" in AGENTS.md) and
   confirm it is clean before pushing.
4. **Push only when the user asks.** `git push -u origin <branch>`.
5. **Write the PR summary** in the format below and hand it over for the user
   to open the PR with. Do not open, merge, or close PRs without being asked.

## Choosing a commit type

Use `docs` when a change is confined to documentation — a Markdown file, or
comments with no behaviour change — and `chore` for other housekeeping such
as build configuration or tooling. A commit that changes code is never a
`docs` commit even when it also updates a comment; by AGENTS.md's rule to
commit each logical change separately, the documentation update is a separate
commit anyway.

## PR summary format

Two sections, `Summary` then `Test plan`. Return the complete PR summary inside
one fenced `markdown` code block so the user can copy and paste it directly
into GitHub without reformatting it.

`Summary` is a flat bullet list, one bullet per change, each stating what
changed and why it matters — not a file-by-file inventory. Any defect found
during review gets its own bullet beginning `Fixed in review:`, naming what
was wrong rather than only what the fix was. Note dependencies on earlier PRs
and anything deliberately left for a later one.

`Test plan` is a checkbox list of the verification commands actually run:

```markdown
## Summary

- Adds ... so that ...
- Fixed in review: ... originally ..., which masked ... — now ...

## Test plan

- [ ] `gofmt -l .` — no output
- [ ] `go build ./...`
- [ ] `go vet ./...`
- [ ] `go test ./...`
```

## Review expectations

Code reviews should:

1. Lead with findings ordered by severity.
2. Cite concrete files and lines.
3. Distinguish correctness blockers from performance suggestions.
4. Review the code generally unless the user asks for validation against a
   plan or specification. If comparison may be relevant but the intended plan
   or specification is unclear, ask the user which one to use.
5. Call out missing regression tests.
6. State what was tested.
7. Include a refactoring and maintainability assessment. Clearly separate
   necessary refactors from optional improvements, and state when the current
   design is already good enough.

If no findings exist, say so explicitly and mention any remaining testing or
scope limitations.
