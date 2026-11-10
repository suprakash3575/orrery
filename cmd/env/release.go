package main

// The two constants below are filled in by git archive when the runtime bundle
// is built: .gitattributes marks this file export-subst, so git replaces each
// placeholder with the archived commit's details. In an ordinary checkout they
// stay exactly as written, and engineVersion falls back to the commit the Go
// toolchain embeds instead.
//
// git archive replaces every placeholder anywhere in this file, so keep this
// file to these two constants. The code that checks whether they were filled
// in lives in main.go.
const (
	// The nearest tag reachable from the archived commit, such as v0.1.0 or
	// v0.1.0-3-gabc1234 three commits later; empty when no tag exists.
	archivedDescribe = "$Format:%(describe:tags)$"
	// The archived commit's full 40-character hash.
	archivedCommit = "$Format:%H$"
)
