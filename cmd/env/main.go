// Command env exposes the batched reinforcement-learning environment over the
// env-v3 binary protocol. Standard output is reserved
// exclusively for protocol responses; startup and fatal I/O diagnostics go to
// standard error.
package main

import (
	"context"
	"flag"
	"fmt"
	"io"
	"os"
	"os/signal"
	"runtime/debug"
	"strings"

	"orrery/mlenv"
)

// Exit statuses returned by run. A usage error is kept apart from a serving
// error so a caller can tell "this command line can never work" from "the
// session failed part-way".
const (
	exitOK         = 0
	exitServeError = 1
	exitUsage      = 2
)

const usage = "usage: env [--players 2|4]"

func main() {
	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt)
	status := run(ctx, os.Args[1:], os.Stdin, os.Stdout, os.Stderr)
	// os.Exit skips deferred calls, so release the signal handler first.
	stop()
	os.Exit(status)
}

// run is everything main does apart from process setup, so the test helper
// process can run the exact startup path, flags included. Its stages are
// ordered so a bad command line is rejected before stdin is read: a caller
// that keeps stdin open still gets an immediate exit instead of a server
// waiting for its first frame.
func run(ctx context.Context, arguments []string, stdin io.Reader, stdout, stderr io.Writer) int {
	// Stage 1: read the command line. Nothing has touched stdin or stdout.
	players, err := parseArguments(arguments)
	if err != nil {
		fmt.Fprintln(stderr, err)
		return exitUsage
	}
	// Stage 2: choose the map generator. A count the parser accepted as a
	// number can still be one the environment does not support, and that is
	// a usage error too.
	config, err := mlenv.DefaultProtocolConfig(engineVersion(), players)
	if err != nil {
		fmt.Fprintf(stderr, "%v\n%s\n", err, usage)
		return exitUsage
	}
	// Stage 3: serve frames until close, EOF, or interrupt. Game execution
	// stays inside mlenv.Batch; this only connects it to the process's pipes.
	if err := mlenv.ServeBinaryProtocol(ctx, stdin, stdout, config); err != nil {
		fmt.Fprintln(stderr, err)
		return exitServeError
	}
	return exitOK
}

// parseArguments reads the player count, 2 by default. It checks only that
// the value is a number; DefaultProtocolConfig decides which counts are
// supported, so the 2-or-4 rule lives in one place. env-v3 is the only
// protocol, so there is no flag to name one.
func parseArguments(arguments []string) (int, error) {
	flags := flag.NewFlagSet("env", flag.ContinueOnError)
	// The flag package would print its own message and usage here; discard
	// that and let run print one message, so stderr says each thing once.
	flags.SetOutput(io.Discard)
	players := flags.Int("players", 2, "number of players on each map: 2 or 4")
	if err := flags.Parse(arguments); err != nil {
		return 0, fmt.Errorf("%w\n%s", err, usage)
	}
	if flags.NArg() > 0 {
		return 0, fmt.Errorf("unexpected argument %q\n%s", flags.Arg(0), usage)
	}
	return *players, nil
}

// engineVersion reports which code this binary was built from. Training
// records it in every checkpoint, and a resume refuses a different one, so it
// must tell code versions apart.
func engineVersion() string {
	info, ok := debug.ReadBuildInfo()
	if !ok {
		info = nil
	}
	return chooseEngineVersion(archivedDescribe, archivedCommit, info)
}

// archivePlaceholder starts every placeholder git archive fills in. It is
// built from two parts so this file never contains a placeholder itself.
const archivePlaceholder = "$" + "Format:"

// chooseEngineVersion picks the most exact identity available, in order:
//
//  1. The commit stamped by git archive (see release.go), with its nearest
//     tag: "v0.1.0+<commit>", or "untagged+<commit>" when no tag exists. A
//     Vast.ai bundle has no .git folder, so this is its only source.
//  2. The commit the Go toolchain embedded when building from a checkout. A
//     "+dirty" suffix is important replay metadata: the named commit alone
//     does not exactly describe a binary built with uncommitted changes.
//  3. The module version, for release builds without VCS settings.
//  4. "development", when none of these exist.
//
// describe and commit are the two archived values; info is the build
// information, or nil when it is unavailable. It is a separate function so
// tests can supply each case without building a binary.
func chooseEngineVersion(describe, commit string, info *debug.BuildInfo) string {
	// Stage 1: archived values count only once git archive replaced them. An
	// empty describe is a real answer (no tag), not a missing one.
	if commit != "" && !strings.HasPrefix(commit, archivePlaceholder) &&
		!strings.HasPrefix(describe, archivePlaceholder) {
		if describe == "" {
			describe = "untagged"
		}
		return describe + "+" + commit
	}
	if info == nil {
		return "development"
	}
	// Stage 2 and 3: what the Go toolchain recorded.
	revision := ""
	modified := false
	for _, setting := range info.Settings {
		switch setting.Key {
		case "vcs.revision":
			revision = setting.Value
		case "vcs.modified":
			modified = setting.Value == "true"
		}
	}
	if revision != "" {
		if modified {
			return revision + "+dirty"
		}
		return revision
	}
	if info.Main.Version != "" && info.Main.Version != "(devel)" {
		return info.Main.Version
	}
	return "development"
}
