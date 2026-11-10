// Command sim runs one deterministic, complete Orrery match and prints a
// per-turn event log plus a final standings summary. It exists to prove the
// engine actually is replayable end to end (PLAN.md step 9), not as a
// player-facing tool — the game package is what a future UI, bot, server, or
// replay tool would build on instead.
package main

import (
	"fmt"
	"io"
	"math/rand"
	"os"
	"sort"

	"orrery/game"
)

// simulationSeed fixes Config.Rand so every run of this simulation —
// including the two runs TestRunIsDeterministic launches back to back —
// produces byte-identical output.
const simulationSeed int64 = 1

func main() {
	if err := run(os.Stdout); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
}

// run drives one full simulation from a fresh game and a fresh RNG to
// completion, writing a per-turn event log and a final summary to w. It
// takes an io.Writer rather than assuming os.Stdout so the determinism test
// can drive it against an in-memory buffer instead of comparing two
// subprocess runs' stdout.
func run(w io.Writer) error {
	g, err := newSimulation()
	if err != nil {
		return err
	}
	for !g.Finished() {
		before := takeSnapshot(g)
		actions := actionsForTurn(before.turn)
		if err := g.AdvanceTurn(actions); err != nil {
			return err
		}
		after := takeSnapshot(g)
		events := notableEvents(before, after, actions)
		if g.Finished() {
			events = append(events, fmt.Sprintf("match finished: reason=%s", terminationReason(g)))
		}
		writeTurnEvents(w, before.turn, events)
	}
	writeFinalSummary(w, g)
	return nil
}

// newSimulation builds the match this CLI demonstrates: a fixed,
// deterministic seed and an explicit planet/comet layout, exactly as
// PLAN.md step 9 requires ("create the match from an explicit, fixed planet
// and player setup" / "use a fixed seed for rule-driven randomness such as
// comet spawning"). Each call constructs an independent *rand.Rand, so two
// calls in the same process (as the determinism test makes) never share
// state.
func newSimulation() (*game.Game, error) {
	config := game.DefaultConfig()
	config.Rand = rand.New(rand.NewSource(simulationSeed))
	return game.New(config, []game.Planet{
		{ID: 1, Position: game.Vec2{X: 15, Y: 50}, OwnerID: 0, Ships: 12, Production: 3},
		{ID: 2, Position: game.Vec2{X: 85, Y: 50}, OwnerID: 1, Ships: 12, Production: 3},
		{ID: 3, Position: game.Vec2{X: 50, Y: 80}, OwnerID: game.NeutralPlayer, Ships: 8, Production: 4},
	}, []game.Comet{
		{ID: 4, OwnerID: game.NeutralPlayer, Ships: 5, Path: demoCometPath()},
	})
}

// demoCometPath is a short hand-made comet visit down the board's left side:
// one point per turn, 4 units apart (the default CometSpeed). It lets the
// demo show a comet travelling and expiring well before the first scheduled
// spawn wave at turn 50.
func demoCometPath() []game.Vec2 {
	path := make([]game.Vec2, 15)
	for i := range path {
		path[i] = game.Vec2{X: 5, Y: 22 + 4*float64(i)}
	}
	return path
}

// actionsForTurn returns the actions submitted on the given turn. This
// simulation deliberately submits none: PLAN.md step 9 explicitly allows an
// empty action list, and an empty list guarantees the match reaches the
// turn limit (500) rather than risking early elimination from a scripted
// attack — which matters here because reaching turn 500 is what exercises
// the comet spawn schedule at turns 50/150/250/350/450. A later, scripted
// match only needs to fill this in; the surrounding loop doesn't change.
func actionsForTurn(turn int) []game.Action {
	return nil
}

// snapshot is a point-in-time copy of exactly what notableEvents needs to
// detect what changed this turn — IDs, owners, ship counts — not full
// Planet/Comet/Fleet values. Planets never spawn or expire, so
// planetOwners' key set never changes across a match; comet and fleet
// presence is read directly off cometOwners' and fleetShips' key sets
// instead of tracking separate ID sets.
type snapshot struct {
	turn         int
	planetOwners map[int]int
	cometOwners  map[int]int
	cometShips   map[int]int
	fleetShips   map[int]int
}

func takeSnapshot(g *game.Game) snapshot {
	s := snapshot{
		turn:         g.Turn,
		planetOwners: map[int]int{},
		cometOwners:  map[int]int{},
		cometShips:   map[int]int{},
		fleetShips:   map[int]int{},
	}
	for _, p := range g.Planets {
		s.planetOwners[p.ID] = p.OwnerID
	}
	for _, c := range g.Comets {
		s.cometOwners[c.ID] = c.OwnerID
		s.cometShips[c.ID] = c.Ships
	}
	for _, f := range g.Fleets {
		s.fleetShips[f.ID] = f.Ships
	}
	return s
}

// notableEvents reports every observable fact that changed between before
// and after, in a fixed category order so output is stable turn to turn:
// accepted launches, planet captures, comet captures, comet spawns, comet
// expiries, fleets entering flight, fleets leaving flight.
//
// It deliberately reports observable facts, not inferred causes. A snapshot
// diff cannot tell an arrival apart from a sun collision, a board exit, or a
// sweep collision — every one of them just removes a fleet from g.Fleets
// the same way — so a fleet that stops appearing is reported as having
// "left flight", not as "destroyed" or "arrived", either of which could be
// wrong.
func notableEvents(before, after snapshot, actions []game.Action) []string {
	var events []string

	// Accepted launches are read from actions directly, not diffed from the
	// snapshots: a fleet whose entire path lands inside a body's radius the
	// instant it moves (see moveFleets) never appears in either snapshot's
	// fleet set, so this is the only way its launch is ever reported.
	for _, a := range actions {
		events = append(events, fmt.Sprintf("launch player=%d source=%d ships=%d angle=%.3f", a.PlayerID, a.PlanetID, a.Ships, a.Angle))
	}

	for _, id := range sortedKeys(before.planetOwners) {
		if before.planetOwners[id] != after.planetOwners[id] {
			events = append(events, fmt.Sprintf("planet %d owner %s -> %s", id, ownerLabel(before.planetOwners[id]), ownerLabel(after.planetOwners[id])))
		}
	}

	for _, id := range sortedKeys(before.cometOwners) {
		if _, stillHere := after.cometOwners[id]; !stillHere {
			continue // reported as an expiry below, not a capture
		}
		if before.cometOwners[id] != after.cometOwners[id] {
			events = append(events, fmt.Sprintf("comet %d owner %s -> %s", id, ownerLabel(before.cometOwners[id]), ownerLabel(after.cometOwners[id])))
		}
	}

	for _, id := range idsOnlyIn(after.cometOwners, before.cometOwners) {
		// Every spawned comet starts neutral: spawnComets hardcodes
		// OwnerID: NeutralPlayer (game/comets.go), and it's the only code
		// path that ever creates a new comet ID — so that much is a
		// deduction from an invariant, not a guess. Its ship count is a
		// different matter: spawning runs before fleet movement in
		// AdvanceTurn's order, so an existing fleet can reach the new
		// comet that same turn — and either capture it, or attack and
		// fail (Battle()'s Stage 2 leaves a repelled defender's owner
		// unchanged but its ships reduced, see resolution.go). A comet
		// with no prior snapshot to diff against has no way to tell
		// "never touched" apart from "attacked, still neutral, fewer
		// ships" — ownership alone isn't a reliable signal, so ships are
		// omitted from the spawn line unconditionally rather than
		// reported right only sometimes.
		events = append(events, fmt.Sprintf("comet %d spawned owner=%s", id, ownerLabel(game.NeutralPlayer)))
		if owner := after.cometOwners[id]; owner != game.NeutralPlayer {
			// A capture, unlike the spawn line above, only ever claims
			// to describe state after this transition — after.cometShips[id]
			// is accurate here regardless of what the comet spawned with.
			events = append(events, fmt.Sprintf("comet %d owner %s -> %s ships=%d", id, ownerLabel(game.NeutralPlayer), ownerLabel(owner), after.cometShips[id]))
		}
	}
	for _, id := range idsOnlyIn(before.cometOwners, after.cometOwners) {
		events = append(events, fmt.Sprintf("comet %d expired", id))
	}

	for _, id := range idsOnlyIn(after.fleetShips, before.fleetShips) {
		events = append(events, fmt.Sprintf("fleet %d entered flight ships=%d", id, after.fleetShips[id]))
	}
	for _, id := range idsOnlyIn(before.fleetShips, after.fleetShips) {
		events = append(events, fmt.Sprintf("fleet %d left flight ships=%d", id, before.fleetShips[id]))
	}

	return events
}

// ownerLabel renders an OwnerID the way a player would read it, rather than
// printing NeutralPlayer's -1 sentinel value directly.
func ownerLabel(id int) string {
	if id == game.NeutralPlayer {
		return "neutral"
	}
	return fmt.Sprintf("%d", id)
}

// sortedKeys returns m's keys in ascending order. Go randomizes map
// iteration order per process by design, so iterating any of these maps
// directly would make notableEvents' own output order non-deterministic
// even though the underlying facts are identical run to run.
func sortedKeys(m map[int]int) []int {
	keys := make([]int, 0, len(m))
	for k := range m {
		keys = append(keys, k)
	}
	sort.Ints(keys)
	return keys
}

// idsOnlyIn returns, in ascending order, every key present in a but not b —
// used both directions to find what newly appeared and what disappeared.
func idsOnlyIn(a, b map[int]int) []int {
	var ids []int
	for id := range a {
		if _, ok := b[id]; !ok {
			ids = append(ids, id)
		}
	}
	sort.Ints(ids)
	return ids
}

// writeTurnEvents prints one line or one indented block per turn — always
// something, "no notable events" on a quiet turn — so the log makes clear
// every turn was actually processed rather than silently skipped, and two
// runs can be compared turn for turn without gaps.
func writeTurnEvents(w io.Writer, turn int, events []string) {
	if len(events) == 0 {
		fmt.Fprintf(w, "turn %03d: no notable events\n", turn)
		return
	}
	fmt.Fprintf(w, "turn %03d:\n", turn)
	for _, e := range events {
		fmt.Fprintf(w, "  %s\n", e)
	}
}

// terminationReason reports why a finished match ended. Turn-limit and
// elimination are the only two ways Finished() can become true, so ruling
// out the turn limit against public Config/Turn state is enough to name the
// other without any new engine surface.
func terminationReason(g *game.Game) string {
	if g.Turn >= g.Config.MaxTurns {
		return "turn-limit"
	}
	return "elimination"
}

// writeFinalSummary prints the match's terminal state in full: why it
// ended, every planet, comet, and active fleet (PLAN.md step 9 asks for "a
// concise final state", not just standings), each player's final standing
// (sorted by player ID, since Standings returns an unordered map —
// printing it directly would make output order non-deterministic the same
// way sortedKeys guards against above), and the tied set of leaders Winner
// already returns sorted.
func writeFinalSummary(w io.Writer, g *game.Game) {
	fmt.Fprintf(w, "finished turn=%d reason=%s\n", g.Turn, terminationReason(g))

	planets := append([]game.Planet(nil), g.Planets...)
	sort.Slice(planets, func(i, j int) bool { return planets[i].ID < planets[j].ID })
	fmt.Fprintln(w, "planets:")
	for _, p := range planets {
		fmt.Fprintf(w, "  planet %d owner=%s ships=%d position=(%.3f, %.3f)\n", p.ID, ownerLabel(p.OwnerID), p.Ships, p.Position.X, p.Position.Y)
	}

	comets := append([]game.Comet(nil), g.Comets...)
	sort.Slice(comets, func(i, j int) bool { return comets[i].ID < comets[j].ID })
	fmt.Fprintln(w, "comets:")
	for _, c := range comets {
		fmt.Fprintf(w, "  comet %d owner=%s ships=%d position=(%.3f, %.3f)\n", c.ID, ownerLabel(c.OwnerID), c.Ships, c.Position.X, c.Position.Y)
	}

	fleets := append([]game.Fleet(nil), g.Fleets...)
	sort.Slice(fleets, func(i, j int) bool { return fleets[i].ID < fleets[j].ID })
	fmt.Fprintln(w, "fleets:")
	for _, f := range fleets {
		fmt.Fprintf(w, "  fleet %d owner=%s ships=%d position=(%.3f, %.3f)\n", f.ID, ownerLabel(f.OwnerID), f.Ships, f.Position.X, f.Position.Y)
	}

	fmt.Fprintln(w, "standings:")
	standings := g.Standings()
	for _, id := range sortedKeys(standings) {
		fmt.Fprintf(w, "  player %d: %d\n", id, standings[id])
	}
	leaders, _ := g.Winner()
	fmt.Fprintf(w, "leaders: %v\n", leaders)
}
