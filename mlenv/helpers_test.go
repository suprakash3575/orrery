package mlenv

import (
	"math"
	"math/rand"
	"testing"

	"orrery/game"
	"orrery/scenario"
)

// Helpers shared by several test files in this package.

// generated builds the generator-v1 starting board for one seed. Its small,
// still, two-player maps make quick, readable fixtures.
func generated(t *testing.T, seed int64) *game.Game {
	t.Helper()
	g, err := scenario.GeneratorV1{}.New(seed)
	if err != nil {
		t.Fatalf("seed %d: %v", seed, err)
	}
	return g
}

// playRandomTurns advances a game a little so observations are taken from
// mid-game states with fleets in flight and captured planets, not only from
// pristine starting boards where most features hold their initial values.
func playRandomTurns(t *testing.T, g *game.Game, seed int64) {
	t.Helper()
	rng := rand.New(rand.NewSource(seed))
	for range 5 + rng.Intn(20) {
		if g.Finished() {
			return
		}
		var actions []game.Action
		for _, p := range g.Planets {
			if p.OwnerID == game.NeutralPlayer || p.Ships < 2 || rng.Intn(3) != 0 {
				continue
			}
			actions = append(actions, game.Action{
				PlayerID: p.OwnerID,
				PlanetID: p.ID,
				Ships:    1 + rng.Intn(p.Ships-1),
				Angle:    rng.Float64() * 2 * math.Pi,
			})
		}
		if err := g.AdvanceTurn(actions); err != nil {
			t.Fatalf("seed %d: %v", seed, err)
		}
	}
}

// bodyPosition returns the current centre of body id, failing the test when
// the game has no such body.
func bodyPosition(t *testing.T, g *game.Game, id int) game.Vec2 {
	t.Helper()
	for _, body := range collectBodies(g) {
		if body.id == id {
			return body.position
		}
	}
	t.Fatalf("game has no body %d", id)
	return game.Vec2{}
}
