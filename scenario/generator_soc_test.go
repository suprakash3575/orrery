package scenario

import (
	"reflect"
	"testing"

	"orrery/game"
)

// The factory must construct the complete two-player variant from one seed.
// Rebuilding and advancing it checks the public replay boundary, beyond the
// private group and home helpers tested elsewhere.
func TestGeneratorV2BuildsFullRulesGame(t *testing.T) {
	factory := GeneratorV2{}
	if got := factory.ID(); got != "generator-v2" {
		t.Fatalf("generator ID = %q, want generator-v2", got)
	}
	for _, seed := range []int64{0, 1, 42, 999} {
		first, err := factory.New(seed)
		if err != nil {
			t.Fatalf("seed %d: %v", seed, err)
		}
		second, err := factory.New(seed)
		if err != nil {
			t.Fatalf("replay seed %d: %v", seed, err)
		}
		if !reflect.DeepEqual(first.Planets, second.Planets) || first.OrbitRate != second.OrbitRate {
			t.Fatalf("seed %d rebuilt a different initial map or orbit rate", seed)
		}
		if !first.Config.SunHazardEnabled || !first.Config.OrbitsEnabled || !first.Config.CometSpawningEnabled {
			t.Fatalf("seed %d disabled a full-rules switch", seed)
		}
		if first.OrbitRate < game.MinOrbitRate || first.OrbitRate >= game.MaxOrbitRate || len(first.Comets) != 0 {
			t.Fatalf("seed %d has orbit rate %v or %d initial comets", seed, first.OrbitRate, len(first.Comets))
		}
		if n := len(first.Planets); n < 20 || n > 40 || n%4 != 0 {
			t.Fatalf("seed %d has %d planets, want 5..10 groups of four", seed, n)
		}

		home0, home1 := -1, -1
		orbitingGroups := 0
		for i, planet := range first.Planets {
			switch planet.OwnerID {
			case 0:
				if home0 != -1 {
					t.Fatalf("seed %d assigned player 0 more than one home", seed)
				}
				home0 = i
				if planet.Ships != 10 {
					t.Fatalf("seed %d player 0 has %d home ships", seed, planet.Ships)
				}
			case 1:
				if home1 != -1 {
					t.Fatalf("seed %d assigned player 1 more than one home", seed)
				}
				home1 = i
				if planet.Ships != 10 {
					t.Fatalf("seed %d player 1 has %d home ships", seed, planet.Ships)
				}
			case game.NeutralPlayer:
			default:
				t.Fatalf("seed %d has unexpected owner %d", seed, planet.OwnerID)
			}
			if i%4 == 0 && planet.Orbit != nil {
				orbitingGroups++
			}
		}
		if home0 < 0 || home0%4 != 0 || home1 != home0+3 || orbitingGroups == 0 {
			t.Fatalf("seed %d home positions = %d, %d and orbiting groups = %d",
				seed, home0, home1, orbitingGroups)
		}

		// The engine assigns movement and production from the new map. Both
		// independent games must finish the turn in the same public state.
		if err := first.AdvanceTurn(nil); err != nil {
			t.Fatalf("seed %d first turn: %v", seed, err)
		}
		if err := second.AdvanceTurn(nil); err != nil {
			t.Fatalf("seed %d replay turn: %v", seed, err)
		}
		if first.Turn != 1 || second.Turn != 1 || !reflect.DeepEqual(first.Planets, second.Planets) ||
			!reflect.DeepEqual(first.Fleets, second.Fleets) {
			t.Fatalf("seed %d diverged after one turn", seed)
		}
	}
}

// The four-player factory must build the two-player map for the same seed,
// with the whole home group owned by players 0 to 3, and keep all four
// players alive through a turn.
func TestGeneratorV2FourPlayerBuildsSameMapWithFourHomes(t *testing.T) {
	factory := GeneratorV2{FourPlayer: true}
	if got := factory.ID(); got != "generator-v2-4p" {
		t.Fatalf("generator ID = %q, want generator-v2-4p", got)
	}
	for _, seed := range []int64{0, 1, 42, 999} {
		two, err := (GeneratorV2{}).New(seed)
		if err != nil {
			t.Fatalf("seed %d: %v", seed, err)
		}
		four, err := factory.New(seed)
		if err != nil {
			t.Fatalf("seed %d: %v", seed, err)
		}
		if four.OrbitRate != two.OrbitRate || len(four.Planets) != len(two.Planets) {
			t.Fatalf("seed %d changed the orbit rate or planet count", seed)
		}

		base := -1
		for i, planet := range two.Planets {
			if planet.OwnerID == 0 {
				base = i
			}
		}
		for i, planet := range four.Planets {
			want := two.Planets[i]
			if i >= base && i < base+4 {
				want.OwnerID, want.Ships = i-base, 10
			}
			if !reflect.DeepEqual(planet, want) {
				t.Fatalf("seed %d planet %d = %+v, want %+v", seed, i, planet, want)
			}
		}

		if err := four.AdvanceTurn(nil); err != nil {
			t.Fatalf("seed %d first turn: %v", seed, err)
		}
		if four.Finished() || len(four.Standings()) != 4 {
			t.Fatalf("seed %d: finished = %v, standings = %+v, want four live players",
				seed, four.Finished(), four.Standings())
		}
	}
}

func TestGeneratorV2ScheduledCometsReplay(t *testing.T) {
	const seed int64 = 42
	first, err := (GeneratorV2{}).New(seed)
	if err != nil {
		t.Fatal(err)
	}
	second, err := (GeneratorV2{}).New(seed)
	if err != nil {
		t.Fatal(err)
	}
	// Turn 50 is the first scheduled spawn. Running empty turns keeps both
	// homes alive while allowing the engine's own comet path logic to run.
	for range 51 {
		if err := first.AdvanceTurn(nil); err != nil {
			t.Fatal(err)
		}
		if err := second.AdvanceTurn(nil); err != nil {
			t.Fatal(err)
		}
	}
	if len(first.Comets) != 4 {
		t.Fatalf("turn 51 has %d comets, want the first wave of four", len(first.Comets))
	}
	if !reflect.DeepEqual(first.Comets, second.Comets) || !reflect.DeepEqual(first.Planets, second.Planets) {
		t.Fatal("same seed diverged after the first scheduled comet wave")
	}
}

// The domain names and derived seeds are part of V2's replay key. Pinning them
// now catches accidental changes before completed V2 maps are recorded.
func TestGeneratorV2RNGDomainsAreStableAndSeparate(t *testing.T) {
	const seed int64 = -17
	tests := []struct {
		name   string
		domain string
		want   int64
	}{
		{name: "map", domain: v2MapRNGDomain, want: -1126456454629610260},
		{name: "engine", domain: v2EngineRNGDomain, want: -1725287402043182456},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := deriveRNGSeed(seed, tt.domain); got != tt.want {
				t.Errorf("derived seed = %d, want %d", got, tt.want)
			}
		})
	}

	mapSeed := deriveRNGSeed(seed, v2MapRNGDomain)
	engineSeed := deriveRNGSeed(seed, v2EngineRNGDomain)
	if mapSeed == engineSeed || mapSeed == deriveRNGSeed(seed, mapRNGDomain) ||
		engineSeed == deriveRNGSeed(seed, engineRNGDomain) {
		t.Fatal("V2 map and engine streams must be distinct from each other and V1")
	}
}

// TestGeneratorV2CometWavesFitRealMaps runs the turn-50 comet wave on 50
// generator-v2 maps. It checks that planet avoidance still leaves most waves
// able to spawn on real crowded maps, and that every spawned path keeps the
// path rules: 5 to 40 points, starting and ending on the board, and never
// overlapping a static planet. The overlap check here is deliberately looser
// than the engine's own clearance, so it does not just repeat that code.
func TestGeneratorV2CometWavesFitRealMaps(t *testing.T) {
	const seeds = 50
	found := 0
	for seed := range int64(seeds) {
		g, err := (GeneratorV2{}).New(seed)
		if err != nil {
			t.Fatal(err)
		}
		g.Turn = 50
		if err := g.AdvanceTurn(nil); err != nil {
			t.Fatal(err)
		}
		if len(g.Comets) == 0 {
			continue
		}
		found++
		for _, c := range g.Comets {
			if n := len(c.Path); n < 5 || n > 40 {
				t.Fatalf("seed %d: comet %d path has %d points, want 5 to 40", seed, c.ID, n)
			}
			for _, end := range [2]game.Vec2{c.Path[0], c.Path[len(c.Path)-1]} {
				if end.X < 0 || end.X > g.Config.Width || end.Y < 0 || end.Y > g.Config.Height {
					t.Fatalf("seed %d: comet %d path end %v is off the board", seed, c.ID, end)
				}
			}
			for _, p := range g.Planets {
				if p.Orbit != nil {
					continue
				}
				for _, point := range c.Path {
					if point.Distance(p.Position) < p.Radius+1 {
						t.Fatalf("seed %d: comet %d passes over static planet %d at %v", seed, c.ID, p.ID, point)
					}
				}
			}
		}
	}
	// The benchmark found a path for 97% of waves; allow some slack below that.
	if found < seeds*8/10 {
		t.Fatalf("only %d of %d generator-v2 maps spawned a comet wave, want at least 80%%", found, seeds)
	}
}

// BenchmarkGeneratorV2CometSpawnTurn times the turn-50 spawn turn on
// generator-v2 maps, which is where comet path search runs. Each iteration
// builds a fresh map outside the timer and jumps it to turn 50, so the timed
// work is one whole turn including that wave's search. found/op is the share
// of waves that found a path: a wave that finds none runs every try, so a low
// share means slow spawn turns.
func BenchmarkGeneratorV2CometSpawnTurn(b *testing.B) {
	factory := GeneratorV2{}
	found := 0
	b.ReportAllocs()
	for i := range b.N {
		b.StopTimer()
		g, err := factory.New(int64(i))
		if err != nil {
			b.Fatal(err)
		}
		g.Turn = 50
		b.StartTimer()
		if err := g.AdvanceTurn(nil); err != nil {
			b.Fatal(err)
		}
		if len(g.Comets) > 0 {
			found++
		}
	}
	b.ReportMetric(float64(found)/float64(b.N), "found/op")
}
