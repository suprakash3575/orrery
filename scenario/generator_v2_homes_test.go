package scenario

import (
	"math/rand"
	"slices"
	"strings"
	"testing"

	"orrery/game"
)

func TestAssignV2HomesUsesOneSeededGroup(t *testing.T) {
	config := game.DefaultConfig()
	sawFirstGroup, sawLastGroup := false, false
	for seed := range 512 {
		mapSeed := deriveRNGSeed(int64(seed), v2MapRNGDomain)
		rng := rand.New(rand.NewSource(mapSeed))
		neutral, err := generateV2Groups(rng, config)
		if err != nil {
			t.Fatalf("seed %d: %v", seed, err)
		}
		assigned := slices.Clone(neutral)
		if err := assignV2Homes(rng, assigned, false); err != nil {
			t.Fatalf("seed %d: %v", seed, err)
		}

		first, fourth := -1, -1
		for i, planet := range assigned {
			switch planet.OwnerID {
			case 0:
				if first != -1 {
					t.Fatalf("seed %d assigned player 0 twice", seed)
				}
				first = i
			case 1:
				if fourth != -1 {
					t.Fatalf("seed %d assigned player 1 twice", seed)
				}
				fourth = i
			case game.NeutralPlayer:
			default:
				t.Fatalf("seed %d assigned unexpected owner %d", seed, planet.OwnerID)
			}
		}
		if first < 0 || first%4 != 0 || fourth != first+3 {
			t.Fatalf("seed %d home indices = %d, %d, want first and fourth of one group", seed, first, fourth)
		}
		for i, planet := range assigned {
			want := neutral[i]
			if i == first {
				want.OwnerID, want.Ships = 0, 10
			} else if i == fourth {
				want.OwnerID, want.Ships = 1, 10
			}
			if planet != want {
				t.Fatalf("seed %d planet %d changed beyond its home owner and ships: got %+v, want %+v",
					seed, i, planet, want)
			}
		}

		// Rebuild the map and choose homes from a fresh copy of the same
		// stream. A replay must not depend on the previous game's RNG state.
		replayRNG := rand.New(rand.NewSource(mapSeed))
		replay, err := generateV2Groups(replayRNG, config)
		if err != nil {
			t.Fatalf("replay seed %d: %v", seed, err)
		}
		if err := assignV2Homes(replayRNG, replay, false); err != nil {
			t.Fatalf("replay seed %d: %v", seed, err)
		}
		if !slices.Equal(assigned, replay) {
			t.Fatalf("seed %d chose different homes or map values on replay", seed)
		}
		sawFirstGroup = sawFirstGroup || first == 0
		sawLastGroup = sawLastGroup || first == len(assigned)-4
	}
	if !sawFirstGroup || !sawLastGroup {
		t.Fatalf("home selection did not reach both ends of the group list: first=%v last=%v",
			sawFirstGroup, sawLastGroup)
	}
}

// The four-player variant must seat all four players in the same group the
// two-player variant picks from the same stream, and change nothing else.
func TestAssignV2HomesFourPlayerSeatsWholeGroup(t *testing.T) {
	config := game.DefaultConfig()
	for seed := range 128 {
		mapSeed := deriveRNGSeed(int64(seed), v2MapRNGDomain)
		twoRNG := rand.New(rand.NewSource(mapSeed))
		neutral, err := generateV2Groups(twoRNG, config)
		if err != nil {
			t.Fatalf("seed %d: %v", seed, err)
		}
		fourRNG := rand.New(rand.NewSource(mapSeed))
		if _, err := generateV2Groups(fourRNG, config); err != nil {
			t.Fatalf("seed %d: %v", seed, err)
		}

		two, four := slices.Clone(neutral), slices.Clone(neutral)
		if err := assignV2Homes(twoRNG, two, false); err != nil {
			t.Fatalf("seed %d: %v", seed, err)
		}
		if err := assignV2Homes(fourRNG, four, true); err != nil {
			t.Fatalf("seed %d: %v", seed, err)
		}
		base := slices.IndexFunc(two, func(p game.Planet) bool { return p.OwnerID == 0 })

		for i, planet := range four {
			want := neutral[i]
			if i >= base && i < base+4 {
				want.OwnerID, want.Ships = i-base, 10
			}
			if planet != want {
				t.Fatalf("seed %d planet %d = %+v, want %+v", seed, i, planet, want)
			}
		}
		// Both variants must leave the map stream in the same state.
		if twoRNG.Int63() != fourRNG.Int63() {
			t.Fatalf("seed %d: variants drew different amounts of map randomness", seed)
		}
	}
}

func TestAssignV2HomesRejectsMalformedGroupsWithoutDrawing(t *testing.T) {
	validGroup := v2PlanetGroup(game.Vec2{X: 50, Y: 50}, 70, 80, 3, 12, 1)
	preOwned := append([]game.Planet(nil), validGroup[:]...)
	preOwned[2].OwnerID = 0
	tests := []struct {
		name    string
		planets []game.Planet
		message string
	}{
		{name: "empty", message: "complete groups"},
		{name: "partial group", planets: validGroup[:3], message: "complete groups"},
		{name: "existing owner", planets: preOwned, message: "neutral planets"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			before := slices.Clone(tt.planets)
			rng := rand.New(rand.NewSource(99))
			if err := assignV2Homes(rng, tt.planets, false); err == nil || !strings.Contains(err.Error(), tt.message) {
				t.Fatalf("assignV2Homes error = %v, want %q", err, tt.message)
			}
			if !slices.Equal(tt.planets, before) {
				t.Fatal("invalid home assignment mutated planets")
			}
			if got, want := rng.Int63(), rand.New(rand.NewSource(99)).Int63(); got != want {
				t.Fatalf("invalid home assignment drew randomness: got %d, want %d", got, want)
			}
		})
	}
}
