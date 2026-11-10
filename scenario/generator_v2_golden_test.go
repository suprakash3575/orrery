package scenario

import (
	"crypto/sha256"
	"encoding/json"
	"fmt"
	"testing"

	"orrery/game"
)

// v2GoldenPlanet contains the starting values that a scenario seed promises
// to reproduce. Orbiting records the engine's classification; the shared
// orbit rate is stored once beside the planet list in the initial-state hash.
type v2GoldenPlanet struct {
	ID         int
	X, Y       float64
	OwnerID    int
	Ships      int
	Production int
	Orbiting   bool
}

func v2GoldenMapProjection(planets []game.Planet) []v2GoldenPlanet {
	projected := make([]v2GoldenPlanet, len(planets))
	for i, planet := range planets {
		projected[i] = v2GoldenPlanet{
			ID:         planet.ID,
			X:          planet.Position.X,
			Y:          planet.Position.Y,
			OwnerID:    planet.OwnerID,
			Ships:      planet.Ships,
			Production: planet.Production,
			Orbiting:   planet.Orbit != nil,
		}
	}
	return projected
}

func v2InitialHash(t *testing.T, g *game.Game) string {
	t.Helper()
	initial := struct {
		Generator string
		OrbitRate float64
		Planets   []v2GoldenPlanet
	}{Generator: GeneratorV2ID, OrbitRate: g.OrbitRate, Planets: v2GoldenMapProjection(g.Planets)}
	encoded, err := json.Marshal(initial)
	if err != nil {
		t.Fatal(err)
	}
	return fmt.Sprintf("%x", sha256.Sum256(encoded))
}

// Golden hashes protect the public (generator version, seed) replay mapping.
// A change to positions, owners, ships, production, orbit classification, or
// shared orbit rate must be an intentional new generator or engine version.
func TestGeneratorV2GoldenInitialStates(t *testing.T) {
	tests := []struct {
		seed int64
		want string
	}{
		{seed: 0, want: "fe341032b7a9157555d9a402f7292d43c2eed92f42863e0eef75f86b926f5069"},
		{seed: 1, want: "40825e3b2ad957203eaaf39a05de69d879180494f792a1851d181ca6de1dc063"},
		{seed: 42, want: "f8df8fb9958c0bedf24cf22e6c20a5aeb3c99e7f8b3f43cb99b0b8b62f33cfb6"},
		{seed: -17, want: "26e1f43e34154049856f795e596049787c16a07a172939dc51304759bf6dc616"},
	}
	for _, tt := range tests {
		g, err := (GeneratorV2{}).New(tt.seed)
		if err != nil {
			t.Fatalf("seed %d: %v", tt.seed, err)
		}
		if got := v2InitialHash(t, g); got != tt.want {
			t.Errorf("seed %d initial hash = %s, want %s", tt.seed, got, tt.want)
		}
	}
}

func TestGeneratorV2MapDistributionHasVariety(t *testing.T) {
	seen := make(map[[32]byte]int, 1000)
	var groupCounts [6]int
	var productionCounts [5]int
	staticHomes, orbitingHomes := 0, 0
	for seed := range 1000 {
		g, err := (GeneratorV2{}).New(int64(seed))
		if err != nil {
			t.Fatalf("seed %d: %v", seed, err)
		}
		// Hash only map values here: including the independent orbit-rate
		// draw could make a repeated layout appear unique.
		encoded, err := json.Marshal(v2GoldenMapProjection(g.Planets))
		if err != nil {
			t.Fatal(err)
		}
		hash := sha256.Sum256(encoded)
		if prior, duplicate := seen[hash]; duplicate {
			t.Fatalf("seeds %d and %d produced the same planet layout", prior, seed)
		}
		seen[hash] = seed
		groupCounts[len(g.Planets)/4-5]++
		for i := 0; i < len(g.Planets); i += 4 {
			productionCounts[g.Planets[i].Production-1]++
			if g.Planets[i].OwnerID == 0 {
				if g.Planets[i].Orbit == nil {
					staticHomes++
				} else {
					orbitingHomes++
				}
			}
		}
	}
	for i, count := range groupCounts {
		if count == 0 {
			t.Errorf("no sampled map had %d groups", i+5)
		}
	}
	for i, count := range productionCounts {
		if count == 0 {
			t.Errorf("no sampled group had production %d", i+1)
		}
	}
	if staticHomes == 0 || orbitingHomes == 0 {
		t.Errorf("home selections covered %d static and %d orbiting groups", staticHomes, orbitingHomes)
	}
}
