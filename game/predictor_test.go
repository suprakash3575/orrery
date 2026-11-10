package game

import (
	"math"
	"testing"
)

func TestNewPredictorRejectsInvalidSnapshots(t *testing.T) {
	// valid returns a small board that NewPredictor accepts; each case breaks
	// exactly one thing about it.
	valid := func() Snapshot {
		return Snapshot{
			Width: 100, Height: 100, Sun: Vec2{50, 50}, SunRadius: 10, SunHazardEnabled: true, MaxFleetSpeed: 6,
			Planets: []Planet{{ID: 1, OwnerID: 0, Ships: 10, Production: 2, Radius: PlanetRadius(2), Position: Vec2{20, 20}}},
			Comets:  []Comet{{ID: 2, OwnerID: NeutralPlayer, Radius: 1, Production: 1, Path: []Vec2{{10, 10}, {14, 10}}, Position: Vec2{10, 10}}},
			Fleets:  []Fleet{{ID: 3, OwnerID: 1, Ships: 5, Position: Vec2{30, 30}, Velocity: Vec2{1, 0}}},
		}
	}
	tests := []struct {
		name   string
		mutate func(*Snapshot)
	}{
		{"non-finite sun", func(s *Snapshot) { s.Sun.X = math.NaN() }},
		{"zero width", func(s *Snapshot) { s.Width = 0 }},
		{"max fleet speed below 1", func(s *Snapshot) { s.MaxFleetSpeed = 0.5 }},
		{"negative turn", func(s *Snapshot) { s.Turn = -1 }},
		{"ID shared by a planet and a comet", func(s *Snapshot) { s.Comets[0].ID = 1 }},
		{"unknown planet owner", func(s *Snapshot) { s.Planets[0].OwnerID = 7 }},
		{"non-finite orbit", func(s *Snapshot) {
			s.Planets[0].Orbit = &Orbit{Radius: 30, RadiansTurn: math.Inf(1)}
		}},
		{"comet index past its path", func(s *Snapshot) { s.Comets[0].PathIndex = 2 }},
		{"non-finite comet path point", func(s *Snapshot) { s.Comets[0].Path[1].Y = math.NaN() }},
		{"neutral fleet", func(s *Snapshot) { s.Fleets[0].OwnerID = NeutralPlayer }},
		{"empty fleet", func(s *Snapshot) { s.Fleets[0].Ships = 0 }},
	}

	if _, err := NewPredictor(valid()); err != nil {
		t.Fatalf("NewPredictor(valid) = %v, want no error", err)
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			s := valid()
			tt.mutate(&s)
			if _, err := NewPredictor(s); err == nil {
				t.Fatal("NewPredictor accepted an invalid snapshot")
			}
		})
	}
}
