package agent

import (
	"context"
	"errors"
	"fmt"
	"math"
	"math/rand"
	"reflect"
	"strings"
	"testing"

	"orrery/game"
)

func TestRandomFactoryValidation(t *testing.T) {
	valid := RandomConfig{LaunchProbability: 0.4, MinFraction: 0.2, MaxFraction: 0.8}
	tests := []struct {
		name   string
		mutate func(*RandomConfig)
	}{
		{"negative launch probability", func(config *RandomConfig) { config.LaunchProbability = -0.1 }},
		{"large launch probability", func(config *RandomConfig) { config.LaunchProbability = 1.1 }},
		{"nan launch probability", func(config *RandomConfig) { config.LaunchProbability = math.NaN() }},
		{"infinite minimum", func(config *RandomConfig) { config.MinFraction = math.Inf(1) }},
		{"negative minimum", func(config *RandomConfig) { config.MinFraction = -0.1 }},
		{"large maximum", func(config *RandomConfig) { config.MaxFraction = 1.1 }},
		{"reversed fractions", func(config *RandomConfig) { config.MinFraction, config.MaxFraction = 0.9, 0.2 }},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			config := valid
			test.mutate(&config)
			if _, err := NewRandomFactory(config); err == nil {
				t.Fatal("NewRandomFactory accepted invalid parameters")
			}
		})
	}

	factory, err := NewRandomFactory(valid)
	if err != nil {
		t.Fatal(err)
	}
	if got, want := factory.Spec().String(), "random-v1:launch_probability=0.4,max_fraction=0.8,min_fraction=0.2"; got != want {
		t.Fatalf("Spec = %q, want %q", got, want)
	}
}

func TestHeuristicFactoryValidation(t *testing.T) {
	valid := HeuristicConfig{Aggression: 1.2, Economy: 2, Efficiency: 0.5, Defense: 0.25}
	tests := []struct {
		name   string
		mutate func(*HeuristicConfig)
	}{
		{"negative aggression", func(config *HeuristicConfig) { config.Aggression = -0.1 }},
		{"nan economy", func(config *HeuristicConfig) { config.Economy = math.NaN() }},
		{"infinite efficiency", func(config *HeuristicConfig) { config.Efficiency = math.Inf(1) }},
		{"negative defense", func(config *HeuristicConfig) { config.Defense = -0.1 }},
		{"large defense", func(config *HeuristicConfig) { config.Defense = 1.1 }},
		{"nan defense", func(config *HeuristicConfig) { config.Defense = math.NaN() }},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			config := valid
			test.mutate(&config)
			if _, err := NewHeuristicFactory(config); err == nil {
				t.Fatal("NewHeuristicFactory accepted invalid parameters")
			}
		})
	}

	factory, err := NewHeuristicFactory(valid)
	if err != nil {
		t.Fatal(err)
	}
	if got, want := factory.Spec().String(), "heuristic-v1:aggression=1.2,defense=0.25,economy=2,efficiency=0.5"; got != want {
		t.Fatalf("Spec = %q, want %q", got, want)
	}
}

func TestScriptedFactoriesNormalizeSignedZero(t *testing.T) {
	negativeZero := math.Copysign(0, -1)
	randomFactory, err := NewRandomFactory(RandomConfig{MaxFraction: negativeZero})
	if err != nil {
		t.Fatal(err)
	}
	if strings.Contains(randomFactory.Spec().String(), "-0") {
		t.Fatalf("random Spec contains signed zero: %q", randomFactory.Spec().String())
	}

	heuristicFactory, err := NewHeuristicFactory(HeuristicConfig{Aggression: negativeZero})
	if err != nil {
		t.Fatal(err)
	}
	if strings.Contains(heuristicFactory.Spec().String(), "-0") {
		t.Fatalf("heuristic Spec contains signed zero: %q", heuristicFactory.Spec().String())
	}
}

func TestRandomAgentIsDeterministicAndLegalAcrossRandomViews(t *testing.T) {
	factory, err := NewRandomFactory(RandomConfig{
		LaunchProbability: 0.65,
		MinFraction:       0,
		MaxFraction:       1,
	})
	if err != nil {
		t.Fatal(err)
	}
	viewRNG := rand.New(rand.NewSource(451))

	for caseIndex := range 200 {
		view := randomizedScriptedView(viewRNG)
		if caseIndex%2 == 0 {
			reversePlanetsAndComets(&view)
		}
		init := AgentInit{AgentSeed: int64(9000 + caseIndex)}
		first, err := factory.New(context.Background(), init)
		if err != nil {
			t.Fatal(err)
		}
		second, err := factory.New(context.Background(), init)
		if err != nil {
			t.Fatal(err)
		}
		request := TurnRequest{PlayerID: 1, View: view}
		firstDecision, err := first.Decide(context.Background(), request)
		if err != nil {
			t.Fatal(err)
		}
		secondDecision, err := second.Decide(context.Background(), request)
		if err != nil {
			t.Fatal(err)
		}
		if !reflect.DeepEqual(firstDecision, secondDecision) {
			t.Fatalf("case %d decisions differ:\nfirst  %#v\nsecond %#v", caseIndex, firstDecision, secondDecision)
		}
		assertLegalScriptedDecision(t, request, firstDecision, false, 0)
	}
}

func TestRandomAgentUsesGlobalBodyIDOrder(t *testing.T) {
	factory, err := NewRandomFactory(RandomConfig{LaunchProbability: 1, MinFraction: 0.5, MaxFraction: 0.5})
	if err != nil {
		t.Fatal(err)
	}
	view := View{
		Planets: []game.Planet{{ID: 8, OwnerID: 1, Ships: 8}, {ID: 2, OwnerID: 1, Ships: 8}},
		Comets:  []game.Comet{{ID: 5, OwnerID: 1, Ships: 8, Path: []game.Vec2{{}}}},
	}
	created, err := factory.New(context.Background(), AgentInit{AgentSeed: 1})
	if err != nil {
		t.Fatal(err)
	}
	decision, err := created.Decide(context.Background(), TurnRequest{PlayerID: 1, View: view})
	if err != nil {
		t.Fatal(err)
	}
	if got, want := actionSourceIDs(decision.Actions), []int{2, 5, 8}; !reflect.DeepEqual(got, want) {
		t.Fatalf("source IDs = %v, want %v", got, want)
	}
}

func TestHeuristicAgentScoreDefenseAndTieBreak(t *testing.T) {
	factory, err := NewHeuristicFactory(HeuristicConfig{
		Aggression: 1,
		Economy:    1,
		Efficiency: 1,
		Defense:    0.21,
	})
	if err != nil {
		t.Fatal(err)
	}
	created, err := factory.New(context.Background(), AgentInit{})
	if err != nil {
		t.Fatal(err)
	}
	view := View{
		Config: ConfigView{MaxFleetSpeed: 6},
		Planets: []game.Planet{
			{ID: 10, Position: game.Vec2{X: 0, Y: 0}, Radius: 1, OwnerID: 1, Ships: 10, Production: 1},
			// Both targets have the same strength, production, and edge distance.
			// The lower ID must win even though it appears later in this slice.
			{ID: 2, Position: game.Vec2{X: 10, Y: 0}, Radius: 1, OwnerID: 0, Ships: 3, Production: 2},
			{ID: 1, Position: game.Vec2{X: 0, Y: 10}, Radius: 1, OwnerID: game.NeutralPlayer, Ships: 3, Production: 2},
		},
	}

	decision, err := created.Decide(context.Background(), TurnRequest{PlayerID: 1, View: view})
	if err != nil {
		t.Fatal(err)
	}
	if len(decision.Actions) != 1 {
		t.Fatalf("actions = %#v, want one", decision.Actions)
	}
	action := decision.Actions[0]
	if action.PlanetID != 10 || action.Ships != 7 {
		t.Fatalf("action = %#v, want source 10 with 7 ships", action)
	}
	if action.Angle != math.Pi/2 {
		t.Fatalf("angle = %v, want pi/2 toward lower-ID tied target", action.Angle)
	}
}

func TestHeuristicAgentIsDeterministicAndLegalAcrossRandomViews(t *testing.T) {
	const defense = 0.37
	factory, err := NewHeuristicFactory(HeuristicConfig{
		Aggression: 1.4,
		Economy:    0.8,
		Efficiency: 1.1,
		Defense:    defense,
	})
	if err != nil {
		t.Fatal(err)
	}
	viewRNG := rand.New(rand.NewSource(993))
	created, err := factory.New(context.Background(), AgentInit{})
	if err != nil {
		t.Fatal(err)
	}

	for caseIndex := range 200 {
		view := randomizedScriptedView(viewRNG)
		request := TurnRequest{PlayerID: 1, View: view}
		first, err := created.Decide(context.Background(), request)
		if err != nil {
			t.Fatal(err)
		}
		reversePlanetsAndComets(&view)
		second, err := created.Decide(context.Background(), TurnRequest{PlayerID: 1, View: view})
		if err != nil {
			t.Fatal(err)
		}
		if !reflect.DeepEqual(first, second) {
			t.Fatalf("case %d changed with slice order:\nfirst  %#v\nsecond %#v", caseIndex, first, second)
		}
		assertLegalScriptedDecision(t, request, first, true, defense)
	}
}

// In a four-player game every other seat is an enemy, not just one fixed
// opponent. Each case seats the agent, surrounds its home with one planet per
// other seat, and leaves only one of them undefended: the agent must aim at
// that one, whichever seat owns it.
func TestHeuristicAgentTargetsEveryOtherSeat(t *testing.T) {
	factory, err := NewHeuristicFactory(HeuristicConfig{Aggression: 1})
	if err != nil {
		t.Fatal(err)
	}
	created, err := factory.New(context.Background(), AgentInit{})
	if err != nil {
		t.Fatal(err)
	}
	home := game.Vec2{X: 50, Y: 50}
	// Seat k's planet sits ten units from home at a quarter-turn per seat, so
	// the launch angle shows which seat was chosen.
	position := func(seat int) game.Vec2 {
		angle := float64(seat) * math.Pi / 2
		return home.Add(game.Vec2{X: math.Cos(angle), Y: math.Sin(angle)}.Scale(10))
	}

	for seat := range 4 {
		for weak := range 4 {
			if weak == seat {
				continue
			}
			t.Run(fmt.Sprintf("seat %d attacks seat %d", seat, weak), func(t *testing.T) {
				view := View{
					Config:  ConfigView{MaxFleetSpeed: 6},
					Planets: []game.Planet{{ID: 1, Position: home, Radius: 1, OwnerID: seat, Ships: 20, Production: 1}},
				}
				for enemy := range 4 {
					if enemy == seat {
						continue
					}
					ships := 50
					if enemy == weak {
						ships = 0
					}
					view.Planets = append(view.Planets, game.Planet{
						ID: 10 + enemy, Position: position(enemy), Radius: 1, OwnerID: enemy, Ships: ships, Production: 1,
					})
				}

				decision, err := created.Decide(context.Background(), TurnRequest{PlayerID: seat, View: view})
				if err != nil {
					t.Fatal(err)
				}
				if len(decision.Actions) != 1 {
					t.Fatalf("actions = %#v, want one", decision.Actions)
				}
				action := decision.Actions[0]
				delta := position(weak).Sub(home)
				if action.PlayerID != seat || action.PlanetID != 1 || action.Angle != math.Atan2(delta.Y, delta.X) {
					t.Fatalf("action = %#v, want seat %d launching from 1 toward seat %d", action, seat, weak)
				}
			})
		}
	}
}

func TestScriptedAgentsHonorCancellation(t *testing.T) {
	randomFactory, err := NewRandomFactory(RandomConfig{})
	if err != nil {
		t.Fatal(err)
	}
	heuristicFactory, err := NewHeuristicFactory(HeuristicConfig{})
	if err != nil {
		t.Fatal(err)
	}
	ctx, cancel := context.WithCancel(context.Background())
	cancel()

	for _, factory := range []Factory{randomFactory, heuristicFactory} {
		if _, err := factory.New(ctx, AgentInit{}); !errors.Is(err, context.Canceled) {
			t.Fatalf("%s factory error = %v, want context.Canceled", factory.Spec().ID(), err)
		}
	}

	randomAgent, err := randomFactory.New(context.Background(), AgentInit{})
	if err != nil {
		t.Fatal(err)
	}
	heuristicAgent, err := heuristicFactory.New(context.Background(), AgentInit{})
	if err != nil {
		t.Fatal(err)
	}
	for _, scripted := range []Agent{randomAgent, heuristicAgent} {
		if _, err := scripted.Decide(ctx, TurnRequest{}); !errors.Is(err, context.Canceled) {
			t.Fatalf("%s decision error = %v, want context.Canceled", scripted.ID(), err)
		}
	}
}

func TestScriptedAgentsIgnoreCometsPendingExpiry(t *testing.T) {
	view := View{
		Config: ConfigView{Width: 100, Height: 100, MaxFleetSpeed: 6},
		Planets: []game.Planet{
			{ID: 1, Position: game.Vec2{X: 10, Y: 10}, Radius: 1, OwnerID: 1, Ships: 10, Production: 1},
			{ID: 10, Position: game.Vec2{X: 90, Y: 90}, Radius: 1, OwnerID: 0, Ships: 3, Production: 1},
		},
		// Both comets stepped past the end of their paths last turn, so the
		// engine expires them before this turn's launches are checked.
		Comets: []game.Comet{
			{ID: 2, Position: game.Vec2{X: 99, Y: 50}, Radius: 1, OwnerID: 1, Ships: 20, Production: 1,
				Path: []game.Vec2{{X: 99, Y: 50}}, PathIndex: 1},
			{ID: 3, Position: game.Vec2{X: 1, Y: 50}, Radius: 1, OwnerID: 0, Ships: 0, Production: 1,
				Path: []game.Vec2{{X: 1, Y: 50}}, PathIndex: 1},
		},
	}
	request := TurnRequest{PlayerID: 1, View: view}

	randomFactory, err := NewRandomFactory(RandomConfig{LaunchProbability: 1, MinFraction: 0.5, MaxFraction: 0.5})
	if err != nil {
		t.Fatal(err)
	}
	randomAgent, err := randomFactory.New(context.Background(), AgentInit{AgentSeed: 4})
	if err != nil {
		t.Fatal(err)
	}
	randomDecision, err := randomAgent.Decide(context.Background(), request)
	if err != nil {
		t.Fatal(err)
	}
	if got, want := actionSourceIDs(randomDecision.Actions), []int{1}; !reflect.DeepEqual(got, want) {
		t.Fatalf("random source IDs = %v, want %v", got, want)
	}

	heuristicFactory, err := NewHeuristicFactory(HeuristicConfig{Economy: 1})
	if err != nil {
		t.Fatal(err)
	}
	heuristicAgent, err := heuristicFactory.New(context.Background(), AgentInit{})
	if err != nil {
		t.Fatal(err)
	}
	heuristicDecision, err := heuristicAgent.Decide(context.Background(), request)
	if err != nil {
		t.Fatal(err)
	}
	if len(heuristicDecision.Actions) != 1 || heuristicDecision.Actions[0].PlanetID != 1 {
		t.Fatalf("heuristic actions = %#v, want one launch from planet 1", heuristicDecision.Actions)
	}
	if got, want := heuristicDecision.Actions[0].Angle, math.Pi/4; got != want {
		t.Fatalf("heuristic angle = %v, want %v toward in-bounds planet", got, want)
	}
}

func randomizedScriptedView(rng *rand.Rand) View {
	view := View{Config: ConfigView{MaxFleetSpeed: 6}}
	for idIndex := range 12 {
		id := idIndex + 1
		owner := rng.Intn(3) - 1
		position := game.Vec2{X: rng.Float64() * 100, Y: rng.Float64() * 100}
		ships := rng.Intn(101)
		production := 1 + rng.Intn(5)
		radius := 1 + math.Log(float64(production))
		if id%3 == 0 {
			view.Comets = append(view.Comets, game.Comet{
				ID: id, Position: position, Radius: 1, OwnerID: owner, Ships: ships, Production: 1,
				Path: []game.Vec2{position},
			})
			continue
		}
		view.Planets = append(view.Planets, game.Planet{
			ID: id, Position: position, Radius: radius, OwnerID: owner, Ships: ships, Production: production,
		})
	}
	return view
}

func reversePlanetsAndComets(view *View) {
	for left, right := 0, len(view.Planets)-1; left < right; left, right = left+1, right-1 {
		view.Planets[left], view.Planets[right] = view.Planets[right], view.Planets[left]
	}
	for left, right := 0, len(view.Comets)-1; left < right; left, right = left+1, right-1 {
		view.Comets[left], view.Comets[right] = view.Comets[right], view.Comets[left]
	}
}

func assertLegalScriptedDecision(t *testing.T, request TurnRequest, decision Decision, heuristic bool, defense float64) {
	t.Helper()
	sources := map[int]body{}
	hasTarget := false
	for _, candidate := range sortedBodies(request.View) {
		if candidate.ownerID == request.PlayerID {
			sources[candidate.id] = candidate
		} else {
			hasTarget = true
		}
	}
	if heuristic {
		expectedActions := 0
		if hasTarget {
			for _, source := range sources {
				available := source.ships - int(math.Ceil(defense*float64(source.ships)))
				if available > 0 {
					expectedActions++
				}
			}
		}
		if len(decision.Actions) != expectedActions {
			t.Fatalf("heuristic actions = %d, want %d", len(decision.Actions), expectedActions)
		}
	}
	seen := map[int]bool{}
	lastSourceID := -1
	for _, action := range decision.Actions {
		source, ok := sources[action.PlanetID]
		if !ok {
			t.Fatalf("action uses unowned or missing source: %#v", action)
		}
		if seen[action.PlanetID] {
			t.Fatalf("source %d used more than once", action.PlanetID)
		}
		seen[action.PlanetID] = true
		if action.PlanetID <= lastSourceID {
			t.Fatalf("actions not in increasing source order: %#v", decision.Actions)
		}
		lastSourceID = action.PlanetID
		if action.PlayerID != request.PlayerID || action.Ships <= 0 || action.Ships > source.ships {
			t.Fatalf("illegal action: %#v for source %#v", action, source)
		}
		if !finite(action.Angle) {
			t.Fatalf("non-finite angle: %#v", action)
		}
		if heuristic {
			wantShips := source.ships - int(math.Ceil(defense*float64(source.ships)))
			if !hasTarget || wantShips <= 0 {
				t.Fatalf("heuristic emitted from unavailable source: %#v", action)
			}
			if action.Ships != wantShips {
				t.Fatalf("heuristic ships = %d, want %d", action.Ships, wantShips)
			}
		} else if action.Angle < 0 || action.Angle >= 2*math.Pi {
			t.Fatalf("random angle = %v, want [0,2*pi)", action.Angle)
		}
	}
}

func actionSourceIDs(actions []game.Action) []int {
	ids := make([]int, len(actions))
	for i, action := range actions {
		ids[i] = action.PlanetID
	}
	return ids
}
