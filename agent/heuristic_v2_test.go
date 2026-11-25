package agent

import (
	"context"
	"math"
	"reflect"
	"testing"

	"orrery/game"
	"orrery/scenario"
)

func TestHeuristicV2FactoryValidation(t *testing.T) {
	valid := HeuristicV2Config{Margin: 0.1, EnemyBonus: 1, Candidates: 4}
	tests := []struct {
		name   string
		mutate func(*HeuristicV2Config)
	}{
		{"negative margin", func(c *HeuristicV2Config) { c.Margin = -0.1 }},
		{"nan margin", func(c *HeuristicV2Config) { c.Margin = math.NaN() }},
		{"infinite enemy bonus", func(c *HeuristicV2Config) { c.EnemyBonus = math.Inf(1) }},
		{"no candidates", func(c *HeuristicV2Config) { c.Candidates = 0 }},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			config := valid
			tt.mutate(&config)
			if _, err := NewHeuristicV2Factory(config); err == nil {
				t.Fatal("NewHeuristicV2Factory accepted invalid parameters")
			}
		})
	}

	factory, err := ParseFactory("heuristic-v2:margin=0.1,enemy_bonus=1,candidates=4")
	if err != nil {
		t.Fatal(err)
	}
	if got, want := factory.Spec().String(), "heuristic-v2:candidates=4,enemy_bonus=1,margin=0.1"; got != want {
		t.Fatalf("Spec = %q, want %q", got, want)
	}
	if _, err := ParseFactory("heuristic-v2:margin=0.1,enemy_bonus=1,candidates=4.5"); err == nil {
		t.Fatal("ParseFactory accepted a fractional candidates count")
	}

	// The pinned default must already be canonical, so the string results
	// record is the string written here.
	pinned, err := ParseFactory(HeuristicV2Default)
	if err != nil {
		t.Fatal(err)
	}
	if got := pinned.Spec().String(); got != HeuristicV2Default {
		t.Fatalf("HeuristicV2Default parses to %q, want it unchanged", got)
	}
}

// TestHeuristicV2Price checks the fleet price across ownership changes and
// multi-player fights, worked out by hand with the engine's rules: production
// comes before combat each turn, only a player's body produces, arriving
// owners fight each other first (the largest keeps its size minus the second
// largest's), and the survivor needs strictly more ships than the garrison.
// Player 0 buys; margin is 0, so each price is the bare minimum.
func TestHeuristicV2Price(t *testing.T) {
	tests := []struct {
		name     string
		target   body
		incoming []arrival
		segment  int // when player 0's fleet lands
		want     int
	}{
		{
			// Player 1's planet: 10 ships, producing 3 a turn for the 5
			// turns up to segment 4 (+15), plus player 1's 5 ships landing in
			// segment 2. 30 defenders, so 31.
			name:     "enemy target with reinforcements",
			target:   body{id: 2, ownerID: 1, ships: 10, production: 3},
			incoming: []arrival{{owner: 1, ships: 5, segment: 2}},
			segment:  4,
			want:     31,
		},
		{
			// A neutral planet of 10. Players 1 (15) and 2 (12) land in the
			// same turn as player 0. Player 0 must beat 15 among the
			// attackers, keeping s-15, and then beat the 10 defenders:
			// s-15 > 10, so 26.
			name:   "three owners arrive together",
			target: body{id: 2, ownerID: game.NeutralPlayer, ships: 10, production: 3},
			incoming: []arrival{
				{owner: 1, ships: 15, segment: 3},
				{owner: 2, ships: 12, segment: 3},
			},
			segment: 3,
			want:    26,
		},
		{
			// A neutral planet of 10 doesn't produce. Player 1 takes it in
			// segment 1 with 14, keeping 4, and then produces 3 a turn in
			// segments 2 to 5 (+12). 16 defenders when player 0 lands in
			// segment 5, so 17.
			name:     "production starts after an earlier capture",
			target:   body{id: 2, ownerID: game.NeutralPlayer, ships: 10, production: 3},
			incoming: []arrival{{owner: 1, ships: 14, segment: 1}},
			segment:  5,
			want:     17,
		},
	}
	agent := &HeuristicV2Agent{config: HeuristicV2Config{Margin: 0, EnemyBonus: 1, Candidates: 4}}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, ok := agent.price(0, tt.target, tt.incoming, tt.segment, 100)
			if !ok || got != tt.want {
				t.Fatalf("price = %d, %v; want %d, true", got, ok, tt.want)
			}
		})
	}
}

// TestLaunchOptionBeats checks the final choice between launches: a higher
// score wins, and an equal score goes to the lower target ID even when the
// higher ID is checked first, as it is when its quick estimate ranked higher.
func TestLaunchOptionBeats(t *testing.T) {
	low := launchOption{target: body{id: 2}, score: 5}
	high := launchOption{target: body{id: 7}, score: 5}
	if !low.beats(high) || high.beats(low) {
		t.Fatal("an equal score must go to the lower target ID, in either order")
	}
	if better := (launchOption{target: body{id: 9}, score: 6}); !better.beats(low) || low.beats(better) {
		t.Fatal("a higher score must win regardless of target ID")
	}
}

// TestHeuristicV2DecisionsOnSmallBoards checks each stage of the policy on a
// static board small enough to work out by hand. Player 0 decides; margin 0
// means exactly one ship more than the predicted defenders.
func TestHeuristicV2DecisionsOnSmallBoards(t *testing.T) {
	mine := func(id int, x, y float64, ships int) game.Planet {
		return game.Planet{ID: id, OwnerID: 0, Ships: ships, Production: 1, Position: game.Vec2{X: x, Y: y}}
	}
	neutral := func(id int, x, y float64, ships int) game.Planet {
		return game.Planet{ID: id, OwnerID: game.NeutralPlayer, Ships: ships, Production: 3, Position: game.Vec2{X: x, Y: y}}
	}
	// incoming is a fleet already flying straight left along y = 20 towards
	// planet 1 at (20, 20).
	incoming := func(id, owner, ships int, x float64) game.Fleet {
		return game.Fleet{ID: id, OwnerID: owner, Ships: ships, Position: game.Vec2{X: x, Y: 20},
			Velocity: game.Vec2{X: -game.FleetSpeed(ships, 6)}, Angle: math.Pi, LaunchTurn: -1}
	}
	tests := []struct {
		name    string
		planets []game.Planet
		fleets  []game.Fleet
		// turn is how many of the board's 500 turns are already played.
		turn int
		// want lists the expected launches as source → ships, in order.
		want [][2]int
	}{
		{
			name:    "sends just enough ships",
			planets: []game.Planet{mine(1, 20, 20, 50), neutral(2, 30, 20, 10)},
			want:    [][2]int{{1, 11}},
		},
		{
			// The straight path from (20, 50) to (80, 50) crosses the sun, so
			// LeadAim finds no verified aim.
			name:    "skips a target it cannot reach",
			planets: []game.Planet{mine(1, 20, 50, 50), neutral(2, 80, 50, 10)},
		},
		{
			// 28 enemy ships land on planet 1 in its sixth turn (segment 5),
			// after it has produced 6 more, so 22 of its 30 must stay home and
			// the 11 that planet 2 needs can't be spared.
			name:    "holds back ships against a predicted attack",
			planets: []game.Planet{mine(1, 20, 20, 30), neutral(2, 20, 40, 10)},
			fleets:  []game.Fleet{incoming(10, 1, 28, 35)},
		},
		{
			// The same 28 enemy ships, but 28 friendly ships land in the same
			// turn. The two fight each other first and cancel out, so nothing
			// reaches the garrison and nothing needs to stay home, whichever
			// fleet has the lower ID.
			name:    "same-turn friendly and enemy fleets cancel (enemy listed first)",
			planets: []game.Planet{mine(1, 20, 20, 30), neutral(2, 20, 40, 10)},
			fleets:  []game.Fleet{incoming(10, 1, 28, 35), incoming(11, 0, 28, 35)},
			want:    [][2]int{{1, 11}},
		},
		{
			name:    "same-turn friendly and enemy fleets cancel (friendly listed first)",
			planets: []game.Planet{mine(1, 20, 20, 30), neutral(2, 20, 40, 10)},
			fleets:  []game.Fleet{incoming(10, 0, 28, 35), incoming(11, 1, 28, 35)},
			want:    [][2]int{{1, 11}},
		},
		{
			name:    "attacks without that threat",
			planets: []game.Planet{mine(1, 20, 20, 30), neutral(2, 20, 40, 10)},
			want:    [][2]int{{1, 11}},
		},
		{
			name:    "sends one source, not two, to one target",
			planets: []game.Planet{mine(1, 20, 20, 50), mine(3, 20, 30, 50), neutral(2, 30, 25, 10)},
			want:    [][2]int{{1, 11}},
		},
		{
			// Planet 0's own 20 ships are already landing on planet 2 at
			// (10, 20), so there is nothing left to send.
			name:    "counts its own fleets already on the way",
			planets: []game.Planet{mine(1, 40, 20, 50), neutral(2, 10, 20, 10)},
			fleets:  []game.Fleet{incoming(10, 0, 20, 15)},
		},
		{
			// Player 0's 20 ships land on planet 2 (10 ships) about ten turns
			// from now. Planet 1 is close enough to land first, but the
			// target is already predicted taken, so it doesn't attack.
			name:    "counts its own fleets landing after a new launch would",
			planets: []game.Planet{mine(1, 30, 30, 50), neutral(2, 30, 20, 10)},
			fleets:  []game.Fleet{incoming(10, 0, 20, 56)},
		},
		{
			// The same 20 ships would land about ten turns from now, but only
			// 5 turns are left, so they never will and claim nothing. Planet 1
			// takes the 1-ship planet in about 3 turns with 2 ships, and its
			// production for the last 2 turns still beats their cost.
			name:    "ignores its own fleets landing after the game ends",
			planets: []game.Planet{mine(1, 30, 26, 50), neutral(2, 30, 20, 1)},
			fleets:  []game.Fleet{incoming(10, 0, 20, 56)},
			turn:    495,
			want:    [][2]int{{1, 2}},
		},
	}

	factory, err := NewHeuristicV2Factory(HeuristicV2Config{Margin: 0, EnemyBonus: 1, Candidates: 4})
	if err != nil {
		t.Fatal(err)
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			g := smallV2Board(t, tt.planets, tt.fleets)
			g.Turn = tt.turn
			decision := decideV2(t, factory, g, 0)

			var got [][2]int
			for _, action := range decision.Actions {
				got = append(got, [2]int{action.PlanetID, action.Ships})
			}
			if !reflect.DeepEqual(got, tt.want) {
				t.Fatalf("launches (source, ships) = %v, want %v", got, tt.want)
			}
			// Every launch is aimed exactly as the engine's LeadAim aims it.
			for _, action := range decision.Actions {
				if !aimedByLeadAim(g, action) {
					t.Errorf("launch %+v is not a verified LeadAim heading", action)
				}
			}
		})
	}
}

// TestHeuristicV2WholeGarrison checks whole_garrison, and the reserve and
// garrison_cost settings that refine it, on small boards built like
// TestHeuristicV2DecisionsOnSmallBoards's.
func TestHeuristicV2WholeGarrison(t *testing.T) {
	mine := func(id int, x, y float64, ships int) game.Planet {
		return game.Planet{ID: id, OwnerID: 0, Ships: ships, Production: 1, Position: game.Vec2{X: x, Y: y}}
	}
	neutral := func(id int, x, y float64, ships int) game.Planet {
		return game.Planet{ID: id, OwnerID: game.NeutralPlayer, Ships: ships, Production: 3, Position: game.Vec2{X: x, Y: y}}
	}
	// plain is a board where planet 1 can take planet 2 with 11 ships.
	plain := func(ships int) []game.Planet {
		return []game.Planet{mine(1, 20, 20, ships), neutral(2, 30, 20, 10)}
	}
	// threatened is planet 1 with 30 ships, and 28 enemy ships landing on it
	// in its sixth turn, so normal v2 keeps 22 of them at home.
	threatened := []game.Planet{mine(1, 20, 20, 30), neutral(2, 20, 40, 10)}
	threat := []game.Fleet{{ID: 10, OwnerID: 1, Ships: 28, Position: game.Vec2{X: 35, Y: 20},
		Velocity: game.Vec2{X: -game.FleetSpeed(28, 6)}, Angle: math.Pi, LaunchTurn: -1}}

	whole := func(reserve ReserveRule, cost GarrisonCost) HeuristicV2Config {
		return HeuristicV2Config{Margin: 0, EnemyBonus: 1, Candidates: 4, WholeGarrison: true, Reserve: reserve, GarrisonCost: cost}
	}
	tests := []struct {
		name    string
		config  HeuristicV2Config
		planets []game.Planet
		fleets  []game.Fleet
		// turn is how many of the board's 500 turns are already played.
		turn int
		// want lists the expected launches as source → ships, in order.
		want [][2]int
	}{
		{
			name:    "sends every ship, not only the 11 needed",
			config:  whole(ReserveHold, GarrisonCostSent),
			planets: plain(50),
			want:    [][2]int{{1, 50}},
		},
		{
			name:    "reserve=hold holds a body that needs defenders",
			config:  whole(ReserveHold, GarrisonCostSent),
			planets: threatened,
			fleets:  threat,
		},
		{
			name:    "reserve=ignore launches every ship anyway",
			config:  whole(ReserveIgnore, GarrisonCostSent),
			planets: threatened,
			fleets:  threat,
			want:    [][2]int{{1, 30}},
		},
		{
			// With 20 turns left, planet 2's production gains more than the
			// 11 ships the capture needs, but less than the 100 sent.
			name:    "garrison_cost=sent holds when the ships sent outweigh the gain",
			config:  whole(ReserveHold, GarrisonCostSent),
			planets: plain(100),
			turn:    480,
		},
		{
			name:    "garrison_cost=needed launches when only the ships needed count",
			config:  whole(ReserveHold, GarrisonCostNeeded),
			planets: plain(100),
			turn:    480,
			want:    [][2]int{{1, 100}},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			factory, err := NewHeuristicV2Factory(tt.config)
			if err != nil {
				t.Fatal(err)
			}
			g := smallV2Board(t, tt.planets, tt.fleets)
			g.Turn = tt.turn
			decision := decideV2(t, factory, g, 0)

			var got [][2]int
			for _, action := range decision.Actions {
				got = append(got, [2]int{action.PlanetID, action.Ships})
			}
			if !reflect.DeepEqual(got, tt.want) {
				t.Fatalf("launches (source, ships) = %v, want %v", got, tt.want)
			}
			// The whole garrison is aimed as the engine's LeadAim aims it.
			for _, action := range decision.Actions {
				if !aimedByLeadAim(g, action) {
					t.Errorf("launch %+v is not a verified LeadAim heading", action)
				}
			}
		})
	}
}

// TestHeuristicV2IsDeterministicAndLegal plays whole four-player games with
// heuristic-v2 in every seat. Each decision is made twice on separate views
// and must match, and the engine must accept every turn.
func TestHeuristicV2IsDeterministicAndLegal(t *testing.T) {
	factory, err := NewHeuristicV2Factory(HeuristicV2Config{Margin: 0.1, EnemyBonus: 1, Candidates: 4})
	if err != nil {
		t.Fatal(err)
	}
	g, err := scenario.GeneratorV2{FourPlayer: true}.New(1000000)
	if err != nil {
		t.Fatal(err)
	}
	launches := 0
	for !g.Finished() {
		var actions []game.Action
		for player := range 4 {
			first := decideV2(t, factory, g, player)
			if again := decideV2(t, factory, g, player); !reflect.DeepEqual(again, first) {
				t.Fatalf("turn %d player %d: two decisions on the same board differ", g.Turn, player)
			}
			actions = append(actions, first.Actions...)
		}
		launches += len(actions)
		if err := g.AdvanceTurn(actions); err != nil {
			t.Fatalf("turn %d: the engine rejected heuristic-v2's moves: %v", g.Turn, err)
		}
	}
	if launches == 0 {
		t.Fatal("heuristic-v2 never launched")
	}
}

// smallV2Board is a static, comet-free 100×100 board with the sun hazard on.
func smallV2Board(t *testing.T, planets []game.Planet, fleets []game.Fleet) *game.Game {
	t.Helper()
	config := game.DefaultConfig()
	config.OrbitsEnabled = false
	config.CometSpawningEnabled = false
	g, err := game.New(config, planets, nil)
	if err != nil {
		t.Fatal(err)
	}
	g.Fleets = fleets
	return g
}

// decideV2 asks a fresh heuristic-v2 agent for player's moves on g.
func decideV2(t *testing.T, factory Factory, g *game.Game, player int) Decision {
	t.Helper()
	instance, err := factory.New(context.Background(), AgentInit{})
	if err != nil {
		t.Fatal(err)
	}
	decision, err := instance.Decide(context.Background(), TurnRequest{PlayerID: player, View: NewView(g)})
	if err != nil {
		t.Fatal(err)
	}
	return decision
}

// aimedByLeadAim reports whether action's angle is the live game's verified
// LeadAim heading for that source, target and fleet size, for some target.
func aimedByLeadAim(g *game.Game, action game.Action) bool {
	for _, p := range g.Planets {
		if aim, ok := g.LeadAim(action.PlanetID, p.ID, action.Ships); ok && aim.Angle == action.Angle {
			return true
		}
	}
	return false
}

// TestSharedLandings covers the rules for reusing a caller's prediction: when
// it is used, how later landings are dropped, and which mismatches are errors.
func TestSharedLandings(t *testing.T) {
	view := View{Fleets: []game.Fleet{{ID: 7}, {ID: 9}}}
	planet := func(fleet, segment int) game.Landing {
		return game.Landing{FleetID: fleet, Resolved: true, Hit: game.Hit{Kind: game.HitPlanet, BodyID: 3, T: 0.5}, Segment: segment}
	}
	cases := []struct {
		name       string
		prediction *FleetPrediction
		horizon    int
		want       []game.Landing
		wantErr    bool
	}{
		{name: "no prediction", prediction: nil, horizon: 100},
		{
			name:       "horizon shorter than wanted",
			prediction: &FleetPrediction{Horizon: 99, Landings: []game.Landing{planet(7, 1), planet(9, 2)}},
			horizon:    100,
		},
		{
			// Given in the other order: the result follows View.Fleets.
			name:       "reordered to the view",
			prediction: &FleetPrediction{Horizon: 142, Landings: []game.Landing{planet(9, 2), planet(7, 1)}},
			horizon:    100,
			want:       []game.Landing{planet(7, 1), planet(9, 2)},
		},
		{
			// Segment 5 is beyond a 5-turn look-ahead, so it reads as unresolved.
			name:       "landing at the horizon dropped",
			prediction: &FleetPrediction{Horizon: 142, Landings: []game.Landing{planet(7, 4), planet(9, 5)}},
			horizon:    5,
			want:       []game.Landing{planet(7, 4), {FleetID: 9}},
		},
		{
			name:       "missing fleet",
			prediction: &FleetPrediction{Horizon: 142, Landings: []game.Landing{planet(7, 1), planet(8, 1)}},
			horizon:    100,
			wantErr:    true,
		},
		{
			name:       "repeated fleet",
			prediction: &FleetPrediction{Horizon: 142, Landings: []game.Landing{planet(7, 1), planet(7, 2)}},
			horizon:    100,
			wantErr:    true,
		},
		{
			name:       "wrong count",
			prediction: &FleetPrediction{Horizon: 142, Landings: []game.Landing{planet(7, 1)}},
			horizon:    100,
			wantErr:    true,
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got, err := sharedLandings(view, tc.prediction, tc.horizon)
			if (err != nil) != tc.wantErr {
				t.Fatalf("sharedLandings() error = %v, want error %v", err, tc.wantErr)
			}
			if !reflect.DeepEqual(got, tc.want) {
				t.Fatalf("sharedLandings() = %+v, want %+v", got, tc.want)
			}
		})
	}
}
