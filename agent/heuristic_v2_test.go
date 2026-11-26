package agent

import (
	"context"
	"math"
	"math/rand"
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
					t.Errorf("launch %+v is not aimed ahead as the engine's LeadAim aims it", action)
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
					t.Errorf("launch %+v is not aimed ahead as the engine's LeadAim aims it", action)
				}
			}
		})
	}
}

// TestHeuristicV2LearnerAim checks learner_aim and no_aim_check on a board
// where aiming ahead fails: the straight path from planet 1 at (20, 50) to
// planet 2 at (80, 50) crosses the sun, so game.LeadAim finds no way there.
// Normal v2 skips such a target; the learner, and v2 with its aim, aim
// straight instead.
func TestHeuristicV2LearnerAim(t *testing.T) {
	planets := func(ships int) []game.Planet {
		return []game.Planet{
			{ID: 1, OwnerID: 0, Ships: ships, Production: 1, Position: game.Vec2{X: 20, Y: 50}},
			{ID: 2, OwnerID: game.NeutralPlayer, Ships: 10, Production: 3, Position: game.Vec2{X: 80, Y: 50}},
		}
	}
	// Aiming straight from (20, 50) at (80, 50) points along +x: angle 0.
	straight := math.Atan2(50-50, 80-20)

	base := HeuristicV2Config{Margin: 0, EnemyBonus: 1, Candidates: 4}
	learnerAim := base
	learnerAim.LearnerAim = true
	noAimCheck := learnerAim
	noAimCheck.NoAimCheck = true
	wholeLearnerAim := learnerAim
	wholeLearnerAim.WholeGarrison = true

	tests := []struct {
		name   string
		config HeuristicV2Config
		// want lists the expected launches as source → ships, in order.
		want [][2]int
	}{
		{name: "normal v2 skips a target it can't aim ahead at", config: base},
		{name: "learner_aim aims straight instead", config: learnerAim, want: [][2]int{{1, 11}}},
		{name: "no_aim_check launches the top target, aimed straight", config: noAimCheck, want: [][2]int{{1, 11}}},
		{name: "with whole_garrison, every ship goes, aimed straight", config: wholeLearnerAim, want: [][2]int{{1, 50}}},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			factory, err := NewHeuristicV2Factory(tt.config)
			if err != nil {
				t.Fatal(err)
			}
			g := smallV2Board(t, planets(50), nil)
			decision := decideV2(t, factory, g, 0)

			var got [][2]int
			for _, action := range decision.Actions {
				got = append(got, [2]int{action.PlanetID, action.Ships})
				if action.Angle != straight {
					t.Errorf("launch %+v: angle %v, want %v (aimed straight)", action, action.Angle, straight)
				}
			}
			if !reflect.DeepEqual(got, tt.want) {
				t.Fatalf("launches (source, ships) = %v, want %v", got, tt.want)
			}
		})
	}
}

// TestHeuristicV2Epsilon checks the epsilon dial: at 1 v2 plays exactly like
// random-v1, at 0 it never rolls the die, and in between a game replays
// exactly from its seed.
func TestHeuristicV2Epsilon(t *testing.T) {
	v2 := func(t *testing.T, spec string) Factory {
		t.Helper()
		factory, err := ParseFactory(HeuristicV2Default + spec)
		if err != nil {
			t.Fatal(err)
		}
		return factory
	}

	t.Run("epsilon=1 plays exactly like random-v1 with the same settings and seed", func(t *testing.T) {
		pairs := []struct{ v2, random string }{
			// The epsilon_* defaults are the pinned random-v1 settings.
			{",epsilon=1", "random-v1:launch_probability=0.3,max_fraction=0.8,min_fraction=0.2"},
			// Custom epsilon_* settings become random-v1's settings.
			{",epsilon=1,epsilon_launch_probability=1,epsilon_min_fraction=0.5", "random-v1:launch_probability=1,max_fraction=0.8,min_fraction=0.5"},
		}
		for _, pair := range pairs {
			random, err := ParseFactory(pair.random)
			if err != nil {
				t.Fatal(err)
			}
			// random-v1 plays seat 0, and the epsilon=1 v2 is asked on the
			// same boards with the same seed: every decision must match.
			want := playSeatZero(t, random, 7, nil)
			got := playSeatZero(t, random, 7, v2(t, pair.v2))
			if !reflect.DeepEqual(got, want) {
				t.Fatalf("%s decided differently from %s", pair.v2, pair.random)
			}
			if countLaunches(want) == 0 {
				t.Fatalf("%s never launched, so the comparison proves nothing", pair.random)
			}
		}
	})

	t.Run("epsilon=0 never rolls the die", func(t *testing.T) {
		for _, tt := range []struct {
			spec    string
			wantRNG bool
		}{{"", false}, {",epsilon=0.5", true}} {
			created, err := v2(t, tt.spec).New(context.Background(), AgentInit{AgentSeed: 7})
			if err != nil {
				t.Fatal(err)
			}
			if hasRNG := created.(*HeuristicV2Agent).rng != nil; hasRNG != tt.wantRNG {
				t.Errorf("spec %q: has a random source = %v, want %v", tt.spec, hasRNG, tt.wantRNG)
			}
		}
	})

	// Which planets move randomly must follow the epsilon actually set, not
	// just "some". On epsilonBoard, every planet would launch under plain v2,
	// so a planet that holds proves its random move skipped v2's planning.
	// Each seed below gives a mix of v2 moves, random launches and random
	// holds, and a different mix from the other two settings, so a dial
	// stuck at one value would fail.
	t.Run("each planet follows the die at epsilon 0.25, 0.5 and 0.75", func(t *testing.T) {
		g := epsilonBoard(t)
		plain, err := NewHeuristicV2Factory(HeuristicV2Config{Margin: 0, EnemyBonus: 1, Candidates: 4})
		if err != nil {
			t.Fatal(err)
		}
		if launches := len(decideV2(t, plain, g, 0).Actions); launches != 6 {
			t.Fatalf("plain v2 launched from %d planets, want all 6", launches)
		}
		kinds := func(moves []epsilonMove) []string {
			var k []string
			for _, m := range moves {
				k = append(k, m.kind)
			}
			return k
		}
		settings := []struct {
			epsilon float64
			seed    int64
		}{{0.25, 7}, {0.5, 1}, {0.75, 4}}
		for _, s := range settings {
			want := predictEpsilon(t, g, s.epsilon, s.seed)
			// The seed still gives the intended mix and pattern.
			seen := map[string]bool{}
			for _, m := range want {
				seen[m.kind] = true
			}
			if !seen["v2"] || !seen["random launch"] || !seen["random hold"] {
				t.Fatalf("epsilon %v seed %d: kinds %v lack a v2 move, a random launch or a random hold", s.epsilon, s.seed, kinds(want))
			}
			for _, other := range settings {
				if other.epsilon != s.epsilon && reflect.DeepEqual(kinds(predictEpsilon(t, g, other.epsilon, s.seed)), kinds(want)) {
					t.Fatalf("epsilon %v seed %d: same pattern as epsilon %v, so it can't tell them apart", s.epsilon, s.seed, other.epsilon)
				}
			}

			factory, err := NewHeuristicV2Factory(HeuristicV2Config{Margin: 0, EnemyBonus: 1, Candidates: 4, Epsilon: s.epsilon})
			if err != nil {
				t.Fatal(err)
			}
			created, err := factory.New(context.Background(), AgentInit{AgentSeed: s.seed})
			if err != nil {
				t.Fatal(err)
			}
			decision, err := created.Decide(context.Background(), TurnRequest{PlayerID: 0, View: NewView(g)})
			if err != nil {
				t.Fatal(err)
			}
			var wantActions []game.Action
			for _, m := range want {
				if m.kind != "random hold" {
					wantActions = append(wantActions, m.action)
				}
			}
			if !reflect.DeepEqual(decision.Actions, wantActions) {
				t.Errorf("epsilon %v seed %d (%v):\n got %v\nwant %v", s.epsilon, s.seed, kinds(want), decision.Actions, wantActions)
			}
		}
	})

	t.Run("epsilon=0.5 replays from its seed", func(t *testing.T) {
		half := v2(t, ",epsilon=0.5")
		first := playSeatZero(t, half, 7, nil)
		if again := playSeatZero(t, half, 7, nil); !reflect.DeepEqual(again, first) {
			t.Fatal("the same seed played a different game")
		}
		if other := playSeatZero(t, half, 8, nil); reflect.DeepEqual(other, first) {
			t.Fatal("seeds 7 and 8 played the same game, so the seed isn't used")
		}
		if countLaunches(first) == 0 {
			t.Fatal("epsilon=0.5 never launched")
		}
	})
}

// epsilonBoard is six owned planets, each with 50 ships and its own neutral
// target 10 units away, far from the others. Each planet's best target is its
// own, so v2's choice for one planet never depends on another's, and the
// planets can be checked one by one.
func epsilonBoard(t *testing.T) *game.Game {
	t.Helper()
	pairs := [][4]float64{
		{15, 15, 25, 15}, {85, 15, 75, 15}, {15, 85, 25, 85},
		{85, 85, 75, 85}, {10, 50, 20, 50}, {90, 50, 80, 50},
	}
	var planets []game.Planet
	for i, p := range pairs {
		planets = append(planets,
			game.Planet{ID: i + 1, OwnerID: 0, Ships: 50, Production: 1, Position: game.Vec2{X: p[0], Y: p[1]}},
			game.Planet{ID: i + 11, OwnerID: game.NeutralPlayer, Ships: 10, Production: 3, Position: game.Vec2{X: p[2], Y: p[3]}})
	}
	return smallV2Board(t, planets, nil)
}

// epsilonMove is what one owned planet should do in one turn.
type epsilonMove struct {
	kind   string // "v2", "random launch" or "random hold"
	action game.Action
}

// predictEpsilon works out, independently of the agent, what a v2 with this
// epsilon and seed should do on g this turn. It replays the agent's random
// source draw by draw: one die roll per owned planet with ships, in ID order,
// and, after a roll below epsilon, random-v1's draws for that planet. A planet
// whose roll says random makes random-v1's move, launch or hold; every other
// planet makes the move a v2 without epsilon plans for it.
func predictEpsilon(t *testing.T, g *game.Game, epsilon float64, seed int64) []epsilonMove {
	t.Helper()
	plain, err := NewHeuristicV2Factory(HeuristicV2Config{Margin: 0, EnemyBonus: 1, Candidates: 4})
	if err != nil {
		t.Fatal(err)
	}
	planned := map[int]game.Action{}
	for _, action := range decideV2(t, plain, g, 0).Actions {
		planned[action.PlanetID] = action
	}

	rng := rand.New(rand.NewSource(seed))
	var moves []epsilonMove
	for _, source := range sortedBodies(NewView(g)) {
		if source.ownerID != 0 || source.ships <= 0 {
			continue
		}
		if rng.Float64() >= epsilon {
			moves = append(moves, epsilonMove{kind: "v2", action: planned[source.id]})
			continue
		}
		if action, ok := randomMove(rng, heuristicV2DefaultRandomMove, 0, source); ok {
			moves = append(moves, epsilonMove{kind: "random launch", action: action})
		} else {
			moves = append(moves, epsilonMove{kind: "random hold"})
		}
	}
	return moves
}

// playSeatZero plays the first 60 turns of a four-player game in which seat 0
// is played by player, seeded with seed, and the other seats hold. It returns
// seat 0's moves, turn by turn. When shadow is set, a second agent with the
// same seed is asked on each of the same boards, and its moves are returned
// instead; the game itself still follows player's moves.
func playSeatZero(t *testing.T, player Factory, seed int64, shadow Factory) [][]game.Action {
	t.Helper()
	ctx := context.Background()
	g, err := scenario.GeneratorV2{FourPlayer: true}.New(1000000)
	if err != nil {
		t.Fatal(err)
	}
	newAgent := func(factory Factory) Agent {
		created, err := factory.New(ctx, AgentInit{PlayerID: 0, AgentSeed: seed})
		if err != nil {
			t.Fatal(err)
		}
		return created
	}
	playing := newAgent(player)
	asked := playing
	if shadow != nil {
		asked = newAgent(shadow)
	}

	var moves [][]game.Action
	for !g.Finished() && g.Turn < 60 {
		decision, err := playing.Decide(ctx, TurnRequest{PlayerID: 0, View: NewView(g)})
		if err != nil {
			t.Fatal(err)
		}
		recorded := decision
		if shadow != nil {
			if recorded, err = asked.Decide(ctx, TurnRequest{PlayerID: 0, View: NewView(g)}); err != nil {
				t.Fatal(err)
			}
		}
		moves = append(moves, recorded.Actions)
		if err := g.AdvanceTurn(decision.Actions); err != nil {
			t.Fatalf("turn %d: the engine rejected the moves: %v", g.Turn, err)
		}
	}
	return moves
}

// countLaunches is the number of launches in a game's moves.
func countLaunches(moves [][]game.Action) int {
	launches := 0
	for _, turn := range moves {
		launches += len(turn)
	}
	return launches
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

// aimedByLeadAim reports whether action is aimed ahead: whether its angle is
// the live game's LeadAim direction for that source and fleet size, for some
// target.
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
