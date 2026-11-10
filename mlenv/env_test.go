package mlenv

import (
	"context"
	"errors"
	"fmt"
	"math"
	"math/rand"
	"reflect"
	"slices"
	"strings"
	"sync"
	"testing"
	"time"

	"orrery/agent"
	"orrery/arena"
	"orrery/game"
	"orrery/scenario"
)

// testScenario builds a small still board that NewSeatFrame accepts.
//
// The seat frame requires every home to be player 0's home turned about the
// sun, in the generator's seat order: going clockwise the seats run
// 0 → 1 → 3 → 2, and in a two-player game player 1 sits opposite player 0.
// Each player's home has body ID player+1.
type testScenario struct {
	id       string
	maxTurns int
	// players is 2 or 4; zero means 2.
	players int
	// close puts a two-player board's homes either side of the sun, a short
	// hop apart, with a large first garrison, so one all-in launch eliminates
	// the second player on the first turn. The sun hazard is off, so the
	// crossing is safe. On a four-player board it does the reverse: the homes
	// sit on a small square round the sun (the seat frame needs them to be
	// quarter-turns of each other), and seat 0 holds one ship beside seat 1's
	// large garrison, so seat 1 can eliminate it in one turn while seats 2
	// and 3 play on.
	close bool
}

func (factory testScenario) ID() string { return factory.id }

func (factory testScenario) Players() int {
	if factory.players == 0 {
		return 2
	}
	return factory.players
}

// misdeclaredScenario builds testScenario's maps but declares a player count
// of its own, to exercise the checks against a factory that misreports.
type misdeclaredScenario struct {
	testScenario
	declared int
}

func (factory misdeclaredScenario) Players() int { return factory.declared }

// shrinkingScenario declares testScenario's player count but builds a
// two-player map for shrinkSeed, so one reset can meet a map that breaks the
// declared count after others in the same request have been built.
type shrinkingScenario struct {
	testScenario
	shrinkSeed int64
}

func (factory shrinkingScenario) New(seed int64) (*game.Game, error) {
	built := factory.testScenario
	if seed == factory.shrinkSeed {
		built.players = 2
	}
	return built.New(seed)
}

func (factory testScenario) New(seed int64) (*game.Game, error) {
	config := game.DefaultConfig()
	config.MaxTurns = factory.maxTurns
	config.SunHazardEnabled = false
	config.CometSpawningEnabled = false
	config.OrbitsEnabled = false
	config.Rand = rand.New(rand.NewSource(seed))

	home := func(player int, at game.Vec2) game.Planet {
		return game.Planet{ID: player + 1, Position: at, OwnerID: player, Ships: 10, Production: 1}
	}
	var planets []game.Planet
	switch {
	case factory.players == 4 && factory.close:
		first, second := home(0, game.Vec2{X: 48, Y: 48}), home(1, game.Vec2{X: 52, Y: 48})
		first.Ships, second.Ships, second.Production = 1, 100, 5
		planets = []game.Planet{
			first, second,
			home(2, game.Vec2{X: 48, Y: 52}),
			home(3, game.Vec2{X: 52, Y: 52}),
		}
	case factory.players == 4:
		planets = []game.Planet{
			home(0, game.Vec2{X: 20, Y: 20}),
			home(1, game.Vec2{X: 80, Y: 20}),
			home(2, game.Vec2{X: 20, Y: 80}),
			home(3, game.Vec2{X: 80, Y: 80}),
		}
	case factory.close:
		first, second := home(0, game.Vec2{X: 48, Y: 48}), home(1, game.Vec2{X: 52, Y: 52})
		first.Ships, first.Production, second.Ships = 100, 5, 1
		planets = []game.Planet{first, second}
	default:
		planets = []game.Planet{home(0, game.Vec2{X: 20, Y: 20}), home(1, game.Vec2{X: 80, Y: 80})}
	}
	return game.New(config, planets, nil)
}

func newTestBatch(t testing.TB, factory SeatedScenarioFactory, opponent agent.Factory, size int, seeds scenario.SeedRange, schedule ScheduleMode, cursor int64, workers int) *Batch {
	t.Helper()
	batch, err := NewBatch(context.Background(), BatchConfig{
		Size: size, Seeds: seeds, Schedule: schedule, Cursor: cursor,
		Scenario: factory, Opponent: opponent, Workers: workers,
	})
	if err != nil {
		t.Fatalf("NewBatch() error = %v", err)
	}
	t.Cleanup(func() {
		if err := batch.Close(); err != nil {
			t.Errorf("Close() error = %v", err)
		}
	})
	return batch
}

func TestBatchMatchesDirectReferenceAndRequiresResetAfterTerminal(t *testing.T) {
	ctx := context.Background()
	factory := testScenario{id: "short-v1", maxTurns: 3}
	batch := newTestBatch(t, factory, agent.IdleFactory{}, 1, scenario.SeedRange{Start: 40, End: 50}, ScheduleTrainV2, 0, 1)

	reset, err := batch.Reset(ctx, []int{0}, nil)
	if err != nil {
		t.Fatalf("Reset() error = %v", err)
	}
	if got, want := reset.Cursor, int64(1); got != want {
		t.Fatalf("reset cursor = %d, want %d", got, want)
	}

	reference, err := factory.New(reset.Envs[0].Seed)
	if err != nil {
		t.Fatalf("reference scenario error = %v", err)
	}
	var final StepResult
	for !reference.Finished() {
		results, stepErr := batch.Step(ctx, map[int][BodySlots]SlotAction{0: {}})
		if stepErr != nil {
			t.Fatalf("Step() error = %v", stepErr)
		}
		if len(results) != 1 {
			t.Fatalf("Step() returned %d results, want 1", len(results))
		}
		final = results[0]
		if err := reference.AdvanceTurn(nil); err != nil {
			t.Fatalf("reference AdvanceTurn() error = %v", err)
		}
	}

	if !final.Terminated || final.Truncated || final.EndReason != EndGameFinished {
		t.Fatalf("terminal result = terminated %t, truncated %t, end %d; want true, false, game finished", final.Terminated, final.Truncated, final.EndReason)
	}
	if final.Reward != 0 {
		t.Fatalf("tie reward = %v, want 0", final.Reward)
	}
	if got, want := agent.NewView(batch.envs[0].game), agent.NewView(reference); !reflect.DeepEqual(got, want) {
		t.Fatalf("batched terminal state differs from directly advanced reference\nbatch: %#v\nreference: %#v", got, want)
	}

	_, err = batch.Step(ctx, map[int][BodySlots]SlotAction{0: {}})
	assertEnvErrorKind(t, err, ErrorTerminated)
}

func TestBatchEliminationTerminatesWithWinningReward(t *testing.T) {
	ctx := context.Background()
	batch := newTestBatch(t, testScenario{id: "elimination-v1", maxTurns: 20, close: true}, agent.IdleFactory{}, 1, scenario.SeedRange{Start: 0, End: 2}, ScheduleTrainV2, 0, 1)
	// The learner must hold the big garrison on body 1, which is seat 0's.
	reset, err := batch.Reset(ctx, []int{0}, []EpisodeOverride{{EnvID: 0, Seed: 0, Seat: 0}})
	if err != nil {
		t.Fatalf("Reset() error = %v", err)
	}

	var actions [BodySlots]SlotAction
	source, target := slotsForBodies(t, reset.Envs[0].Observation, 1, 2)
	actions[source] = SlotAction{Launch: true, Target: int32(target)}
	results, err := batch.Step(ctx, map[int][BodySlots]SlotAction{0: actions})
	if err != nil {
		t.Fatalf("Step() error = %v", err)
	}
	if got := results[0]; !got.Terminated || got.Reward != 1 || got.EndReason != EndGameFinished || got.Turn >= batch.envs[0].game.Config.MaxTurns {
		t.Fatalf("elimination result = %+v, want early game-finished termination with reward +1", got)
	}
}

// TestBatchLearnerEliminatedWhileOthersPlayOn runs a learner elimination
// through Step on a four-player board. Three opponents survive, so the game
// itself has not finished. The episode must still end, and with the
// learner-eliminated reason rather than the game-finished one.
func TestBatchLearnerEliminatedWhileOthersPlayOn(t *testing.T) {
	ctx := context.Background()
	batch := newTestBatch(t, testScenario{id: "learner-eliminated-v1", maxTurns: 20, players: 4, close: true}, strikingOpponentFactory{}, 1, scenario.SeedRange{Start: 0, End: 2}, ScheduleTrainV2, 0, 1)
	// Seat 0 is the one-ship home next to seat 1's 100 ships.
	if _, err := batch.Reset(ctx, []int{0}, []EpisodeOverride{{EnvID: 0, Seed: 0, Seat: 0}}); err != nil {
		t.Fatalf("Reset() error = %v", err)
	}

	// The learner does nothing; seat 1 captures its only planet this turn.
	results, err := batch.Step(ctx, map[int][BodySlots]SlotAction{0: {}})
	if err != nil {
		t.Fatalf("Step() error = %v", err)
	}
	env := batch.envs[0]
	got := results[0]
	if !got.Terminated || got.Truncated || got.Reward != -1 || got.EndReason != EndLearnerEliminated || got.Turn != 1 {
		t.Fatalf("result = terminated %t, truncated %t, reward %v, end %d, turn %d; want true, false, -1, learner eliminated, 1",
			got.Terminated, got.Truncated, got.Reward, got.EndReason, got.Turn)
	}
	if env.game.Finished() {
		t.Fatal("game finished, want it still running so the end reason is the learner's elimination")
	}
	for seat := 1; seat < PlayerSlots; seat++ {
		if !ownsAnything(env.game, seat) {
			t.Fatalf("seat %d owns nothing, want all three opponents alive", seat)
		}
	}
	// The last observation is the learner's view of the board it lost.
	want, err := Build(env.game, env.frame)
	if err != nil {
		t.Fatalf("Build() error = %v", err)
	}
	if !reflect.DeepEqual(got.Observation, want) {
		t.Fatal("terminal observation differs from Build of the final game")
	}

	// The same result must reach Python unchanged through the step frame.
	encoded, err := encodeBinaryStepResponse(results)
	if err != nil {
		t.Fatalf("encodeBinaryStepResponse() error = %v", err)
	}
	wire := decodeBinaryTestStepResponse(t, encoded.bytes())[0]
	if wire.Reward != got.Reward || wire.Terminated != got.Terminated || wire.Truncated != got.Truncated ||
		wire.EndReason != got.EndReason || wire.Turn != got.Turn || !reflect.DeepEqual(wire.Observation, got.Observation) {
		t.Fatalf("wire result = %+v, want %+v", wire, got)
	}

	// An ended episode refuses further steps until it is reset.
	_, err = batch.Step(ctx, map[int][BodySlots]SlotAction{0: {}})
	assertEnvErrorKind(t, err, ErrorTerminated)
	if _, err := batch.Reset(ctx, []int{0}, nil); err != nil {
		t.Fatalf("Reset() after elimination error = %v", err)
	}
	if _, err := batch.Step(ctx, map[int][BodySlots]SlotAction{0: {}}); err != nil {
		t.Fatalf("Step() after reset error = %v", err)
	}
}

// TestTerminalReward pins that only finishing strictly first by ships earns
// +1, on both map sizes. The four-player rows matter to rotate validation,
// which counts a game as a win exactly when its reward is positive.
func TestTerminalReward(t *testing.T) {
	for _, test := range []struct {
		name    string
		ships   []int // one count per seat
		learner int
		want    float32
	}{
		{name: "win", ships: []int{10, 1}, learner: 0, want: 1},
		{name: "loss", ships: []int{10, 1}, learner: 1, want: -1},
		{name: "tie", ships: []int{10, 10}, learner: 0, want: 0},
		{name: "4p sole winner", ships: []int{1, 5, 10, 5}, learner: 2, want: 1},
		{name: "4p tied for first", ships: []int{10, 3, 10, 1}, learner: 2, want: 0},
		{name: "4p opponents tied above", ships: []int{10, 10, 5, 1}, learner: 2, want: -1},
	} {
		t.Run(test.name, func(t *testing.T) {
			g := finishedGame(t, test.ships)
			if got := terminalReward(g, test.learner); got != test.want {
				t.Fatalf("terminalReward() = %v, want %v", got, test.want)
			}
		})
	}
}

// TestEpisodeEnd pins the order of the end checks and the ownership rule
// that keeps a learner alive. Each case starts from a four-player board at
// turn 0 and changes what the learner, seat 0, owns.
func TestEpisodeEnd(t *testing.T) {
	tests := []struct {
		name       string
		change     func(g *game.Game)
		wantReason EndReason
		wantReward float32
	}{
		{name: "playing on", change: func(*game.Game) {}, wantReason: EndOngoing},
		{
			// An owned planet with no ships still produces, so its owner plays on.
			name:       "zero-ship owned planet",
			change:     func(g *game.Game) { g.Planets[0].Ships = 0 },
			wantReason: EndOngoing,
		},
		{
			name: "fleet only",
			change: func(g *game.Game) {
				g.Planets[0].OwnerID = game.NeutralPlayer
				g.Fleets = append(g.Fleets, game.Fleet{ID: 100, OwnerID: 0, Ships: 5, Position: game.Vec2{X: 30, Y: 30}, LaunchTurn: -1})
			},
			wantReason: EndOngoing,
		},
		{
			name: "comet only",
			change: func(g *game.Game) {
				g.Planets[0].OwnerID = game.NeutralPlayer
				g.Comets = append(g.Comets, game.Comet{ID: 100, OwnerID: 0, Ships: 5, Radius: 1, Production: 1,
					Position: game.Vec2{X: 30, Y: 30}, Path: []game.Vec2{{X: 30, Y: 30}}})
			},
			wantReason: EndOngoing,
		},
		{
			name:       "learner eliminated while others play on",
			change:     func(g *game.Game) { g.Planets[0].OwnerID = game.NeutralPlayer },
			wantReason: EndLearnerEliminated, wantReward: -1,
		},
		{
			// Nobody owns anything, so the game itself is over with no winner.
			// That is a tie, not a learner loss, because the game check runs
			// before the learner-elimination check.
			name: "everyone wiped out at once",
			change: func(g *game.Game) {
				for i := range g.Planets {
					g.Planets[i].OwnerID = game.NeutralPlayer
				}
			},
			wantReason: EndGameFinished, wantReward: 0,
		},
		{
			name: "learner last one standing",
			change: func(g *game.Game) {
				for i := 1; i < len(g.Planets); i++ {
					g.Planets[i].OwnerID = game.NeutralPlayer
				}
			},
			wantReason: EndGameFinished, wantReward: 1,
		},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			g, err := testScenario{id: "end-v1", maxTurns: 10, players: 4}.New(1)
			if err != nil {
				t.Fatal(err)
			}
			test.change(g)
			reason, reward := episodeEnd(g, 0)
			if reason != test.wantReason || reward != test.wantReward {
				t.Fatalf("episodeEnd() = reason %d reward %v, want reason %d reward %v", reason, reward, test.wantReason, test.wantReward)
			}
		})
	}
}

func TestResetSchedulingAndSortedAllocation(t *testing.T) {
	ctx := context.Background()
	factory := testScenario{id: "schedule-v1", maxTurns: 2}
	for _, test := range []struct {
		name     string
		mode     ScheduleMode
		wantSeed []int64
		wantSeat []int
	}{
		{name: "train-v2", mode: ScheduleTrainV2, wantSeed: []int64{100, 101, 102}, wantSeat: []int{1, 1, 0}},
		{name: "rotate", mode: ScheduleRotate, wantSeed: []int64{100, 100, 101}, wantSeat: []int{0, 1, 0}},
	} {
		t.Run(test.name, func(t *testing.T) {
			makeReset := func(order []int) ResetResult {
				batch := newTestBatch(t, factory, agent.IdleFactory{}, 3, scenario.SeedRange{Start: 100, End: 104}, test.mode, 0, 1)
				result, err := batch.Reset(ctx, order, nil)
				if err != nil {
					t.Fatalf("Reset(%v) error = %v", order, err)
				}
				return result
			}
			first := makeReset([]int{2, 0, 1})
			second := makeReset([]int{1, 2, 0})
			if !reflect.DeepEqual(episodeInfos(first), episodeInfos(second)) {
				t.Fatalf("request order changed assignments: %v versus %v", episodeInfos(first), episodeInfos(second))
			}
			for i, env := range first.Envs {
				if env.EnvID != i || env.Seed != test.wantSeed[i] || env.Seat != test.wantSeat[i] || env.Explicit {
					t.Errorf("env %d assignment = %+v, want seed %d seat %d scheduled", i, env.EpisodeInfo, test.wantSeed[i], test.wantSeat[i])
				}
			}
		})
	}
}

// TestScheduleGoldenVectors freezes the exact cursor mapping shared with the
// Python trainer. The expected seats were computed independently in Python
// from the formula in trainV2Seat. Values around epoch boundaries catch
// changes to the domain text, integer encoding, byte order, digest
// extraction, modulo, or repetition rule; rotate vectors protect the
// measurement schedule at the same time.
func TestScheduleGoldenVectors(t *testing.T) {
	standard := scenario.SeedRange{Start: 100, End: 104}
	tests := []struct {
		name     string
		seeds    scenario.SeedRange
		mode     ScheduleMode
		cursor   int64
		players  int
		wantSeed int64
		wantSeat int
	}{
		{name: "train 2p epoch 0 first", seeds: standard, mode: ScheduleTrainV2, cursor: 0, players: 2, wantSeed: 100, wantSeat: 1},
		{name: "train 2p epoch 0 second", seeds: standard, mode: ScheduleTrainV2, cursor: 1, players: 2, wantSeed: 101, wantSeat: 1},
		{name: "train 2p epoch 0 last", seeds: standard, mode: ScheduleTrainV2, cursor: 3, players: 2, wantSeed: 103, wantSeat: 0},
		{name: "train 2p epoch 1 first", seeds: standard, mode: ScheduleTrainV2, cursor: 4, players: 2, wantSeed: 100, wantSeat: 1},
		{name: "train 2p epoch 1 last", seeds: standard, mode: ScheduleTrainV2, cursor: 7, players: 2, wantSeed: 103, wantSeat: 1},
		{name: "train 2p epoch 2 first", seeds: standard, mode: ScheduleTrainV2, cursor: 8, players: 2, wantSeed: 100, wantSeat: 0},
		{name: "train 2p negative seed encoding", seeds: scenario.SeedRange{Start: -2, End: 2}, mode: ScheduleTrainV2, cursor: 0, players: 2, wantSeed: -2, wantSeat: 1},
		{name: "train 4p epoch 0 first", seeds: standard, mode: ScheduleTrainV2, cursor: 0, players: 4, wantSeed: 100, wantSeat: 3},
		{name: "train 4p epoch 0 second", seeds: standard, mode: ScheduleTrainV2, cursor: 1, players: 4, wantSeed: 101, wantSeat: 1},
		{name: "train 4p epoch 0 last", seeds: standard, mode: ScheduleTrainV2, cursor: 3, players: 4, wantSeed: 103, wantSeat: 2},
		{name: "train 4p epoch 1 first", seeds: standard, mode: ScheduleTrainV2, cursor: 4, players: 4, wantSeed: 100, wantSeat: 3},
		{name: "train 4p epoch 1 last", seeds: standard, mode: ScheduleTrainV2, cursor: 7, players: 4, wantSeed: 103, wantSeat: 1},
		{name: "train 4p epoch 2 first", seeds: standard, mode: ScheduleTrainV2, cursor: 8, players: 4, wantSeed: 100, wantSeat: 2},
		{name: "train 4p negative seed encoding", seeds: scenario.SeedRange{Start: -2, End: 2}, mode: ScheduleTrainV2, cursor: 0, players: 4, wantSeed: -2, wantSeat: 1},
		{name: "rotate 2p first seat", seeds: standard, mode: ScheduleRotate, cursor: 0, players: 2, wantSeed: 100, wantSeat: 0},
		{name: "rotate 2p second seat", seeds: standard, mode: ScheduleRotate, cursor: 1, players: 2, wantSeed: 100, wantSeat: 1},
		{name: "rotate 2p next seed", seeds: standard, mode: ScheduleRotate, cursor: 2, players: 2, wantSeed: 101, wantSeat: 0},
		{name: "rotate 4p seat 0", seeds: standard, mode: ScheduleRotate, cursor: 0, players: 4, wantSeed: 100, wantSeat: 0},
		{name: "rotate 4p seat 1", seeds: standard, mode: ScheduleRotate, cursor: 1, players: 4, wantSeed: 100, wantSeat: 1},
		{name: "rotate 4p seat 2", seeds: standard, mode: ScheduleRotate, cursor: 2, players: 4, wantSeed: 100, wantSeat: 2},
		{name: "rotate 4p seat 3", seeds: standard, mode: ScheduleRotate, cursor: 3, players: 4, wantSeed: 100, wantSeat: 3},
		{name: "rotate 4p next seed", seeds: standard, mode: ScheduleRotate, cursor: 4, players: 4, wantSeed: 101, wantSeat: 0},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			plan, err := scheduledEpisode(test.seeds, test.mode, test.cursor, test.players)
			if err != nil {
				t.Fatalf("scheduledEpisode() error = %v", err)
			}
			seat, err := plan.seatFor(test.players)
			if err != nil {
				t.Fatalf("seatFor(%d) error = %v", test.players, err)
			}
			if plan.info.Seed != test.wantSeed || seat != test.wantSeat {
				t.Fatalf("schedule = seed %d seat %d, want seed %d seat %d", plan.info.Seed, seat, test.wantSeed, test.wantSeat)
			}
		})
	}
}

// TestTrainV2EpochCoverageAndSeatBalance checks the standard training range,
// not a toy substitute. Every epoch must walk each seed exactly once before
// repeating, while the hash-derived seats stay balanced over both player
// counts without relying on cursor parity or forced alternation.
func TestTrainV2EpochCoverageAndSeatBalance(t *testing.T) {
	seeds := scenario.StandardRanges().Training
	length := seeds.Len()
	for epoch := range int64(3) {
		var seatCounts [2][PlayerSlots]int64 // [two players, four players][seat]
		visited := make([]bool, length)
		for offset := range length {
			cursor := epoch*length + offset
			plan, err := scheduledEpisode(seeds, ScheduleTrainV2, cursor, 2)
			if err != nil {
				t.Fatalf("epoch %d offset %d: scheduledEpisode() error = %v", epoch, offset, err)
			}
			index := plan.info.Seed - seeds.Start
			if index < 0 || index >= length {
				t.Fatalf("epoch %d offset %d assigned out-of-range seed %d", epoch, offset, plan.info.Seed)
			}
			if visited[index] {
				t.Fatalf("epoch %d assigned seed %d more than once", epoch, plan.info.Seed)
			}
			visited[index] = true
			for i, players := range []int{2, 4} {
				seat, err := plan.seatFor(players)
				if err != nil {
					t.Fatalf("epoch %d offset %d: seatFor(%d) error = %v", epoch, offset, players, err)
				}
				seatCounts[i][seat]++
			}
		}
		for index, seen := range visited {
			if !seen {
				t.Fatalf("epoch %d did not assign seed %d", epoch, seeds.Start+int64(index))
			}
		}
		for i, players := range []int{2, 4} {
			counts := seatCounts[i][:players]
			imbalance := slices.Max(counts) - slices.Min(counts)
			if limit := length / 100; imbalance > limit {
				t.Errorf("epoch %d %d-player seat counts = %v, imbalance %d exceeds 1%% limit %d", epoch, players, counts, imbalance, limit)
			}
		}
	}

	firstRepeat, err := scheduledEpisode(seeds, ScheduleTrainV2, length, 2)
	if err != nil {
		t.Fatalf("first assignment of epoch 1: %v", err)
	}
	if firstRepeat.info.Seed != seeds.Start {
		t.Fatalf("first repeated seed = %d, want first range seed %d", firstRepeat.info.Seed, seeds.Start)
	}
}

func TestNewBatchRejectsRemovedAndUnknownSchedules(t *testing.T) {
	for _, mode := range []ScheduleMode{"train", "train-v1", "future-v3"} {
		t.Run(string(mode), func(t *testing.T) {
			_, err := NewBatch(context.Background(), BatchConfig{
				Size: 1, Seeds: scenario.SeedRange{Start: 0, End: 10}, Schedule: mode,
				Scenario: testScenario{id: "schedule-rejection-v1", maxTurns: 2},
				Opponent: agent.IdleFactory{},
			})
			assertEnvErrorKind(t, err, ErrorValidation)
			if !strings.Contains(err.Error(), "unknown schedule") || !strings.Contains(err.Error(), string(mode)) {
				t.Fatalf("NewBatch() error = %q, want rejected schedule %q named clearly", err, mode)
			}
		})
	}
}

// TestNewBatchRejectsUnsupportedPlayerCounts also covers a nil factory: the
// schedule length needs Players(), so the nil check must run first rather
// than panic.
func TestNewBatchRejectsUnsupportedPlayerCounts(t *testing.T) {
	declaring := func(players int) SeatedScenarioFactory {
		return misdeclaredScenario{testScenario: testScenario{id: "players-rejection-v1", maxTurns: 2}, declared: players}
	}
	for _, test := range []struct {
		name       string
		factory    SeatedScenarioFactory
		wantInText string
	}{
		{name: "nil factory", factory: nil, wantInText: "must not be nil"},
		{name: "0 players", factory: declaring(0), wantInText: "declares 0 players"},
		{name: "3 players", factory: declaring(3), wantInText: "declares 3 players"},
		{name: "5 players", factory: declaring(5), wantInText: "declares 5 players"},
	} {
		t.Run(test.name, func(t *testing.T) {
			_, err := NewBatch(context.Background(), BatchConfig{
				Size: 1, Seeds: scenario.SeedRange{Start: 0, End: 10}, Schedule: ScheduleRotate,
				Scenario: test.factory, Opponent: agent.IdleFactory{},
			})
			assertEnvErrorKind(t, err, ErrorValidation)
			if !strings.Contains(err.Error(), test.wantInText) {
				t.Fatalf("NewBatch() error = %q, want it to contain %q", err, test.wantInText)
			}
		})
	}
}

func TestScheduledEpisodeRejectsNegativeCursorAsValidation(t *testing.T) {
	_, err := scheduledEpisode(scenario.SeedRange{Start: 0, End: 10}, ScheduleTrainV2, -1, 2)
	assertEnvErrorKind(t, err, ErrorValidation)
}

func TestScheduleLengthHandlesEveryInputDirectly(t *testing.T) {
	// The 4p saturation range is the smallest one whose seeds × 4 overflows,
	// while the 2p rotate row just below it still fits.
	overflowsAtFour := scenario.SeedRange{Start: 0, End: math.MaxInt64/4 + 1}
	tests := []struct {
		name    string
		seeds   scenario.SeedRange
		mode    ScheduleMode
		players int
		want    int64
	}{
		{name: "empty training", seeds: scenario.SeedRange{}, mode: ScheduleTrainV2, players: 2, want: 0},
		{name: "empty rotate", seeds: scenario.SeedRange{}, mode: ScheduleRotate, players: 2, want: 0},
		{name: "repeating training", seeds: scenario.SeedRange{Start: 10, End: 12}, mode: ScheduleTrainV2, players: 2, want: math.MaxInt64},
		{name: "finite rotate 2p", seeds: scenario.SeedRange{Start: 10, End: 12}, mode: ScheduleRotate, players: 2, want: 4},
		{name: "finite rotate 4p", seeds: scenario.SeedRange{Start: 10, End: 12}, mode: ScheduleRotate, players: 4, want: 8},
		{name: "rotate 2p length saturation", seeds: scenario.SeedRange{Start: math.MinInt64, End: math.MaxInt64}, mode: ScheduleRotate, players: 2, want: math.MaxInt64},
		{name: "rotate 2p same range fits", seeds: overflowsAtFour, mode: ScheduleRotate, players: 2, want: (math.MaxInt64/4 + 1) * 2},
		{name: "rotate 4p length saturation", seeds: overflowsAtFour, mode: ScheduleRotate, players: 4, want: math.MaxInt64},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			got, err := scheduleLength(test.seeds, test.mode, test.players)
			if err != nil {
				t.Fatalf("scheduleLength() error = %v", err)
			}
			if got != test.want {
				t.Fatalf("scheduleLength() = %d, want %d", got, test.want)
			}
		})
	}

	_, err := scheduleLength(scenario.SeedRange{}, ScheduleMode("future-v3"), 2)
	assertEnvErrorKind(t, err, ErrorValidation)
}

// TestResetExhaustionIsStructuredAndDoesNotPartiallyAdvance starts a
// one-seed rotate schedule at its last cursor, players-1. Two resets would
// need the cursor one past the end, so the pair is refused whole; one reset
// still gets the last assignment, and the next one finds the schedule
// exhausted.
func TestResetExhaustionIsStructuredAndDoesNotPartiallyAdvance(t *testing.T) {
	for _, players := range []int{2, 4} {
		t.Run(fmt.Sprintf("%dp", players), func(t *testing.T) {
			ctx := context.Background()
			last := int64(players - 1)
			batch := newTestBatch(t, testScenario{id: "exhaust-v1", maxTurns: 2, players: players}, agent.IdleFactory{}, 2, scenario.SeedRange{Start: 7, End: 8}, ScheduleRotate, last, 1)
			_, err := batch.Reset(ctx, []int{0, 1}, nil)
			assertEnvErrorKind(t, err, ErrorExhausted)
			if got := batch.Cursor(); got != last {
				t.Fatalf("cursor after rejected reset = %d, want starting cursor %d", got, last)
			}
			if batch.envs[0] != nil || batch.envs[1] != nil {
				t.Fatal("exhausted reset partially installed environments")
			}

			reset, err := batch.Reset(ctx, []int{0}, nil)
			if err != nil {
				t.Fatalf("Reset() of the last assignment error = %v", err)
			}
			if info := reset.Envs[0].EpisodeInfo; info.Seed != 7 || info.Seat != players-1 || reset.Cursor != int64(players) {
				t.Fatalf("last assignment = %+v cursor %d, want seed 7 seat %d cursor %d", info, reset.Cursor, players-1, players)
			}
			_, err = batch.Reset(ctx, []int{1}, nil)
			assertEnvErrorKind(t, err, ErrorExhausted)
		})
	}
}

func TestTrainV2MaxCursorExhaustionDoesNotPartiallyReset(t *testing.T) {
	batch := newTestBatch(t, testScenario{id: "max-cursor-v1", maxTurns: 2}, agent.IdleFactory{}, 2, scenario.SeedRange{Start: 7, End: 9}, ScheduleTrainV2, math.MaxInt64-1, 1)
	if _, err := batch.Reset(context.Background(), []int{0}, nil); err != nil {
		t.Fatalf("initial Reset() error = %v", err)
	}
	installed := batch.envs[0]

	_, err := batch.Reset(context.Background(), []int{1, 0}, nil)
	assertEnvErrorKind(t, err, ErrorExhausted)
	if got := batch.Cursor(); got != math.MaxInt64 {
		t.Fatalf("cursor after rejected reset = %d, want %d", got, int64(math.MaxInt64))
	}
	if batch.envs[0] != installed {
		t.Fatal("exhausted reset replaced the installed environment")
	}
	if batch.envs[1] != nil {
		t.Fatal("exhausted reset partially installed a new environment")
	}
}

func TestExplicitResetBypassesCursorAndIsMarked(t *testing.T) {
	batch := newTestBatch(t, testScenario{id: "explicit-v1", maxTurns: 2}, agent.IdleFactory{}, 2, scenario.SeedRange{Start: 10, End: 20}, ScheduleTrainV2, 3, 1)
	result, err := batch.Reset(context.Background(), []int{1, 0}, []EpisodeOverride{{EnvID: 0, Seed: 4711, Seat: 1}})
	if err != nil {
		t.Fatalf("Reset() error = %v", err)
	}
	want := []EpisodeInfo{
		{EnvID: 0, Seed: 4711, Seat: 1, Explicit: true},
		{EnvID: 1, Seed: 13, Seat: 1, Explicit: false},
	}
	if got := episodeInfos(result); !reflect.DeepEqual(got, want) {
		t.Fatalf("assignments = %+v, want %+v", got, want)
	}
	if result.Cursor != 4 {
		t.Fatalf("cursor = %d, want 4 (only scheduled reset advances it)", result.Cursor)
	}
}

// TestResetCreatesOneOpponentPerOtherSeat checks who plays which seat. Each
// opponent must be asked to play its own seat, under its own contestant ID
// and seed, and the learner's seat must be left to the learner.
func TestResetCreatesOneOpponentPerOtherSeat(t *testing.T) {
	tests := []struct {
		name    string
		players int
		learner int
		want    []int // opponent seats, ascending
	}{
		{name: "two players", players: 2, learner: 1, want: []int{0}},
		{name: "four players", players: 4, learner: 2, want: []int{0, 1, 3}},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			opponents := &observedOpponentFactory{}
			batch := newTestBatch(t, testScenario{id: "seats-v1", maxTurns: 5, players: test.players}, opponents, 1, scenario.SeedRange{Start: 0, End: 10}, ScheduleTrainV2, 0, 1)
			const seed = 7
			reset, err := batch.Reset(context.Background(), []int{0}, []EpisodeOverride{{EnvID: 0, Seed: seed, Seat: test.learner}})
			if err != nil {
				t.Fatalf("Reset() error = %v", err)
			}
			if info := reset.Envs[0].EpisodeInfo; info.Seat != test.learner || !info.Explicit {
				t.Fatalf("episode = %+v, want explicit seat %d", info, test.learner)
			}
			inits := opponents.initsSnapshot()
			if len(inits) != len(test.want) {
				t.Fatalf("created %d opponents, want %d", len(inits), len(test.want))
			}
			seen := map[int64]bool{}
			for i, init := range inits {
				contestant := fmt.Sprintf("opponent-seat-%d", test.want[i])
				if init.PlayerID != test.want[i] || init.ContestantID != contestant {
					t.Fatalf("opponent %d = player %d contestant %q, want player %d contestant %q", i, init.PlayerID, init.ContestantID, test.want[i], contestant)
				}
				if want := arena.DeriveAgentSeed(seed, contestant, opponents.Spec()); init.AgentSeed != want {
					t.Fatalf("opponent %d seed = %d, want %d", i, init.AgentSeed, want)
				}
				if seen[init.AgentSeed] {
					t.Fatalf("opponent %d repeats seed %d", i, init.AgentSeed)
				}
				seen[init.AgentSeed] = true
			}
			if batch.envs[0].opponents[test.learner] != nil {
				t.Fatal("an opponent was created for the learner's seat")
			}
		})
	}
}

// TestResetRejectsSeatsTheMapCannotSeat covers seats and rosters that can
// only be checked once the map exists. Env 0's replacement is built first, opponents
// and all; env 1 is then rejected, so the reset must close that replacement's
// opponents and leave the installed episode and the cursor as they were.
func TestResetRejectsSeatsTheMapCannotSeat(t *testing.T) {
	tests := []struct {
		name     string
		players  int
		schedule ScheduleMode
		// bad is env 1's explicit override, or nil to leave env 1 to the schedule.
		bad *EpisodeOverride
		// shrink makes seed 0, env 1's first scheduled seed, a two-player map
		// although the factory declares players.
		shrink     bool
		replaced   int // opponents created for env 0's discarded replacement
		wantInText string
	}{
		{
			name: "explicit seat not on a two-player map", players: 2, schedule: ScheduleTrainV2,
			bad: &EpisodeOverride{EnvID: 1, Seed: 3, Seat: 3}, replaced: 1,
			wantInText: "seat 3 is not seated on this 2-player map",
		},
		{
			name: "map seats fewer players than declared", players: 4, schedule: ScheduleRotate,
			shrink: true, replaced: 3,
			wantInText: "declares 4 players but its map for seed 0 seats 2",
		},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			ctx := context.Background()
			opponents := &observedOpponentFactory{}
			var factory SeatedScenarioFactory = testScenario{id: "seat-rejection-v1", maxTurns: 5, players: test.players}
			if test.shrink {
				factory = shrinkingScenario{testScenario: factory.(testScenario), shrinkSeed: 0}
			}
			batch := newTestBatch(t, factory, opponents, 2, scenario.SeedRange{Start: 0, End: 10}, test.schedule, 0, 1)
			if _, err := batch.Reset(ctx, []int{0}, []EpisodeOverride{{EnvID: 0, Seed: 5, Seat: 1}}); err != nil {
				t.Fatalf("initial Reset() error = %v", err)
			}
			installed := batch.envs[0]
			installedOpponents := len(opponents.initsSnapshot())

			overrides := []EpisodeOverride{{EnvID: 0, Seed: 6, Seat: 0}}
			if test.bad != nil {
				overrides = append(overrides, *test.bad)
			}
			_, err := batch.Reset(ctx, []int{0, 1}, overrides)
			assertEnvErrorKind(t, err, ErrorValidation)
			if !strings.Contains(err.Error(), test.wantInText) {
				t.Fatalf("Reset() error = %q, want it to contain %q", err, test.wantInText)
			}
			if batch.envs[0] != installed || batch.envs[1] != nil {
				t.Fatal("rejected reset changed the installed episodes")
			}
			if got := batch.Cursor(); got != 0 {
				t.Fatalf("cursor after rejected reset = %d, want 0", got)
			}
			if got := len(opponents.initsSnapshot()) - installedOpponents; got != test.replaced {
				t.Fatalf("rejected reset created %d opponents, want %d for env 0's replacement", got, test.replaced)
			}
			if got := opponents.closeCount(); got != test.replaced {
				t.Fatalf("closed %d opponents, want the %d of the discarded replacement", got, test.replaced)
			}
		})
	}
}

func TestCursorResumeMatchesUninterruptedSchedule(t *testing.T) {
	ctx := context.Background()
	for _, test := range []struct {
		mode    ScheduleMode
		players int
	}{
		{mode: ScheduleTrainV2, players: 2},
		{mode: ScheduleRotate, players: 2},
		// Cursor 5 on four players is seed 1001, seat 1: mid-seed, so a resume
		// must pick up the rotation where it stopped rather than at seat 0.
		{mode: ScheduleRotate, players: 4},
	} {
		mode := test.mode
		factory := testScenario{id: "resume-v1", maxTurns: 2, players: test.players}
		t.Run(fmt.Sprintf("%s/%dp", mode, test.players), func(t *testing.T) {
			const cursor = int64(5)
			uninterrupted := newTestBatch(t, factory, agent.IdleFactory{}, 1, scenario.SeedRange{Start: 1000, End: 1020}, mode, 0, 1)
			for range cursor {
				if _, err := uninterrupted.Reset(ctx, []int{0}, nil); err != nil {
					t.Fatalf("advance Reset() error = %v", err)
				}
			}
			continued, err := uninterrupted.Reset(ctx, []int{0}, nil)
			if err != nil {
				t.Fatalf("continued Reset() error = %v", err)
			}

			resumed := newTestBatch(t, factory, agent.IdleFactory{}, 1, scenario.SeedRange{Start: 1000, End: 1020}, mode, cursor, 1)
			restarted, err := resumed.Reset(ctx, []int{0}, nil)
			if err != nil {
				t.Fatalf("resumed Reset() error = %v", err)
			}
			if got, want := episodeInfos(restarted), episodeInfos(continued); !reflect.DeepEqual(got, want) {
				t.Fatalf("resumed assignment = %+v, uninterrupted assignment = %+v", got, want)
			}
		})
	}
}

func TestExplicitReplayReproducesTerminalState(t *testing.T) {
	ctx := context.Background()
	factory := testScenario{id: "replay-v1", maxTurns: 5}
	opponent := heuristicFactory(t)
	original := newTestBatch(t, factory, opponent, 1, scenario.SeedRange{Start: 60, End: 70}, ScheduleTrainV2, 0, 1)
	reset, err := original.Reset(ctx, []int{0}, nil)
	if err != nil {
		t.Fatalf("original Reset() error = %v", err)
	}
	decisions, firstResults := playToCompletion(t, original)

	replay := newTestBatch(t, factory, opponent, 1, scenario.SeedRange{Start: 900, End: 910}, ScheduleTrainV2, 0, 1)
	replayedReset, err := replay.Reset(ctx, []int{0}, []EpisodeOverride{{EnvID: 0, Seed: reset.Envs[0].Seed, Seat: reset.Envs[0].Seat}})
	if err != nil {
		t.Fatalf("replay Reset() error = %v", err)
	}
	if !replayedReset.Envs[0].Explicit || replay.Cursor() != 0 {
		t.Fatalf("replay identity = %+v cursor %d, want explicit and cursor 0", replayedReset.Envs[0].EpisodeInfo, replay.Cursor())
	}
	secondResults := replayDecisions(t, replay, decisions)

	for i := range secondResults {
		// Explicit is provenance, not game state: a correct replay must carry a
		// different marker from the originally scheduled episode.
		secondResults[i].Explicit = firstResults[i].Explicit
	}
	if !reflect.DeepEqual(firstResults, secondResults) {
		t.Fatalf("replayed step results differ\noriginal: %#v\nreplay: %#v", firstResults, secondResults)
	}
	if got, want := agent.NewView(replay.envs[0].game), agent.NewView(original.envs[0].game); !reflect.DeepEqual(got, want) {
		t.Fatalf("replayed terminal state differs\nreplay: %#v\noriginal: %#v", got, want)
	}
}

// TestBatchDeterministicAcrossWorkerCounts also checks every returned
// observation against a fresh Build of the resulting game, which guards
// against stale masks or garrisons slipping into a result.
func TestBatchDeterministicAcrossWorkerCounts(t *testing.T) {
	tests := []struct {
		schedule ScheduleMode
		players  int
	}{
		{schedule: ScheduleTrainV2, players: 2},
		{schedule: ScheduleRotate, players: 2},
		{schedule: ScheduleTrainV2, players: 4},
	}
	for _, test := range tests {
		t.Run(fmt.Sprintf("%s/%dp", test.schedule, test.players), func(t *testing.T) {
			ctx := context.Background()
			const size = 8
			factory := testScenario{id: "workers-v1", maxTurns: 6, players: test.players}
			opponent := heuristicFactory(t)
			one := newTestBatch(t, factory, opponent, size, scenario.SeedRange{Start: 200, End: 300}, test.schedule, 0, 1)
			many := newTestBatch(t, factory, opponent, size, scenario.SeedRange{Start: 200, End: 300}, test.schedule, 0, 4)
			ids := make([]int, size)
			for i := range ids {
				ids[i] = size - i - 1
			}
			oneReset, err := one.Reset(ctx, ids, nil)
			if err != nil {
				t.Fatalf("one-worker Reset() error = %v", err)
			}
			manyReset, err := many.Reset(ctx, ids, nil)
			if err != nil {
				t.Fatalf("many-worker Reset() error = %v", err)
			}
			if !reflect.DeepEqual(oneReset, manyReset) {
				t.Fatal("worker count changed reset results")
			}

			active := make(map[int][BodySlots]SlotAction, size)
			for id := range size {
				active[id] = [BodySlots]SlotAction{}
			}
			for len(active) > 0 {
				oneResults, err := one.Step(ctx, active)
				if err != nil {
					t.Fatalf("one-worker Step() error = %v", err)
				}
				manyResults, err := many.Step(ctx, active)
				if err != nil {
					t.Fatalf("many-worker Step() error = %v", err)
				}
				if !reflect.DeepEqual(oneResults, manyResults) {
					t.Fatalf("worker count changed step results\none: %#v\nmany: %#v", oneResults, manyResults)
				}
				for _, result := range oneResults {
					env := one.envs[result.EnvID]
					fresh, err := Build(env.game, env.frame)
					if err != nil {
						t.Fatalf("env %d: Build() error = %v", result.EnvID, err)
					}
					if result.Observation != fresh {
						t.Fatalf("env %d turn %d: returned observation differs from the game state", result.EnvID, result.Turn)
					}
					if result.Terminated {
						delete(active, result.EnvID)
					}
				}
			}
			for id := range size {
				if got, want := agent.NewView(many.envs[id].game), agent.NewView(one.envs[id].game); !reflect.DeepEqual(got, want) {
					t.Fatalf("env %d terminal state differs by worker count", id)
				}
			}
		})
	}
}

func TestBatchRunsIndependentOpponentDecisionsConcurrently(t *testing.T) {
	ctx := context.Background()
	opponents := newBlockingOpponentFactory(2)
	defer opponents.unblock()
	batch := newTestBatch(t, testScenario{id: "parallel-opponents-v1", maxTurns: 3}, opponents, 2, scenario.SeedRange{Start: 0, End: 10}, ScheduleTrainV2, 0, 2)
	if _, err := batch.Reset(ctx, []int{0, 1}, nil); err != nil {
		t.Fatalf("Reset() error = %v", err)
	}

	stepDone := make(chan error, 1)
	go func() {
		_, err := batch.Step(ctx, map[int][BodySlots]SlotAction{0: {}, 1: {}})
		stepDone <- err
	}()

	// Both agents block inside Decide. Receiving both start notifications before
	// either is released proves the batch did not call them one after another.
	for decision := 1; decision <= 2; decision++ {
		select {
		case <-opponents.started:
		case <-time.After(time.Second):
			t.Fatalf("only %d opponent decisions started concurrently", decision-1)
		}
	}
	opponents.unblock()
	if err := <-stepDone; err != nil {
		t.Fatalf("Step() error = %v", err)
	}
}

// TestStepDecodesWholeBatchBeforeStatefulOpponentDecisions uses four-player
// games, so six opponents across two environments must all stay untouched
// when one environment's learner decision is invalid.
func TestStepDecodesWholeBatchBeforeStatefulOpponentDecisions(t *testing.T) {
	ctx := context.Background()
	opponents := &observedOpponentFactory{}
	batch := newTestBatch(t, testScenario{id: "decode-first-v1", maxTurns: 3, players: 4}, opponents, 2, scenario.SeedRange{Start: 0, End: 10}, ScheduleTrainV2, 0, 1)
	reset, err := batch.Reset(ctx, []int{0, 1}, nil)
	if err != nil {
		t.Fatalf("Reset() error = %v", err)
	}

	var invalid [BodySlots]SlotAction
	obs := reset.Envs[1].Observation
	invalid[unownedSlot(t, obs)] = SlotAction{Launch: true, Target: 0}
	_, err = batch.Step(ctx, map[int][BodySlots]SlotAction{0: {}, 1: invalid})
	assertEnvErrorKind(t, err, ErrorDecode)
	if got := opponents.decisionCounts(); !reflect.DeepEqual(got, []int{0, 0, 0, 0, 0, 0}) {
		t.Fatalf("opponent decision counts after rejected decode = %v, want all zero", got)
	}
	if batch.envs[0].game.Turn != 0 || batch.envs[1].game.Turn != 0 {
		t.Fatalf("rejected decode advanced games to turns %d and %d", batch.envs[0].game.Turn, batch.envs[1].game.Turn)
	}

	if _, err := batch.Step(ctx, map[int][BodySlots]SlotAction{0: {}, 1: {}}); err != nil {
		t.Fatalf("corrected Step() error = %v", err)
	}
	if got := opponents.decisionCounts(); !reflect.DeepEqual(got, []int{1, 1, 1, 1, 1, 1}) {
		t.Fatalf("opponent decision counts after corrected step = %v, want all one", got)
	}
}

// TestStepJoinsActionsInSeatOrder has every seat launch one fleet. The engine
// creates fleets in the order actions arrive, so the fleets' owners show the
// order the batch joined the seats in. It also checks that an opponent cannot
// act for another seat.
func TestStepJoinsActionsInSeatOrder(t *testing.T) {
	ctx := context.Background()
	const learner = 2
	newBatch := func(opponents agent.Factory) (*Batch, ResetResult) {
		batch := newTestBatch(t, testScenario{id: "seat-order-v1", maxTurns: 5, players: 4}, opponents, 1, scenario.SeedRange{Start: 0, End: 10}, ScheduleTrainV2, 0, 1)
		reset, err := batch.Reset(ctx, []int{0}, []EpisodeOverride{{EnvID: 0, Seed: 1, Seat: learner}})
		if err != nil {
			t.Fatalf("Reset() error = %v", err)
		}
		return batch, reset
	}
	learnerLaunch := func(obs Observation) map[int][BodySlots]SlotAction {
		var decisions [BodySlots]SlotAction
		source, target := slotsForBodies(t, obs, learner+1, 1)
		decisions[source] = SlotAction{Launch: true, Target: int32(target)}
		return map[int][BodySlots]SlotAction{0: decisions}
	}

	batch, reset := newBatch(launchingOpponentFactory{claim: -1})
	if _, err := batch.Step(ctx, learnerLaunch(reset.Envs[0].Observation)); err != nil {
		t.Fatalf("Step() error = %v", err)
	}
	var owners []int
	for _, fleet := range batch.envs[0].game.Fleets {
		owners = append(owners, fleet.OwnerID)
	}
	if want := []int{0, 1, 2, 3}; !reflect.DeepEqual(owners, want) {
		t.Fatalf("fleet owners in launch order = %v, want %v", owners, want)
	}

	// Seat 1's opponent launches from the learner's home in the learner's name.
	// The engine would accept that, since the learner owns the planet.
	batch, reset = newBatch(launchingOpponentFactory{claim: learner})
	_, err := batch.Step(ctx, learnerLaunch(reset.Envs[0].Observation))
	assertEnvErrorKind(t, err, ErrorPoisoned)
	if !strings.Contains(err.Error(), "claims player 2") {
		t.Fatalf("Step() error = %q, want the false claim named", err)
	}
	if got := batch.envs[0].game.Turn; got != 0 {
		t.Fatalf("game advanced to turn %d after a false claim", got)
	}
}

// TestOpponentFailurePoisonsWholeStepUntilReset makes the last opponent of
// env 0's four-player game fail, after its two earlier opponents have already
// acted, so the whole request must be poisoned.
func TestOpponentFailurePoisonsWholeStepUntilReset(t *testing.T) {
	ctx := context.Background()
	opponents := &observedOpponentFactory{failDecisionNumber: 3}
	batch := newTestBatch(t, testScenario{id: "opponent-failure-v1", maxTurns: 3, players: 4}, opponents, 2, scenario.SeedRange{Start: 0, End: 10}, ScheduleTrainV2, 0, 1)
	if _, err := batch.Reset(ctx, []int{0, 1}, nil); err != nil {
		t.Fatalf("Reset() error = %v", err)
	}

	results, err := batch.Step(ctx, map[int][BodySlots]SlotAction{0: {}, 1: {}})
	assertEnvErrorKind(t, err, ErrorPoisoned)
	if results != nil {
		t.Fatalf("failed step results = %#v, want nil", results)
	}
	if got := opponents.decisionCounts(); !reflect.DeepEqual(got, []int{1, 1, 1, 1, 1, 1}) {
		t.Fatalf("opponent decision counts = %v, want every opponent asked once", got)
	}
	if batch.envs[0].game.Turn != 0 || batch.envs[1].game.Turn != 0 {
		t.Fatal("opponent failure should occur before any game advances")
	}

	_, err = batch.Step(ctx, map[int][BodySlots]SlotAction{1: {}})
	assertEnvErrorKind(t, err, ErrorPoisoned)
	if got := opponents.decisionCounts(); !reflect.DeepEqual(got, []int{1, 1, 1, 1, 1, 1}) {
		t.Fatalf("poisoned retry contacted an opponent: counts = %v", got)
	}

	if _, err := batch.Reset(ctx, []int{0}, nil); err != nil {
		t.Fatalf("recovery Reset() error = %v", err)
	}
	if _, err := batch.Step(ctx, map[int][BodySlots]SlotAction{0: {}}); err != nil {
		t.Fatalf("Step() after reset error = %v", err)
	}
}

func TestEngineFailureReturnsNoPartialResultsAndPoisonsWholeStep(t *testing.T) {
	ctx := context.Background()
	opponents := &observedOpponentFactory{invalidActionNumber: 2}
	batch := newTestBatch(t, testScenario{id: "engine-failure-v1", maxTurns: 3}, opponents, 2, scenario.SeedRange{Start: 0, End: 10}, ScheduleTrainV2, 0, 1)
	if _, err := batch.Reset(ctx, []int{0, 1}, nil); err != nil {
		t.Fatalf("Reset() error = %v", err)
	}

	results, err := batch.Step(ctx, map[int][BodySlots]SlotAction{0: {}, 1: {}})
	assertEnvErrorKind(t, err, ErrorPoisoned)
	if results != nil {
		t.Fatalf("failed step results = %#v, want nil", results)
	}
	// One worker makes the partial engine outcome deterministic: env 0
	// completed before env 1's invalid opponent action failed validation.
	if got, want := batch.envs[0].game.Turn, 1; got != want {
		t.Fatalf("env 0 turn = %d, want %d", got, want)
	}
	if got, want := batch.envs[1].game.Turn, 0; got != want {
		t.Fatalf("env 1 turn = %d, want %d", got, want)
	}

	_, err = batch.Step(ctx, map[int][BodySlots]SlotAction{0: {}})
	assertEnvErrorKind(t, err, ErrorPoisoned)
	if got := batch.envs[0].game.Turn; got != 1 {
		t.Fatalf("poisoned retry advanced env 0 to turn %d, want 1", got)
	}
}

// TestResetClosesOpponentsWhenConstructionFails covers a factory that
// returns a partly built agent together with its error. That agent must be
// closed, and so must every opponent already created for the same game.
func TestResetClosesOpponentsWhenConstructionFails(t *testing.T) {
	tests := []struct {
		name      string
		players   int
		failOn    int // which creation fails, counting from 1
		wantClose int
	}{
		{name: "only opponent", players: 2, failOn: 1, wantClose: 1},
		{name: "second of three", players: 4, failOn: 2, wantClose: 2},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			opponents := &observedOpponentFactory{failConstructionNumber: test.failOn}
			batch := newTestBatch(t, testScenario{id: "construction-failure-v1", maxTurns: 3, players: test.players}, opponents, 1, scenario.SeedRange{Start: 0, End: 10}, ScheduleTrainV2, 0, 1)
			_, err := batch.Reset(context.Background(), []int{0}, nil)
			assertEnvErrorKind(t, err, ErrorOpponent)
			if got := len(opponents.initsSnapshot()); got != test.failOn {
				t.Fatalf("created %d opponents, want construction to stop at %d", got, test.failOn)
			}
			if got := opponents.closeCount(); got != test.wantClose {
				t.Fatalf("closed %d opponents, want %d", got, test.wantClose)
			}
			if batch.envs[0] != nil {
				t.Fatal("failed reset installed an environment")
			}
		})
	}
}

func heuristicFactory(t testing.TB) agent.Factory {
	t.Helper()
	factory, err := agent.NewHeuristicFactory(agent.HeuristicConfig{
		Aggression: 1.2,
		Economy:    1,
		Efficiency: 1,
		Defense:    0.1,
	})
	if err != nil {
		t.Fatalf("NewHeuristicFactory() error = %v", err)
	}
	return factory
}

// finishedGame plays the one-turn board for len(ships) players (2 or 4),
// with seat p's home starting at ships[p]. Every home produces 1 ship, so the
// final standings keep the order the counts give.
func finishedGame(t *testing.T, ships []int) *game.Game {
	t.Helper()
	factory := testScenario{id: "reward-v1", maxTurns: 1, players: len(ships)}
	g, err := factory.New(1)
	if err != nil {
		t.Fatalf("scenario error = %v", err)
	}
	for seat, count := range ships {
		g.Planets[seat].Ships = count
		g.Planets[seat].Production = 1
	}
	if err := g.AdvanceTurn(nil); err != nil {
		t.Fatalf("AdvanceTurn() error = %v", err)
	}
	return g
}

func slotsForBodies(t *testing.T, obs Observation, sourceID, targetID int) (int, int) {
	t.Helper()
	source, target := -1, -1
	for slot, id := range obs.SlotBodyID {
		if !obs.BodyMask[slot] {
			continue
		}
		switch int(id) {
		case sourceID:
			source = slot
		case targetID:
			target = slot
		}
	}
	if source < 0 || target < 0 {
		t.Fatalf("body slots not found: source %d => %d, target %d => %d", sourceID, source, targetID, target)
	}
	return source, target
}

// unownedSlot returns the first real slot the observer does not own.
func unownedSlot(t *testing.T, obs Observation) int {
	t.Helper()
	for slot := range BodySlots {
		if obs.BodyMask[slot] && !obs.OwnedMask[slot] {
			return slot
		}
	}
	t.Fatal("observation has no unowned body")
	return 0
}

func episodeInfos(result ResetResult) []EpisodeInfo {
	infos := make([]EpisodeInfo, len(result.Envs))
	for i, env := range result.Envs {
		infos[i] = env.EpisodeInfo
	}
	return infos
}

func assertEnvErrorKind(t *testing.T, err error, want ErrorKind) {
	t.Helper()
	var envErr *EnvError
	if !errors.As(err, &envErr) {
		t.Fatalf("error = %v (%T), want *EnvError", err, err)
	}
	if envErr.Kind != want {
		t.Fatalf("error kind = %q, want %q: %v", envErr.Kind, want, err)
	}
}

func playToCompletion(t *testing.T, batch *Batch) ([][BodySlots]SlotAction, []StepResult) {
	t.Helper()
	decisions := make([][BodySlots]SlotAction, 0)
	results := make([]StepResult, 0)
	for !batch.envs[0].terminated {
		decision := [BodySlots]SlotAction{}
		step, err := batch.Step(context.Background(), map[int][BodySlots]SlotAction{0: decision})
		if err != nil {
			t.Fatalf("Step() error = %v", err)
		}
		decisions = append(decisions, decision)
		results = append(results, step[0])
	}
	return decisions, results
}

func replayDecisions(t *testing.T, batch *Batch, decisions [][BodySlots]SlotAction) []StepResult {
	t.Helper()
	results := make([]StepResult, 0, len(decisions))
	for _, decision := range decisions {
		step, err := batch.Step(context.Background(), map[int][BodySlots]SlotAction{0: decision})
		if err != nil {
			t.Fatalf("replay Step() error = %v", err)
		}
		results = append(results, step[0])
	}
	return results
}

const observedOpponentID = "observed-test-v1"

// observedOpponentFactory records every opponent it creates, in creation
// order, and can make one creation or decision fail. The "number" fields
// count from 1 across the whole factory, not per game; zero means never.
type observedOpponentFactory struct {
	mu                     sync.Mutex
	agents                 []*observedOpponent
	inits                  []agent.AgentInit
	failDecisionNumber     int
	invalidActionNumber    int
	failConstructionNumber int
	closed                 int
}

type blockingOpponentFactory struct {
	started     chan struct{}
	release     chan struct{}
	releaseOnce sync.Once
}

func (factory *blockingOpponentFactory) unblock() {
	factory.releaseOnce.Do(func() { close(factory.release) })
}

func newBlockingOpponentFactory(size int) *blockingOpponentFactory {
	return &blockingOpponentFactory{
		started: make(chan struct{}, size),
		release: make(chan struct{}),
	}
}

func (*blockingOpponentFactory) Spec() agent.Spec {
	return agent.NewSpec("blocking-v1", nil)
}

func (factory *blockingOpponentFactory) New(context.Context, agent.AgentInit) (agent.Agent, error) {
	return &blockingOpponent{factory: factory}, nil
}

type blockingOpponent struct {
	factory *blockingOpponentFactory
}

func (*blockingOpponent) ID() string { return "blocking-v1" }

func (opponent *blockingOpponent) Decide(ctx context.Context, _ agent.TurnRequest) (agent.Decision, error) {
	opponent.factory.started <- struct{}{}
	select {
	case <-opponent.factory.release:
		return agent.Decision{}, nil
	case <-ctx.Done():
		return agent.Decision{}, ctx.Err()
	}
}

func (*blockingOpponent) Close() error { return nil }

func (*observedOpponentFactory) Spec() agent.Spec {
	return agent.NewSpec(observedOpponentID, nil)
}

func (factory *observedOpponentFactory) New(ctx context.Context, init agent.AgentInit) (agent.Agent, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	factory.mu.Lock()
	defer factory.mu.Unlock()
	index := len(factory.agents)
	created := &observedOpponent{
		factory:       factory,
		failDecision:  index+1 == factory.failDecisionNumber,
		invalidAction: index+1 == factory.invalidActionNumber,
	}
	factory.agents = append(factory.agents, created)
	factory.inits = append(factory.inits, init)
	if index+1 == factory.failConstructionNumber {
		return created, errors.New("construction failed after allocating opponent")
	}
	return created, nil
}

func (factory *observedOpponentFactory) decisionCounts() []int {
	factory.mu.Lock()
	defer factory.mu.Unlock()
	counts := make([]int, len(factory.agents))
	for i, created := range factory.agents {
		counts[i] = created.decisions
	}
	return counts
}

func (factory *observedOpponentFactory) initsSnapshot() []agent.AgentInit {
	factory.mu.Lock()
	defer factory.mu.Unlock()
	return append([]agent.AgentInit(nil), factory.inits...)
}

func (factory *observedOpponentFactory) closeCount() int {
	factory.mu.Lock()
	defer factory.mu.Unlock()
	return factory.closed
}

type observedOpponent struct {
	factory       *observedOpponentFactory
	failDecision  bool
	invalidAction bool
	decisions     int
}

func (*observedOpponent) ID() string { return observedOpponentID }

func (opponent *observedOpponent) Decide(ctx context.Context, request agent.TurnRequest) (agent.Decision, error) {
	if err := ctx.Err(); err != nil {
		return agent.Decision{}, err
	}
	opponent.factory.mu.Lock()
	opponent.decisions++
	fail, invalid := opponent.failDecision, opponent.invalidAction
	opponent.factory.mu.Unlock()
	if fail {
		return agent.Decision{}, errors.New("stateful opponent failed after consuming state")
	}
	if invalid {
		return agent.Decision{Actions: []game.Action{{
			PlayerID: request.PlayerID,
			PlanetID: 999,
			Ships:    1,
		}}}, nil
	}
	return agent.Decision{}, nil
}

func (opponent *observedOpponent) Close() error {
	opponent.factory.mu.Lock()
	defer opponent.factory.mu.Unlock()
	opponent.factory.closed++
	return nil
}

const launchingOpponentID = "launching-test-v1"

// launchingOpponentFactory makes opponents that launch one ship from their
// own home every turn. On testScenario boards seat p's home is body p+1.
// claim, when not -1, makes the seat-1 opponent launch from claim's home in
// claim's name instead, to test that a seat cannot act for another.
type launchingOpponentFactory struct {
	claim int
}

func (launchingOpponentFactory) Spec() agent.Spec { return agent.NewSpec(launchingOpponentID, nil) }

func (factory launchingOpponentFactory) New(context.Context, agent.AgentInit) (agent.Agent, error) {
	return launchingOpponent(factory), nil
}

type launchingOpponent struct {
	claim int
}

func (launchingOpponent) ID() string { return launchingOpponentID }

func (opponent launchingOpponent) Decide(_ context.Context, request agent.TurnRequest) (agent.Decision, error) {
	player := request.PlayerID
	if opponent.claim >= 0 && player == 1 {
		player = opponent.claim
	}
	return agent.Decision{Actions: []game.Action{{PlayerID: player, PlanetID: player + 1, Ships: 1}}}, nil
}

const strikingOpponentID = "striking-test-v1"

// strikingOpponentFactory makes opponents where only seat 1 acts: on turn 0
// it sends its whole home garrison at seat 0's home, then idles. On the
// four-player close testScenario board that captures seat 0's only planet.
type strikingOpponentFactory struct{}

func (strikingOpponentFactory) Spec() agent.Spec { return agent.NewSpec(strikingOpponentID, nil) }

func (strikingOpponentFactory) New(context.Context, agent.AgentInit) (agent.Agent, error) {
	return strikingOpponent{}, nil
}

type strikingOpponent struct{}

func (strikingOpponent) ID() string { return strikingOpponentID }

func (strikingOpponent) Decide(_ context.Context, request agent.TurnRequest) (agent.Decision, error) {
	if request.PlayerID != 1 || request.View.Turn != 0 {
		return agent.Decision{}, nil
	}
	// View lists planets by ID; on testScenario boards seat p's home is body
	// p+1, so seat 0's home is index 0 and seat 1's is index 1.
	target, source := request.View.Planets[0], request.View.Planets[1]
	angle := math.Atan2(target.Position.Y-source.Position.Y, target.Position.X-source.Position.X)
	return agent.Decision{Actions: []game.Action{{PlayerID: 1, PlanetID: source.ID, Ships: source.Ships, Angle: angle}}}, nil
}
