package arena

import (
	"math"
	"reflect"
	"testing"

	"orrery/agent"
)

func TestSchedulePairedMatchesOrderSeatsAndSeeds(t *testing.T) {
	left, err := agent.ParseFactory("random-v1:max_fraction=0.8,min_fraction=0.2,launch_probability=0.3")
	if err != nil {
		t.Fatal(err)
	}
	right, err := agent.ParseFactory("heuristic-v1:defense=0.1,efficiency=1,economy=1,aggression=1.2")
	if err != nil {
		t.Fatal(err)
	}
	spec := PairingSpec{
		Scenario:      FixedScenarioFactory{},
		ScenarioCount: 3,
		SeedStart:     -2,
		Contestants: []ContestantSpec{
			{ID: "left", Factory: left},
			{ID: "right", Factory: right},
		},
		RecordActions: true,
	}

	matches, err := SchedulePairedMatches(spec)
	if err != nil {
		t.Fatal(err)
	}
	if got, want := len(matches), 6; got != want {
		t.Fatalf("match count = %d, want %d", got, want)
	}
	for index, match := range matches {
		scenarioIndex := index / 2
		leg := index % 2
		if match.ID != matchID(scenarioIndex, leg) || match.ScenarioIndex != scenarioIndex || match.Leg != leg {
			t.Fatalf("match %d identity = ID %q scenario %d leg %d", index, match.ID, match.ScenarioIndex, match.Leg)
		}
		if got, want := match.Seed, int64(-2+scenarioIndex); got != want {
			t.Fatalf("match %d seed = %d, want %d", index, got, want)
		}
		if match.Scenario != spec.Scenario || !match.RecordActions {
			t.Fatalf("match %d lost scenario or recording setting: %#v", index, match)
		}

		wantOrder := []string{"left", "right"}
		if leg == 1 {
			wantOrder = []string{"right", "left"}
		}
		if got := []string{match.Seats[0].ContestantID, match.Seats[1].ContestantID}; !reflect.DeepEqual(got, wantOrder) {
			t.Fatalf("match %d seat order = %v, want %v", index, got, wantOrder)
		}
		for _, seat := range match.Seats {
			contestant := spec.Contestants[0]
			if seat.ContestantID == "right" {
				contestant = spec.Contestants[1]
			}
			if seat.Factory != contestant.Factory {
				t.Fatalf("match %d contestant %q factory changed", index, seat.ContestantID)
			}
			wantSeed := DeriveAgentSeed(match.Seed, contestant.ID, contestant.Factory.Spec())
			if seat.AgentSeed != wantSeed {
				t.Fatalf("match %d contestant %q seed = %d, want %d", index, contestant.ID, seat.AgentSeed, wantSeed)
			}
		}
	}

	// Each scenario's two legs must hold each contestant's private random
	// stream constant while changing only the player seat.
	for scenarioIndex := range spec.ScenarioCount {
		legZero := matches[2*scenarioIndex]
		legOne := matches[2*scenarioIndex+1]
		if legZero.Seats[0].AgentSeed != legOne.Seats[1].AgentSeed ||
			legZero.Seats[1].AgentSeed != legOne.Seats[0].AgentSeed {
			t.Fatalf("scenario %d agent seeds changed across seat swap", scenarioIndex)
		}
	}

	// Jobs must not share their mutable seat slices.
	matches[0].Seats[0].ContestantID = "mutated"
	if matches[1].Seats[1].ContestantID != "left" {
		t.Fatal("leg 0 and leg 1 share a seat slice")
	}
}

func TestDeriveAgentSeedStableContract(t *testing.T) {
	factory, err := agent.ParseFactory("random-v1:launch_probability=0.3,max_fraction=0.8,min_fraction=0.2")
	if err != nil {
		t.Fatal(err)
	}
	const want int64 = -3188614544360009373
	if got := DeriveAgentSeed(-17, "contestant-a", factory.Spec()); got != want {
		t.Fatalf("seed = %d, want stable golden %d", got, want)
	}
}

func TestDeriveAgentSeedInputsAndCanonicalIdentity(t *testing.T) {
	canonical, err := agent.ParseFactory("random-v1:launch_probability=0.3,max_fraction=0.8,min_fraction=0.2")
	if err != nil {
		t.Fatal(err)
	}
	reordered, err := agent.ParseFactory("random-v1:min_fraction=2e-1,launch_probability=3e-1,max_fraction=8e-1")
	if err != nil {
		t.Fatal(err)
	}
	base := DeriveAgentSeed(91, "contestant-a", canonical.Spec())
	if got := DeriveAgentSeed(91, "contestant-a", reordered.Spec()); got != base {
		t.Fatalf("canonical-equivalent specification seed = %d, want %d", got, base)
	}

	changed := []int64{
		DeriveAgentSeed(92, "contestant-a", canonical.Spec()),
		DeriveAgentSeed(91, "contestant-b", canonical.Spec()),
		DeriveAgentSeed(91, "contestant-a", agent.NewSpec(agent.IdleV1, nil)),
	}
	for index, seed := range changed {
		if seed == base {
			t.Fatalf("changed input %d retained base seed %d", index, base)
		}
	}
}

func TestSchedulePairedMatchesValidation(t *testing.T) {
	valid := PairingSpec{
		Scenario:      FixedScenarioFactory{},
		ScenarioCount: 1,
		SeedStart:     10,
		Contestants: []ContestantSpec{
			{ID: "left", Factory: agent.IdleFactory{}},
			{ID: "right", Factory: agent.IdleFactory{}},
		},
	}
	var nilFactory *testFactory
	maxInt := int(^uint(0) >> 1)
	tests := []struct {
		name   string
		mutate func(*PairingSpec)
	}{
		{"nil scenario", func(spec *PairingSpec) { spec.Scenario = nil }},
		{"empty scenario ID", func(spec *PairingSpec) { spec.Scenario = scenarioFunc{} }},
		{"zero scenarios", func(spec *PairingSpec) { spec.ScenarioCount = 0 }},
		{"negative scenarios", func(spec *PairingSpec) { spec.ScenarioCount = -1 }},
		{"match count overflow", func(spec *PairingSpec) { spec.ScenarioCount = maxInt/2 + 1 }},
		{"seed overflow", func(spec *PairingSpec) { spec.ScenarioCount, spec.SeedStart = 2, math.MaxInt64 }},
		{"one contestant", func(spec *PairingSpec) { spec.Contestants = spec.Contestants[:1] }},
		{"three contestants", func(spec *PairingSpec) {
			spec.Contestants = append(spec.Contestants, ContestantSpec{ID: "third", Factory: agent.IdleFactory{}})
		}},
		{"empty contestant ID", func(spec *PairingSpec) { spec.Contestants[0].ID = "" }},
		{"duplicate contestant ID", func(spec *PairingSpec) { spec.Contestants[1].ID = spec.Contestants[0].ID }},
		{"typed nil factory", func(spec *PairingSpec) { spec.Contestants[0].Factory = nilFactory }},
		{"empty factory ID", func(spec *PairingSpec) { spec.Contestants[0].Factory = &testFactory{} }},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			spec := valid
			spec.Contestants = append([]ContestantSpec(nil), valid.Contestants...)
			test.mutate(&spec)
			if matches, err := SchedulePairedMatches(spec); err == nil {
				t.Fatalf("SchedulePairedMatches = %#v, want error", matches)
			}
		})
	}
}
