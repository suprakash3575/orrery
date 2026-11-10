package arena

import (
	"encoding/binary"
	"fmt"
	"hash/fnv"
	"math"

	"orrery/agent"
)

// ContestantSpec is one stable experiment entrant. ID distinguishes two
// contestants that intentionally use the same canonical agent factory.
type ContestantSpec struct {
	ID      string
	Factory agent.Factory
}

// PairingSpec describes the first-release two-contestant schedule. Later
// tournament scheduling can produce MatchSpec values through a different
// scheduler without changing RunMatch or RunExperiment.
type PairingSpec struct {
	Scenario      ScenarioFactory
	ScenarioCount int
	SeedStart     int64
	Contestants   []ContestantSpec
	RecordActions bool
}

// SchedulePairedMatches expands every scenario into two adjacent jobs. Job
// index 2*i is leg 0 and 2*i+1 is leg 1, so result ordering never depends on
// worker completion order.
func SchedulePairedMatches(spec PairingSpec) ([]MatchSpec, error) {
	if err := validatePairingSpec(spec); err != nil {
		return nil, err
	}

	matches := make([]MatchSpec, 0, 2*spec.ScenarioCount)
	for scenarioIndex := range spec.ScenarioCount {
		scenarioSeed := spec.SeedStart + int64(scenarioIndex)
		seats := make([]SeatSpec, len(spec.Contestants))
		for contestantIndex, contestant := range spec.Contestants {
			seats[contestantIndex] = scheduledSeat(contestant, scenarioSeed)
		}
		matches = append(matches, MatchSpec{
			ID:            matchID(scenarioIndex, 0),
			Scenario:      spec.Scenario,
			ScenarioIndex: scenarioIndex,
			Leg:           0,
			Seed:          scenarioSeed,
			Seats:         seats,
			RecordActions: spec.RecordActions,
		})

		// Copy the seat values rather than sharing a slice with leg 0. Match
		// jobs may execute concurrently, and no mutation of one job should be
		// able to rewrite another job's assignments.
		swapped := []SeatSpec{seats[1], seats[0]}
		matches = append(matches, MatchSpec{
			ID:            matchID(scenarioIndex, 1),
			Scenario:      spec.Scenario,
			ScenarioIndex: scenarioIndex,
			Leg:           1,
			Seed:          scenarioSeed,
			Seats:         swapped,
			RecordActions: spec.RecordActions,
		})
	}
	return matches, nil
}

func validatePairingSpec(spec PairingSpec) error {
	if nilInterface(spec.Scenario) {
		return fmt.Errorf("scenario factory must not be nil")
	}
	if spec.Scenario.ID() == "" {
		return fmt.Errorf("scenario ID must not be empty")
	}
	if spec.ScenarioCount <= 0 {
		return fmt.Errorf("scenario count must be positive")
	}
	maxInt := int(^uint(0) >> 1)
	if spec.ScenarioCount > maxInt/2 {
		return fmt.Errorf("paired match count overflows int")
	}
	if spec.SeedStart > math.MaxInt64-int64(spec.ScenarioCount-1) {
		return fmt.Errorf("scenario seed range overflows int64")
	}
	if len(spec.Contestants) != 2 {
		return fmt.Errorf("paired schedule requires exactly two contestants")
	}
	seen := make(map[string]bool, len(spec.Contestants))
	for index, contestant := range spec.Contestants {
		if contestant.ID == "" {
			return fmt.Errorf("contestant %d ID must not be empty", index)
		}
		if seen[contestant.ID] {
			return fmt.Errorf("duplicate contestant ID %q", contestant.ID)
		}
		seen[contestant.ID] = true
		if nilInterface(contestant.Factory) {
			return fmt.Errorf("contestant %q factory must not be nil", contestant.ID)
		}
		if contestant.Factory.Spec().ID() == "" {
			return fmt.Errorf("contestant %q factory ID must not be empty", contestant.ID)
		}
	}
	return nil
}

func scheduledSeat(contestant ContestantSpec, scenarioSeed int64) SeatSpec {
	return SeatSpec{
		ContestantID: contestant.ID,
		Factory:      contestant.Factory,
		AgentSeed:    DeriveAgentSeed(scenarioSeed, contestant.ID, contestant.Factory.Spec()),
	}
}

func matchID(scenarioIndex, leg int) string {
	return fmt.Sprintf("scenario-%06d-leg-%d", scenarioIndex, leg)
}

// DeriveAgentSeed implements the stable agent-seed-v1 replay contract. It
// hashes, in order: the domain string below; the scenario seed as one unsigned
// big-endian 64-bit word (preserving the two's-complement bits of negative
// seeds); then contestant ID and canonical agent spec as big-endian
// length-prefixed UTF-8 byte strings. FNV-1a is fixed by the standard library.
// Seat and leg are deliberately absent, keeping a contestant's private random
// stream unchanged when paired matches swap seats.
func DeriveAgentSeed(scenarioSeed int64, contestantID string, spec agent.Spec) int64 {
	hash := fnv.New64a()
	writeSeedString(hash, "orrery-agent-seed-v1")
	var encoded [8]byte
	binary.BigEndian.PutUint64(encoded[:], uint64(scenarioSeed))
	_, _ = hash.Write(encoded[:])
	writeSeedString(hash, contestantID)
	writeSeedString(hash, spec.String())
	return int64(hash.Sum64())
}

type byteWriter interface {
	Write([]byte) (int, error)
}

func writeSeedString(writer byteWriter, value string) {
	var encodedLength [8]byte
	binary.BigEndian.PutUint64(encodedLength[:], uint64(len(value)))
	_, _ = writer.Write(encodedLength[:])
	_, _ = writer.Write([]byte(value))
}
