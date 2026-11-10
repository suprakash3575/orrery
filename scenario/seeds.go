package scenario

import (
	"fmt"
	"math"
	"slices"
)

// SeedRange is a half-open interval of scenario seeds: Start is included and
// End is not.
//
// Half-open is what makes neighbouring ranges natural to write — one range's
// End is the next one's Start, with no seed shared between them and none
// skipped in the gap. Writing them closed would force every boundary to be
// stated as End-1 and invite exactly the off-by-one this package exists to
// prevent.
type SeedRange struct {
	Start int64
	End   int64
}

// Empty reports whether the range contains no seeds.
func (r SeedRange) Empty() bool { return r.Start >= r.End }

// Len is how many seeds the range contains. An inverted or empty range
// contains none rather than a negative count.
//
// The width is computed in uint64, which is exact for any pair of int64
// bounds, and then saturated. Subtracting directly in int64 overflows once a
// range spans more than half the seed domain — plausible if a range is ever
// written as a "every seed" sentinel — and would report a negative length,
// which silently breaks any loop or capacity derived from it. Saturating keeps
// an absurdly wide range reporting an absurdly large count instead.
func (r SeedRange) Len() int64 {
	if r.Empty() {
		return 0
	}
	if width := uint64(r.End) - uint64(r.Start); width <= math.MaxInt64 {
		return int64(width)
	}
	return math.MaxInt64
}

// String renders the interval in the half-open notation this package
// documents, so an error message reads the same way the declarations do.
func (r SeedRange) String() string { return fmt.Sprintf("[%d,%d)", r.Start, r.End) }

// overlap returns the seeds two ranges share, which is an empty range when
// they share none.
func (r SeedRange) overlap(other SeedRange) SeedRange {
	return SeedRange{Start: max(r.Start, other.Start), End: min(r.End, other.End)}
}

// namedSeedRange pairs an interval with the purpose an operator needs to edit
// when validation finds a collision.
type namedSeedRange struct {
	// name is a budget purpose such as "training"; an empty name uses index.
	name string
	// index is the original Disjoint argument position used by generic errors.
	index int
	// seeds is the interval validated under name or index.
	seeds SeedRange
}

func (r namedSeedRange) label() string {
	if r.name != "" {
		return r.name
	}
	return fmt.Sprintf("range %d", r.index)
}

// Disjoint reports whether every pair of the given ranges is free of shared
// seeds, returning nil when they are.
//
// This exists to catch an off-by-one or copy-paste mistake in a seed budget
// before it can quietly invalidate a result. If training and validation shared
// even one seed, a policy could be scored on a map it had already trained on,
// and the validation number would measure memorization rather than skill —
// with nothing anywhere else in the system reporting a problem.
//
// Every pair is compared, not merely neighbouring ones: ranges are usually
// written in ascending order, but nothing enforces that, and the first and
// third can overlap while each looks fine beside its neighbour. Accepting any
// number of ranges keeps the check correct if a fourth is added later.
func Disjoint(ranges ...SeedRange) error {
	named := make([]namedSeedRange, len(ranges))
	for i, seeds := range ranges {
		named[i] = namedSeedRange{index: i, seeds: seeds}
	}
	return disjoint(named)
}

func disjoint(ranges []namedSeedRange) error {
	// Inverted ranges are rejected before any comparison. The overlap test
	// below treats Start > End as simply empty, and therefore harmless — so a
	// transposed pair of numbers, exactly the typo this function is here to
	// catch, would otherwise pass while silently reserving no seeds at all.
	for _, r := range ranges {
		if r.seeds.Start > r.seeds.End {
			return fmt.Errorf("%s %s is inverted: Start must not exceed End", r.label(), r.seeds)
		}
	}
	for i := range ranges {
		for j := i + 1; j < len(ranges); j++ {
			if shared := ranges[i].seeds.overlap(ranges[j].seeds); !shared.Empty() {
				return fmt.Errorf("%s %s and %s %s both contain seeds %s",
					ranges[i].label(), ranges[i].seeds, ranges[j].label(), ranges[j].seeds, shared)
			}
		}
	}
	return nil
}

// EvaluationPool names one protected seed interval that may be spent on a
// frozen checkpoint. Naming pools lets evaluation records identify exactly
// which held-out data was consumed without copying range bounds into every
// artifact.
type EvaluationPool struct {
	// Name is the stable pool identifier recorded when an evaluation spends
	// this interval, such as "eval-1".
	Name string
	// Seeds is the protected half-open interval assigned to the pool. Its exact
	// bounds are replay metadata and must accompany the name in run records.
	Seeds SeedRange
}

// Ranges is one complete seed budget: which seeds a run may train on, which it
// measures itself against while training, and which named intervals are held
// back for final evaluations.
type Ranges struct {
	// Training seeds are sampled freely and repeatedly throughout a run.
	Training SeedRange
	// Validation seeds decide when a run is finished. They are read often, so
	// they are not a neutral measurement — choosing a checkpoint by validation
	// score makes those seeds part of the selection process.
	Validation SeedRange
	// EvaluationPools are spent in order, one per already-frozen checkpoint.
	// That is the whole of their value: a number measured on seeds nothing was
	// tuned against is the only one that estimates play on an unseen map.
	EvaluationPools []EvaluationPool
}

// Validate reports whether the budget is usable: every purpose has seeds to
// draw on, and no seed is claimed by two purposes at once.
//
// Emptiness is checked here rather than in Disjoint because the two ask
// different questions. Disjoint asks whether intervals overlap, and empty
// intervals legitimately overlap nothing. A budget is a stronger claim: a run
// cannot train on nothing, and an evaluation over zero seeds reports a win
// rate computed from no games — a number that looks like a result but measures
// nothing at all.
func (r Ranges) Validate() error {
	for _, purpose := range []struct {
		name  string
		seeds SeedRange
	}{
		{"training", r.Training},
		{"validation", r.Validation},
	} {
		if purpose.seeds.Empty() {
			return fmt.Errorf("%s range %s contains no seeds", purpose.name, purpose.seeds)
		}
	}
	if len(r.EvaluationPools) == 0 {
		return fmt.Errorf("evaluation pools contain no seed ranges")
	}

	allRanges := []namedSeedRange{
		{name: "training", seeds: r.Training},
		{name: "validation", seeds: r.Validation},
	}
	poolNames := make(map[string]struct{}, len(r.EvaluationPools))
	for i, pool := range r.EvaluationPools {
		if pool.Name == "" {
			return fmt.Errorf("evaluation pool %d has an empty name", i)
		}
		if _, duplicate := poolNames[pool.Name]; duplicate {
			return fmt.Errorf("evaluation pool name %q appears more than once", pool.Name)
		}
		poolNames[pool.Name] = struct{}{}
		if pool.Seeds.Empty() {
			return fmt.Errorf("evaluation pool %q range %s contains no seeds", pool.Name, pool.Seeds)
		}
		allRanges = append(allRanges, namedSeedRange{name: fmt.Sprintf("evaluation pool %q", pool.Name), seeds: pool.Seeds})
	}

	if err := disjoint(allRanges); err != nil {
		return fmt.Errorf("seed budget: %w", err)
	}
	return nil
}

// standardRanges is the single place this project's seed budget is declared.
// Nothing else should write these numbers down: a second copy is free to drift
// from this one, and the drift would appear as an inexplicably good validation
// score rather than as an error.
//
// The sizes are deliberately lopsided. Training needs breadth, so it takes the
// bulk. Validation is read repeatedly and only has to separate a good
// checkpoint from a bad one. Each evaluation pool is small because it is spent
// once and never reused, and a thousand maps already measure a win rate more
// precisely than the difference any freeze decision turns on. Five pools let
// later checkpoints receive fresh held-out exams without reusing a failed one.
var standardRanges = Ranges{
	Training:   SeedRange{Start: 0, End: 1_000_000},
	Validation: SeedRange{Start: 1_000_000, End: 1_010_000},
	EvaluationPools: []EvaluationPool{
		{Name: "eval-1", Seeds: SeedRange{Start: 1_010_000, End: 1_011_000}},
		{Name: "eval-2", Seeds: SeedRange{Start: 1_011_000, End: 1_012_000}},
		{Name: "eval-3", Seeds: SeedRange{Start: 1_012_000, End: 1_013_000}},
		{Name: "eval-4", Seeds: SeedRange{Start: 1_013_000, End: 1_014_000}},
		{Name: "eval-5", Seeds: SeedRange{Start: 1_014_000, End: 1_015_000}},
	},
}

// StandardRanges returns this project's seed budget.
//
// It hands back a deep copy rather than exposing the declaration itself. A
// Ranges value contains a slice, so copying only the struct would still let a
// caller overwrite the shared evaluation entries. Isolating that slice removes
// both silent partition changes and data races between concurrent readers.
// Callers wanting a different budget should build their own Ranges and
// Validate it.
func StandardRanges() Ranges {
	ranges := standardRanges
	ranges.EvaluationPools = slices.Clone(standardRanges.EvaluationPools)
	return ranges
}
