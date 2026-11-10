package scenario

import (
	"math"
	"slices"
	"strings"
	"testing"
)

// TestDisjoint covers the boundary cases a seed budget actually gets wrong.
// Touching and empty ranges are the ones most likely to be mis-rejected, and
// the non-adjacent overlap is what proves every pair is compared rather than
// only neighbouring ones.
func TestDisjoint(t *testing.T) {
	tests := []struct {
		name    string
		ranges  []SeedRange
		wantErr bool
	}{
		{name: "no ranges", ranges: nil},
		{name: "single range", ranges: []SeedRange{{0, 100}}},
		{
			name:   "separated by a gap",
			ranges: []SeedRange{{0, 100}, {200, 300}},
		},
		{
			// One range's End is the next one's Start. Half-open intervals make
			// this the normal way to write neighbouring ranges, so rejecting it
			// would make the whole scheme unusable.
			name:   "touching at a shared boundary",
			ranges: []SeedRange{{0, 100}, {100, 200}},
		},
		{
			name:   "declared out of ascending order",
			ranges: []SeedRange{{200, 300}, {0, 100}},
		},
		{
			name:   "empty range inside another",
			ranges: []SeedRange{{0, 100}, {50, 50}},
		},
		{
			name:   "two empty ranges at the same point",
			ranges: []SeedRange{{50, 50}, {50, 50}},
		},
		{
			name:    "overlapping by a single seed",
			ranges:  []SeedRange{{0, 100}, {99, 200}},
			wantErr: true,
		},
		{
			name:    "one range nested in another",
			ranges:  []SeedRange{{0, 1000}, {400, 500}},
			wantErr: true,
		},
		{
			name:    "identical ranges",
			ranges:  []SeedRange{{0, 100}, {0, 100}},
			wantErr: true,
		},
		{
			// The two overlapping ranges are not beside each other, so a check
			// that only compared consecutive entries would pass this.
			name:    "first and third overlap across a clean middle",
			ranges:  []SeedRange{{0, 100}, {500, 600}, {50, 80}},
			wantErr: true,
		},
		{
			name:    "inverted range",
			ranges:  []SeedRange{{100, 0}},
			wantErr: true,
		},
		{
			// An inverted range overlaps nothing by the arithmetic, so this
			// only fails if inversion is rejected in its own right.
			name:    "inverted range that would otherwise look harmless",
			ranges:  []SeedRange{{0, 100}, {300, 200}},
			wantErr: true,
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			err := Disjoint(test.ranges...)
			if test.wantErr && err == nil {
				t.Fatalf("Disjoint(%v) = nil, want an error", test.ranges)
			}
			if !test.wantErr && err != nil {
				t.Fatalf("Disjoint(%v) = %v, want nil", test.ranges, err)
			}
		})
	}
}

// TestDisjointErrorIdentifiesTheOverlap checks the error is actionable rather
// than merely present: it has to say which ranges clashed and on which seeds,
// because the reader's next step is editing one of those numbers.
func TestDisjointErrorIdentifiesTheOverlap(t *testing.T) {
	err := Disjoint(SeedRange{0, 1000}, SeedRange{2000, 3000}, SeedRange{900, 1100})
	if err == nil {
		t.Fatal("expected an error for overlapping ranges")
	}
	for _, want := range []string{
		"range 0",
		"range 2",
		"[0,1000)",
		"[900,1100)",
		"[900,1000)", // the shared seeds themselves
	} {
		if !strings.Contains(err.Error(), want) {
			t.Errorf("error %q does not mention %q", err, want)
		}
	}
}

// TestStandardRangesAreDisjoint is the check the whole file exists for: the
// project's declared budget must never let a seed serve two purposes.
func TestStandardRangesAreDisjoint(t *testing.T) {
	if err := StandardRanges().Validate(); err != nil {
		t.Fatalf("standard seed budget is invalid: %v", err)
	}
}

// TestStandardRangesHaveExactBounds freezes the V1 partition. The names and
// half-open endpoints become evaluation metadata, so changing either is a
// contract change even when all ranges remain disjoint and correctly sized.
func TestStandardRangesHaveExactBounds(t *testing.T) {
	ranges := StandardRanges()
	if ranges.Training != (SeedRange{Start: 0, End: 1_000_000}) {
		t.Errorf("training range = %s, want [0,1000000)", ranges.Training)
	}
	if ranges.Validation != (SeedRange{Start: 1_000_000, End: 1_010_000}) {
		t.Errorf("validation range = %s, want [1000000,1010000)", ranges.Validation)
	}
	wantPools := []EvaluationPool{
		{Name: "eval-1", Seeds: SeedRange{Start: 1_010_000, End: 1_011_000}},
		{Name: "eval-2", Seeds: SeedRange{Start: 1_011_000, End: 1_012_000}},
		{Name: "eval-3", Seeds: SeedRange{Start: 1_012_000, End: 1_013_000}},
		{Name: "eval-4", Seeds: SeedRange{Start: 1_013_000, End: 1_014_000}},
		{Name: "eval-5", Seeds: SeedRange{Start: 1_014_000, End: 1_015_000}},
	}
	if !slices.Equal(ranges.EvaluationPools, wantPools) {
		t.Errorf("evaluation pools = %#v, want %#v", ranges.EvaluationPools, wantPools)
	}
}

// TestStandardRangesIsNotSharedState checks the slice-backed budget cannot be
// edited through the accessor. A shallow struct copy would still expose the
// evaluation entries shared by every later caller.
func TestStandardRangesIsNotSharedState(t *testing.T) {
	mine := StandardRanges()
	mine.EvaluationPools[0].Name = "spent"
	mine.EvaluationPools[0].Seeds.Start = 0
	mine.EvaluationPools = append(mine.EvaluationPools, EvaluationPool{Name: "extra", Seeds: SeedRange{2_000_000, 2_001_000}})

	got := StandardRanges().EvaluationPools
	if len(got) != 5 {
		t.Fatalf("standard evaluation pool count changed to %d after caller append, want 5", len(got))
	}
	if got[0] != (EvaluationPool{Name: "eval-1", Seeds: SeedRange{Start: 1_010_000, End: 1_011_000}}) {
		t.Fatalf("first standard evaluation pool changed to %#v after caller edit", got[0])
	}
}

// TestValidateRejectsUnusableBudgets covers the cases that are disjoint but
// still unusable. A budget of three empty ranges overlaps nothing, so an
// overlap check alone would pass it while leaving every purpose with no seeds
// to draw on.
func TestValidateRejectsUnusableBudgets(t *testing.T) {
	usable := StandardRanges()
	tests := []struct {
		name   string
		budget Ranges
		want   string
	}{
		{name: "wholly empty budget", budget: Ranges{}, want: "training"},
		{
			name:   "empty training",
			budget: Ranges{Training: SeedRange{5, 5}, Validation: usable.Validation, EvaluationPools: usable.EvaluationPools},
			want:   "training",
		},
		{
			name:   "empty validation",
			budget: Ranges{Training: usable.Training, Validation: SeedRange{}, EvaluationPools: usable.EvaluationPools},
			want:   "validation",
		},
		{
			name:   "no evaluation pools",
			budget: Ranges{Training: usable.Training, Validation: usable.Validation},
			want:   "evaluation",
		},
		{
			name: "unnamed evaluation pool",
			budget: Ranges{
				Training:        usable.Training,
				Validation:      usable.Validation,
				EvaluationPools: []EvaluationPool{{Seeds: SeedRange{1_010_000, 1_011_000}}},
			},
			want: "empty name",
		},
		{
			name: "duplicate evaluation pool name",
			budget: Ranges{
				Training:   usable.Training,
				Validation: usable.Validation,
				EvaluationPools: []EvaluationPool{
					{Name: "eval", Seeds: SeedRange{1_010_000, 1_011_000}},
					{Name: "eval", Seeds: SeedRange{1_011_000, 1_012_000}},
				},
			},
			want: "appears more than once",
		},
		{
			name: "empty evaluation pool",
			budget: Ranges{
				Training:        usable.Training,
				Validation:      usable.Validation,
				EvaluationPools: []EvaluationPool{{Name: "eval", Seeds: SeedRange{300, 300}}},
			},
			want: "eval",
		},
		{
			// Inverted bounds contain no seeds either, so this must be caught
			// rather than silently reserving nothing.
			name: "inverted evaluation",
			budget: Ranges{
				Training:        usable.Training,
				Validation:      usable.Validation,
				EvaluationPools: []EvaluationPool{{Name: "eval", Seeds: SeedRange{300, 200}}},
			},
			want: "eval",
		},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			err := test.budget.Validate()
			if err == nil {
				t.Fatal("Validate() = nil, want an error for an unusable budget")
			}
			if !strings.Contains(err.Error(), test.want) {
				t.Fatalf("error %q does not name the offending purpose %q", err, test.want)
			}
		})
	}
}

func TestValidateOverlapNamesEvaluationPools(t *testing.T) {
	budget := StandardRanges()
	budget.EvaluationPools[3].Seeds = budget.EvaluationPools[2].Seeds
	err := budget.Validate()
	if err == nil {
		t.Fatal("Validate() = nil, want an overlap error")
	}
	for _, want := range []string{"eval-3", "eval-4"} {
		if !strings.Contains(err.Error(), want) {
			t.Errorf("error %q does not name evaluation pool %q", err, want)
		}
	}
}

// TestStandardRangeSizes pins the budget's intended shape. Without it a
// careless edit could shrink evaluation to a handful of seeds, which would
// still validate as disjoint while making the final number too noisy to mean
// anything.
func TestStandardRangeSizes(t *testing.T) {
	ranges := StandardRanges()
	tests := []struct {
		name  string
		seeds SeedRange
		want  int64
	}{
		{name: "training", seeds: ranges.Training, want: 1_000_000},
		{name: "validation", seeds: ranges.Validation, want: 10_000},
	}
	for _, pool := range ranges.EvaluationPools {
		tests = append(tests, struct {
			name  string
			seeds SeedRange
			want  int64
		}{name: pool.Name, seeds: pool.Seeds, want: 1_000})
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			if got := test.seeds.Len(); got != test.want {
				t.Fatalf("%s holds %d seeds, want %d", test.name, got, test.want)
			}
		})
	}
}

// TestValidateRejectsOverlappingBudgets covers Validate's overlap path, which
// every case in TestValidateRejectsUnusableBudgets misses: those budgets are
// empty or inverted, so they return at the emptiness check and never reach
// Disjoint. Without these cases the overlap check could be dropped from
// Validate entirely and the suite would still pass — leaving the one entry
// point callers actually use silently unable to detect seed leakage.
func TestValidateRejectsOverlappingBudgets(t *testing.T) {
	usable := StandardRanges()
	tests := []struct {
		name       string
		budget     Ranges
		wantShared string
	}{
		{
			name: "training runs one seed into validation",
			budget: Ranges{
				Training:        SeedRange{Start: 0, End: 1_000_001},
				Validation:      usable.Validation,
				EvaluationPools: usable.EvaluationPools,
			},
			wantShared: "[1000000,1000001)",
		},
		{
			name: "evaluation starts before validation ends",
			budget: Ranges{
				Training:        usable.Training,
				Validation:      usable.Validation,
				EvaluationPools: []EvaluationPool{{Name: "eval", Seeds: SeedRange{Start: 1_009_000, End: 1_011_000}}},
			},
			wantShared: "[1009000,1010000)",
		},
		{
			// Training and evaluation clash while validation is clean between
			// them, so this only fails if Validate compares every pair rather
			// than neighbouring purposes.
			name: "training and evaluation overlap around a clean validation",
			budget: Ranges{
				Training:        SeedRange{Start: 0, End: 2_000_000},
				Validation:      SeedRange{Start: 2_000_000, End: 2_010_000},
				EvaluationPools: []EvaluationPool{{Name: "eval", Seeds: SeedRange{Start: 500_000, End: 600_000}}},
			},
			wantShared: "[500000,600000)",
		},
		{
			name: "nonadjacent evaluation pools overlap",
			budget: Ranges{
				Training:   usable.Training,
				Validation: usable.Validation,
				EvaluationPools: []EvaluationPool{
					{Name: "first", Seeds: SeedRange{1_010_000, 1_011_000}},
					{Name: "middle", Seeds: SeedRange{1_012_000, 1_013_000}},
					{Name: "last", Seeds: SeedRange{1_010_500, 1_011_500}},
				},
			},
			wantShared: "[1010500,1011000)",
		},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			err := test.budget.Validate()
			if err == nil {
				t.Fatal("Validate() = nil, want an error for overlapping ranges")
			}
			// The wrapper is asserted too: it is what maps Disjoint's range
			// numbers back to training, validation and evaluation.
			if !strings.Contains(err.Error(), "seed budget") {
				t.Errorf("error %q lacks the budget context that names each range index", err)
			}
			if !strings.Contains(err.Error(), test.wantShared) {
				t.Errorf("error %q does not report the shared seeds %s", err, test.wantShared)
			}
		})
	}
}

// TestSeedRangeAccessors covers the small value type's own behaviour,
// including the inverted case that must report as empty rather than as a
// negative length.
func TestSeedRangeAccessors(t *testing.T) {
	tests := []struct {
		name      string
		seeds     SeedRange
		wantEmpty bool
		wantLen   int64
		wantText  string
	}{
		{name: "populated", seeds: SeedRange{10, 20}, wantLen: 10, wantText: "[10,20)"},
		{name: "single seed", seeds: SeedRange{7, 8}, wantLen: 1, wantText: "[7,8)"},
		{name: "empty", seeds: SeedRange{5, 5}, wantEmpty: true, wantText: "[5,5)"},
		{name: "inverted", seeds: SeedRange{9, 4}, wantEmpty: true, wantText: "[9,4)"},
		{name: "negative seeds", seeds: SeedRange{-5, 5}, wantLen: 10, wantText: "[-5,5)"},
		{
			// Subtracting these bounds in int64 overflows to -1. A negative
			// length is nonsense that would break any loop or capacity built
			// from it, so the width saturates instead.
			name:     "spanning the whole seed domain",
			seeds:    SeedRange{math.MinInt64, math.MaxInt64},
			wantLen:  math.MaxInt64,
			wantText: "[-9223372036854775808,9223372036854775807)",
		},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			if got := test.seeds.Empty(); got != test.wantEmpty {
				t.Errorf("Empty() = %v, want %v", got, test.wantEmpty)
			}
			if got := test.seeds.Len(); got != test.wantLen {
				t.Errorf("Len() = %d, want %d", got, test.wantLen)
			}
			if got := test.seeds.String(); got != test.wantText {
				t.Errorf("String() = %q, want %q", got, test.wantText)
			}
		})
	}
}
