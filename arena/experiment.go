package arena

import (
	"context"
	"fmt"
	"sync"
)

// ExperimentSpec configures one deterministic paired batch. Pairing owns all
// scenario and contestant identity; Workers changes only execution parallelism
// and must never change a completed MatchResult.
type ExperimentSpec struct {
	Pairing PairingSpec
	Workers int
}

// RunExperiment executes a fully validated paired schedule through an indexed
// worker pool. Each worker writes only to the result slot belonging to its job,
// so completion order cannot reorder output or affect aggregation.
func RunExperiment(ctx context.Context, spec ExperimentSpec) ([]MatchResult, error) {
	if spec.Workers <= 0 {
		return nil, fmt.Errorf("worker count must be positive")
	}
	matches, err := SchedulePairedMatches(spec.Pairing)
	if err != nil {
		return nil, fmt.Errorf("schedule experiment: %w", err)
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}

	workerCount := min(spec.Workers, len(matches))
	results := make([]MatchResult, len(matches))
	jobs := make(chan int, workerCount)
	var workers sync.WaitGroup
	workers.Add(workerCount)
	for range workerCount {
		go func() {
			defer workers.Done()
			for jobIndex := range jobs {
				results[jobIndex] = RunMatch(ctx, matches[jobIndex])
			}
		}()
	}

	cancelled := false
enqueue:
	for jobIndex := range matches {
		select {
		case <-ctx.Done():
			cancelled = true
			break enqueue
		case jobs <- jobIndex:
		}
	}
	close(jobs)
	workers.Wait()

	if cancelled || ctx.Err() != nil {
		return results, ctx.Err()
	}
	return results, nil
}
