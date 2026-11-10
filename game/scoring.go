package game

import "sort"

// Standings returns each player's total ship count: ships garrisoned on that
// player's planets, ships stationed on that player's comets, plus ships in
// that player's active fleets. A comet is a temporary hold, not a permanent
// one, but for as long as it's on the board its ships count exactly like a
// planet's — see produce() and moveComets' expiry, which is what eventually
// takes an uncollected comet's ships out of every player's count for good.
// NeutralPlayer is never included since it is not a player.
func (g *Game) Standings() map[int]int {
	totals := map[int]int{}
	for _, p := range g.Planets {
		if p.OwnerID != NeutralPlayer {
			totals[p.OwnerID] += p.Ships
		}
	}
	for _, c := range g.Comets {
		if c.OwnerID != NeutralPlayer {
			totals[c.OwnerID] += c.Ships
		}
	}
	for _, f := range g.Fleets {
		totals[f.OwnerID] += f.Ships
	}
	return totals
}

// Winner returns every player tied for the highest total ship count once the
// match has ended (turn limit or elimination — see Finished). ok is false
// only if the game has not finished yet.
//
// leaders has more than one entry exactly when the top score is an exact
// tie: the rules require a tied result be represented explicitly rather than
// silently declined, so a length-1 result is a clear winner and any longer
// result is that tie, spelled out rather than hidden behind a bare false.
// leaders is sorted by player ID so the result is deterministic regardless of
// Standings' unordered map — a seeded replay must produce the same slice
// every time.
//
// ok reflects only whether the match has ended — never whether the result
// itself is interesting. A finished board with no non-neutral owner
// anywhere (nobody left to have a standing at all) still returns ok == true,
// with an empty leaders: there is no winner, but the match is unambiguously
// over, which is a different fact than "not finished yet" and must not
// share its false with it.
func (g *Game) Winner() (leaders []int, ok bool) {
	if !g.Finished() {
		return nil, false
	}
	standings := g.Standings()
	if len(standings) == 0 {
		return nil, true
	}
	best := -1
	for _, ships := range standings {
		if ships > best {
			best = ships
		}
	}
	for player, ships := range standings {
		if ships == best {
			leaders = append(leaders, player)
		}
	}
	sort.Ints(leaders)
	return leaders, true
}
