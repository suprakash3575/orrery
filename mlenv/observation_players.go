package mlenv

import "orrery/game"

// cometProduction is a comet's production. The engine forces every comet to
// 1 whatever the caller sets (see game.Comet), so this is a fact about the
// rules rather than a tuning choice.
const cometProduction = 1

// playerProductionScale divides a player's total production. It is the most
// a player could hold if they owned every slot obs-v2 has: 40 planets at the
// top production of 5, plus 4 comets at 1, which is 204.
const playerProductionScale = planetSlots*productionMax + cometSlots*cometProduction

// encodePlayers fills the four player rows, one per role: row 0 is the
// observer, then clockwise, opposite, and counter-clockwise.
//
// Each row also carries its role as a one-hot value. The model treats rows as
// an unordered set, so without it the policy could not tell which row is
// which, however carefully they were ordered.
//
// A role nobody holds — clockwise and counter-clockwise in a two-player game —
// stays an all-zero row. An eliminated player keeps its row and one-hot: it
// was still in the game, and dropping it would read as an opponent that never
// existed rather than one that has been beaten. Its totals are simply zero.
func encodePlayers(obs *Observation, g *game.Game, frame SeatFrame) {
	// Stage 1: totals per role. buildSlotTable has already rejected every owner
	// outside the starting roster, so each non-neutral owner has a role.
	var ships, production [roleCount]int
	add := func(owner, shipCount, productionCount int) {
		if role, ok := frame.RoleOf(owner); ok {
			ships[role] += shipCount
			production[role] += productionCount
		}
	}
	for _, p := range g.Planets {
		add(p.OwnerID, p.Ships, p.Production)
	}
	for _, c := range g.Comets {
		add(c.OwnerID, c.Ships, c.Production)
	}
	// Ships in flight still belong to their owner: a player who has just sent
	// a garrison to attack has not become weaker. Fleets produce nothing.
	for _, f := range g.Fleets {
		add(f.OwnerID, f.Ships, 0)
	}

	// Stage 2: one row per seated role.
	for role := range roleCount {
		if _, seated := frame.PlayerFor(role); !seated {
			continue
		}
		row := &obs.Players[role]
		row[PlayerFeatureShips] = shipFeature(ships[role])
		row[PlayerFeatureProduction] = float32(production[role]) / playerProductionScale
		row[PlayerFeatureRoleMe+int(role)] = 1
	}
}
