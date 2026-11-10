package mlenv

import (
	"fmt"

	"orrery/game"
)

// encodeIncoming fills body columns 16–23: for each body, the ships each
// role has inbound and how soon that role's first fleet makes contact.
//
// A single "enemy" total would hide what combat will do with four players:
// arriving owners fight each other first, so 10 ships each from two opponents
// destroy each other before the garrison is touched, while 20 from one
// opponent hit it in full. Keeping a column per role lets the policy tell the
// two apart.
//
// It follows every fleet with Game.PredictFleets, so what it
// reports is what AdvanceTurn will do, moving orbits and comets included. It
// summarizes inbound fleets only: it does not predict who will own the body
// afterwards, and it cannot see comet waves that have not spawned yet.
func encodeIncoming(obs *Observation, g *game.Game, frame SeatFrame, slots slotTable) error {
	// Stage 1: predict every fleet once. predictFleets fails if any fleet is
	// still flying after the longest crossing the board allows.
	landings, err := predictFleets(g)
	if err != nil {
		return err
	}
	horizon := float64(maxPredictionSegments(g))

	// Stage 2: aggregate by destination slot and owner role. Ships are summed
	// as raw counts and compressed only at the end: log1p of a sum is not a
	// sum of log1p values.
	var ships [BodySlots][roleCount]int
	var earliest [BodySlots][roleCount]float64
	var arrives [BodySlots][roleCount]bool
	for i, f := range g.Fleets {
		landing := landings[i]
		if !landing.Arrives() {
			// Lost to the sun or the board edge, or reaching a comet just as
			// it departs: these ships reach no body, so they count nowhere.
			continue
		}
		slot, ok := slots.slotOf[landing.Hit.BodyID]
		if !ok {
			// The slot table and the prediction come from the same game, so
			// this can only mean the engine reported a body the board does not
			// hold — a broken state, not one to encode around.
			return fmt.Errorf("obs-v2: fleet %d is predicted to reach body %d, which has no slot", f.ID, landing.Hit.BodyID)
		}
		// buildSlotTable has already rejected neutral fleets and fleets owned
		// outside the roster. The check stays anyway: RoleOf's zero role is
		// RoleMe, so ignoring a false here would quietly count someone else's
		// ships as the observer's.
		role, ok := frame.RoleOf(f.OwnerID)
		if !ok {
			return fmt.Errorf("obs-v2: fleet %d belongs to player %d, who has no role", f.ID, f.OwnerID)
		}
		ships[slot][role] += f.Ships
		// ETA is the moment of physical contact, in turns from now: whole
		// turns already flown plus the fraction of the final turn. Combat
		// still resolves at the end of that turn.
		eta := float64(landing.Segment) + landing.Hit.T
		if !arrives[slot][role] || eta < earliest[slot][role] {
			earliest[slot][role] = eta
			arrives[slot][role] = true
		}
	}

	// Stage 3: write the columns. A contact happens within the horizon, so
	// dividing by horizon+1 keeps every real ETA strictly below the 1 that
	// means "nothing from this role is inbound".
	for slot, body := range slots.bodies {
		if !body.real() {
			continue
		}
		row := &obs.Bodies[slot]
		for role := range roleCount {
			row[FeatureIncomingMe+int(role)] = shipFeature(ships[slot][role])
			arrival := float32(noArrival)
			if arrives[slot][role] {
				arrival = float32(earliest[slot][role] / (horizon + 1))
			}
			row[FeatureArrivalMe+int(role)] = arrival
		}
	}
	return nil
}
