package mlenv

// encodeMasks fills the masks and the slot-to-body map, which tell the
// model which rows are real and which launches it may choose.
//
// Each mask is derived from the slot table and the frame, never from the
// encoded features: legality is a fact about the game, and a normalized
// column is only a rendering of it for the network.
func encodeMasks(obs *Observation, frame SeatFrame, slots slotTable) {
	// Stage 1: which slots are real, whose they are, and the way back from a
	// slot to its engine body, which decoding a chosen action needs.
	for slot, body := range slots.bodies {
		if !body.real() {
			continue
		}
		obs.BodyMask[slot] = true
		obs.SlotBodyID[slot] = int32(body.id())
		obs.OwnedMask[slot] = body.owner() == frame.Observer()
	}

	// Stage 2: player rows. The mask comes from the starting roster held by
	// the frame, not from who owns anything now, so it never changes during
	// an episode. A two-player game is [me, -, opposite, -].
	for role := range roleCount {
		_, seated := frame.PlayerFor(role)
		obs.PlayerMask[role] = seated
	}

	// Stage 3: targets. A source must be a real body the observer owns with
	// at least one ship, read from the raw garrison. A target is any other
	// real body, friendly ones included: reinforcing is a legal and often
	// correct move. An owned body with no ships keeps OwnedMask but gets an
	// all-false row, so Python's launchable mask, OwnedMask AND any(target),
	// leaves it out. The mask does not check reachability: Decode aims with
	// lead aiming when it can and at the target's current centre otherwise,
	// so an allowed target may still not receive the fleet.
	for source, body := range slots.bodies {
		if !obs.OwnedMask[source] || body.ships() <= 0 {
			continue
		}
		for target := range BodySlots {
			if target != source && obs.BodyMask[target] {
				obs.TargetMask[source][target] = true
			}
		}
	}
}
