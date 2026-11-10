package mlenv

import (
	"fmt"
	"math"

	"orrery/game"
)

// SlotAction is the policy's decision for one body slot. Target is another
// body-slot index and is meaningful only when Launch is true. Keeping the
// target in observation coordinates lets the policy act on exactly the fixed
// space it sees; Decode maps the slot back to an engine body ID.
type SlotAction struct {
	Launch bool
	Target int32
}

// Decode converts per-slot obs-v2 decisions into engine launch actions for
// frame's observer.
//
// Every launch sends the source's whole current garrison. It is aimed with
// lead rather than at the target's current centre: bodies move in obs-v2, so
// aiming at where a target is now misses anything that orbits or is a comet.
// Decode asks game.LeadAim for a heading that meets the target where it will
// be, and falls back to the current centre when LeadAim finds no verified
// heading. That failure is an expected case, not an error: LeadAim tries one heading
// only, so "no aim found" does not mean the target cannot be reached. A
// fallback fleet may hit something else, reinforce, miss, or die, and an
// allowed target is never a promise that the fleet arrives.
//
// The stages for each launch run in this order:
//
//	validate the decision against the observation and the live game
//	  → size the launch from the live garrison
//	  → aim with LeadAim, or fall back to the target's current centre
//	  → collect the action
//
// Any invalid decision returns an error and no actions at all, not the ones
// decoded before it, so a caller can never submit half a turn by mistake.
//
// The angle is LeadAim's as it is, in engine coordinates. The seat frame turns
// what the policy sees, not what it does: an action names engine bodies by ID
// and goes straight to the engine, so rotating its angle would aim the fleet
// at the wrong place.
//
// obs must have been built by Build from this same game state, unchanged
// since. The frame check below proves only that obs and g belong to the same
// episode, not that the observation is fresh; keeping them in step is the
// caller's job. The live-game checks catch the clearest cases of a stale
// observation — a source that changed owner or ran out of ships.
//
// Decode only reads g. LeadAim simulates on its own copy, and only
// Game.AdvanceTurn applies the returned actions.
func Decode(g *game.Game, frame SeatFrame, obs Observation, acts [BodySlots]SlotAction) ([]game.Action, error) {
	// Stage 1: inputs. These mirror Build's, so an action is only ever
	// decoded with the frame its observation was built with.
	if g == nil {
		return nil, fmt.Errorf("decode obs-v2 actions: nil game")
	}
	if !frame.built {
		return nil, fmt.Errorf("decode obs-v2 actions: seat frame was not built by NewSeatFrame")
	}
	if frame.game != g {
		return nil, fmt.Errorf("decode obs-v2 actions: seat frame belongs to a different game")
	}
	player := frame.Observer()

	bodies := collectBodies(g)
	bodyByID := make(map[int]bodyRef, len(bodies))
	for _, body := range bodies {
		bodyByID[body.id] = body
	}

	// Stage 2: decisions, in slot order so the action order is deterministic.
	// The map above is only looked up by key, never iterated.
	actions := make([]game.Action, 0)
	for sourceSlot, decision := range acts {
		if !decision.Launch {
			continue
		}

		// Validate the source: the observation must offer it, and the live
		// game must still agree that the observer owns it and it has ships.
		if !obs.BodyMask[sourceSlot] {
			return nil, fmt.Errorf("decode obs-v2 actions: source slot %d is padding", sourceSlot)
		}
		if !obs.OwnedMask[sourceSlot] {
			return nil, fmt.Errorf("decode obs-v2 actions: source slot %d is not owned by player %d", sourceSlot, player)
		}
		sourceID := int(obs.SlotBodyID[sourceSlot])
		source, ok := bodyByID[sourceID]
		if !ok {
			return nil, fmt.Errorf("decode obs-v2 actions: source slot %d names body %d, which is not in the game", sourceSlot, sourceID)
		}
		if source.owner != player {
			return nil, fmt.Errorf("decode obs-v2 actions: source body %d belongs to player %d, not player %d", source.id, source.owner, player)
		}
		if source.ships <= 0 {
			return nil, fmt.Errorf("decode obs-v2 actions: source body %d has no ships", source.id)
		}

		// Validate the target: a different real body the mask allows.
		targetSlot := int(decision.Target)
		if targetSlot < 0 || targetSlot >= BodySlots {
			return nil, fmt.Errorf("decode obs-v2 actions: target slot %d is outside 0..%d", targetSlot, BodySlots-1)
		}
		if !obs.BodyMask[targetSlot] {
			return nil, fmt.Errorf("decode obs-v2 actions: target slot %d is padding", targetSlot)
		}
		if targetSlot == sourceSlot {
			return nil, fmt.Errorf("decode obs-v2 actions: source slot %d cannot target itself", sourceSlot)
		}
		targetID := int(obs.SlotBodyID[targetSlot])
		target, ok := bodyByID[targetID]
		if !ok {
			return nil, fmt.Errorf("decode obs-v2 actions: target slot %d names body %d, which is not in the game", targetSlot, targetID)
		}
		if target.id == source.id {
			return nil, fmt.Errorf("decode obs-v2 actions: source body %d cannot target itself through slot %d", source.id, targetSlot)
		}
		if !obs.TargetMask[sourceSlot][targetSlot] {
			return nil, fmt.Errorf("decode obs-v2 actions: launch from slot %d to slot %d is masked out", sourceSlot, targetSlot)
		}

		// Size: the whole live garrison. Launches are applied before
		// production in a turn, so ships produced this turn are never part of
		// the launch and the garrison seen now is exactly what can leave.
		ships := source.ships

		// Aim: lead aiming first, the target's current centre otherwise.
		var angle float64
		if aim, ok := g.LeadAim(source.id, target.id, ships); ok {
			angle = aim.Angle
		} else {
			delta := target.position.Sub(source.position)
			angle = math.Atan2(delta.Y, delta.X)
		}

		actions = append(actions, game.Action{
			PlayerID: player,
			PlanetID: source.id,
			Ships:    ships,
			Angle:    angle,
		})
	}
	return actions, nil
}
