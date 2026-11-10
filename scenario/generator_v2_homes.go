package scenario

import (
	"fmt"
	"math/rand"

	"orrery/game"
)

const v2HomeShips = 10

// assignV2Homes changes one newly generated neutral group in place. It uses
// the same map RNG after group placement, so a seed fixes both the layout and
// the selected home group. The caller owns planets; the home garrisons replace
// their neutral rolls, while production and every other planet value stay as
// generated.
//
// Two players take the group's first and fourth planets, which sit opposite
// each other across the sun. Four players take one planet each, in group
// order. Every member of a group is a quarter-turn rotation of the others about
// the sun (see v2PlanetGroup), so each seat starts from an equivalent position
// either way. Rotation, not mirroring, is what keeps seats equivalent once the
// board moves: every orbit turns the same way, and a mirror image would turn
// the opposite way.
func assignV2Homes(rng *rand.Rand, planets []game.Planet, fourPlayer bool) error {
	// Validate the whole input before mutating or drawing randomness. A
	// malformed group list must not leave half-assigned homes or shift the
	// replay stream before the caller reports the error.
	if len(planets) == 0 || len(planets)%4 != 0 {
		return fmt.Errorf("home assignment needs complete groups of four planets, got %d", len(planets))
	}
	for _, planet := range planets {
		if planet.OwnerID != game.NeutralPlayer {
			return fmt.Errorf("home assignment needs neutral planets, found owner %d on planet %d", planet.OwnerID, planet.ID)
		}
	}

	// Both variants make this same single draw, so the map stream stays
	// identical between them for any later draws.
	base := 4 * rng.Intn(len(planets)/4)
	// owners[i] is the player who gets group member base+i; neutral leaves
	// that member as generated.
	owners := [4]int{0, game.NeutralPlayer, game.NeutralPlayer, 1}
	if fourPlayer {
		owners = [4]int{0, 1, 2, 3}
	}
	for i, owner := range owners {
		if owner == game.NeutralPlayer {
			continue
		}
		planets[base+i].OwnerID = owner
		planets[base+i].Ships = v2HomeShips
	}
	return nil
}
