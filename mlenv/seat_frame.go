package mlenv

import (
	"fmt"

	"orrery/game"
)

// Role is a seated player's position relative to the observer, going
// clockwise around the sun. obs-v2 describes owners, incoming fleets, and
// player rows by role rather than by player ID, so the same situation looks
// the same to the policy whichever seat it plays.
type Role int

const (
	RoleMe Role = iota
	RoleCW
	RoleOpposite
	RoleCCW
	roleCount
)

// Seat cycle positions, indexed by player ID.
//
// A generated home group lists its members as: home, a quarter-turn
// clockwise, a quarter-turn counter-clockwise, then opposite (see
// v2PlanetGroup in package scenario). Four players take members 0–3 in that
// order, so going clockwise around the sun the seats run 0 → 1 → 3 → 2. Two
// players take members 0 and 3, which sit opposite each other.
//
// A player's cycle position p means its starting home is player 0's home
// turned p quarter-turns clockwise. Everything else follows from that one
// number:
//
//   - The frame rotates an observer at position p by (4 − p) mod 4 quarter
//     turns, which carries its home back onto player 0's corner.
//   - Player q's role from observer o is (p(q) − p(o)) mod 4, read as me,
//     clockwise, opposite, counter-clockwise.
//
// Worked out, those give these tables:
//
//	four players                          two players
//	observer  turns  me, cw, opp, ccw     observer  turns  me  opposite
//	0         0      0, 1, 3, 2           0         0      0   1
//	1         3      1, 3, 2, 0           1         2      1   0
//	2         1      2, 0, 1, 3
//	3         2      3, 2, 0, 1
var (
	fourPlayerCycle = [...]int{0, 1, 3, 2}
	twoPlayerCycle  = [...]int{0, 2}
)

// seatHomeTolerance is how far, in board units, a rotated starting home may
// land from player 0's home and still count as the same point.
//
// The generator builds each rotated copy as 2·sun − coordinate, while the
// frame rotates as sun + turn(position − sun). Those two float paths can round
// differently by a few units in the last place — around 1e-14 on a 100-unit
// board. 1e-9 is far above that rounding and far below any real misplacement:
// planets have radius at least 1, so a wrongly seated home is off by whole
// units.
const seatHomeTolerance = 1e-9

// SeatFrame fixes, for one observer and one episode, how the board is turned
// and which seated player holds which role.
//
// It is built once from the starting state and reused for every observation
// of the episode. Neither field changes after a capture, an orbit, or an
// elimination: roles describe where players started, not what they own now.
// A frame recomputed mid-game could swap roles under the policy's feet.
//
// All fields are unexported so a frame can only come from NewSeatFrame, which
// checks the board first. The zero value is not a usable frame.
type SeatFrame struct {
	// built is false on the zero value, so a builder can refuse a frame that
	// NewSeatFrame never checked.
	built bool
	// game identifies the episode this frame belongs to. It is kept only for
	// identity checks — Build refuses any other game, even an identical
	// copy — and frame methods never read or change the state through it.
	// Copying a game would need an explicit operation that rebinds the frame
	// while keeping its rotation and roles; rebuilding a frame from a game
	// past turn 0 would be wrong.
	game *game.Game
	// observer is the player ID this frame describes the board for.
	observer int
	// seated[p] reports whether player p was in the starting roster.
	seated [PlayerSlots]bool
	// homeID and homePosition record the observer's starting home planet and
	// where it stood at turn 0. They are kept for diagnostics; the frame does
	// not follow the planet afterwards.
	homeID       int
	homePosition game.Vec2
	// quarterTurns is how many clockwise quarter-turns about the sun carry the
	// observer's starting home onto player 0's. It is in 0..3.
	quarterTurns int
	// sun is the rotation centre. The frame requires it at the board centre,
	// so rotated positions stay on the board.
	sun game.Vec2
	// roleOf[p] is player p's role from the observer. Meaningful only when
	// seated[p].
	roleOf [PlayerSlots]Role
	// playerOf[r] is the seated player holding role r, or game.NeutralPlayer
	// when nobody does — clockwise and counter-clockwise in two-player games.
	playerOf [roleCount]int
}

// NewSeatFrame checks the starting board and returns observer's frame.
//
// The seat tables above assume the generator's seating. A hand-built board
// that breaks that assumption would otherwise receive confident but wrong
// roles, so the board is checked against geometry first:
//
//  1. The board is square with the sun at its centre. A quarter-turn about
//     any other point would carry bodies off the board.
//  2. The game is at turn 0, and each seated player owns exactly one planet,
//     its starting home. The roster is players {0, 1} or {0, 1, 2, 3}.
//  3. Every seated player's home, rotated by that player's own quarter-turn
//     count, lands on player 0's home. Checking only the observer would not
//     be enough: observer 0's rotation is zero, so its own check always
//     passes even if two opponents' homes are swapped.
//
// None of these checks depend on the observer, so every seat's frame rests on
// the same result.
func NewSeatFrame(g *game.Game, observer int) (SeatFrame, error) {
	if g == nil {
		return SeatFrame{}, fmt.Errorf("seat frame: nil game")
	}

	// Stage 1: board shape.
	cfg := g.Config
	if err := checkSeatBoard(cfg); err != nil {
		return SeatFrame{}, fmt.Errorf("seat frame: %w", err)
	}

	// Stage 2: the starting roster and each player's home.
	if g.Turn != 0 {
		return SeatFrame{}, fmt.Errorf("seat frame: must be built from the starting state, got turn %d", g.Turn)
	}
	var homes [PlayerSlots]*game.Planet
	var frame SeatFrame
	seatedCount := 0
	for i := range g.Planets {
		p := &g.Planets[i]
		if p.OwnerID == game.NeutralPlayer {
			continue
		}
		// game.New rejects other owners, but a Game can also be assembled by
		// hand, and an out-of-range ID must fail here rather than index past
		// the array.
		if p.OwnerID < 0 || p.OwnerID >= PlayerSlots {
			return SeatFrame{}, fmt.Errorf("seat frame: planet %d has owner %d, outside players 0..%d", p.ID, p.OwnerID, PlayerSlots-1)
		}
		if homes[p.OwnerID] != nil {
			return SeatFrame{}, fmt.Errorf("seat frame: player %d owns planets %d and %d at the start, want exactly one home",
				p.OwnerID, homes[p.OwnerID].ID, p.ID)
		}
		homes[p.OwnerID] = p
		frame.seated[p.OwnerID] = true
		seatedCount++
	}
	var cycle []int
	switch {
	case seatedCount == 2 && frame.seated[0] && frame.seated[1]:
		cycle = twoPlayerCycle[:]
	case seatedCount == 4:
		cycle = fourPlayerCycle[:]
	default:
		return SeatFrame{}, fmt.Errorf("seat frame: starting roster must be players {0, 1} or {0, 1, 2, 3}, got %v", frame.seated)
	}
	if observer < 0 || observer >= len(cycle) {
		return SeatFrame{}, fmt.Errorf("seat frame: observer %d is not seated in this %d-player game", observer, len(cycle))
	}

	// Stage 3: every home is the right rotation of player 0's home.
	for player, position := range cycle {
		turns := (4 - position) % 4
		got := rotateAbout(homes[player].Position, cfg.Sun, turns)
		if got.Distance(homes[0].Position) > seatHomeTolerance {
			return SeatFrame{}, fmt.Errorf("seat frame: player %d's home %d at (%g, %g) is not player 0's home turned %d quarter-turns clockwise",
				player, homes[player].ID, homes[player].Position.X, homes[player].Position.Y, position)
		}
	}

	// The board checks out; fill in the observer's view of it.
	frame.built = true
	frame.game = g
	frame.observer = observer
	frame.homeID = homes[observer].ID
	frame.homePosition = homes[observer].Position
	frame.quarterTurns = (4 - cycle[observer]) % 4
	frame.sun = cfg.Sun
	for r := range frame.playerOf {
		frame.playerOf[r] = game.NeutralPlayer
	}
	for player, position := range cycle {
		role := Role((position - cycle[observer] + 4) % 4)
		frame.roleOf[player] = role
		frame.playerOf[role] = player
	}
	return frame, nil
}

// checkSeatBoard reports whether a board can be turned by quarter-turns about
// its sun without carrying bodies off it: the board must be square and the
// sun must sit at its centre.
//
// NewSeatFrame and Build both call it. A frame is checked once at turn 0,
// but Config is a plain field a caller could change afterwards — resizing a
// square board while leaving the sun where it was would keep it square yet
// move the sun off-centre — so Build checks again rather than trusting the
// frame's earlier result. One function keeps the two checks from drifting.
//
// Config values are set exactly, not computed, so plain comparison is safe.
func checkSeatBoard(cfg game.Config) error {
	if cfg.Width != cfg.Height || cfg.Sun.X != cfg.Width/2 || cfg.Sun.Y != cfg.Height/2 {
		return fmt.Errorf("needs a square board with the sun at its centre, got %gx%g with the sun at (%g, %g)",
			cfg.Width, cfg.Height, cfg.Sun.X, cfg.Sun.Y)
	}
	return nil
}

// Observer returns the player ID this frame describes the board for.
func (f SeatFrame) Observer() int { return f.observer }

// Players returns how many players the starting roster seats: 2 or 4.
func (f SeatFrame) Players() int {
	count := 0
	for _, seated := range f.seated {
		if seated {
			count++
		}
	}
	return count
}

// RoleOf returns player's role from the observer. It reports false for
// neutral and for any player not in the starting roster.
func (f SeatFrame) RoleOf(player int) (Role, bool) {
	if player < 0 || player >= PlayerSlots || !f.seated[player] {
		return 0, false
	}
	return f.roleOf[player], true
}

// PlayerFor returns the seated player holding role. It reports false when no
// seated player holds it, as for clockwise and counter-clockwise in a
// two-player game.
func (f SeatFrame) PlayerFor(role Role) (int, bool) {
	if role < 0 || role >= roleCount || f.playerOf[role] == game.NeutralPlayer {
		return game.NeutralPlayer, false
	}
	return f.playerOf[role], true
}

// Position turns a board position into the observer's frame: about the sun,
// by the frame's quarter-turn count.
func (f SeatFrame) Position(p game.Vec2) game.Vec2 {
	return rotateAbout(p, f.sun, f.quarterTurns)
}

// Vector turns a displacement into the observer's frame. A displacement has
// no position, so it turns about the origin rather than the sun.
func (f SeatFrame) Vector(v game.Vec2) game.Vec2 {
	return rotateQuarterTurns(v, f.quarterTurns)
}

// rotateAbout turns p clockwise about centre by turns quarter-turns.
func rotateAbout(p, centre game.Vec2, turns int) game.Vec2 {
	return centre.Add(rotateQuarterTurns(p.Sub(centre), turns))
}

// rotateQuarterTurns turns v clockwise by turns quarter-turns.
//
// The board's Y axis points down, so (x, y) → (−y, x) carries "right" (1, 0)
// to "down" (0, 1): clockwise on screen. Quarter-turns only swap and negate
// coordinates, so they are exact; sin and cos of 90° would add rounding error
// to every rotated position.
func rotateQuarterTurns(v game.Vec2, turns int) game.Vec2 {
	switch turns % 4 {
	case 1:
		return game.Vec2{X: -v.Y, Y: v.X}
	case 2:
		return game.Vec2{X: -v.X, Y: -v.Y}
	case 3:
		return game.Vec2{X: v.Y, Y: -v.X}
	default:
		return v
	}
}
