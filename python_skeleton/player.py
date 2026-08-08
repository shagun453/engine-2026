"""
MIT Pokerbot Player – Conservative Improvements
Only the safest, highest-impact changes from original
"""

from skeleton.actions import FoldAction, CallAction, CheckAction, RaiseAction, DiscardAction
from skeleton.states import STARTING_STACK
from skeleton.bot import Bot
from skeleton.runner import parse_args, run_bot

import random
from itertools import combinations


class Player(Bot):

    ############################
    # INIT
    ############################

    def __init__(self):
        self.deck = self._create_deck()

        self.rank_values = {
            '2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7,
            '8': 8, '9': 9, 'T': 10, 'J': 11, 'Q': 12, 'K': 13, 'A': 14
        }

        # Opponent discard bias stats (Laplace smoothed)
        self.opp_discard = {"total": 1, "high": 1, "entropy": 1}

    def _create_deck(self):
        ranks = ['2', '3', '4', '5', '6', '7', '8', '9', 'T', 'J', 'Q', 'K', 'A']
        suits = ['h', 'd', 'c', 's']
        return [r + s for r in ranks for s in suits]

    ############################
    # ROUND HOOKS
    ############################

    def handle_new_round(self, game_state, round_state, active):
        pass

    def handle_round_over(self, game_state, terminal_state, active):
        prev = terminal_state.previous_state
        if prev is None or prev.board is None or len(prev.board) < 3:
            return

        discarded = prev.board[-1]
        self.opp_discard["total"] += 1

        if self.rank_values[discarded[0]] >= 11:
            self.opp_discard["high"] += 1

        if self._entropy_added_by_discard(discarded, prev.board[:-1]) >= 2:
            self.opp_discard["entropy"] += 1

    ############################
    # HAND EVAL (unchanged)
    ############################

    def _evaluate_5card_hand(self, five):
        ranks = [self.rank_values[c[0]] for c in five]
        suits = [c[1] for c in five]

        counts = {}
        for r in ranks:
            counts[r] = counts.get(r, 0) + 1

        sorted_counts = sorted(counts.values(), reverse=True)
        is_flush = len(set(suits)) == 1

        unique = sorted(set(ranks))
        is_straight = False
        high = 0

        if len(unique) >= 5:
            for i in range(len(unique) - 4):
                if unique[i + 4] - unique[i] == 4:
                    is_straight = True
                    high = unique[i + 4]
                    break
            if set([14, 2, 3, 4, 5]).issubset(unique):
                is_straight = True
                high = 5

        if is_straight and is_flush:
            return (9, high)
        if sorted_counts == [4, 1]:
            quad = max(counts, key=lambda r: counts[r])
            kicker = min(counts, key=lambda r: counts[r])
            return (8, quad * 15 + kicker)
        if sorted_counts == [3, 2]:
            trips = max(counts, key=lambda r: counts[r])
            pair = min(counts, key=lambda r: counts[r])
            return (7, trips * 15 + pair)
        if is_flush:
            return (6, sorted(ranks, reverse=True))
        if is_straight:
            return (5, high)
        if sorted_counts == [3, 1, 1]:
            return (4, sorted(ranks, reverse=True))
        if sorted_counts == [2, 2, 1]:
            return (3, sorted(ranks, reverse=True))
        if sorted_counts == [2, 1, 1, 1]:
            return (2, sorted(ranks, reverse=True))
        return (1, sorted(ranks, reverse=True))

    def _best_hand(self, hole, board):
        if len(hole) + len(board) < 5:
            return (0, [])
        return max(self._evaluate_5card_hand(list(c))
                   for c in combinations(hole + board, 5))

    ############################
    # ENTROPY MODELS
    ############################

    def _entropy_added_by_discard(self, card, board):
        entropy = 0
        rank = self.rank_values[card[0]]
        suit = card[1]

        suit_count = sum(1 for c in board if c[1] == suit)
        if suit_count == 2:
            entropy += 2
        elif suit_count >= 3:
            entropy += 3

        ranks = sorted(self.rank_values[c[0]] for c in board + [card])
        for i in range(len(ranks) - 3):
            if ranks[i + 3] - ranks[i] <= 4:
                entropy += 2
                break

        if any(self.rank_values[c[0]] == rank for c in board):
            entropy += 2
        if rank >= 11:
            entropy += 1

        return entropy

    def _board_entropy(self, board):
        """
        Quantifies uncertainty/chaos of a poker board (0-5 range).
        Low entropy: structured boards (pairs, straights likely).
        High entropy: scattered boards with few draw opportunities.
        """
        if not board:
            return 0.0
        
        entropy = 0.0
        ranks = [self.rank_values[c[0]] for c in board]
        suits = [c[1] for c in board]
        
        # 1. Pair penalty: -1 per pair
        rank_counts = {}
        for r in ranks:
            rank_counts[r] = rank_counts.get(r, 0) + 1
        pairs = sum(1 for count in rank_counts.values() if count >= 2)
        entropy -= pairs * 1.0
        
        # 2. Flush draw bonus: +0.5 per suit with 2+ cards
        suit_counts = {}
        for s in suits:
            suit_counts[s] = suit_counts.get(s, 0) + 1
        flush_draws = sum(1 for count in suit_counts.values() if count >= 2)
        entropy += flush_draws * 0.5
        
        # 3. Straight draw bonus: +0.3 per 3+ rank gap
        sorted_ranks = sorted(set(ranks))
        gap_count = 0
        for i in range(len(sorted_ranks) - 1):
            if sorted_ranks[i + 1] - sorted_ranks[i] >= 3:
                gap_count += 1
        entropy += gap_count * 0.3
        
        # 4. Rank spread: +0.1 × normalized rank range
        if len(sorted_ranks) > 1:
            rank_range = sorted_ranks[-1] - sorted_ranks[0]
            normalized_spread = rank_range / 13.0
            entropy += 0.1 * normalized_spread
        
        # 5. Gap clustering: -0.2 per tight cluster (consecutive ranks)
        tight_clusters = 0
        for i in range(len(sorted_ranks) - 1):
            if sorted_ranks[i + 1] - sorted_ranks[i] == 1:
                tight_clusters += 1
        entropy -= tight_clusters * 0.2
        
        # Clamp to [0, 5]
        return max(0.0, min(5.0, entropy))

    def _opponent_entropy_bias(self):
        s = self.opp_discard
        total = s["total"]
        bias = 1.0
        bias += (s["entropy"] / total - 0.4)
        bias += 0.5 * (s["high"] / total - 0.33)
        return max(0.7, min(1.4, bias))

    ############################
    # MONTE CARLO DISCARD - IMPROVED OPP MODEL ONLY
    ############################

    def _monte_carlo_toss(self, cards, board, sims):
        known = set(cards + board)
        remaining = [c for c in self.deck if c not in known]

        best_idx, best_score = 0, -1
        opp_bias = self._opponent_entropy_bias()

        for i in range(3):
            tossed = cards[i]
            wins = 0

            for _ in range(sims):
                my_final = [c for j, c in enumerate(cards) if j != i]
                opp_cards = random.sample(remaining, 3)
                
                # ONLY CHANGE: Better opponent model
                # Weight opponent's discard choice by rank (prefer discarding low)
                opp_ranks = [self.rank_values[c[0]] for c in opp_cards]
                min_rank = min(opp_ranks)
                # Find all cards with minimum rank
                min_indices = [k for k, c in enumerate(opp_cards) if self.rank_values[c[0]] == min_rank]
                # Randomly choose among lowest cards
                opp_toss = random.choice(min_indices)
                
                opp_final = [c for k, c in enumerate(opp_cards) if k != opp_toss]

                board2 = board + [tossed, opp_cards[opp_toss]]
                used = set(my_final + opp_final + board2)
                future = [c for c in self.deck if c not in used]

                if len(future) < 2:
                    continue

                board_full = board2 + random.sample(future, 2)

                my_hand = self._best_hand(my_final, board_full)
                opp_hand = self._best_hand(opp_final, board_full)

                if my_hand[0] > opp_hand[0]:
                    wins += 1
                elif my_hand == opp_hand:
                    wins += 0.5

            win_rate = wins / max(1, sims)
            # Only penalize extreme entropy (> 2.5), gentler curve
            entropy_val = self._entropy_added_by_discard(tossed, board)
            penalty = 0.02 * max(0, entropy_val - 2.5) * opp_bias
            score = win_rate - penalty

            if score > best_score:
                best_score, best_idx = score, i

        return best_idx


    def _simulate_raise_ev(self, hole, board, continue_cost, sims=30):
        """
        Simulate expected value of raising vs calling.
        Returns (EV_raise, EV_call)
        """
        known = set(hole + board)
        remaining = [c for c in self.deck if c not in known]

        wins_call = 0
        wins_raise = 0

        for _ in range(sims):
            # Simulate opponent hand
            opp_hand = random.sample(remaining, 2)
            remaining_s = [c for c in remaining if c not in opp_hand]

            # Simulate 2 future board cards (turn+river)
            if len(remaining_s) < 2:
                continue
            future_board = board + random.sample(remaining_s, 2)

            my_best = self._best_hand(hole, future_board)
            opp_best = self._best_hand(opp_hand, future_board)

            # Evaluate EV if we just call
            if my_best[0] > opp_best[0]:
                wins_call += 1
            elif my_best[0] == opp_best[0]:
                wins_call += 0.5

            # Evaluate EV if we raise (opponent may fold randomly for bluff effect)
            fold_prob = 0.1  # baseline chance opponent folds to raise
            if random.random() < fold_prob:
                wins_raise += 1
            else:
                if my_best[0] > opp_best[0]:
                    wins_raise += 1
                elif my_best[0] == opp_best[0]:
                    wins_raise += 0.5

        ev_call = wins_call / max(1, sims) - continue_cost / STARTING_STACK
        ev_raise = wins_raise / max(1, sims) - (continue_cost * 1.5) / STARTING_STACK  # scale cost for raise

        return ev_raise, ev_call

    ############################
    # ACTION
    ############################

        
    def get_action(self, game_state, round_state, active):
        legal = round_state.legal_actions()
        street = round_state.street
        board = round_state.board
        hole = round_state.hands[active]

        my_pip = round_state.pips[active]
        opp_pip = round_state.pips[1 - active]
        continue_cost = opp_pip - my_pip
        pot = (STARTING_STACK - round_state.stacks[0]) + (STARTING_STACK - round_state.stacks[1])
        in_position = (active == 0)

        # -------- DISCARD --------
        if DiscardAction in legal:
            clock = game_state.game_clock
            rnd = game_state.round_num
            if clock < 5:
                sims = 10
            elif rnd > 600:
                sims = 25
            elif rnd > 400:
                sims = 40
            elif clock < 20:
                sims = 60
            else:
                sims = 120
            return DiscardAction(self._monte_carlo_toss(hole, board, sims))

        # -------- PRE-FLOP AGGRESSION BOOST (conservative) --------
        if street == 0:  # preflop
            hole_ranks = [self.rank_values[c[0]] for c in hole]
            high_cards = sum(1 for r in hole_ranks if r >= 12)  # J, Q, K, A only (not T)
            # Raise cautiously with premium cards pre-flop, only in position
            if high_cards >= 1 and in_position and RaiseAction in legal:
                min_r, max_r = round_state.raise_bounds()
                return RaiseAction(min(min_r * 2, max_r))  # 2x min, not 3x

        # -------- HAND STRENGTH --------
        strength = self._best_hand(hole, board)[0] / 9.0  # normalize [0,1]
        pot_odds = continue_cost / (pot + continue_cost) if continue_cost > 0 else 0
        entropy = self._board_entropy(board)
        opp_bias = self._opponent_entropy_bias()

        # Base aggression scaler
        agg = 1.0
        if entropy >= 4:
            agg *= 0.7
        elif entropy <= 1:
            agg *= 1.15
        if opp_bias > 1.2:
            agg *= 0.8
        elif opp_bias < 0.9:
            agg *= 1.2
        agg = max(0.6, min(1.4, agg))

        # -------- STRONG HAND --------
        if strength >= 0.75:
            if RaiseAction in legal:
                ev_raise, _ = self._simulate_raise_ev(hole, board, continue_cost, sims=30)
                min_r, max_r = round_state.raise_bounds()
                size = int(pot * 0.6 * agg + ev_raise * pot)
                return RaiseAction(max(min_r, min(size, max_r)))
            return CallAction() if CallAction in legal else CheckAction()

        # -------- MEDIUM HAND (EV-based) --------
        if strength >= 0.40:
            ev_raise, ev_call = self._simulate_raise_ev(hole, board, continue_cost, sims=25)

            if ev_raise > 0 and RaiseAction in legal:
                min_r, max_r = round_state.raise_bounds()
                size = int(pot * 0.4 * agg + ev_raise * pot)
                return RaiseAction(max(min_r, min(size, max_r)))

            if continue_cost == 0:
                # opportunistic in-position raise with some probability (reduced)
                if RaiseAction in legal and in_position and random.random() < 0.15 * agg:
                    min_r, max_r = round_state.raise_bounds()
                    size = int(pot * 0.3 * agg)
                    return RaiseAction(min_r)
                return CheckAction()

            # Call if profitable by pot odds (tighter threshold)
            if strength > pot_odds * (1.3 / agg):
                return CallAction()
            return FoldAction()

        # -------- WEAK HAND --------
        if continue_cost == 0:
            bluff_freq = (0.12 if in_position else 0.05) * agg
            if RaiseAction in legal and random.random() < bluff_freq:
                min_r, _ = round_state.raise_bounds()
                return RaiseAction(min_r)
            return CheckAction()

        return FoldAction()


if __name__ == "__main__":
    run_bot(Player(), parse_args())
