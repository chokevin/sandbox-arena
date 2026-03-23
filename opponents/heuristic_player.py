from poke_env.player import Player


class HeuristicPlayer(Player):
    """Opponent that uses type effectiveness and base power."""

    def choose_move(self, battle):
        if battle.available_moves:
            opponent = battle.opponent_active_pokemon

            def move_score(move):
                power = move.base_power if move.base_power else 0
                if opponent:
                    multiplier = opponent.damage_multiplier(move)
                else:
                    multiplier = 1.0
                return power * multiplier

            best_move = max(battle.available_moves, key=move_score)
            if move_score(best_move) > 0:
                return self.create_order(best_move)

        return self.choose_random_move(battle)
