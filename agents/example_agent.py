from poke_env.player import Player


class ExampleAgent(Player):
    """A simple agent that always picks the highest base power move."""

    def choose_move(self, battle):
        if battle.available_moves:
            best_move = max(battle.available_moves, key=lambda m: m.base_power)
            return self.create_order(best_move)
        return self.choose_random_move(battle)
