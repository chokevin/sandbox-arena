def play(hand_total, dealer_showing, num_cards):
    """Basic strategy blackjack agent.

    Args:
        hand_total: current hand value (int)
        dealer_showing: dealer's visible card value (int)
        num_cards: number of cards in hand (int)
    Returns:
        "hit" or "stand"
    """
    # Stand on 17+
    if hand_total >= 17:
        return "stand"
    # Always hit on 11 or less
    if hand_total <= 11:
        return "hit"
    # Hit on 12-16 if dealer shows 7+
    if dealer_showing >= 7:
        return "hit"
    # Stand on 12-16 if dealer shows 2-6 (dealer likely to bust)
    return "stand"
