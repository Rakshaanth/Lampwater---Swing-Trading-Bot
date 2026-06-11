from src.execute import select_orders


class TestSelectOrders:
    def test_sizes_orders_as_whole_shares(self):
        orders = select_orders([("AAA", 333.0)], held=set(), max_open_positions=10, position_size_usd=1000)
        assert orders == [{"symbol": "AAA", "qty": 3, "ref_price": 333.0}]

    def test_skips_already_held_symbols(self):
        orders = select_orders(
            [("AAA", 100.0), ("BBB", 100.0)],
            held={"AAA"}, max_open_positions=10, position_size_usd=1000,
        )
        assert [o["symbol"] for o in orders] == ["BBB"]

    def test_respects_max_open_positions(self):
        signals = [("AAA", 100.0), ("BBB", 100.0), ("CCC", 100.0)]
        orders = select_orders(signals, held={"X", "Y"}, max_open_positions=4, position_size_usd=1000)
        assert [o["symbol"] for o in orders] == ["AAA", "BBB"]

    def test_no_orders_when_position_cap_already_reached(self):
        orders = select_orders([("AAA", 100.0)], held={"X", "Y"}, max_open_positions=2, position_size_usd=1000)
        assert orders == []

    def test_skips_symbols_too_expensive_for_one_share(self):
        orders = select_orders(
            [("PRICY", 5000.0), ("OK", 100.0)],
            held=set(), max_open_positions=10, position_size_usd=1000,
        )
        assert [o["symbol"] for o in orders] == ["OK"]

    def test_held_symbol_does_not_consume_a_slot(self):
        # 2 free slots (3 cap − 1 held); held signal is skipped without using one
        signals = [("HELD", 100.0), ("AAA", 100.0), ("BBB", 100.0)]
        orders = select_orders(signals, held={"HELD"}, max_open_positions=3, position_size_usd=1000)
        assert [o["symbol"] for o in orders] == ["AAA", "BBB"]
