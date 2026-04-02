from graph import Graph
from path_planner import RailwayPlanner


def test_1():
    g = Graph.from_edge_list([
        (0, 1, 1), (1, 2, 1),  # path for ticket 1
        (3, 4, 1), (4, 5, 1),  # path for ticket 2
    ], num_nodes=6)
    planner = RailwayPlanner(g)
    result = planner.plan([(0, 2), (3, 5)])
    assert result.satisfied == 2
    assert result.unsatisfied == 0
    assert result.total_cost > 0

def test_2():
    g = Graph.from_edge_list([
        (0, 1, 1), (1, 2, 1), (2, 3, 1)
    ], num_nodes=4)

    planner = RailwayPlanner(g)
    result = planner.plan([(0, 3), (3, 0)])

    assert result.satisfied == 1
    assert result.unsatisfied == 1

def test_3():
    g = Graph.from_edge_list([
        (0, 1, 1), (1, 2, 1), (0, 2, 5),
        (2, 3, 1), (1, 3, 4)
    ], num_nodes=4)

    planner = RailwayPlanner(g, max_k=5)
    result = planner.plan([(0, 2), (1, 3)])

    assert result.ticket_results[0].chosen_path.cost == 2.0
    assert result.ticket_results[1].chosen_path.cost == 4.0
    assert result.ticket_results[1].attempts >= 2

if __name__ == "__main__":
    test_1()
    print("Test 1 passed")
    test_2()
    print("Test 2 passed")
    test_3()
    print("Test 3 passed")
    print("All integration tests passed!")