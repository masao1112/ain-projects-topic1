from starter.graph import load_graph
from starter.student_core import check_euclidean_consistency

graph = load_graph()

total_violations = 0
max_violation = 0.0
max_case = None

for goal in graph._nodes:
    violations = check_euclidean_consistency(
        graph,
        goal
    )

    total_violations += len(violations)

    for violation in violations:
        difference = violation["difference"]

        if difference > max_violation:
            max_violation = difference
            max_case = violation

print("Total violations:", total_violations)
print("Maximum violation:", max_violation)

if max_case:
    print("Worst case:")
    print(max_case)