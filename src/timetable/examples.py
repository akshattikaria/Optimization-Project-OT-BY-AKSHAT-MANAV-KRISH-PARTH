"""Deliberately tiny, exhaustible conflict demonstrations; separate from benchmarks."""
from .model import Faculty, Group, Offering, Problem, Room


def conflict_instance(case: str) -> Problem:
    if case not in {"room", "workload"}:
        raise ValueError("case must be room or workload")
    rooms = ((Room("small", 20, "classroom", (1, 3)), Room("large", 40, "classroom", (0, 2)))
             if case == "room" else (Room("classroom", 20, "classroom"),))
    faculty = ((Faculty("A", 2, 2), Faculty("B", 2, 2)) if case == "room" else
               (Faculty("A", 2, 2, (2, 3)), Faculty("B", 2, 2, (0, 1, 2))))
    return Problem(f"exact_{case}_conflict", ("Monday",), 4, (), (Group("G", 20),), rooms, faculty,
                   tuple(Offering(k, ("G",), ("A", "B"), 1, 1, "classroom") for k in ("S1", "S2")),
                   {"source": "synthetic analytical conflict example; exhaustive proof possible"})


def room_conflict_instance(seed: int = 0, groups: int = 2, days: int = 2, periods: int = 5,
                           offerings: int = 5) -> Problem:
    """Small synthetic instance with a deliberate gaps-versus-room-fit conflict.

    Best-fit classrooms are unavailable in about 60% of the periods, a
    mid-sized room in about 35% and the largest room never.  A compact day
    therefore tends to need oversized rooms, while using the best-fit rooms
    tends to leave idle periods.  The instance stays small enough for CP-SAT
    to certify its complete (gaps, room wastage) front.  It is an analytical
    test family, not a model of a real college.
    """
    import random

    if min(groups, days, periods, offerings) < 1:
        raise ValueError("groups, days, periods and offerings must be positive")
    rng = random.Random(seed)
    total = days * periods
    sizes = [(20, 24, 30)[g % 3] for g in range(groups)]
    cohort = tuple(Group(f"G{g + 1}", sizes[g]) for g in range(groups))
    rooms = []
    for tier, (multiplier, rate) in enumerate(zip((1.0, 1.5, 2.0), (0.6, 0.35, 0.0))):
        for size in (sorted(set(sizes)) if tier == 0 else [max(sizes)]):
            capacity = size if tier == 0 else max(size, int(round(size * multiplier / 10.0)) * 10)
            unavailable = tuple(t for t in range(total) if rng.random() < rate)
            rooms.append(Room(f"R{len(rooms) + 1}", capacity, "classroom", unavailable))
    teachers = tuple(Faculty(f"F{i + 1}", total, periods) for i in range(max(2, (groups * offerings + 3) // 4)))
    names = [t.id for t in teachers]
    items = []
    for g in range(groups):
        for o in range(offerings):
            eligible = tuple(sorted(rng.sample(names, min(2, len(names)))))
            items.append(Offering(f"G{g + 1}-S{o + 1}", (cohort[g].id,), eligible, 1, 1, "classroom", False))
    return Problem(f"room_conflict_{groups}x{offerings}_s{seed}", tuple(f"Day{d + 1}" for d in range(days)), periods, (),
                   cohort, tuple(rooms), teachers, tuple(items),
                   {"source": "synthetic analytical room-availability conflict family; small enough for CP-SAT proofs"})
