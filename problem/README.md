# College timetable decision problem

A department's timetable coordinator must decide **when**, **where**, and **by whom** every required weekly teaching session is delivered. Students must be able to attend their sessions, faculty cannot teach two sessions simultaneously, rooms have finite capacities and types, and a laboratory occupies several consecutive periods. A clash-free result is the minimum acceptable product, rather than one objective that can be traded away.

Among feasible schedules, the coordinator compares:

1. Student and faculty teaching-free internal gaps, excluding the mandatory lunch period.
2. The duration-weighted fraction of empty seats in the rooms used.
3. Variation in faculty teaching load relative to their individual weekly allowances.

Students benefit from compact teaching days; the institute benefits from appropriately sized rooms; faculty benefit from balanced relative loads and compact teaching. These aims can conflict when scarce resources or restricted availability force a choice. Conflict is not guaranteed on every instance, so the experiment must measure rather than presume the trade-off. The [formulation](../formulation/README.md) includes small explicit examples.

## Inputs and resulting decision

The inputs are atomic disjoint student groups and sizes, rooms and capacities/types, faculty eligibility and teaching limits, resource unavailability, required offerings and session counts, session durations, and a weekly day/period grid with a common lunch. The generator supplies configurable **synthetic** engineering-college instances. No real student records, real staff performance data, or anonymized institutional timetable is included.

An offering is a subject delivered to specified groups, with one teacher for all its sessions. Each required session is an event. The solver assigns one time block and one compatible room per event, and one eligible teacher per offering. A multi-group event reserves every attending group and sums their headcounts. Cohort modeling is atomic: a full-division lecture should list its disjoint batches, rather than mixing overlapping parent and batch cohorts.

The output is a set of feasible schedules that are non-dominated among the schedules discovered by the search, with explicit objectives and readable weekly grids. A recommended compromise is a policy choice, not a theorem about the uniquely best timetable. Administrators should review gap-minimal, room-efficient, workload-balanced, and compromise choices; several labels may refer to the same timetable if the front is small.

## Hard requirements

Every required event appears exactly once. Room, student-group, and faculty usage never overlaps in any occupied period. The entire block fits within one day, avoids lunch, obeys faculty/room availability, uses a correctly typed room of adequate capacity, and respects faculty daily/weekly teaching limits. Repeated sessions of an offering use one consistent eligible teacher. Offerings marked `distinct_days` cannot have two sessions on the same day.

A numerical violation score helps the algorithm search. It does not turn clashes or capacity failures into acceptable soft preferences. An infeasible candidate is never a valid timetable, even if its objective values are numerically attractive.

## Assumptions and scope

The model describes a deterministic week that repeats during a term. Sessions and durations are fixed and known before scheduling. Teaching periods are equal in length. The faculty load measure counts teaching periods, rather than preparation effort, research, administrative duties, or student contact difficulty. Exact room-type matching is used. There are no travel times, rotating weeks, student-level elective choices, team teaching, or mid-semester arrivals. Fixed event starts and minimum faculty loads require explicit schema/model extensions.

Known-feasible synthetic witnesses are stored separately from the problem input and are used only to test feasibility, not to initialize the compared solvers. Feasible generation is not an infeasibility oracle for arbitrary edited inputs. Failure by a greedy or evolutionary algorithm does not prove that no timetable exists.

## Course context

This project follows the supplied four-member, eight-week Optimization Techniques brief and the timetable proposal, with scope corrections documented in [reference_review.md](../docs/reference_review.md). Topic approval, repository privacy/access, personal contribution records, and eight weeks of authentic development history are course requirements to be supplied by the students. They are not asserted by this implementation.
