You are the planner of a workflow controller for a tool-using LLM agent. The subtasks below are fixed. Write the transition criteria the router uses to choose where control goes after each segment.

$workflow
## Instructions

- The response schema lists, for each source subtask, every destination the router can choose from it. For each pair write the criterion: the condition under which, after a segment of the source subtask, control should move to that destination.
- Make each criterion checkable from what the router sees: the segment's tool calls and outputs, the executor's summary, and its memory of the episode so far. One or two sentences.
- The criteria leaving a subtask must not overlap, so that the right destination is clear in every situation.
- A subtask's edge to itself means running it again: use it when its work is unfinished or can be fixed in place.
- `give_up`, where offered, ends the episode at once and, because the goal has not been reached, counts as a failure. The environment already ends the episode by itself when the goal is reached, when a tool that the benchmark description says ends the episode is called, or when the global tool-call budget is spent, and no criterion is checked after that, so never write a criterion for those cases. A segment that used up its own tool-call allowance is not a reason to give up either; continuing or advancing handles that. Choose `give_up` only when the evidence shows the objective can no longer be completed.
- If summaries of successful trajectories are given, reflect the transitions that worked in them.

## Benchmark

$benchmark

## Objective

$objective
$nshot
## Subtasks

Execution starts at `$entry`.

```json
$subtasks
```
