You grade an AI research agent's answer to a financial question against an expert rubric.

<question>
$question
</question>

<expert_answer>
$expected_answer
</expert_answer>

<agent_answer>
$answer
</agent_answer>

Sources cited by the agent:
$sources

Rubric criteria:

$criteria

For each criterion, reason about the agent's answer and then give a yes/no verdict to the criterion's question:
- correctness: does the agent's answer state the criterion, or something equivalent to it? Different wording, units or reasonable rounding of the same value count as stated. A missing, different or only partially covered fact is "no".
- contradiction: does the agent's answer contradict the reference statement? Answer "yes" if it asserts figures, facts or conclusions incompatible with the statement. Leaving something out is not a contradiction.

Grade only what the agent's answer says; cited sources do not make up for content missing from the answer. Grade every criterion independently and put the verdict for criterion_1 in criterion_1, for criterion_2 in criterion_2, and so on.
