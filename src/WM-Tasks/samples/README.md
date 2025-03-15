# Sample Instructions

## Notes:

* The CSV file includes the following columns:
* trial: Trial number
* condition: The n‑back load (e.g., “1-back”, “2-back”, “3-back”)
* stimulus: The stimulus identifier
* response: The participant’s response (expected “yes” or “no”)
* correct: Indicator (1 for correct, 0 for incorrect)
* reaction_time: Reaction time in milliseconds
* target: Whether a target was present (1) or not (0)

## Notes for Reverse Span:

* Each trial’s data is represented by multiple lines; only the final line per trial is used for metric computation.
* The fields (separated by commas) are: trial_id, best_score, reaction_time, and error_details.
