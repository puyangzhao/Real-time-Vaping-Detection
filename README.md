# Real-time Vaping Detection Through Heart Rate Dynamics and Temporal Patterns

This repository provides reproducible analysis scripts for the statistical framework described in:

**Real-time Vaping Detection Through Heart Rate Dynamics and Temporal Patterns**

The repository does not contain participant-level wearable data or self-reported vaping records because the original data are subject to privacy and data-use restrictions. Instead, we provide a synthetic-data demonstration that follows the same workflow:

1. Minute-level heart-rate preprocessing
2. 60-minute retrospective window construction
3. Segment-level feature extraction using linear regression
4. Heart-rate-only KDE scoring
5. Time-aware probability adjustment
6. Leave-one-day-out cross-validation for tuning
7. Held-out test-day evaluation
8. Exploratory logistic-regression baseline

## Data format

The analytic input should be a segment-level or minute-level data table with the following columns:

| Column | Description |
|---|---|
| `participant_id` | Participant identifier |
| `timestamp` | Minute-level timestamp |
| `heart_rate` | Heart-rate value |
| `event_time` | Self-reported vaping event timestamp, if available |

The synthetic-data example generates mock data with the same structure but does not use any real participant information.

## Privacy note

No real wearable data, self-report records, participant identifiers, or study-specific file paths are included in this repository.
