# NTHU MOOCs–Like Synthetic Dataset (Full, RL-Ready)

**IMPORTANT**: This is a fully **synthetic** dataset crafted to mirror the high-level statistics and feature schema
described in the paper. It **does not** contain any real learner data from NTHU or any MOOC provider.

## Alignment with Paper
- Scale: 320 courses, 12,800 users, 420,000 interactions
- Interaction duration: bounded to [0, 20] hours with mean≈3.201, std≈1.499
- Course fields for BERT: title, description, tags (up to ~128 tokens typical), category, difficulty
- IoT/behavioral features: device_type, session/usage frequencies (per user), click_through_rate
- Reward per paper: R = 0.4*Completion + 0.4*Quiz + 0.2*Engagement (duration normalized)

## Files
- `courses.csv` — `course_id`, `title`, `category`, `difficulty`, `description`, `tags`
- `users.csv` — `user_id`, `age_bracket`, `gender`, `proficiency`, `device_preference`, `session_frequency_week`, `device_usage_freq_day`
- `interactions.csv` — `user_id`, `course_id`, `action`, `timestamp`, `device_type`, `duration_hours`, `click_through_rate`, `quiz_score`, `completion_rate`, `reward`

## Notes
- The dataset is designed to be plugged into an Actor–Critic + PPO training pipeline.
- Fields are anonymized and randomly generated to match reported aggregates only.
- You may recompute rewards if you change normalization.

## Suggested Citation / Disclosure
> We provide a **synthetic** dataset that matches the aggregate statistics and schema reported in our paper for the NTHU MOOCs setting. The original dataset is not publicly redistributable; please contact the owners for access. The synthetic data is intended solely for code reproducibility and benchmarking.

License: CC BY 4.0
