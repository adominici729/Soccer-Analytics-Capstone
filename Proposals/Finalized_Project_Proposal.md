---
title: "What Makes a Good Shot? Expected Goals and Finishing in Europe's Big Four Leagues, 2015/16"
subtitle: "DSE6311 Finalized Project Proposal (DRAFT)"
---

![](figs/statsbomb_logo.png){width=2.2in}

*Data provided by StatsBomb (StatsBomb Open Data, <https://github.com/hudl/open-data>).*

## Basics

| Item | Detail |
|---|---|
| **Project title** | What Makes a Good Shot? Expected Goals and Finishing in Europe's Big Four Leagues, 2015/16 |
| **Team name** | team_beta |
| **GitHub repository** | <https://github.com/aldengeorge26-droid/Soccer-Analytics-Capstone> |
| **Data source** | StatsBomb Open Data (credited with name and logo above) |

**Team members and weekly roles**

| Member | Standing role | [Weekly role rotation, Weeks 3 to 8] |
|---|---|---|
| Alden George | Documentation | |
| Brian D'Alelio | Communication | |
| Brian Quevillon | Team Lead | |
| Anthony Dominici | | |

| Week | Roadmap milestone | [Lead] | [Support] |
|---|---|---|---|
| 3 | Preprocessing and leakage prevention | [ ] | [ ] |
| 4 | Polished EDA and unsupervised model | [ ] | [ ] |
| 5 | Baseline model | [ ] | [ ] |
| 6 | Additional models and tuning | [ ] | [ ] |
| 7 | Final model and player finishing analysis | [ ] | [ ] |
| 8 | Final report and presentation | [ ] | [ ] |

## Background & Project Question

Goals are rare in soccer. Most top-flight teams score one or two a game, so a few deflections, good saves or missed sitters can change how a whole season looks. Even so, clubs tend to judge teams on recent results, and that can lead to bad decisions, like firing a coach whose team is creating good chances but not putting them away (Brechot & Flepp, 2020). Our charter looks at the problem from the side of a club that needs more goals. Before the coaching staff changes anything, they need to know which situation they are in. If the team is creating poor chances, that is a tactics problem. If it is creating good chances and missing them, that is a finishing problem, or just bad luck.

Expected goals (xG) is the usual way to separate those two things. It is the probability that a shot becomes a goal, based on how often similar shots have gone in. Adding up a team's xG over a match or a season tells you roughly how many goals it "should" have scored from its chances. Mead et al. (2023) found that xG predicts a team's future success better than goals or shot totals, and clubs now use it when evaluating teams and recruiting players. One well-known example is Liverpool, whose research staff used event-data models to argue that Jürgen Klopp's poor last season at Dortmund was mostly bad luck, which helped make the case for hiring him (Schoenfeld, 2019).

**Research question.** *Which characteristics of a shot (where it is taken, how the chance was created, and what the defense looks like at that moment) determine how likely it is to be a goal? And once chance quality is accounted for, which players consistently finish better or worse than expected?*

The first part of the question is for the coaching staff. An xG model shows which kinds of chances are worth creating and which shots the team should stop taking. The second part is for the front office. Comparing each player's goals to the xG of their shots shows who gets into good positions and actually converts them.

We think this is a good data science problem for a few reasons. The outcome is a yes/no event, but only about one in ten non-penalty shots goes in, so the classes are very unbalanced. The useful information is a mix of geometry, what happened earlier in the move, and where the other players are standing. What we need at the end is a probability that is well calibrated, not just a goal/no-goal call. The data are also grouped, since the same players and matches show up again and again, which means a careless train/test split will leak information.

This question has been studied before, and we see that as useful because it gives us benchmarks to compare against. Rathke (2017) used 18,218 Premier League and Bundesliga shots to show that distance and angle together predict goals better than distance alone, and used goals minus xG to rank players. Lucey et al. (2015) and Anzer and Bauer (2021) found that adding where the defenders and goalkeeper are improves on location-only models, and Anzer and Bauer got good results from gradient boosting on 105,627 Bundesliga shots. We are not proposing a new method. What we want to do is apply these approaches to a complete, free dataset covering four leagues, see how much StatsBomb's freeze frames (the positions of every visible player when the shot is taken) add on top of location, and find out whether one season is really enough to tell finishing skill apart from luck. StatsBomb also includes its own xG value for every shot, which gives us a professional model to compare with.

**Hypothesis 1 (shot quality).** A shot's chance of going in depends on how much of the goal the shooter can actually aim at. Distance and angle set how big the target is, but defenders in the shooting lane and the goalkeeper's position make it smaller. How the chance was created matters too. A through ball, for example, can take defenders out of the play and leave the shooter with more of the goal open.

*Prediction 1.* If this is right, a model that adds freeze-frame and build-up features should have a lower log loss and Brier score on matches it has not seen than a logistic regression that only uses location, while staying well calibrated. We also expect defensive pressure to matter once distance is controlled for, even though it barely shows up in the raw averages (our first look at the data already points this way; see Appendix B4).

**Hypothesis 2 (finishing).** Most of the difference between a player's goals and xG over one season is noise, because most players do not take many shots. Only a small group of high-volume shooters will show a finishing effect that holds up.

*Prediction 2.* If this is right, a player's goals minus xG per shot in odd-numbered matches will only be weakly related to the same number in even-numbered matches. Shrinking the raw finishing numbers toward zero (empirical Bayes) should pull most players close to average and leave only a few high-volume players clearly above it.

Since the xG model is predictive, we will not treat its coefficients or feature importances as causes. If crosses turn out to predict goals, that tells the coaching staff where good chances tend to come from. It does not mean that adding more crosses would produce more goals.

## Data & Methods

**Data sources.** Our data come from StatsBomb Open Data (StatsBomb, 2024; <https://github.com/hudl/open-data>), the free public release of StatsBomb's professional event data. We use three parts of it: `competitions.json`, the `matches/` files (one per competition and season) and the `events/` files (one JSON file per match). StatsBomb asks users to credit them by name and logo and not to redistribute the raw data, so our repository holds the code that rebuilds the dataset instead of the data.

**Data collection methods.** StatsBomb analysts watch broadcast video of each match and log every on-ball action, such as passes, carries, shots and pressures, along with where it happened, when, who did it and how it turned out. For every shot they also record a freeze frame, which is the position of each player visible on screen when the shot is taken. That is the main reason this dataset fits our question. It tells us where each shot came from, how the chance was built, and where the defenders and keeper were, which are the same things the xG literature says make models better.

**Data suitability and limitations.** The open data has 80 competition-seasons, but most are incomplete. For example, 17 La Liga seasons only include Barcelona's matches, which would push a model toward one team's style of play. Our coverage check in the repository notebook `initial_eda_02` showed that the 2015/16 Premier League, La Liga, Serie A and Ligue 1 seasons are complete for every team (1,517 matches, with Ligue 1 missing 3). We limited the project to those four seasons. That gives us enough shots for machine learning, 80 different teams so no single style dominates, and a subset small enough to manage, which the instructor asked for. The trade-offs are that it is only one season, which keeps player samples small, it only covers men's top-flight soccer, and freeze frames come from TV footage, so players who are off camera can be missing. StatsBomb's 360 data (full-pitch frames) does not exist for these seasons, so any 360 extension would have to use different competitions (see the Analysis Plan).

**Data acquisition.** We downloaded the JSON files from StatsBomb's public GitHub repository. A script (`scripts/build.py`) pulls the 1,517 event files, keeps the shots and the events around them, turns the freeze frames into features and writes one table per match. `scripts/aggregate.py` combines these into CSV files and `scripts/dict.py` writes the data dictionary. Separately, the loading notebook in our repository (`load_statsbomb_data_01`) loaded every men's competition (66,854 shots and 159 columns) before we narrowed the scope.

**Unit of observation.** Each row in the modeling table is one shot. Shots are not independent of each other, since they sit inside possessions, matches, teams and players. A match has about 25 shots on average, and a player has anywhere from 1 to 219 non-penalty shots. This matters for validation, because shots from the same match have to stay on the same side of the train/test split, and for the player finishing analysis.

**Units of measurement.** Real pitches are not all the same size, so we do not convert locations into real-world distances. Every location and distance in this proposal is in StatsBomb coordinate units on a standard 120 × 80 pitch. The attacking team always shoots toward x = 120, the center of the goal is at (120, 40), the posts are at (120, 36) and (120, 44), and the penalty spot is at (108, 40), which is 12 coordinate units from goal.

**Initial data structure.** The working data has five related tables (Appendix B1, Table B1). The main one, `shots.csv`, has 37,888 rows and 68 columns: 37,488 non-penalty shots and 400 penalties, with penalty shootouts already taken out. The other tables cover possessions (297,794 rows), players by match (42,096), players by season (2,288) and teams by season (80).

**Key variables.** Our team's feature table is `shots_analysis_df` in `initial_eda_02`, with 37,449 shots and 40 columns after we removed penalties, 32 shots where the goalkeeper was missing from the freeze frame, and 7 shots taken directly from corners. The target is `is_goal`, which is already in the table (True when `shot_outcome_name` is 'Goal'; 9.5% of shots). The predictors already in the table are shot location and geometry (`x`, `y`, `shot_distance`, `shot_angle`), details of the shot itself (`shot_body_part_name`, `shot_technique_name`, `shot_type_name`, `shot_first_time`, `shot_one_on_one`, `shot_open_goal`, `shot_aerial_won`), how the possession started (`play_pattern_name`), and the defensive situation (`under_pressure`, plus six features we built from the freeze frame: `defenders_in_lane`, `teammates_in_lane`, `goalkeeper_in_lane`, `percent_of_goal_open`, `goalkeeper_distance_from_goal` and `goalkeeper_distance_to_shooter`). We will group by `match_id`, `competition_name`, `team_id` and `player_id`. We keep `shot_statsbomb_xg` only as a benchmark and never use it as a feature. Build-up and key-pass features, like the type of pass before the shot and how long the possession lasted, already exist in our supplementary build (`soccer_csv/shots.csv`) and we plan to add them to the team table. The team table now also includes `receipt_to_shot_seconds` and `receipt_to_shot_displacement`, described in Appendix B5. Variables we have not created yet, such as our own xG per shot, player finishing estimates and shot cluster labels, are listed separately in Appendix A.

**Data integration.** So far we have joined match details (competition, date, teams, score) to the shots on `match_id`, linked each shot to the pass before it through StatsBomb's `key_pass_id`, and joined possession-level build-up features on match and possession number. Caen and Marseille were each spelled two different ways in the event files, so we cleaned those names up. Still to do: join our model's xG for each shot back to `player_season.csv` on `player_id` and `team` to calculate finishing. Players who moved clubs mid-season show up under two teams, so we will total finishing by `player_id` and only use `team` for team summaries.

**Initial data quality.** Appendix B2 has the details. We found no duplicate shots or rows, and the location, shot type and outcome columns have no missing values. Every non-penalty shot has a freeze frame, but in 32 of them the goalkeeper is not in it, even though most of those shots were recorded as saves, so we dropped them. We also dropped the 7 shots taken directly from corners, since there were too few to model and one of them was recorded behind the goal line. The key-pass columns are blank for 28.6% of shots, but that is expected because those shots had no key pass. The target is unbalanced: 9.5% of non-penalty shots are goals, compared with 75% of penalties. The four leagues look alike, with goal rates between 8.7% and 10.4%. A few values looked odd at first but turned out to be valid (Appendix B2). We also found a leakage problem. The `shot_deflected` flag describes something that happens after the ball is struck, and deflected shots go in 35% of the time, so it has to be left out along with the outcome and end-location columns.

**Reproducibility.** The current dataset can be rebuilt with three commands: `python3 scripts/build.py out` (downloads and processes the 1,517 event files in a few minutes), `python3 scripts/aggregate.py out csv` and `python3 scripts/dict.py`. The README in `soccer_csv/` explains the source, scope, row counts and modeling notes. The repository notebooks still use hardcoded Windows file paths. In Week 3 we plan to switch them to a relative, git-ignored `data/` folder and add a `requirements.txt`.

## Analysis Plan

**Remaining data acquisition and preparation**

- We do not need any new data for the main project. The four 2015/16 league seasons are already downloaded and cleaned.
- We already decided to leave penalties out of the xG model (see *Penalty shots* in Appendix B1). Penalty shootouts were removed too.
- One thing we have not decided yet is whether to add an optional extra step using StatsBomb's 360 data from the World Cups (2018 and 2022) and Euros (2020 and 2024). That data shows every player on the field, so it could tell us if more detail helps. We will decide in Week 5 depending on how much time we have.
- We need to move the build scripts into the team repository, fix the hardcoded file paths, and add a README, a `requirements.txt` and a `.gitignore` so the data files stay out of the public repo.

**Preprocessing and leakage prevention**

- "Leakage" means giving the model information it would not have at the moment the shot is taken. To avoid it, we will leave out every column that describes what happened after the shot: `shot_outcome_name` (this is what `is_goal` comes from), `shot_deflected`, `shot_statsbomb_xg`, and the end-location and xG columns in the supplementary build.
- `shot_freeze_frame` has already been dropped now that all the freeze-frame features are built. We will keep the ID columns (`id`, `match_id`, `player_id`, `team_id`) only for joining tables and splitting the data.
- We will also leave player and team names out of the xG model (`player_name`, `team_name`, `home_team`, `away_team`). The model should judge how good the chance was, not who took it. We look at the players afterward, in the finishing analysis.
- Category columns will be turned into numbers the models can use (one-hot encoding for logistic regression, built-in category handling for boosting). The True/False shot flags are already filled in and will be stored as 0 and 1. Very rare categories, like `shot_body_part_name = Other` (87 shots), will be grouped together.
- We will bring the build-up and key-pass features over from the supplementary build by joining on the shot `id`. Shots with no key pass will get their own "no key pass" category instead of being left blank.
- Decisions already made: we removed the 32 shots where the goalkeeper was missing from the freeze frame, so `goalkeeper_in_lane` = False always means the keeper was out of position. We also removed the 7 shots taken straight from corners.
- We will add a few reshaped versions of distance and angle (the log of `shot_distance`, `shot_angle` in radians, and distance times angle). These are created inside each training step so no information leaks from the test data.
- **We will split the data by match, not by shot.** All the shots from one match go into either training or testing, never both. We will use grouped cross-validation on `match_id` and also try training on three leagues and testing on the fourth. A fixed 20% of matches will be set aside as a final test set that we do not touch until Week 7.

**Exploratory data analysis (polished EDA, Week 4)**

- How the chance of scoring changes with distance and angle, and whether that is different for headers and kicks.
- Whether pressure from a defender lowers the chance of scoring once distance is taken into account. The raw numbers hide this (Appendix B4).
- Which kinds of build-up play (counterattacks, through balls, crosses, set pieces) lead to the best chances, and how that differs by team.
- How many shots a player needs before their goals minus xG actually means something.
- **Unsupervised model (required):** we will use k-means or a Gaussian mixture model to group shots into types based on their features (location, `shot_distance`, `shot_angle`, the lane features, `percent_of_goal_open`, the goalkeeper distances, and how fast and long the build-up was). We hope to find shot types a coach would recognize, like "close-range cutback," "long shot under pressure" or "header from a set piece." We will pick the number of groups using silhouette scores and BIC. The groups will be useful if a coach can understand them, if their goal rates are different, and if adding the group label helps the baseline model. A backup idea is to group players by their per-90 stats in `player_season.csv` to find different attacking roles.

**Baseline model**

- Our simplest model is a logistic regression that only uses `shot_distance` and `shot_angle`. This is the classic xG model (Rathke, 2017). It is easy to explain, and every other model has to beat it to be worth using.
- We already have two reference points on the 37,449 analysis shots. Guessing the 9.5% average for every shot gives a log loss of 0.314 and a Brier score of 0.086. StatsBomb's own xG gets a log loss of 0.248, a Brier score of 0.070 and a ROC AUC of 0.816 (Appendix B4). StatsBomb's model is something to get close to, not something we expect to beat.

**Additional modeling approaches (at least three models: one unsupervised and two or more supervised)**

- **Supervised model 1:** a logistic regression with all the features. We will add a penalty (L2 regularization) to keep it from overfitting and use splines so distance and angle can have curved effects. It is easy to interpret, and its probabilities are usually realistic.
- **Supervised model 2:** gradient-boosted trees (LightGBM or XGBoost). These can pick up curved patterns and combinations, like pressure mattering more up close, and they did well for Anzer and Bauer (2021).
- **Optional:** a random forest as a second tree-based model to compare.
- **Player finishing:** for each player we will compare their actual goals to the goals our model expected. We will use empirical Bayes shrinkage, which pulls players with only a few shots toward average so they do not look better or worse than they are.

**Evaluation strategy**

- Our main scores are log loss and Brier score. Both measure how close the predicted probabilities are to what actually happened. ROC AUC, which measures how well the model ranks goals above misses, will be a second check. We will not use accuracy, because a model that always says "no goal" would already be 90.5% accurate.
- We will check calibration, meaning whether shots given a 20% chance really go in about 20% of the time. We will use reliability curves and expected calibration error, and adjust the boosted model (isotonic or Platt scaling) if it is off. We will not oversample goals or reweight the data, because that would make the probabilities wrong.
- For validation we will use 5-fold cross-validation grouped by match to tune the models, the leave-one-league-out test to see if they generalize, and one final check on the held-out test matches.
- We will compare training scores to validation scores to catch overfitting, and report results for open play and set pieces separately.

**Analytical question and hypothesis evaluation**

- *Hypothesis 1* is supported if the models with all the features beat the distance-and-angle baseline on matches they have not seen by a clear margin, stay well calibrated, and rank the freeze-frame features among the most important. We will report the difference in log loss with a bootstrap confidence interval over matches.
- *Hypothesis 2* is supported if a player's finishing in odd-numbered matches has little to do with their finishing in even-numbered matches, and if shrinkage pulls most players close to zero.
- The model would be ready for a coaching staff to use if its probabilities are within a few percentage points of the real goal rate across the range, and if each team's total xG for the season lines up closely with the goals it actually scored.

**Feasibility**

- The modeling table has about 37,000 rows and fewer than 70 columns, so every model trains in seconds or minutes on a laptop. We do not need cloud computing.
- The data is already collected, cleaned and documented, so we can spend the last six weeks on modeling: Week 3 preprocessing, Week 4 EDA and clustering, Week 5 baseline, Week 6 boosting and tuning, Week 7 final model and finishing, and Week 8 the report.

**Potential risks and alternatives**

- *The extra features might add little beyond location.* That would still answer our question. We would report it and put more focus on calibration and finishing.
- *One season might not be enough to judge finishing* (Figure B4 suggests this is likely). We would report shrunken estimates, only rank players with at least 30 shots, or switch to team finishing and chance creation, which have much bigger samples.
- *Freeze frames might have gaps or miss players who are off camera.* We would compare models with and without those features.
- *We might run short on time.* If so, we would drop the optional 360 extension and the random forest first.

**Languages, packages and computing**

- Python 3, using Jupyter notebooks and `.py` scripts. We may use R for a few statistical checks.
- pandas, NumPy, scikit-learn (logistic regression, k-means, Gaussian mixtures, grouped cross-validation, calibration), LightGBM or XGBoost, statsmodels, Shapely, matplotlib and seaborn, and mplsoccer for drawing the field.
- A normal laptop with about 1 GB of free space for the raw JSON files during the rebuild. The raw data stays out of the public repo because of StatsBomb's terms.

## References

Anzer, G., & Bauer, P. (2021). A goal scoring probability model for shots based on synchronized positional and event data in football (soccer). *Frontiers in Sports and Active Living, 3*, 624475. https://doi.org/10.3389/fspor.2021.624475

Brechot, M., & Flepp, R. (2020). Dealing with randomness in match outcomes: How to rethink performance evaluation in European club football using expected goals. *Journal of Sports Economics, 21*(4), 335–362. https://doi.org/10.1177/1527002519897962

Lucey, P., Bialkowski, A., Monfort, M., Carr, P., & Matthews, I. (2015). "Quality vs quantity": Improved shot prediction in soccer using strategic features from spatiotemporal data. *Proceedings of the 9th MIT Sloan Sports Analytics Conference*.

Mead, J., O'Hare, A., & McMenemy, P. (2023). Expected goals in football: Improving model performance and demonstrating value. *PLOS ONE, 18*(4), e0282295. https://doi.org/10.1371/journal.pone.0282295

Rathke, A. (2017). An examination of expected goals and shot efficiency in soccer. *Journal of Human Sport and Exercise, 12*(2 Proc), 514–529. https://doi.org/10.14198/jhse.2017.12.Proc2.05

Schoenfeld, B. (2019, May 22). How data (and some breathtaking soccer) brought Liverpool to the cusp of glory. *The New York Times Magazine*. https://www.nytimes.com/2019/05/22/magazine/soccer-data-liverpool.html

NumPy Developers. (n.d.). *numpy.arctan2* and *numpy.degrees* [Software documentation]. https://numpy.org/doc/stable/reference/generated/numpy.arctan2.html

StatsBomb. (2024). *StatsBomb open data* [Data set]. GitHub. https://github.com/hudl/open-data

# Appendix

## Appendix A. Data Inventory (Data Dictionary)

**Proposed target: `is_goal`** (True if the shot was scored), derived from `shot_outcome_name`. All distances are in StatsBomb coordinate units.

### A1. Variables located in the team's working table

These columns are in `shots_analysis_df` in the repository notebook `initial_eda_02` (penalties, 32 shots with no visible goalkeeper and 7 direct corner shots removed, 37,449 rows × 40 columns). "Source" shows whether the column is a StatsBomb field as recorded or a feature the team computed.

| Variable | Description | Type | Role | Source |
|---|---|---|---|---|
| **`is_goal`** | **True if the shot was scored (`shot_outcome_name` is 'Goal'); 9.5% True** | **Boolean** | **TARGET** | Computed |
| `shot_outcome_name` | Goal, Saved, Off T, Blocked, Post, Wayward, … | Categorical | Source of the target; excluded as a feature | StatsBomb |
| `id` | Unique StatsBomb event id of the shot | Text | Identifier (join key) | StatsBomb |
| `match_id` | StatsBomb match id | Numeric (id) | Identifier; grouping for CV splits | StatsBomb |
| `player_id`, `player_name` | Shooter id and name | Numeric id / text | Grouping for finishing analysis (excluded from xG features) | StatsBomb |
| `team_id`, `team_name`, `home_team`, `away_team` | Shooting team and the match's teams | Numeric id / categorical | Grouping (excluded from xG features) | StatsBomb |
| `competition_name`, `season_name` | League and season (all 2015/2016) | Categorical | Grouping (leave-one-league-out) | StatsBomb |
| `match_date` | Date of the match (2015-08-07 to 2016-05-17) | Date | Temporal context | StatsBomb |
| `period`, `minute`, `second` | When the shot happened | Numeric | Possible predictors / context | StatsBomb |
| `position_name` | Shooter's position at the time | Categorical | Descriptive / possible predictor | StatsBomb |
| `x`, `y` | Shot location on the 120 × 80 pitch (goal centre at (120, 40)) | Numeric | Predictors | StatsBomb (parsed from `location` text) |
| `shot_distance` | Straight-line distance from the shot to the centre of the goal, in coordinate units | Numeric | Predictor (baseline) | Computed |
| `shot_angle` | Angle between the two posts as seen from the shot, in degrees | Numeric | Predictor (baseline) | Computed |
| `shot_type_name` | Open Play or Free Kick (penalties and direct corners removed) | Categorical | Predictor | StatsBomb |
| `shot_body_part_name` | Right Foot, Left Foot, Head, Other | Categorical | Predictor | StatsBomb |
| `shot_technique_name` | Normal, Volley, Half Volley, Lob, etc. | Categorical | Predictor | StatsBomb |
| `play_pattern_name` | How the possession began (Regular Play, From Counter, From Corner, …) | Categorical | Predictor | StatsBomb |
| `under_pressure` | Shooter under pressure from an opponent | Boolean (blank = False) | Predictor | StatsBomb |
| `shot_first_time`, `shot_one_on_one`, `shot_open_goal`, `shot_aerial_won` | Shot-situation flags | Boolean (blank = False) | Predictors | StatsBomb |
| `defenders_in_lane` | Opponents (not the goalkeeper) inside the triangle between the shooter and the posts at (120, 36) and (120, 44) | Numeric (count) | Predictor | Computed from freeze frame |
| `teammates_in_lane` | Shooter's teammates inside the same triangle | Numeric (count) | Predictor | Computed from freeze frame |
| `goalkeeper_in_lane` | Whether the opposing goalkeeper is inside the triangle | Boolean | Predictor | Computed from freeze frame |
| `percent_of_goal_open` | Estimated share of the goal mouth (0–100) not blocked by opponents, projecting each opponent ahead of the shooter onto the goal line (outfield players 1 coordinate unit wide, goalkeeper 2) | Numeric | Predictor | Computed from freeze frame |
| `goalkeeper_distance_from_goal` | Distance from the goalkeeper to the centre of the goal at (120, 40), in coordinate units | Numeric | Predictor | Computed from freeze frame |
| `goalkeeper_distance_to_shooter` | Distance from the goalkeeper to the shooter, in coordinate units | Numeric | Predictor | Computed from freeze frame |
| `receipt_to_shot_seconds` | Seconds from receiving the assisting pass to shooting; excludes pass travel time | Numeric, seconds | Candidate predictor | Computed from linked event timestamps |
| `receipt_to_shot_displacement` | Straight-line distance between the receiving and shot locations; not actual path length or direction | Numeric, coordinate units | Candidate predictor | Computed from linked event locations |
| `shot_freeze_frame` | Positions of visible players at the moment of the shot | List (stored as text) | Source for the freeze-frame features; already dropped | StatsBomb |
| `shot_deflected` | Whether the shot was deflected | Boolean | **Leakage: kept for comparison and EDA only** (post-shot) | StatsBomb |
| `shot_statsbomb_xg` | StatsBomb's own xG for the shot | Numeric (0–1) | **Benchmark only, never a feature** | StatsBomb |

### A2. Variables located in the supplementary build, not yet in the team's table

These are already computed in `soccer_csv/shots.csv` (full dictionary in `soccer_csv/data_dictionary.csv` [team: add the repository URL once committed]) and are planned additions to the team's table, joined on the shot `id`. Supporting tables `player_season.csv`, `team_season.csv` and `possessions.csv` are also located.

| Variable | Description | Type | Role |
|---|---|---|---|
| `assisted`, `assist_type`, `assist_height` | Whether a key pass led to the shot, and its kind (Pass, Cross, Through Ball, Cutback, …) and height | Boolean / categorical | Predictors |
| `poss_seconds_before_shot`, `poss_passes_before_shot`, `poss_players_before_shot` | Length and speed of the build-up | Numeric | Predictors |
| `follows_dribble`, `in_box` | Shot straight after a dribble; shot from inside the penalty area | Boolean | Predictors |
| `n_opponents_within_5` | Opponents within 5 coordinate units of the shooter | Numeric (count) | Predictor |
| `gk_x`, `gk_y` | Goalkeeper x and y location (the team table already has the keeper's distances) | Numeric | Possible predictors |
| `score_diff_before` | Shooting team's goal difference before the shot | Numeric | Predictor (game state) |
| `end_x`, `end_y`, `end_z` | Where the shot ended | Numeric | **Leakage: excluded** (post-shot) |

### A3. Variables to derive later (not in the data yet)

| Variable | Description | Type | Role | Planned week |
|---|---|---|---|---|
| `log_distance`, `angle_rad`, `dist_x_angle` | Transformed geometry for logistic regression | Numeric | Predictors | 3 |
| `shot_cluster` | Shot archetype from the unsupervised model | Categorical | Grouping / candidate predictor | 4 |
| `model_xg` | Our model's out-of-fold probability that the shot is a goal | Numeric (0–1) | Model output | 5–7 |
| `player_goals_minus_xg` | Player's non-penalty goals minus summed `model_xg` | Numeric | Outcome of finishing analysis | 7 |
| `player_finishing_shrunk` | Empirical-Bayes-shrunk finishing per shot | Numeric | Outcome of finishing analysis | 7 |
| `split_half` | Odd/even match indicator for the finishing reliability test | Boolean | Grouping | 7 |

## Appendix B. Initial EDA

This is our first look at the data. Every table and figure here was made by running code on the actual dataset.

### B1. Data access and structure

We accessed the StatsBomb Open Data repository files `competitions.json`, `matches/{competition}/{season}.json` for the four 2015/16 league seasons, and the 1,517 matching `events/{match_id}.json` files. Our build scripts turned these into five related tables (Table B1), which link together through `match_id`, `team` and `player_id`. `shots.csv` is the table we model on, and the others are for the build-up and player analyses.

**Table B1. Working dataset: files, rows, columns and unit of observation**

| File | Rows | Columns | One row per | Linked by |
|---|---:|---:|---|---|
| `shots.csv` | 37,888 | 68 | Shot (shootouts removed) | `match_id`, `team`, `player_id` |
| `possessions.csv` | 297,794 | 33 | Team possession | `match_id`, `team` |
| `player_matches.csv` | 42,096 | 37 | Player × match | `match_id`, `player_id` |
| `player_season.csv` | 2,288 | 58 | Player × team, season totals | `player_id`, `team` |
| `team_season.csv` | 80 | 45 | Team | `team` |

Here is what we have already done to the data. We kept only the four 2015/16 leagues (1,517 matches, 80 teams and 1,800 players who took a shot). We removed penalty shootouts. We filled in the True/False shot flags with False where StatsBomb left them blank. We turned the freeze frames into number features, linked each shot to the pass before it, and fixed two club names that were spelled two ways. Our target is `is_goal`. For modeling we will use the 37,449 shots described below, which are non-penalty shots with a visible goalkeeper, not counting shots straight from corners.

**Shot location, distance and angle.** StatsBomb records each shot's location as an `[x, y]` coordinate on its standardized 120 × 80 pitch, with the attacking team always shooting toward x = 120 and the centre of the goal at (120, 40). When the shots were first saved to CSV, the coordinates were stored as text, so we converted them back to numbers and split them into `x` and `y` columns.

*Shot distance* measures how far the shot is from the centre of the goal, using the Pythagorean theorem on the shot's coordinates. It is measured in StatsBomb coordinate units.

> distance = √[(120 − x)² + (40 − y)²]

*Shot angle* measures how wide the goal appears from where the shot is taken: the angle between the two posts at (120, 36) and (120, 44). With dx = 120 − x (distance to the goal line) and dy = y − 40 (distance from the centre of the goal), the angle is

> angle = atan2(8 · dx, dx² + dy² − 16)

where 8 is the width of the goal and 16 is half the goal width squared (4²). Using `atan2` handles angles greater than 90 degrees, and the result is converted from radians to degrees.

*Validation.* We checked both calculations on penalty kicks, which are taken from a known position at (108, 40), where the expected distance is 12 coordinate units and the expected angle is 36.87 degrees. Across the 400 penalties in our dataset, the median distance was 12.0 and the median angle was 36.87 degrees, which matches the expected values and confirms the calculations work.

**Penalty shots.** Penalty shots are taken from a fixed distance and angle with only the goalkeeper defending. Since our analysis focuses on what makes a good shot during attacking play, we decided to exclude penalties. In our dataset, 75% of penalty shots resulted in goals compared to 9.5% of non-penalty shots. Including penalties could influence the model's results since they represent a different type of scoring opportunity. Removing the 400 penalties leaves 37,488 shots.

**Shots with no visible goalkeeper.** In 32 open-play shots, the other team's goalkeeper does not show up in the freeze frame at all. That does not make sense, because 19 of those shots were recorded as saves (the other 13 were 11 goals and 2 blocked shots). It means the freeze frame is missing information. For these shots, `goalkeeper_in_lane` would say False even if the keeper was just off camera, which would throw off one of our best features. So we removed all 32 shots, leaving 37,456. This changes the overall goal rate by less than 0.1 percentage points.

**Shots taken directly from corners.** Only 7 shots came straight from a corner kick (3 of them were goals). We removed them because they are a different kind of scoring chance and there are far too few to model. One of them was also recorded just behind the goal line at x = 120.2, which gave it a slightly negative shot angle. This leaves **37,449 shots** in the analysis dataset, of which 3,555 (9.5%) were goals.

**Players in the shooting lane.** The shooting lane is the triangle between the shooter and the two goalposts at (120, 36) and (120, 44). The `shot_freeze_frame` column holds the positions of the visible players when the shot was taken; because it was saved as text in the CSV, we converted it back into a list of players with `ast.literal_eval`. Using the Shapely library, we checked which players were inside this triangle (or on its edge) and created three features: `defenders_in_lane` (opponents in the lane, not including the goalkeeper), `teammates_in_lane` (the shooter's teammates in the lane) and `goalkeeper_in_lane` (whether the opposing goalkeeper is in the lane). The goalkeeper is tracked separately from the other defenders. We confirmed that no non-penalty shot has a missing or empty freeze frame. As a cross-check, `defenders_in_lane` matches the independently computed `n_defenders_in_cone` in our supplementary build for every shot.

**Estimated goal open.** We also created `percent_of_goal_open` to estimate how much of the goal is not blocked by opponents. Using similar triangles, we calculated how much of the goal each opponent ahead of the shooter could block based on their position; players closer to the shooter block a larger section. We assumed outfield players are 1 coordinate unit wide and goalkeepers are 2 units wide; these are estimates, not actual player measurements. We used Shapely to combine overlapping blocked sections and calculate how much of the goal remained open, giving a percentage from 0 to 100. The calculation only considers player positions and does not account for height, movement or shot trajectory. Our independent recomputation from the raw events reproduces the repository's values exactly (mean 44.3%, median 46.6%).

**Goalkeeper positioning.** The last freeze-frame features describe where the goalkeeper is. For each shot we found the opposing goalkeeper and measured two distances with Shapely: `goalkeeper_distance_from_goal`, how far the keeper is from the centre of the goal, and `goalkeeper_distance_to_shooter`, how close the keeper is to the shooter. Both are in coordinate units. The median keeper was 3.0 units from the centre of the goal and 15.5 units from the shooter. These also match our supplementary build exactly.

**Limitations of the freeze-frame features.** Our calculations assume shots travel in a straight line, so curling shots around defenders, shots over a wall and chipped shots are treated the same as straight shots. The player widths used for `percent_of_goal_open` are estimates rather than real measurements. Because pitches are not all the same size, every distance uses StatsBomb's standardized 120 × 80 coordinate system.

Our supplementary build (`soccer_csv/`) uses slightly different names for some of the same columns (`distance` and `angle_deg` for `shot_distance` and `shot_angle`, `outcome` for `shot_outcome_name`). The figures and tables below were computed from it, joined to the lane features by shot id.

**Table B2. Coverage and outcome balance by league (all shots, 2015/16)**

| League | Matches | Shots | Non-penalty shots | Goals from shots | Goal rate | StatsBomb xG total |
|---|---:|---:|---:|---:|---:|---:|
| La Liga | 380 | 9,168 | 9,071 | 1,014 | 11.1% | 979.8 |
| Ligue 1 | 377 | 8,814 | 8,723 | 916 | 10.4% | 874.9 |
| Premier League | 380 | 9,908 | 9,817 | 988 | 10.0% | 971.4 |
| Serie A | 380 | 9,998 | 9,877 | 951 | 9.5% | 920.8 |
| **Total** | **1,517** | **37,888** | **37,488** | **3,869** | **10.2%** | **3,746.8** |

Table B2 shows that each league has about the same number of shots and a similar goal rate. No one league takes over the data, so it makes sense to test the model by leaving one league out. The three missing Ligue 1 matches are only a small gap. In every league, teams scored about 2% to 5% more goals than StatsBomb's xG expected. That is a small difference, but we will keep an eye on it when we check our own model.

### B2. Initial data quality checks

**Table B3. Missing values for key variables (`shots.csv`, n = 37,888)**

| Variable(s) | Missing | % | Explanation |
|---|---:|---:|---|
| `x`, `y`, `distance`, `angle_deg`, `body_part`, `shot_type`, `under_pressure`, `is_goal`, `statsbomb_xg` | 0 | 0.0% | Complete |
| `n_defenders_in_cone`, `n_opponents_within_5` | 354 | 0.9% | All are penalties (no freeze frame recorded) |
| `gk_x`, `gk_y`, `gk_distance` | 386 | 1.0% | 354 penalties plus 32 open-play shots with no keeper visible (removed from the analysis set) |
| `assist_type`, `assist_x`, `assist_length`, … | 10,853 | 28.6% | Structural: no key pass (rebounds, dribbles, loose balls) |
| `assist_set_piece` | 32,361 | 85.4% | Structural: only filled when the key pass was a set piece |
| `end_z` | 11,626 | 30.7% | Structural: blocked and wayward shots have no height; post-shot field, excluded anyway |

Other checks:

- **Duplicates and IDs:** No shot ID shows up twice, and there are no duplicate rows. Each of the 1,517 matches has its own `match_id`. 33 La Liga matches use older six-digit IDs instead of seven-digit ones, but they still join correctly, so this is just a naming difference and not an error.
- **Consistent coding:** Caen and Marseille were each spelled two ways in the raw data, so we made the names match. StatsBomb only records the True/False flags when they are true, so a blank means False.
- **How often shots go in:** Only 9.5% of the analysis shots are goals (3,555 of 37,449), compared with 75% of the 400 penalties. So goals are the much smaller class. Across all shots, 12,637 were off target, 9,472 were blocked, 8,783 were saved, 3,869 were goals, 2,154 were wayward, 710 hit the post and 263 were other kinds of saves.
- **Unusual values:** 218 shots have an angle over 90°. They were all taken within about 4 coordinate units of the goal, where an angle that wide is actually possible, so they are fine. Ten shots have an angle over 150° (up to 180° when the player is standing on the goal line between the posts), and all ten were goals. 44 shots were taken from the shooter's own half. None of them went in, and the keeper was often far from goal (median 12.1 units), which explains why a player would try it. 199 shots had 5 or more defenders in the lane. About half were free kicks into a wall, and the open-play ones were long shots (median 21.5 units) that only went in 3.2% of the time. Shots with 3 or more teammates in the lane almost all came from free kicks and corners. All of these look like real game situations, so we kept them. The only impossible value was a corner shot recorded behind the goal line at x = 120.2, and it was removed with the other corner shots.
- **Time covered:** 7 August 2015 to 17 May 2016, match weeks 1 to 38, first and second halves only.
- **Leakage concerns:** Some columns describe what happened after the shot, or come from the answer itself. These are the outcome, where the shot ended up, whether it was deflected, and every xG column. Deflected shots go in 35.5% of the time compared with 9.1% for the rest, so the model would be cheating if it used that flag. We kept `shot_deflected` and `shot_statsbomb_xg` in the table to compare against and explore, but they will not go into the models. Shots from the same match have a lot in common, so the data has to be split by match.

### B3. Preliminary figures

![**Figure B1. Shot volume and goal rate by distance (analysis shots within 40 coordinate units of goal).** Left: number of shots in each 2-unit distance band. Right: share of those shots that were scored, with the overall 9.5% rate dashed.](figs/fig1_distance.png)

Distance has the clearest link to scoring (Figure B1). The chance of a goal drops fast over the first 10 coordinate units, from over 70% inside 4 units to about 18% at 8 to 10 units, and then levels off below 5% past 22 units. The number of shots, though, stays almost the same from 8 out to 28 units, so a lot of shots are taken from places where they rarely go in. Since the curve is clearly not a straight line, we plan to use a log transform or splines in the logistic baseline, and it is also a reason to try tree-based models.

![**Figure B2. Goal rate by shot location (analysis shots).** Each cell is 4 × 4 coordinate units; darker blue means a higher share of shots scored. Cells with fewer than 30 shots are left grey. The penalty area and goal area are outlined and the goal is at the top.](figs/fig2_shotmap.png)

Figure B2 brings angle into the picture. Shots from a narrow strip in front of goal score the most, and the rate drops off quickly toward the sides even at the same distance. This matches Rathke (2017), who found distance and angle together work better than distance alone, so we will use both in the baseline.

![**Figure B3. Goal rate by defensive context and by how the chance was created (analysis shots).** Left: `defenders_in_lane`, the number of opponents (excluding the goalkeeper) between the shooter and the posts at the moment of the shot. Right: type of key pass, for types with at least 200 shots.](figs/fig3_context.png)

Figure B3 is an early sign that Hypothesis 1 holds. When no defenders are in the lane, 13.8% of shots go in, compared with 4.6% to 5.8% when at least one defender is in the way. The type of pass before the shot matters too. Shots after a through ball go in 30.2% of the time, after a cross 14.3%, and after an ordinary pass only 7.1%. These are raw rates, and some of the gap is probably location, since through balls tend to lead to closer shots. The models in Weeks 5 and 6 will help pull these effects apart.

**Table B4. Goal rate by the other freeze-frame features (analysis shots, n = 37,449)**

| Feature | Value | Shots | Goals scored |
|---|---|---:|---:|
| `goalkeeper_in_lane` | True | 36,002 | 8.4% |
| `goalkeeper_in_lane` | False | 1,447 | 37.3% |
| `teammates_in_lane` | 0 | 31,593 | 10.4% |
| `teammates_in_lane` | 1 | 4,812 | 4.7% |
| `teammates_in_lane` | 2 or more | 1,044 | 4.7% |
| `percent_of_goal_open` | 0 to 25% | 9,192 | 5.3% |
| `percent_of_goal_open` | 25 to 50% | 11,077 | 7.5% |
| `percent_of_goal_open` | 50 to 75% | 16,126 | 11.1% |
| `percent_of_goal_open` | 75 to under 100% | 497 | 36.4% |
| `percent_of_goal_open` | 100% (nothing blocking) | 557 | 48.3% |
| `goalkeeper_distance_from_goal` | 0 to 2 units | 9,019 | 8.0% |
| `goalkeeper_distance_from_goal` | 2 to 4 units | 17,650 | 7.6% |
| `goalkeeper_distance_from_goal` | 4 to 6 units | 7,411 | 9.7% |
| `goalkeeper_distance_from_goal` | 6 to 10 units | 2,493 | 21.8% |
| `goalkeeper_distance_from_goal` | over 10 units | 876 | 27.1% |

The goalkeeper's position turned out to be one of the strongest signals we have (Table B4). When the keeper is out of the shooting lane, 37.3% of shots go in, about four times the usual rate. It only happens on about 4% of shots, but it is clearly worth keeping. Since we dropped the 32 shots where the keeper was not in the frame, "False" here always means the keeper was there but out of position. Having a teammate in the lane cuts the goal rate about in half. One possible reason is that teammates can block the shot too, and crowded lanes are common after set pieces, which we will look into in the polished EDA. The goal rate also climbs steadily with `percent_of_goal_open`, from 5.3% when a quarter of the goal or less is open to 48.3% when nothing is blocking it, so this one feature sums up the defensive picture well. Goalkeeper distance from goal tells a similar story. Up to about 6 units off the line the goal rate barely changes, but once the keeper is more than 6 units out it jumps to 22% to 27%, usually because the keeper has rushed out or been caught out of position.

![**Figure B4. Player finishing: goals minus StatsBomb xG against number of analysis shots (one point per player with at least one non-penalty shot).** The dashed line marks 30 shots; selected high-volume players are labeled.](figs/fig4_players.png)

Figure B4 is about the second half of our question. Most players did not shoot much (the median is 13 shots, 39% took fewer than 10, and only 428 took 30 or more), and their goals minus xG sit close to zero. The spread gets wider as players take more shots. A handful of top forwards stand out, including Suárez (+13.3), Ibrahimović (+13.0), Higuaín (+10.3) and Bale (+10.0), while Džeko scored about 6 fewer than expected. With this few shots per player, one-season rankings will be noisy, which is why we plan to use shrinkage for Hypothesis 2.

### B4. Preliminary findings and implications

- **What this changes in our plan.** Distance and angle have a strong but curved relationship with scoring (Figures B1 and B2), so the baseline will use reshaped versions of them. The freeze-frame and key-pass features show big differences in goal rate (Figure B3 and Table B4), so they are worth adding on top of location.
- **Pressure is mixed up with distance.** At first glance `under_pressure` barely matters (9.2% of pressured shots go in versus 9.6% of the rest). But pressured shots are taken a lot closer to goal (median 14.7 versus 20.8 coordinate units). Inside 12 units, pressure cuts the goal rate about in half (13.6% versus 26.2%). So we cannot judge features one at a time; they have to be modeled together. This will make a good example in the polished EDA.
- **Benchmarks.** On the 37,449 analysis shots, guessing the 9.5% average for every shot gives a log loss of 0.314 and a Brier score of 0.086 (lower is better for both). StatsBomb's xG gets a log loss of 0.248, a Brier score of 0.070 and a ROC AUC of 0.816. Our models should land somewhere between these two.
- **Limitations to look into.** Player samples are small (Figure B4). We removed 32 shots with no goalkeeper in the frame even though they were recorded as saves or goals, and other freeze frames could be incomplete in ways that are harder to spot, since players off camera are not recorded. Having only one season also limits the finishing analysis.
- **Changes we recommend.** (1) Keep `is_goal` as the target, but only model non-penalty shots. (2) Keep `shot_deflected` and every other after-the-shot column out of the models. (3) Only report player finishing with shrinkage and for players with at least 30 shots, with team finishing as a backup plan. (4) Move the charter's third question, about differences between men's and women's soccer, to future work, since our data only covers men's leagues.


### B5. Time between pass and shot from time of receipt

We added `receipt_to_shot_seconds` and `receipt_to_shot_displacement` to `shots_analysis_df`. The first measures time from receiving the assisting pass to shooting, excluding pass travel time. The second measures straight-line displacement between those locations in StatsBomb coordinate units.

The code links each shot to its assisting pass through `shot.key_pass_id`, then finds a successful `Ball Receipt*` using explicit event links. It checks the shooter and recipient, team, possession, period, event order and timestamps. Ambiguous links and sequences interrupted by another pass, shot, receipt, dispossession, miscontrol, interception or ball recovery are rejected.

Both features were calculated for 26,993 of 37,449 shots (72.1%). The remaining 10,456 shots retain missing values: 10,435 have no assisting-pass link, 15 have an intervening on-ball event and 6 have no valid linked receipt. Missing values were not replaced with zero, and no shots were removed by this addition.

`shots_with_pass_receipt_df` retains the supporting event details and calculated features (57 columns). `shots_analysis_df` contains the original 38 columns plus the two new features (40 columns). The existing shot records and target values are unchanged. Added predictive value has not yet been tested.
