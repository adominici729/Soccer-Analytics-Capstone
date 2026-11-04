---
output:
  html_document: default
  pdf_document: default
  word_document: default
---

# Capstone Project Charter

The purpose of this charter is to establish a shared understanding of your team's project before you begin. It should identify the problem you are addressing, why it matters, who might benefit from the results, and what data you will use.

__This is not a research proposal or a full analysis plan.__ You will develop your project background/rationale and full analysis plan during Sprint 2 of the Capstone.

Keep your responses concise. Aim for approximately 1–2 pages.

## Project Information

__Project title:__ What Makes a Good Shot

__Team name:__ team_beta

__Team members:__ Alden George, Brian D'Alelio, Brian Quevillon, Anthony Dominici

__Project source:__

- [x] Capstone Curated Project Bank

- [ ] Alternative project approved by instructor

### Problem and Context

__What problem are you trying to address?__

Describe the problem in a few sentences. What is happening that an organization, business, agency, or community would want to understand, predict, or improve?

*Example:* A subscription-based company is experiencing customer attrition. When customers cancel their subscriptions, the company loses recurring revenue and may incur additional costs to acquire replacement customers.

__Your response:__

A soccer club is looking to improve its offense by figuring out what makes some shots better than others. This includes looking at shot location and other factors that affect scoring chances. The club also wants to find players who regularly get into good scoring positions and put the ball in the net.

### Why Does This Problem Matter?

Briefly explain why the problem is important in its organizational, business, healthcare, public sector, or other applied context. You do not need to provide a literature review or fully support your rationale with sources yet; you will develop the background and rationale during Sprint 2.

__Consider__:
- What consequences might result from this problem?
- Why might an organization or stakeholder want to address it?
- What decisions could a better understanding of the problem inform?

*Example:* Understanding customer attrition could help a company evaluate its retention strategies and determine where to focus customer support or engagement resources.

__Your response:__

A soccer club is struggling to win games because they aren't scoring enough goals. Poor results can hurt their league standing, revenue, and fan support, putting pressure on management to improve the offense. Understanding which shots have the best chance of scoring could help coaches create better opportunities and avoid wasting possessions on poor shots. The results could also help management decide which players to recruit or rely on to score goals.

### Stakeholder and Intended Use

__Who would care about the results of this project?__

The club's coaching staff would care most because they decide how the team creates scoring chances. The front office would also be interested, since the results could help them identify players who are reliable scoring threats.

__Primary stakeholder:__ The club's coaching staff

__What might this stakeholder want to learn or do with the results?__

*Example:* A customer retention team might use the results to better understand which customer characteristics or behaviors are associated with cancellation and to inform future retention strategies.

__Your response:__

The coaching staff wants to understand which shot locations and situations give players the best chance of scoring. This includes looking at how defensive pressure affects those chances. The results could help coaches plan attacking plays and training sessions that focus on creating better scoring opportunities instead of settling for poor shots. Player results could also help decide who should be taking those chances.

### Data Overview

__What data will you use to investigate the problem?__

Identify the dataset and briefly describe what it contains. For a project from the Capstone curated project bank, use the information provided in the project description. Remember, you have the option to refine the project goal or question while still using the same data source!

| Item | Description |
|---|---|
| Dataset name | StatsBomb Open Data (event, match and lineup data) |
| Data source | Hudl StatsBomb Open Data: <https://github.com/hudl/open-data> |
| Unit of analysis (e.g., customer, transaction, patient) | Shot (one row per shot attempt) |
| Approximate number of observations, if known | 24 competitions, 80 seasons; shot count TBD in Sprint 2 |
| General types of information available | Shot details (location, type, outcome), player and team, match details, and StatsBomb's xG |
| Data access confirmed? | Yes |

__How might these data help address the problem?__

Briefly explain what information in the dataset could be relevant to the problem.

*Example:* The dataset contains customer demographics, subscription information, service usage, and cancellation status. These data may help the team examine patterns associated with customer attrition.

__Your response:__

The dataset includes information about shots, such as their location, type, body part used, and outcome. Shots also record whether the shooter was under pressure, and many include the positions of nearby players. We can use this data to see which situations lead to more goals and compare players based on the scoring chances they get and how often they convert them.

### The Data Science Opportunity

__What could data science / machine learning / AI practices help you accomplish here?__

At a high level, describe what you hope to learn or accomplish using the available data. You do not need to specify particular algorithms, preprocessing steps, or evaluation metrics here; those belong in the Analysis Plan due in Sprint 2.

*Example:* The team could use historical customer information to identify patterns associated with churn and investigate whether customer cancellation can be predicted using available customer characteristics and behaviors.

__Your response:__

We want to use past shot data to figure out what makes certain shots more likely to become goals. From there, we hope to predict the chance of scoring based on where a shot is taken and the situation around it. We can then use these predictions, known as expected goals (xG), to see which players get the best scoring chances and how often they actually score.

### Initial Questions and Unknowns

What do you still need to understand about the problem, the data, or the intended use of the results before beginning the analysis?

List up to three questions that your team expects to investigate during the background/rationale development and initial data exploration, both of which you will do during Sprint 2.

1. How do shot location, shot type, body part, and defensive pressure affect the chances of scoring?

2. Do we have enough data to compare players fairly, especially when some seasons only include one team's matches? How often are defender and goalkeeper positions available?

3. Are scoring patterns different across competitions or between men's and women's matches? If so, should we analyze them separately? 

These are starting questions, not necessarily your final analytical question(s). You will refine your understanding of the problem and develop your Analysis Plan during Sprint 2.

### Project Risks

Project risks are uncertainties that can undermine a project's success. Sometimes we have ways to better control for or assess risks, and sometimes we do not. Your job is to try to identify at least 2 more risks to this project. I am starting you off with a risk everyone will always have at this stage in a project.

1. __The dataset chosen may not be suitable to address the problem/question exactly as posed.__

2. Uneven coverage across seasons. Many seasons only include matches from one team, usually Barcelona, or a single final. This could make player comparisons unfair because some players have more matches available than others.

3. Small samples for individual players. Some players may only have a few shots in the dataset, making it difficult to tell whether their scoring results are due to skill or luck.

### Your Intended Stakeholder

This is actually one of the hardest parts - identifying your primary stakeholder. In the real world, you will know exactly who they are and what is expected of you. Here, you are being asked __to choose__ one (and really try to focus on just one) stakeholder. This will be the audience to whom you address your deliverables (as well as other data scientists on your team, of course!).

__Your response:__

The club's coaching staff will be the intended stakeholder. We aim to equip the coaching staff with actionable shot quality data. 

## Final Check

Before submitting, confirm that everyone on your team can answer the following:

- [x] We can clearly describe the problem we will address in our Capstone project.
- [x] We understand why the problem may matter and __to whom__ it may matter.
- [x] We have identified a relevant stakeholder or intended audience.
- [x] We have identified the data we will use and what those data represent.
- [x] We understand, at a high level, what machine learning and data science practices might contribute to this problem.
- [x] We have identified any important questions or uncertainties that remain.
- [x] We have identified, at a high level, any risks to this project.
- [x] We have confirmed that the project and data source are appropriate for the eight-week Capstone.

__Team confirmation:__ We agree that this charter accurately describes our shared understanding of the project at the start of the Capstone.

__Submitted by:__ Brian Quevillon

__Date:__ 11/2/2026

## Data Attribution

Data provided by StatsBomb: <https://github.com/hudl/open-data>
![StatsBomb logo](../images/statsbomb_logo.png)
