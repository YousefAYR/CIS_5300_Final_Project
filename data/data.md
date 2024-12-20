# CIS 530 Term Project – Areeb Alam, Yousef Alrabiah, Yannis Kalaitzidis, Rafiz Sadique

## Data Discussion

We have gathered four datasets:

- 2020 Tweets `data/2020_tweets`
- 2020 Tweets with Stance `data/kawintiranon-stance-detection`
- FACTOID Dataset `data/factoid_dataset`
- 2020 Election Nominees and Results `data/2020_elections`

We have also provided the script, `data.ipynb`, which we used to investigate
and split the datasets.

We are also requested access to two datasets:

- Passive-Tracking Participants’ Views of Facebook Posts with Civic News
Domains.
- 2020 US Presidential Election through Factiva

The following are the descriptions of the datasets provided as well as a
summary on the requested data.

### 2020 Tweets

This dataset was found on
[Kaggle](https://www.kaggle.com/datasets/manchunhui/us-election-2020-tweets).
It originally featured two csv files `hashtag_donaldtrump.csv` and
`hashtag_joebiden.csv`, which features tweets posted between 2020-10-15 and
2020-11-08 (five days after the election) that contain the hashtags
(#DonaldTrump, #Trump) and (#JoeBiden, #Biden) respectively. Tweets are not
unique to a dataset.

Originally we had 776,886 tweets attributed to Biden, 970,919 attributed to
Trump, a total of 1,747,805 tweets. We removed duplicate `tweet_ids`, keeping
the one with the latest `created_at` date We then merged the two datasets by
`tweet_id`, while specifying if the tweet originated from the `Biden`, `Trump`,
or `Both` datasets. The final dataset contains 1,522,909 tweets, reducing the
total by 221,700 tweets.

#### Columns

- **created_at**: Date and time of tweet creation
  - Range [2020-10-15 00:00:01, 2020-11-08 23:59:58]
- **tweet_id**: Unique ID of the tweet
  - Count: 1,522,909
- **tweet**: Full tweet text
  - Length:
    - median: 155.0
    - mean: 167.15
    - max: 996
    - min: 6
- **likes**: Number of likes
  - median: 0.0
  - mean: 8.57
  - max: 165702.0
  - min: 0.0
- **retweet_count**: Number of retweets
  - median: 0.0
  - mean: 1.89
  - max: 63473.0
  - min: 0.0
- **source**: Utility used to post tweet
  - 1036 unique labels.
  - Five most frequent labels:
    - `Twitter Web App`:        482,047
    - `Twitter for iPhone`:     463,445
    - `Twitter for Android`:    426,067
    - `Twitter for iPad`:        56,289
    - `TweetDeck`:               23,615
- **user_id**: User ID of tweet creator
  - 483,194 unique ids
- **user_name**: Username of tweet creator
  - Length:
    - median: 13.0
    - mean: 14.11
    - max: 50.0
    - min: 1.0
- **user_screen_name**: Screen name of tweet creator
  - Length:
    - median: 11.0
    - mean: 11.24
    - max: 15
    - min: 2
- **user_description**: Description of self by tweet creator
  - Length:
    - median: 13.0
    - mean: 14.11
    - max: 50.0
    - min: 1.0
- **user_join_date**: Join date of tweet creator
  - Range [1970-01-01 00:00:00, 2020-11-08 23:29:53]
- **user_followers_count**: Followers count on tweet creator
  - median: 428.0
  - mean: 22027.846158897217
  - max: 82417099.0
  - min: 0.0
- **user_location**: Location given on tweet creator's profile
  - NA count: 462,774
- **lat**: Latitude parsed from user_location
  - NA count: 826,041
- **long**: Longitude parsed from user_location
  - NA count: 826,041
- **city**: City parsed from user_location
  - NA count: 1,165,485
- **country**: Country parsed from user_location
  - NA count: 829,875
- **state**: State parsed from user_location
  - NA count: 1,012,984
- **state_code**: State code parsed from user_location
  - NA count: 1,042,596
- **collected_at**: Date and time tweet data was mined from twitter
  - Range [2020-10-21 00:00:00, 2020-11-09 18:40:08]
- **contains**: If tweet contains hashtags relating to Biden, Trump, or Both
  - `Trump`:    747877
  - `Biden`:    553332
  - `Both`:     221700

### 2020 Tweets with Stance

The dataset was collected from Georgetown University and retrieved from
the following
[site](https://portals.mdi.georgetown.edu/public/stance-detection-KE-MLM).

>These tweets were sampled from tweets related to the 2020 US Presidential
election obtained from the Twitter Streaming API using the keywords Biden and
Trump from January 2020 and September 2020. These tweets are original tweets in
English. Retweets, quoted tweets and non-English tweets were discarded.\
The tweets were labeled using Amazon Mechanical Turk. For each tweet, three
annotators were asked to annotate whether a given tweet contained a
supportive/against/neutral stance toward the target (Biden/Trump). The stance
label is the majority vote.

Originally the data was distributed over four csv files:

- `biden_stance_train_public.csv`: 875 rows
- `biden_stance_test_public.csv`: 375 rows
- `trump_stance_train_public.csv`: 875 rows
- `trump_stance_test_public.csv`: 375 rows

Since there was no distinction between train and test datasets, we merged train
and test (to keep ratio 80/10/10). We also merged the Biden and Trump datasets
with an added column indicating which candidate the row came from, as the label
is relative to the candidate.

Train was truncated to abide by Gradescope Max file size.

#### Columns

- **tweet_id**: Unique ID of the tweet
- **text**: Full tweet text
  - Length:
    - median: 146.0
    - mean: 157.1996
    - max: 510
    - min: 4
- **label**: Indicates whether the tweet supports, opposes, or is neutral
towards the candidate
  - Biden
    - NONE: 487 (38.96%)
    - AGAINST: 385 (30.80%)
    - FAVOR: 378 (30.24%)
  - Trump
    - NONE: 410 (32.80%)
    - AGAINST: 499 (39.92%)
    - FAVOR: 341 (27.28%)
- **candidate**: The candidate the tweet label refers to.

### FACTOID Dataset

This dataset was obtained from the FACTOID project and includes user-level data
from Reddit, covering the period from January 2020 to April 2021. The dataset
consists of 4,150 users who posted on various political topics, with a total of
approximately 3.3 million posts across controversial subreddits like
r/politics, r/Conservative, r/democrats, and others. The posts were collected
iteratively, focusing on threads related to topics such as the 2020 U.S.
Presidential Election, COVID-19, climate change, and gun control.

Initially, the dataset was annotated by categorizing users based on the nature
of the news content they shared. Users were labeled as "Misinformation
Spreaders" or "Real News Spreaders" based on the presence of links to credible
or misleading sources in their post history. Additional labels included metrics
for factuality, political bias, and science belief, to capture the nuances of
each user's posting behavior and alignment.

Dataset can be found from the following
[Google Drive](https://drive.google.com/drive/folders/1MB6zsrhNerZQlLFBdjJ8sDbvXa2NcELZ).

Train was truncated to abide by Gradescope Max file size.

#### Columns

Column data was obtained from the following [paper](https://arxiv.org/abs/2205.06181)

- **user_id**: Unique identifier for each Reddit user.
  - Count: 4,150 unique users

- **post_id**: Unique identifier for each post.
  - Count: 3,354,450 posts

- **created_at**: Date and time of post creation on Reddit.
  - Range: [2020-01-01 00:00:00, 2021-04-30 23:59:59]

- **subreddit**: Subreddit where the post was made.
  - 65 unique subreddits.
  - Five most frequent subreddits:
    - `r/politics`: 2,399,254 posts
    - `r/Conservative`: 346,042 posts
    - `r/Coronavirus`: 92,163 posts
    - `r/MensRights`: 57,654 posts
    - `r/climateskeptics`: 38,606 posts

- **title**: Title of the Reddit post.
  - Length:
    - median: 48 characters
    - mean: 52.3 characters
    - max: 300 characters
    - min: 1 character

- **body**: Full content of the post (if available).
  - Length:
    - median: 178 characters
    - mean: 204.8 characters
    - max: 10,000 characters
    - min: 0 characters (empty posts)

- **num_comments**: Number of comments on the post.
  - median: 5
  - mean: 12.4
  - max: 8,764
  - min: 0

- **score**: Upvote score of the post.
  - median: 10
  - mean: 24.6
  - max: 32,569
  - min: -10

- **user_karma**: Total karma score of the user at the time of posting.
  - median: 1,524
  - mean: 8,532.9
  - max: 1,245,237
  - min: -50

- **user_creation_date**: Date when the user created their Reddit account.
  - Range: [2005-06-23 00:00:00, 2021-04-30 23:59:59]

- **political_bias**: Political bias score of the user, based on the content
shared.
  - Range: [-3 (extreme left), +3 (extreme right)]
  - Distribution:
    - extreme left: 10.2%
    - left: 18.3%
    - center-left: 14.1%
    - least biased: 28.4%
    - center-right: 15.7%
    - right: 9.4%
    - extreme right: 3.9%

- **factuality**: Factuality degree score of the user.
  - Range: [-3 (very low), +3 (very high)]
  - median: 1 (mostly factual)
  - mean: 0.5
  - min: -3
  - max: 3

- **science_belief**: Science belief score, indicating the user’s trust in
scientific information.
  - Range: [-1 (conspiracy theory), +1 (science-based)]
  - Distribution:
    - conspiracy: 15.8%
    - science-based: 84.2%

- **satire_degree**: Proportion of satirical content in the user’s
misinformation posts.
  - Range: [0.0, 1.0]
  - median: 0.1
  - mean: 0.3

- **label**: Label indicating if the user is a misinformation or real news
spreader.
  - `Misinformation Spreader`: 26%
  - `Real News Spreader`: 74%

### 2020 Election Nominees and Results

The 2020 Election results by county. We aim to use this dataset to compare it
to our predictions. State results can be found by finding maximum count by state.
The dataset was retrieved from
[Kaggle](https://www.kaggle.com/datasets/unanimad/us-election-2020).

#### Columns

- state
- county
- candidate
- party: Candidate's party
- total_votes: Votes to the candidate from the county.
- won: Boolean indicating if the county was won by the candidate.

### Passive-Tracking Participants’ Views of Facebook Posts with Civic News Domains.

The metrics in this
[dataset](https://socialmediaarchive.org/record/41?ln=en&v=pdf)
measure participants' views of posts with links to civic news domains over the
study period. The dataset contains domain-level metrics from Facebook activity
data for the subset of participants in the platform intervention experiment
control groups who consented to have their internet browsing behavior tracked
by an external partner for the purpose of this study, aggregated over the study
period. Includes content views, content attributes, user attributes.
353,533 participants were part of the study.

### 2020 US Presidential Election through Factiva

[Retrieving data about the 2020 United States presidential election by consuming
the Factiva APIs.](https://developer.dowjones.com/site/docs/getting_started/data_selection_samples/2020_US_presidential_election/index.gsp)

This data selection sample describes how to create a query to select and
extract data about the 2020 U.S. presidential election by consuming the
Factiva APIs. Such query considers a variety of terms and topics related to the
presidential candidates, the Democratic and Republican parties and the 2020
U.S. presidential race in general. It also filters content based on subject,
language and publication date.
