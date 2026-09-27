# The Marvel Universe as a Social Network

Network-science analysis of the Marvel Comics character co-appearance network — reproducing and extending [Alberich, Miró-Julià & Rosselló (2002)](https://arxiv.org/abs/cond-mat/0202174), "Marvel Universe looks almost like a real social network."

Answers three questions with real statistical validation, not just plots:
1. Who's the *true* main character by network structure, vs. by marketing/movies?
2. What are the real cliques the writers built (and do they match known teams)?
3. Whose removal would fracture the network the most?

## Data

Source: [Marvel Chronology Project](http://www.chronologyproject.com) data, via [melaniewalsh/sample-social-network-datasets](https://github.com/melaniewalsh/sample-social-network-datasets). Raw files live in `data/marvel/`.

Three networks are analyzed throughout:
- **Model #1** — the pre-built 327-character network (shipped as-is)
- **Model #2** — our own Jaccard-weighted projection from the raw bipartite data, curated (2,396 characters)
- **Model #3** — same, uncurated (6,439 characters) — used to check the curated version's conclusions aren't an artifact of that curation



