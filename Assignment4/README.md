# CSL7110 Assignment 4 — Clustering and PageRank

**Name :** Arpita Kundu
**Roll Number:** M25DE1004  


---

## Overview

This assignment is divided into three parts:

1. **Part 1 — Clustering** (40 marks): Implement k-center and k-means++ clustering algorithms on Apache Spark using a spam detection dataset.
2. **Part 2 — Web Search / Inverted Index** (40 marks): Build an inverted index and implement TF-IDF based search over a collection of webpages.
3. **Part 3 — PageRank on Spark** (40 marks): Implement the iterative PageRank algorithm on a directed graph using Apache Spark.

---

## Repository Structure

```
.
├── README.md
├── .ipynb_checkpoints/
├── datasets/                              # All datasets used across parts
├── M25DE1004_CSL7110_Assignment4.ipynb   # Jupyter notebook with all code
└── M25DE1004_CSL7110_Assignment4.docx    # Assignment report
```

---

## Part 1: Clustering

### Dataset
- **Source:** UCI Spam Dataset (in `datasets/` folder)
- **Size:** 4601 points × 58 dimensions
- Points are represented as vectors of numbers (email spam detection features)

### Algorithms

#### Farthest First Traversal (k-center)
Greedily selects k centers by always picking the point farthest from the current set of centers.  
**Reference:** http://www.wikiwand.com/en/Farthest-first_traversal

#### k-means++
Probabilistic seeding strategy for k-means that selects initial centers with probability proportional to squared distance from existing centers.  
**Reference:** https://theory.stanford.edu/~sergei/papers/kMeansPP-soda.pdf

### Functions Implemented

| Function | Description |
|---|---|
| `readVectorsSeq(filename)` | Reads a CSV file and returns a list of `Vector` objects |
| `kcenter(P, k)` | Farthest First Traversal; returns k centers. Runs in O(\|P\| × k) |
| `kmeansPP(P, k)` | k-means++ seeding; returns k centers. Runs in O(\|P\| × k) |
| `kmeansObj(P, C)` | Computes average squared distance from each point to its closest center |

### Program Behavior

Given a dataset file and integers `k` and `k1` (with `k < k1`), the program:

1. Runs `kcenter(P, k)` and prints its running time.
2. Runs `kmeansPP(P, k)` to get k centers C, then runs `kmeansObj(P, C)` and prints the result.
3. Runs `kcenter(P, k1)` to get k1 centers X, then runs `kmeansPP(X, k)` to extract k centers C from X, then runs `kmeansObj(P, C)` and prints the result. (Tests whether a k1-center coreset improves kmeans++ quality.)

### Notes
- Use `org.apache.spark.mllib.linalg.Vectors` for vector operations (`Vectors.dense()`, `Vectors.sqdist()`)
- All implementations are Spark RDD-compatible

---

## Part 2: Web Search — Inverted Index

### Classes Implemented

#### `MySet`
- `addElement(element)` — Add element to the set
- `union(otherSet)` — Return union of two sets
- `intersection(otherSet)` — Return intersection of two sets

#### `Position`
Represents a `<page, word_index>` tuple.
- `Position(p: PageEntry, wordIndex: int)`
- `getPageEntry()` — Returns the page entry
- `getWordIndex()` — Returns the word index

#### `WordEntry`
Stores all positions of a word across documents.
- `WordEntry(word: String)`
- `addPosition(position)` / `addPositions(positions)`
- `getAllPositionsForThisWord()` — Returns all position entries
- `getTermFrequency(word)` — Returns TF of the word in a webpage

#### `PageIndex`
Stores one word-entry per unique word in a document.
- `addPositionForWord(str, position)` — Adds position; creates new entry if needed
- `getWordEntries()` — Returns all word entries

#### `MyHashTable`
Maps words to their word-entries.
- `getHashIndex(str)` — Hash function for the word string
- `addPositionsForWord(w: WordEntry)` — Adds or merges entry

#### `PageEntry`
- `PageEntry(pageName)` — Constructor; reads the file and builds the page index
- `getPageIndex()` — Returns the page index

#### `InvertedPageIndex`
- `addPage(p: PageEntry)` — Adds a page to the index
- `getPagesWhichContainWord(str)` — Returns all pages containing the word

#### `SearchEngine`
- `SearchEngine()` — Constructor; creates an empty inverted index
- `performAction(actionMessage)` — Parses and executes an action string

### Supported Actions

| Action | Output |
|---|---|
| `addPage x` | Adds webpage `x` to the search engine from the `webpages/` folder |
| `queryFindPagesWhichContainWord x` | Prints comma-separated page names containing word `x`, or `"No webpage contains word x"` |
| `queryFindPositionsOfWordInAPage x y` | Prints comma-separated word indices of `x` in page `y`, or appropriate error message |

### Preprocessing Rules
- Convert all words to **lowercase**
- **Remove stop words:** `a, an, the, they, these, this, for, is, are, was, of, or, and, does, will, whose`
- **Replace punctuation with a space:** `{ } [ ] < > = ( ) . , ; ' " ? # ! - :`
- **Treat singular and plural as the same** (e.g., `stack` = `stacks`, `structure` = `structures`)
- Stop words are **counted** when calculating word indices but **not stored** in the index

### Scoring: TF-IDF

```
relevance_w(p) = tf_w(p) × idf_w(p)

tf_w(p)  = (occurrences of w in p) / (total words in p)
idf_w(p) = log(N / n_w)
```

where `N` = total number of pages, `n_w` = number of pages containing word `w`.

---

## Part 3: PageRank on Spark

### Dataset
- **Source:** https://github.com/pnijhara/PySpark-PageRank/tree/main/graph
- **Full graph:** `whole.txt` — 1000 nodes, 8192 edges (stored in `datasets/`)
- **Small graph:** `small.txt` — 53 nodes (for testing; top PageRank score ≈ 0.036)
- The graph contains a directed cycle over all 1000 nodes to ensure connectivity
- Treat multiple directed edges between the same pair of nodes as a single edge
- Column 1 = source node, Column 2 = destination node

### PageRank Formula

The transition matrix M is defined as:

```
M[i][j] = 1 / deg(i)   if edge i → j exists
         = 0            otherwise
```

The PageRank vector `r` satisfies:

```
r = ((1 - β) / n) * A + β * M * r
```

where `A` is the n×1 unit vector and `β` is the damping factor.

### Iterative Algorithm

```
1. Initialize r0 = (1/n) * A
2. For i = 1 to k:
       r[i] = ((1 - β) / n) * A + β * M * r[i-1]
```

### Experiment Parameters
- **Iterations:** 40
- **β (damping factor):** 0.8
- **Implementation:** Matrix M processed as a Spark RDD

### Output
1. **Top 5 nodes** with the highest PageRank scores
2. **Bottom 5 nodes** with the lowest PageRank scores

---

## References

- Farthest First Traversal: http://www.wikiwand.com/en/Farthest-first_traversal
- k-means++ Paper: https://theory.stanford.edu/~sergei/papers/kMeansPP-soda.pdf
- k-means++ Slides: http://theory.stanford.edu/~sergei/slides/BATS-Means.pdf
- PageRank Dataset: https://github.com/pnijhara/PySpark-PageRank/tree/main/graph
