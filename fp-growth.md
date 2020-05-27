# FP-Growth

### Sequential Pattern Mining

Source: <http://data-mining.philippe-fournier-viger.com/introduction-sequential-pattern-mining/>

**Pattern Mining**: discover useful patterns in databases. For example, frequent itemsets, associations, subgraphs, sequential rules, and periodic patterns.

**Sequential Pattern Mining**: task of analyzing sequential data to discover sequential patterns. Discover subsequences in a set of sequences, where the interestingness of a subsequence can be measured in terms of various criteria such as its occurrence frequency, length, and profit.

Lot of real-life applications: bioinformatics, e-learning, market basket analysis, texts, webpage click-stream analysis,...

#### Example: purchases made by a customers in a retail store

```
SID | Sequence
----------------------------
1   | <{a,b},{c},{f,g},{g},{e}>
2   | <{a,d},{c},{b},{a,b,e,f}>
3   | <{a},{b},{f,g},{e}>
4   | <{b},{f,g}>
```

Each sequence represents the items purchases by a customer at different times. For example, SID = 1, means that the customer 1 purchased first a and b, then c, then f and g, then g and finally e.

**frequent sequantial patterns**: subsequences that appear often in a sequence database.

In our example, sequantial pattern mining can be used to find the sequences of items frequently bought by customers. This can be useful to understand the behaviour of customers to make marketing decisions.

**minimum support threshold**: minimum number of sequences in which a pattern must appear to be considered frequent.

A _support_ is the number of sequences containing each pattern

```
Pattern       | Support
<{a}>         | 3
<{a},{g}>     | 2
<{a},{g},{e}> | 2
...
```

Another example of application of sequential pattern mining is text analysis. Find sequences of word frequent in text. An study case: <data-mining.philippe-fournier-viger.com/tutorial-how-to-discover-hidden-patterns-in-text-documents/>

### Time series

Sequential patter mining can also be applied to time series (e.g. stock data), when discretization is performed as a pre-processng step. After the transformation, any sequential pattern mining algorithm can be applied. SAX transformations is one of the techniques to achieve this.

### Algorithms

Classic:

- PrefixSpan
- Spade
- SPAM
- GSP

Novel:

- CM-SPADE
- CM-SPAM
- FCIoSM and FGenSM

Extract frequent sets by either:

- breadth-first search: apriori
- depth-first search: eclat


Theorem antimonotonicity: supersets of infrequent sets must necessarily be infrequent as well.

This allows us to avoid exploration of ample areas of the search space.

#### apriori algorithm

count the frequency of all items (discard infrequent ones)

for growing sizes c = 1,2,3...
  use a join operation on frequent sets of size c to construct candidates of size c+1
  check for infrequent subsets of size c, discard if found
  count (at once) the support of the remaining candidates and discard infrequent ones
  exit when no candidates remains

join:
  - identify pairs of frequent sets of size c that only differ in their largest item
  - join them into a candidate set of size c+1
  - the other c-1 subsets of size c are still to be checked

Hash trees are used.
