# Bitmap-precomputability classification rules (per query template)

Inputs: TPC-DS templates <tpcds-kit>/query_templates/queryN.tpl (with `define` lines),
distributions <tpcds-kit>/tools/*.dst; TPC-H queries <tpch-kit>/dbgen/queries/N.sql
(substitution parameters `:1`, `:2`, ... are defined in TPC-H spec clause 2.4.N.3; also see tpch-kit/dbgen/qgen.c and dists.dss).

For EVERY predicate in the WHERE/HAVING/ON clauses of the whole query (all CTEs, subqueries, UNION branches), emit one record:
- column: table.column (or the expression text)
- op: =, IN, BETWEEN, <, <=, >, >=, LIKE, IS NULL, ... 
- category, exactly one of:
  * "join"            : equality between columns of two different tables (foreign-key join)
  * "col_vs_col"      : comparison of two columns (same or different table) that is NOT a plain FK join, e.g. l_commitdate < l_receiptdate
  * "expression"      : the column side is an arithmetic expression / function of columns (e.g. ss_net_profit/ss_quantity, substr(...)) 
  * "like"            : LIKE / substring / pattern match
  * "subquery"        : the constant comes from a (correlated or scalar) subquery or from another aggregate (e.g. > 1.2*avg(...), IN (select ...))
  * "aggregate_filter": HAVING on an aggregate (e.g. count(*) > 1, sum(x) > c)
  * "constant"        : one base-table column compared against literal constants and/or template substitution parameters
- legal_set_size (only for "constant"): the number of DISTINCT predicate instances that the template can generate for this
  column (union over all legal parameter values). Rules:
  * parameter = random(a,b,uniform) used as an equality constant -> b-a+1
  * dist(name, ...) -> number of entries of that distribution in the .dst files (count the `add` lines of `create name`)
  * a parameter used to build a window/range (e.g. d_year = YEAR, d_month_seq between DMS and DMS+11, date between DATE and DATE+30 days) -> number of distinct windows = number of legal values of the parameter
  * literal constants with no parameter -> 1 per distinct literal predicate
  * IN-lists of k parameters drawn from a set of size s -> s (one bitmap per member value; IN = sum of disjoint member bitmaps)
  * if a parameter's range depends on scale/rowcount (rowcount(...), distmember over active rows), give an upper bound and set uncertain=true
- uncertain: true if you had to guess a size; note: short free text with the source (define line / dist name / spec clause)

GROUP BY keys (and PARTITION BY keys if they define output groups): one record per key column with
domain_size_estimate (distinct values at SF1; use the .dst files where the column is drawn from a distribution, else your best
knowledge of TPC-DS/TPC-H SF1 cardinalities), source, uncertain flag. Skip ROLLUP/GROUPING artefacts beyond listing the columns.

Output: append ONE JSON object per query (JSON Lines) to the output file given in your task, with keys:
{"benchmark": "TPC-H"|"TPC-DS", "query": <int>, "predicates": [ {column, op, category, legal_set_size, uncertain, note} ... ],
 "group_by": [ {column, domain_size_estimate, uncertain, source} ... ], "comments": "..."}
Be exhaustive and literal: every predicate occurrence in the SQL text counts (do not deduplicate across branches; dedup only
exact duplicates in the same scope). Do not judge precomputability yourself beyond the category; thresholds are applied later.
