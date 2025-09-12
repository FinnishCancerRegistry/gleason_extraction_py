Gleason score extraction at the Finnish Cancer Registry
=======================================================

Gleasonextraction is a python tool to extract gleason scores from pathology texts. This is python implementation of the original [research project written in R](https://github.com/FinnishCancerRegistry/gleason_extraction) written for the peer-reviewed study "Accurate pattern-based extraction of complex Gleason score expressions from pathology reports" (https://doi.org/10.1016/j.jbi.2021.103850).

# Installation

```bash
# optional virtual environment
python -m venv .venv
./.venv/Scripts/activate

# install deps
pip install -r requirements.txt
```

# Usage

```python
import gleason_extraction_py as ge

ge.extract_gleason_scores_from_text(
  text= "gleason 4 + 3 something something gleason 4 + 4",
  patterns=["gleason (?P<A>[3-5])[ +]+(?P<B>[3-5])"],
  match_types=["a + b"]
)

   obs_id  a  b     t     c  start  stop match_type warning
0       0  4  3  <NA>  <NA>      0    13      a + b    None
1       1  4  4  <NA>  <NA>     34    47      a + b    None
```

# Extraction process description

This was written to make the extraction process understandable. Mainly what
assumptions are made, therefore what limitations there are.

Extraction is based on regular expressions. The actual extracted values
must be included in named capture groups, one of `A`, `B`, `T`, `C`, and
`A_and_B`. A simple example is given above under section Usage.

The regexes are used in the given order. This matters because
parts of text can sometimes match multiple regexes you have written.
To ensure this only happens on the first matching regex, after every match
the matched part of text is "masked", replaced with the number
of underscores which keeps the length of the text the same. E.g.

`"something gleason 3 + 4 something"`

is in masked form

`"something _____________ something"`

The formation of the actual regular expressions in `gleason_extraction.py`
is not documented here in detail, but the philosophy was to avoid false 
positives as much as possible by defining "whitelists" of expressions that
are allowed before and/or after an actual value. Such whitelists include
defining what words mean "primary Gleason grade" 
("prim[aä]{1,2}", "pääluok", ...). Because matching at least one item in such
a whitelist is mandatory in a prefix and/or suffix, only more "safe" parts of
text are extracted from text (again, we hate false positives).

The regular expressions have been written for Finnish and Swedish texts in mind,
but they can be modified to work better for e.g. English texts by adding English
expressions into the various whitelists. A few English
expressions have already been included --- English language was rare in our
Finnish text data.

Before the actual extraction certain false positives are removed and the text
is simplified and lowercased. Especially names of fields in text such as 
"gleason 6 or less" caused false positives. The simplification step removes
line changes, repeated special characters (e.g. "....." -> "."), and replaces
roman numerals with latin ones. Also, to handle texts where all lines were
pasted together (even before the cleaning up step), a whitespace is added
between all instances of a number and letter 
(e.g. "Gleason 3 + 4Some other text"). We remove two kinds of expressions
from text to simplify and shorten them.

After extraction we identify combinations of A/B/T/C values (at this point
integers) which were not collected together previously 
(in e.g. `match_type = "a + b"` they are already collected together and need no
further processing). This is necessary because many texts contain "tables" in
text which list the components separately, sometimes multiple values for the
same component are given so that we end up with e.g.
`match_type = ["a","a", "b","b"]` and `value = [3,3, 4,3]`.
`determine_element_combinations` was written to combine such separately
extracted components into proper "a + b" etc. combinations, where possible.

`determine_element_combinations` contains a fixed list of allowed combinations
which are interpreted as belonging in the same group (e.g. `{C, A, B, T}`,
`{C, A, B}`). This list is looped through, and each element is allowed to repeat
1-n times (e.g. n = 2 -> c,c, a,a, b,b, t,t; the largest n is the first to
be searched for). Each combination-repetition is
tested whether it fits the data, and the first match is taken as the element
combination. This has been found to work very well, but it is based on the
what is assumed in the list of allowed combinations and the order in which
the elements appear in text only (and not based on e.g. the distance between
two elements).

This concludes the extraction process.
