# Gleason score extraction at the Finnish Cancer Registry

Gleasonextraction is a python tool to extract gleason scores from pathology texts. This is python implementation of the original [research project written in R](https://github.com/WetRobot/gleason_extraction) written for the peer-reviewed study "Accurate pattern-based extraction of complex Gleason score expressions from pathology reports" (https://doi.org/10.1016/j.jbi.2021.103850).

Contents:

- `src/gleason_extraction.py`: the regular expressions and how they are used are defined here.
- `src/utils.py`: general utility functions.
- `tests/*`: unit tests.

To use this code, you can simply add this project as a sub-dir of your
project and import the sub-dir with e.g.

```python
import gleason_extraction_py as ge
```

## Extraction process description

This was written to make the extraction process understandable. Mainly what
assumptions are made, therefore what limitations there are.

Extraction is based on regular expressions. They must be defined into a table 
of regexes. The table must look like this:

|pattern_name |match_type |prefix      |value           |suffix   |
|:------------|:----------|:-----------|:---------------|:--------|
|pattern_01   |a + b      |gleason[ ]* |[0-9] [+] [0-9] |         |
|pattern_02   |a + b      |gleason[ ]* |[0-9] and [0-9] |         |
|pattern_03   |c          |gleason[ ]* |[0-9]           | karsino |

A table like this is used by `extract_gleason_scores`
(ultimately `extract_context_affixed_values`) to perform extraction
from text. They are used in the order given in the table. The prefix and suffix
define mandatory expressions before and after the value itself for higher
certainty. Each found match is replaced in text by a "mask" 
(e.g. "gleason 3 + 4, gleason 7" -> "___________, gleason 7" 
-> "__________, ___________"). This is done to avoid matching the same part of
text multiple times. Therefore, it is important to have the patterns in
a good order in the table.

`pattern_name` is used simply to identify what pattern was used. `match_type`
must be one of the allowed match types 
(see `component_parsing_instructions_by_match_type`) --- it is used later
to parse the extracted value string (e.g. "3 + 4" -> "{'a': 3, 'b': 4}").

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
expressions have already been included --- English language was rare in Finnish our
text data.

Before the actual extraction is performed in `extract_gleason_scores`,
the text is "cleaned up" by `prepare_text`.
This removes certain false positives and simplifies
and lowercases the text. Especially names of fields in text such as 
"gleason 6 or less" caused false positives. The simplification step removes
line changes, repeated special characters (e.g. "....." -> "."), and replaces
roman numerals with latin ones. Also, to handle texts where all lines were
pasted together (even before the cleaning up step), a whitespace is added
between all instances of a number and letter 
(e.g. "Gleason 3 + 4Some other text"). We remove two kinds of expressions
from text to simplify and shorten them.

After cleaning the texts, `extract_context_affixed_values` is called
on the given texts and table of regexes.
After this we have the `match_type` (e.g. "a + b") and value string of
a part of text (e.g. "3 + 4"). 

The value string must now be parsed, i.e.
the separate components must be collected into a structured format.
The parsing rules are defined in `component_parsing_instructions_by_match_type`
and `extract_context_affixed_values` is used in the parsing by
`parse_gleason_value_string_elements`. This step can in rare cases produce
problematic results, e.g. `match_type = "a + b"` but only "a" was succesfully
parsed. A few of such problematic results are programmatically identified in
`parse_gleason_value_string_elements`.

In the last step we identify combinations of A/B/T/C values (at this point
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

Finally, the product of the extraction process is a table with columns

- `text_id` (int64): original text_id
- `obs_id` (int64): original text_id extended with the value, 
          which tells the order of appearance in the text. E.g for text_id 1, obs_id is 1001
- `a` (int/NoneType): Most prevalent gleason value
- `b` (int/NoneType): Second most prevalent gleason value
- `t` (int/NoneType): Tertiary gleason value
- `c` (int/NoneType): Gleason score
- `warning` (str): Warning message (match type does not match extracted values, gleason score does not match primary and secondary values)

This concludes the extraction process.
