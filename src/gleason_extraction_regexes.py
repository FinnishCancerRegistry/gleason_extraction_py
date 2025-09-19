import pandas as pd

# @codedoc_comment_block study_regexes
# The regular expressions used in the
# [study](https://doi.org/10.1016/j.jbi.2021.103850)
# were formed by writing "lego blocks" of simpler regular expressions,
# functions that process regular expressions into more complex ones, and
# making use of these two types of objects to form rather long regular
# expressions that were ultimately used to extract Gleason scores in our data.
#
# While a programme based on regular expressions is always specific for the
# dataset for which they were developed, this is also true in statistical models
# where the statistical model is general but the fit is specific to the dataset.
# Our regexes can be straightforward to adapt to other datasets because the
# "lego blocks" are often lists of mandatory words that must appear before
# and/or after a gleason score match. For instance the regular expression
# for `kw_a`, keyword-and-primary, requires both the words "gleason" (in some
# form, with common typos) and the word "primary", with possible conjugates,
# and many synonyms. Both must appear but in either order, so both e.g.
# "gleason primary 3" and "primary gleason 3" are matched.
#
# Some mandatory words are required in all the regular expressions, even
# "3 + 3 = 6" would not be matched without a preceding "gleason". We chose
# this approach because we considered it far worse to collect false alarms than
# to miss some extractions. Indeed in our study less than 1 % of collected
# values were false alarms.
# @codedoc_comment_block study_regexes

# word elements ----------------------------------------------------------------
# `word_sep` defines what must separates words.
word_sep = "[ ,-]{1,3}"

# `optional_word_sep` defines what may separate words.
optional_word_sep = "[ ,-]{0,2}"

# `word_suffices` defines what characters words can use
# in inflections. E.g. "gradus" -> "gradusta", etc. The dot `"."` was included
# to allow for abbreviated forms, e.g. "yht.pist." meaning "yhteispistemäärä"
# meaning "total score".
word_suffices = "[.a-zåäö]*"

# `one_arbitrary_natural_language_word` is an alias of `word_suffices` because
# both in effect define what characters a word is allowed to have (i.e.
# no difference in characters allowed in suffix vs. body of word).
one_arbitrary_natural_language_word = word_suffices

# `zero_to_three_arbitrary_natural_language_words` allows
# `one_arbitrary_natural_language_word` to repeat zero, one, two, or three 
# times. The word separator is `optional_word_sep`.
zero_to_three_arbitrary_natural_language_words = "(" + one_arbitrary_natural_language_word + optional_word_sep + "){0,3}"


# other basic elements ---------------------------------------------------------

# `plus` defines what addition must look like.
plus = "[ ]?[+][ ]?"

# `equals` defines how the equal sign is used in text.
equals = "[ ]?[=][ ]?"

# `number_range` defines what ranges of single-digit numbers look like.
number_range = "[0-9]+[ ]?[-][ ]?[0-9]+"
# `number_range_in_parenthesis` defines single-digit number ranges in 
# parenthesis, e.g. "( 0-9 )".
number_range_in_parenthesis = "\\([ ]?"+ number_range + "[ ]?\\)"


# `optional_nondigit_buffer_5` is intended to allow for arbitrary non-digit
# characters between two things (between zero and five).
optional_nondigit_buffer_5 = "[^0-9]{0,5}"

# `optional_nondigit_buffer_20` is intended to allow for arbitrary non-digit
# characters between two things (between zero and twenty).
optional_nondigit_buffer_20 = "[^0-9]{0,20}"

# `default_regex_suffix` defines a default ending for regular expression
# used to actually extract the Gleason value and its context 
# (i.e. it is the default RHS context).
default_regex_suffix = "([^0-9]|$)"

# `arbitrary_expression_in_parenthesis` defines any expression in parenthesis.
arbitrary_expression_in_parenthesis = "\\([^)]*\\)"


# funs --------------------------------------------------------------------
def whitelist_sep():
	""""
	Returns:
		str: `([ ,-]{0,2}| ja | tai | och | eller )`
	"""
	return "([ ,-]{0,2}| ja | tai | och | eller )"

	

def whitelist_to_whitelist_regex(whitelist, match_count = "+"):
	""" Turns a list of whitelist expressions into regex of those expressions. 
	Each expression may be separated by `whitelist_sep()` and repeat to the quantity specified
  	via argument `match_count`.

	Args:
		whitelist (list(str)): whitelist expressions
		match_count (str, optional): Repetition quantity. Defaults to "+".
	
	Raises:
		ValueError: match_count length has to be 1

	Returns:
		str: regex
	"""
	if not len(match_count) == 1:
		raise ValueError('match_count length has to be 1')
	
	return "(" + "(" + "|".join(whitelist)  + ")" + whitelist_sep() + ")" + match_count


def word_whitelist_to_word_whitelist_regex(whitelist, match_count = "+"):
	""" Allows for a set of words to repeat the requested number of times
	(defined via argument `match_count`) in any order. The words may be 
	separated by anything that matches `whitelist_sep()`. The words are 
	allowed to inflect by appending regex `word_suffices` to each word in `whitelist`.

	Args:
		whitelist (list(str)): whitelist expressions
		match_count (str, optional): Repetition quantity. Defaults to "+".

	Raises:
		ValueError: match_count lenght has to be 1

	Returns:
		str: regex
	"""
	if not len(match_count) == 1:
		raise ValueError('match_count lenght has to be 1')
	return "(" + "(" + "|".join(whitelist) + ")" + word_suffices + whitelist_sep() + ")" + match_count


def multiple_alternative_value_matches(x):
	"""Turns a regex capturing a value into one that can capture multiple ones occurring in sequence.

	Args:
		x (str): regex

	Returns:
		str: regex
	"""
	return x + "(( | / |/| tai | ja | eller | och | and | or |[ ]?-[ ]?)" + x + ")*"


# grade / score values ----------------------------------------------------
# `score_a_b_t` defines what kinds of grades (A, B, and T in A + B + T = C) are 
# extracted.
score_a_b_t = "[3-5]"
# `score_c` defines what kinds of scoresums (C in A + B = C) are extracted.
score_c = "(?:10|[6-9])"

score_a = "(?P<A>%s)" % score_a_b_t
score_a_and_b = "(?P<A_and_B>%s)" % score_a_b_t
score_b = "(?P<B>%s)" % score_a_b_t
score_t = "(?P<T>%s)" % score_a_b_t
score_c = "(?P<C>%s)" % score_c

# whitelists and their derivatives ----------------------------------------
# `whitelist_scoreword` contains roots of words which refer to either
# the scoresum or gradus.
whitelist_scoreword = ["pist", "tyyp", "luok", "score", "gr", "lk", "kl", "mö", "kuvio","arkkitehtuuri"]
whitelist_scoreword_regex = word_whitelist_to_word_whitelist_regex(whitelist_scoreword)

# `whitelist_gleason_word` defines what variants of "gleason" we search for.
whitelist_gleason_word = "gl[aei]{1,2}s{1,2}[oi]n[a-zåäö]*"

# `whitelist_base_optional` contains expressions which may precede e.g. a 
# scoresum value after the word "gleason"
whitelist_base_optional = [whitelist_scoreword_regex, "n" + word_suffices, number_range_in_parenthesis, arbitrary_expression_in_parenthesis]
whitelist_base_optional_regex = whitelist_to_whitelist_regex(whitelist_base_optional, match_count = "*")

# `base_gleason_regex` is `whitelist_base_optional_regex` but also captures
# the word "gleason".
base_gleason_regex = whitelist_gleason_word + optional_word_sep + whitelist_base_optional_regex

# `optional_base_gleason_regex` is similar to `base_gleason_regex`, but
# it is agnostic wrt the order of word "gleason" and the filler words before
# the value.
optional_base_gleason_regex = whitelist_to_whitelist_regex(whitelist_base_optional + [whitelist_gleason_word], match_count = "*")

# `whitelist_primary` contains the roots of words that indicate primary grade.
whitelist_primary = ["prim[aä]{1,2}", "pääluok", "hufvudkl", "valtaos", "enimm", "tavalli", "vallits", "ylei", "hallits", "vanlig"]
whitelist_primary_regex = word_whitelist_to_word_whitelist_regex(whitelist_primary)
# `optional_or_aggressive_regex` is intended to capture expressions in text such 
# as "or most aggressive" (after e.g. "primary")
optional_or_aggressive_regex = "([ ]?(/|tai|eller)[ ]?aggres[.a-zåäö]*)?"
whitelist_primary_regex = whitelist_primary_regex + optional_or_aggressive_regex

# `whitelist_secondary_regex` captures secondary gradus expressions in text.
whitelist_secondary = whitelist_primary[-5:]
whitelist_secondary_regex =  word_whitelist_to_word_whitelist_regex(whitelist_secondary)
whitelist_secondary_regex = "((2[.])|toise|näst)[.a-zåäö]*[ ]?" + whitelist_secondary_regex
whitelist_secondary_regex = "(" + whitelist_secondary_regex + "|" + word_whitelist_to_word_whitelist_regex(["sekund"]) + ")" + optional_or_aggressive_regex

# `whitelist_scoresumword` contains roots of words associated with the scoresum.
# note that sometimes the word "gradus" was used with the scoresum although 
# this is the incorrect term. only A and B are grades.
whitelist_scoresumword = ["yh","pist","poäng","sum","score","gradus"]
whitelist_scoresumword_regex = word_whitelist_to_word_whitelist_regex(whitelist_scoresumword)

# `whitelist_total` contains roots of expressions indicating the result of
# addition. `whitelist_total_regex` is the list in the form of one regex.
whitelist_total = ["eli", "yht", "yhtä kuin", "pist", "sum", "total", "=", "sammanlag"]
whitelist_total = list(set(whitelist_total + whitelist_scoresumword))
whitelist_total.sort(key=len, reverse = True)
whitelist_total_regex = word_whitelist_to_word_whitelist_regex(whitelist_total)

whitelist_tertiary = [ "terti"] + ["((3\\.)|(kolmann)|(trädj))" + value for value in whitelist_secondary]
whitelist_tertiary_regex = word_whitelist_to_word_whitelist_regex(whitelist_tertiary)

# unit tested variables --------------------------------------------------------
# some variables are defined here instead of in a function because we use them
# in unit tests
# `addition_guide`: addition with letters. sometimes this appears in text to
# guide the reader.
addition_guide = "\\(?[ ]?(a|x)[ ]?[+][ ]?(b|y)[ ]?\\)?"

whitelist_c_optional = [value + word_suffices for value in whitelist_scoreword]
whitelist_c_optional = whitelist_c_optional + [addition_guide, number_range_in_parenthesis, arbitrary_expression_in_parenthesis]
whitelist_c_optional_base_regex = whitelist_to_whitelist_regex(whitelist_c_optional, match_count = "*")
kw_c_prefix = whitelist_c_optional_base_regex + base_gleason_regex + whitelist_c_optional_base_regex + whitelist_scoresumword_regex + whitelist_c_optional_base_regex + optional_word_sep + optional_nondigit_buffer_5

# `a_plus_b` defines what addition should look like.
a_plus_b = score_a + plus + score_b

# `a_comma_b` defines regex for capturing e.g. "3, 4" in "gleason 7 (3,4)".
a_comma_b = score_a + ",[ ]?" + score_b

# `a_plus_b_plus_t` defines what addition with tertiary value should look 
# like.
a_plus_b_plus_t = a_plus_b + "[ (]*" + "[+][ ]?" + score_t + "[ )]*"

# `a_comma_b_comma_t` is `a_comma_b` with additional tertiary score.
a_comma_b_comma_t = a_comma_b + ",[ ]?" + score_t

# `addition_values` defines a plethora of ways in which different additions
	# may appear. note that it is a list of multiple regexes
addition_values = [
	a_plus_b + optional_word_sep + optional_base_gleason_regex + whitelist_total_regex + optional_base_gleason_regex + optional_word_sep + score_c,
	score_c + equals + a_plus_b,
	score_c + "[ ]?\\(" + a_plus_b + "[ ]?\\)",
	score_c + "[ ]?\\(" + a_comma_b + "[ ]?\\)",
	a_plus_b + "[ ]?\\(" + score_c + "[ ]?\\)",
	a_plus_b]
addition_values = ["(" + value + ")" for value in addition_values]

# fcr_pattern_dt ---------------------------------------------------------------
def fcr_pattern_dt():
	"""Generate pattern table which is used for the gleason extraction at the finnish cancer registry.
	Pattern is assembled from smaller pieces. Table is combination of `addition_dt()`, `minor_dt()`, `keyword_dt()`

	This function uses module variables.

	Raises:
		ValueError: Pattern names should be unique

	Returns:
		`DataFrame`: with columns
			`pattern_name` (str): one name per pattern; values:
				"a + b = c","c = a + b","c (a + b)","c (a, b)","a + b (c)", "a + b",
				"a + b + t = c","c = a + b + t","c (a + b + t)","c (a, b, t)","a + b + t (c)", "a + b + t",
				"kw_t","kw_b", "kw_a", "a_kw", "kw_c", "c_kw", "kw_all_a" or
				"sum_near_end"
			`match_type` (str): value combination to look for; values:
				"a + b = c", "a + b",
				"a + b + t = c", "a + b + t"
				"t", "b", "a", "c" or 
				"kw_all_a"
			`prefix` (str): regex; context prefix for value
			`value` (str): regex; the value itself
			`suffix` (str): regex; context suffix for value
			`full_pattern` (str): regex; prefix + value + suffix

	"""
	def addition_dt():
		"""Build addition table which is used for assembling `fcr_pattern_dt`

		This function uses module variables.

		Returns:
			Dataframe: with columns
				`pattern_name` (str): Values:  "a + b = c","c = a + b","c (a + b)","c (a, b)","a + b (c)", "a + b", 
				"a + b + t = c","c = a + b + t","c (a + b + t)","c (a, b, t)","a + b + t (c)", "a + b + t"
				`match_type` (str): Values "a + b = c", "a + b", "a + b + t = c" or "a + b + c"
				`prefix` (str): regex; context prefix for value
				`value` (str): regex
				`suffix` (str): regex
		"""
		addition_dt = pd.DataFrame({
			"pattern_name":["a + b = c","c = a + b","c (a + b)","c (a, b)","a + b (c)", "a + b"],
			"match_type": ["a + b = c","a + b = c","a + b = c","a + b = c","a + b = c", "a + b"],
			"prefix": base_gleason_regex + zero_to_three_arbitrary_natural_language_words,
			"value" : addition_values,
			"suffix" : default_regex_suffix
		})

		# capture also tertiary scores
		addition_values_t = [
			a_plus_b_plus_t + optional_word_sep + optional_base_gleason_regex + whitelist_total_regex + optional_base_gleason_regex + optional_word_sep + score_c,
			score_c + equals + a_plus_b_plus_t,
			score_c + "[ ]?\\(" + a_plus_b_plus_t + "[ ]?\\)",
			score_c + "[ ]?\\(" + a_comma_b_comma_t + "[ ]?\\)",
			a_plus_b_plus_t + "[ ]?\\(" + score_c + "[ ]?\\)",
			a_plus_b_plus_t]
		addition_values_t = ["(" + value + ")" for value in addition_values_t]
		abt_dt = pd.DataFrame({
			"pattern_name":["a + b + t = c","c = a + b + t","c (a + b + t)","c (a, b, t)","a + b + t (c)", "a + b + t"],
			"match_type": ["a + b + t = c","a + b + t = c","a + b + t = c","a + b + t = c","a + b + t = c", "a + b + t"],
			"prefix": base_gleason_regex + zero_to_three_arbitrary_natural_language_words,
			"value" : addition_values_t,
			"suffix" : default_regex_suffix
		})
		return pd.concat([abt_dt, addition_dt])


	def keyword_dt():
		"""Build keyword table which is used for assembling `fcr_pattern_dt`

		t (tertiary) values are not extracted but the program relies on recognising them

		This function uses module variables.

		Returns:
			Dataframe: with columns
				`pattern_name` (str): Values:  "kw_t","kw_b", "kw_a", "a_kw", "kw_c", "c_kw", "kw_all_a"
				`match_type` (str): Values "t", "b", "a", "c" or "kw_all_a"
				`prefix` (str): regex; context prefix for value
				`value` (str): regex
				`suffix` (str): regex
		"""

		# kw_all_a --------------------------------------------------------------
		# `whitelist_only_one_kind` defines roots of words indicating a monograde
    	# result --- e.g. "whole sample grade 4" -> 4+4=8.
		whitelist_only_one_kind = ["yksinom", "ainoas", "pelk", "endast", "enbart"]
		whitelist_only_one_kind_regex = word_whitelist_to_word_whitelist_regex(whitelist_only_one_kind, match_count = "+")
		# `kw_all_*` objects define the (RHS + LHS context and the value) regexes 
   		# for keyword + monograde expressions.
		kw_all_a_prefix = whitelist_only_one_kind_regex + optional_word_sep + base_gleason_regex + optional_word_sep
		kw_all_a_value = score_a_and_b
		kw_all_a_suffix = default_regex_suffix
		
		# kw_a ---------------------------------------------------------------------
    	# `kw_a_*` objects define the regexes for keyword + grade A expressions.
		kw_a_prefix = whitelist_primary_regex + optional_word_sep + optional_base_gleason_regex + optional_word_sep + optional_nondigit_buffer_5
		kw_a_value = score_a
		kw_a_suffix = default_regex_suffix
		
		# kw_b ---------------------------------------------------------------------
		# `kw_b_*` objects define the regexes for keyword + grade B expressions.
		kw_b_prefix = whitelist_secondary_regex + word_sep + "((tai|/|eller) (pahin|korkein|högst)){0,1}" + optional_word_sep + optional_base_gleason_regex + optional_word_sep + optional_nondigit_buffer_5
		kw_b_value = score_b
		kw_b_suffix = default_regex_suffix
		
		# kw_c ---------------------------------------------------------------------
		#`kw_c_*` objects define the regexes for keyword + scoresum expressions.
		kw_c_value = score_c
		kw_c_suffix = default_regex_suffix 	

		# kw_t ---------------------------------------------------------------------
		# keyword + tertiary.
		kw_t_prefix = whitelist_tertiary_regex + optional_word_sep + optional_base_gleason_regex + optional_word_sep
		kw_t_value = score_t
		kw_t_suffix = default_regex_suffix
		
		# a_kw ---------------------------------------------------------------------
		a_kw_prefix = base_gleason_regex + optional_nondigit_buffer_5
		a_kw_value = score_a
		a_kw_suffix = optional_word_sep + whitelist_primary_regex
		
		# b_kw ---------------------------------------------------------------------
		## not in use
		# b_kw_prefix = base_gleason_regex + optional_nondigit_buffer_5
		# b_kw_value = score_b
		# b_kw_suffix = optional_word_sep + whitelist_secondary_regex
		
		# c_kw ---------------------------------------------------------------------
		c_kw_prefix = base_gleason_regex + optional_word_sep + optional_nondigit_buffer_20
		c_kw_value = score_c
		whitelist_scoresum_suffix = ["tauti", "syö", "prostata", "karsino{1,2}ma", "eturauhassyö", "adeno"]
		whitelist_scoresum_suffix = list(set(whitelist_scoresum_suffix + whitelist_scoreword))
		whitelist_scoresum_suffix_regex = word_whitelist_to_word_whitelist_regex(whitelist_scoresum_suffix)
		c_kw_suffix = word_sep + whitelist_scoresum_suffix_regex
		
		# keyword pattern dt -------------------------------------------------------
		kw_names = ["kw_t","kw_b", "kw_a", "a_kw", "kw_c", "c_kw", "kw_all_a"]
		keyword_dt = pd.DataFrame({
			"pattern_name": kw_names,
			"match_type": ["t", "b","a","a","c","c","kw_all_a"],
			"prefix": [kw_t_prefix, kw_b_prefix, kw_a_prefix, a_kw_prefix, kw_c_prefix, c_kw_prefix, kw_all_a_prefix],
			"value" : [kw_t_value, kw_b_value, kw_a_value, a_kw_value, kw_c_value, c_kw_value, kw_all_a_value],
			"suffix" : [kw_t_suffix, kw_b_suffix, kw_a_suffix, a_kw_suffix, kw_c_suffix, c_kw_suffix, kw_all_a_suffix]
		})
		return keyword_dt

	def minor_dt():
		"""Build minor table which is used for assembling `fcr_pattern_dt`

		This function uses module variables.

		Returns:
			Dataframe: with columns
				`pattern_name` (str): Values:  "sum_near_end"
				`match_type` (str): Values "c"
				`prefix` (str): regex; context prefix for value
				`value` (str): regex
				`suffix` (str): regex
		"""
		minor_dt = pd.DataFrame({
			"pattern_name": ["sum_near_end"],
			"match_type": ["c"],
			"prefix": [base_gleason_regex + "[ ]?"],
			"value" : [score_c],
			"suffix" : ["[^0-9]{0,30}$"]
		})
		return minor_dt

	pattern_dt = pd.concat([addition_dt(), minor_dt(), keyword_dt()])
	pattern_dt["value"] = pattern_dt["value"].apply(lambda x: multiple_alternative_value_matches(x))
	pattern_dt["full_pattern"] = pattern_dt["prefix"] + pattern_dt["value"] + pattern_dt["suffix"]
	pattern_dt.reset_index(drop=True, inplace=True)

	if (pattern_dt.pattern_name.duplicated().any()):
		raise ValueError('Pattern names should be unique.')
	return pattern_dt

