import os
import re
import sys

import numpy as np
import pandas as pd

sys.path.append(os.path.dirname(os.path.realpath(__file__)))

import logs
import pattern_extraction
import utils

logger = logs.logging.getLogger('gleason_extraction')

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
		ValueError: match_count lenght has to be 1

	Returns:
		str: regex
	"""
	try:
		if not len(match_count) == 1:
			raise ValueError('match_count lenght has to be 1')
	except Exception as e:
		logger.exception(e)
		raise
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
	try:
		if not len(match_count) == 1:
			raise ValueError('match_count lenght has to be 1')
	except Exception as e:
		logger.exception(e)
		raise
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
# `score_a_or_b` defines what kinds of grades (A and B in A + B = C) are 
# extracted.
score_a_or_b = "[3-5]"
# `score_c` defines what kinds of scoresums (C in A + B = C) are extracted.
score_c = "(10|[6-9])"


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

def fcr_pattern_dt():
	"""Generate pattern table which is used for the gleason extraction at the finnish cancer registry.
	Pattern is assembled from smaller pieces. Table is combination of `additon_dt()`, `minor_dt()`, `keyword_dt()`

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
	def additon_dt():
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
		global a_plus_b
		global addition_values
		global a_comma_b
		global a_plus_b_plus_t
		global a_comma_b_comma_t
		# `a_plus_b` defines what addition should look like.
		a_plus_b = score_a_or_b + plus + score_a_or_b

		# `a_comma_b` defines regex for capturing e.g. "3, 4" in "gleason 7 (3,4)".
		a_comma_b = score_a_or_b + ",[ ]?" + score_a_or_b

		# `a_plus_b_plus_t` defines what addition with tertiary value should look 
		# like.
		a_plus_b_plus_t = a_plus_b + "[ (]*" + "[+][ ]?" + score_a_or_b + "[ )]*"

		# `a_comma_b_comma_t` is `a_comma_b` with additional tertiary score.
		a_comma_b_comma_t = a_comma_b + ",[ ]?" + score_a_or_b


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

		global addition_guide
		global kw_c_prefix
		global whitelist_tertiary_regex

		# kw_all_a --------------------------------------------------------------
		# `whitelist_only_one_kind` defines roots of words indicating a monograde
    	# result --- e.g. "whole sample grade 4" -> 4+4=8.
		whitelist_only_one_kind = ["yksinom", "ainoas", "pelk", "endast", "enbart"]
		whitelist_only_one_kind_regex = word_whitelist_to_word_whitelist_regex(whitelist_only_one_kind, match_count = "+")
		# `kw_all_*` objects define the (RHS + LHS context and the value) regexes 
   		# for keyword + monograde expressions.
		kw_all_a_prefix = whitelist_only_one_kind_regex + optional_word_sep + base_gleason_regex +optional_word_sep
		kw_all_a_value = score_a_or_b
		kw_all_a_suffix = default_regex_suffix
		
		# kw_a ---------------------------------------------------------------------
    	# `kw_a_*` objects define the regexes for keyword + grade A expressions.
		kw_a_prefix = whitelist_primary_regex + optional_word_sep + optional_base_gleason_regex + optional_word_sep + optional_nondigit_buffer_5
		kw_a_value = score_a_or_b
		kw_a_suffix = default_regex_suffix
		
		# kw_b ---------------------------------------------------------------------
		# `kw_b_*` objects define the regexes for keyword + grade B expressions.
		kw_b_prefix = whitelist_secondary_regex + word_sep + "((tai|/|eller) (pahin|korkein|högst)){0,1}" + optional_word_sep + optional_base_gleason_regex + optional_word_sep + optional_nondigit_buffer_5
		kw_b_value = score_a_or_b
		kw_b_suffix = default_regex_suffix
		
		# kw_c ---------------------------------------------------------------------
		#`kw_c_*` objects define the regexes for keyword + scoresum expressions.
		whitelist_c_optional = [value + word_suffices for value in whitelist_scoreword]

		# `addition_guide`: addition with letters. sometimes this appears in text to
    	# guide the reader.
		addition_guide = "\\(?[ ]?(a|x)[ ]?[+][ ]?(b|y)[ ]?\\)?"

		whitelist_c_optional = whitelist_c_optional + [addition_guide, number_range_in_parenthesis, arbitrary_expression_in_parenthesis]
		whitelist_c_optional_base_regex = whitelist_to_whitelist_regex(whitelist_c_optional, match_count = "*")
		kw_c_prefix = whitelist_c_optional_base_regex + base_gleason_regex + whitelist_c_optional_base_regex + whitelist_scoresumword_regex + whitelist_c_optional_base_regex + optional_word_sep + optional_nondigit_buffer_5
		kw_c_value = score_c
		kw_c_suffix = default_regex_suffix 	

		# kw_t ---------------------------------------------------------------------
		# keyword + tertiary.
		whitelist_tertiary = [ "terti"] + ["((3\\.)|(kolmann)|(trädj))" + value for value in whitelist_secondary]
		whitelist_tertiary_regex = word_whitelist_to_word_whitelist_regex(whitelist_tertiary)
		kw_t_prefix = whitelist_tertiary_regex + optional_word_sep + optional_base_gleason_regex + optional_word_sep
		kw_t_value = score_a_or_b
		kw_t_suffix = default_regex_suffix
		
		# a_kw ---------------------------------------------------------------------
		a_kw_prefix = base_gleason_regex + optional_nondigit_buffer_5
		a_kw_value = score_a_or_b
		a_kw_suffix = optional_word_sep + whitelist_primary_regex
		
		# b_kw ---------------------------------------------------------------------
		## not in use
		# b_kw_prefix = base_gleason_regex + optional_nondigit_buffer_5
		# b_kw_value = score_a_or_b
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

	pattern_dt = pd.concat([additon_dt(), minor_dt(), keyword_dt()])
	pattern_dt["value"] = pattern_dt["value"].apply(lambda x: multiple_alternative_value_matches(x))
	pattern_dt["full_pattern"] = pattern_dt["prefix"] + pattern_dt["value"] + pattern_dt["suffix"]
	pattern_dt.reset_index(drop=True, inplace=True)

	try:
		if (pattern_dt.pattern_name.duplicated().any()):
			raise ValueError('Pattern names should be unique.')
	except Exception as e:
		logger.exception(e)
		raise
	return pattern_dt


# extraction funs ---------------------------------------------------------

def rm_false_positives(x):
	""" Remove false positive matches of gleason scores in text.
	Especially names of fields in text such as "gleason 6 or less" caused false positives.

	Args:
		x (str): text

	Returns:
		str: trimmed text
	"""
	rm = [
		base_gleason_regex + "[ ]?4[ ](ja|tai|or|och|eller)[ ]5",
		"fokaalinen syöpä \\([^)]*\\)",
		"\\(gleason score 6 tai alle\\)"
	]
	for pat in rm:
		x = re.sub(pat,"", x) 
	return x

def prepare_text(x):
	"""Does everything needed to prepare text for the actual extraction; 
	i.e. normalises the text and removes false positives.
	
	Args:
		x (str): text

	Returns:
		str: prepared text
	"""
	x = rm_false_positives(utils.normalise_text(x))
	# to remove certain expressions we know in advance to have no bearing
	# on gleason scores --- to shorten and simplify the text.
	x = re.sub("\\([^0-9]+\\)", " ", x) # e.g. "(some words here)"
	x = re.sub("\\([ ]*[0-9]+[ ]*%[ ]*\\)", " ", x) # e.g. "(45 %)"
	# remove a false positive. this should live in rm_false_positives!
	# e.g. "Is bad (Gleason score 9-10): no"
	re_field_name_gleason_range = "[(][ ]*" + whitelist_gleason_word + "[^0-9]*" + "[5-9][ ]*[-][ ]*([6-9]|(10))" + "[ ]*[)]"
	x = re.sub(re_field_name_gleason_range, " ", x)
		
	return re.sub("[ ]+", " ", x)



def component_parsing_instructions_by_match_type() -> dict[str, list[pd.DataFrame | int]]:
	"""
	Returns:
		dict: keys(str): pattern_name. values(list): dataframe, number of max tries per pattern
	"""
	re_abt = "[3-5]"
	re_c = "([6-9]|10)"
	re_plus = "[^0-9+,(]?[+,][^0-9+,]?"
	re_mask_prefix = "_"
	re_nonmask_prefix = "(^|[^_])"
	re_nonmask_nonplus_prefix = "(^|[^_+])"
	# to avoid e.g. %ORDER=001%; see pattern_extraction.extract_context_affixed_values
	re_nonmask_digit_suffix = "($|(?=[^%0-9]))"
	abtc_dt = pd.DataFrame({
		"pattern_name": ["a","b","t","c"],
		"prefix": [re_nonmask_nonplus_prefix, re_mask_prefix, re_plus, re_nonmask_nonplus_prefix],
		"value": [re_abt, re_abt, re_abt, re_c],
		"suffix": [re_plus, re_nonmask_digit_suffix, re_nonmask_digit_suffix, re_nonmask_digit_suffix]
	})
	abc_dt = pd.DataFrame({
		"pattern_name": ["a","b","c"],
		"prefix": [re_nonmask_prefix, re_mask_prefix, ""],
		"value": [re_abt, re_abt, re_c],
		"suffix": [re_plus, "", ""]
	})
	abt_dt = pd.DataFrame({
		"pattern_name": ["a","b","t"],
		"prefix": [re_nonmask_prefix, re_mask_prefix, re_mask_prefix],
		"value": [re_abt, re_abt, re_abt],
		"suffix": [re_plus, re_plus,""]
	})
	ab_dt = pd.DataFrame({
		"pattern_name": ["a","b"],
		"prefix": [re_nonmask_prefix, re_mask_prefix],
		"value": [re_abt, re_abt],
		"suffix": [re_plus, ""]
	})
	ac_dt = pd.DataFrame({
		"pattern_name": ["a","c"],
		"prefix": [re_nonmask_prefix, re_nonmask_prefix],
		"value": [re_abt, re_c],
		"suffix": ["", ""]
	})
	a_dt = pd.DataFrame({
		"pattern_name": ["a"],
		"prefix": [re_nonmask_prefix],
		"value": [re_abt],
		"suffix": [re_nonmask_digit_suffix]
	})
	b_dt = a_dt.copy()
	b_dt['pattern_name'] = ["b"]
	t_dt = a_dt.copy()
	t_dt['pattern_name'] = ["t"]
	c_dt = a_dt.copy()
	c_dt = pd.DataFrame({
		"pattern_name": ["c"],
		"prefix": [re_nonmask_prefix],
		"value": [re_c],
		"suffix": [re_nonmask_digit_suffix]
	})
	return {#pattern_dt, n_max_tries_per_pattern
		"kw_all_a": [a_dt, 1], 
		"a + b + t = c": [abtc_dt, 10],
		"a + b + t": [abt_dt, 10],
		"a + b = c": [abc_dt, 10],
		"a + b": [ab_dt, 10],
		"a...c": [ac_dt, 10],
		"a": [a_dt, 10],
		"b": [b_dt, 10],
		"c": [c_dt, 10],
		"t": [t_dt, 10]
	}

def parse_gleason_value_string_elements(
		value_strings : list[str] | pd.Series,
		match_types : list[str] | pd.Series
	) -> pd.DataFrame:
	"""Parse Gleason Value Strings.
	
	Separate elements of the Gleason score (A, B, C, T) from strings extracted from text.

	Args:
		`value_strings` (list[str] | pd.Series):
			Strings extracted from text containing (only) components of the Gleason score
		`match_types` (list[str] | pd.Series):
			Each `value_strings` elements must have a corresponding match type;
			e.g. match type `"a + b = c"` is handled differently than match type `"c"`

	Function  works as follows:

	- simple regexes are defined for different match types; e.g. for
	"a + b = c", one or more for regex for a, one or more for b, 
	and one or more for c; e.g. for "a" there are patterns
	`c("[0-9]+[ ]?[+]", "[0-9]+")`

	- the patterns for each element (a, b, t, c) are run separately on a copy of the
	string to be processed. As said above, each element can have one or more.
	The first is used to extract the specified part from the original string,
	and this is saved as the "current version" of the string.
	The next is used to extract the specified part from the "current version"
	to update it. And so on. In the end only an integer should remain.
	
	- after some basic manipulation to get a nice clean table, a table is 
	returned containing the extracted a,b,t,c values for each string.

	- `parse_gleason_value_string_elements` can in rare cases produce
	  problematic results, e.g. `match_type = "a + b"` but only "a" was 
	  succesfully parsed. A few of such problematic results are programmatically
	  identified in `parse_gleason_value_string_elements` --- see output column
	  `warning`.

	Returns:
		`DataFrame` with columns
			`pos` (int): order number of `value_strings`
			`value_string` (str): the `value_strings`
			`match_type`(str): the `match_types`
			`a` (int/float): grade A
			`b` (int/float): grade B
			`t` (int/float): tertiary gleason
			`c` (int/float): scoresum
			`warning` (str): warning message

		The rows are in the same order as `value_strings`. Missing values
		are denoted by np.nan, which is always a float, though if a column
		has no missing values then you have an int column.
	"""
	

	logger.info('Start parsing gleason values')

	try:
		if not (len(value_strings) == len(match_types)):
			raise ValueError('Lengths do not match')	
	except Exception as e:
		logger.exception(e)
		raise

	parsed_dt = None
	instructions_by_match_type = component_parsing_instructions_by_match_type()
	match_type_set = set(instructions_by_match_type.keys()).intersection(set(match_types))
	
	for match_type in match_type_set:
		instructions : list[pd.DataFrame | int] = instructions_by_match_type.get(match_type) # type: ignore
		idx_list = [idx for idx, element in enumerate(match_types) if element == match_type]
		if (len(idx_list) > 0):
			logger.info("Start processing `{match_type}`".format(match_type = match_type))
			text_list = [re.sub("[()]", "", value_strings[i]) for i in idx_list] # type: ignore
			dt = pattern_extraction.extract_context_affixed_values(text_list, pattern_dt = instructions[0], n_max_tries_per_pattern = instructions[1])
			if not dt.empty:
				dt["duplicate_id"] = dt.groupby(["pos", "pattern_name"]).cumcount()+1 # make an unique id for a combinaiton of 'pos' and 'pattern_name'
				dt["value"] = dt["value"].astype('int') # can run pivot_table for numericals only
				pattern_names_extracted = np.unique(dt.pattern_name)
				pattern_names_instructions = instructions[0]['pattern_name']
				# note that pivoting causes here int "value" column values
				# to be turned into float values when there are any missing
				# values in the resulting value columns. this is because
				# missing values are denoted with np.nan which is a float.
				dt = dt.pivot_table(index=["pos","duplicate_id"], columns="pattern_name", values= "value").rename_axis(None, axis=1).reset_index()
				dt.drop("duplicate_id", axis=1, inplace = True) # not needed anymore				
				dt['pos'] = dt.pos.apply(lambda x : idx_list[x])
				dt["match_type"] = match_type
				
				# add warning flags: match type does not match extracted values
				if 'warning' not in dt.columns:
					dt["warning"] = None 
				match_type_mismatch_warning = "!`{msg}`".format(msg = match_type) # extracted values do not match matchtype
				match_type_warning = "match type `{msg}` problem".format(msg = match_type)
				if set(pattern_names_extracted) == set(pattern_names_instructions):
					match_type_mask = (dt[pattern_names_instructions.tolist()].isna().any(axis=1)) # rows having Nan values
					if match_type_mask.any():
						dt.loc[match_type_mask, 'warning'] = dt.loc[match_type_mask, 'warning'].apply(lambda x: x + '||' + match_type_mismatch_warning if x else match_type_mismatch_warning)
						dt['warning'] = dt.warning.apply(lambda x: x + '||' + match_type_warning if x else match_type_warning) # match type broblem generates float values
				else:
					dt['warning'] = dt.warning.apply(lambda x: x + '||' + match_type_warning if x else match_type_warning)
					match_type_mask = (dt[pattern_names_extracted].isna().any(axis=1))
					if match_type_mask.any():
						dt.loc[match_type_mask, 'warning'] = dt.loc[match_type_mask, 'warning'].apply(lambda x: x + '||' + match_type_mismatch_warning if x else match_type_mismatch_warning)
				
				if parsed_dt is None:
					parsed_dt = dt
				else:
					parsed_dt = pd.concat([parsed_dt, dt], ignore_index = True)	

	parsed_dt['pos'] = parsed_dt['pos'].astype('int')				

	# kw_all_a implies a == b, but at this point b is missing.
	kw_all_a_mask = (parsed_dt["match_type"] == "kw_all_a")
	parsed_dt.loc[kw_all_a_mask, 'b'] = parsed_dt.loc[kw_all_a_mask, 'a']

	return parsed_dt.rename_axis('temp_index').sort_values(by=['pos','temp_index'], ignore_index=True) # order by pos and keep the order within pos

		
def extract_gleason_scores(texts, text_ids, format = ["standard", "typed"][0], pattern_dt = fcr_pattern_dt()):
	"""Extract Gleason Scores.

		Runs the extraction itself, parses and formats results.

	Args:
		`texts` (list(str)): texts to process
		`text_ids` (list(int)): identifies each text; will be retained in output
		`format` (list(str)): Defaults to "standard".
		`pattern_dt` (DataFrame, optional): Defaults to fcr_pattern_dt(). With columns
			`pattern_name` (str): one name per pattern
			`match_type` (str): value combination to look for
			`prefix` (str): regex; context prefix for value
			`value` (str): regex; the value itself
			`suffix` (str): regex; context suffix for value
			`full_pattern` (str): regex; prefix + value + suffix

	Raises:
		ValueError: Number of texts and text ids have to match
		ValueError: Format needs to be either "standard" or "typed"
		ValueError: Each match type in pattern_dt have to exist in instructions dict as well
		ValueError: text ids must be unique

	Returns:
		`DataFrame`: Gleason extraction table with columns
			`text_id` (int64): original text_id
			`obs_id` (int64): original text_id extended with the value, 
				which tells the order of appearance in the text. E.g for text_id 1, obs_id is 1001
			`a` (int/NoneType): Most prevalent gleason value
			`b` (int/NoneType): Second most prevalent gleason value
			`t` (int/NoneType): Tertiary gleason value
			`c` (int/NoneType): Gleason score
			`warning` (str): Warning message (match type does not match extracted values, gleason score does not match primary and secondary values)
	In `standard` format, gleason elements are combined so that possible abc values of one component are at the same row.
	Else output contains columns `combination_id`,`value_type`, `value` to link
	values to combinations to texts.
	Non-nan values are int and nan values None.
	"""


	logger.info('Start gleason extraction')
	try:
		if not (isinstance(texts, list) and isinstance(text_ids, list)):
			raise TypeError("Only list is supported for texts and textids")
		if not (len(set(text_ids)) == len(text_ids)):
			raise ValueError("text ids must be unique")
		if not (format == "standard"):	
			raise ValueError('Only "standard" format is implemented. Format was: ' + format)
		if not (len(texts) == len(text_ids)):
			raise ValueError('Number of texts and ids do not match')
		if not (format in ["standard", "typed"]):
			raise ValueError('Format not supported.')
		instruction_keys = list(component_parsing_instructions_by_match_type().keys())
		for m in pattern_dt.match_type:
			if not (m in instruction_keys):
				raise ValueError('Match type did not exist in instructions. See pattern_dt.match_type:', m)	
	except Exception as e:
		logger.exception(e)
		raise

	texts = [prepare_text(text) if text else None for text in texts]
	logger.info('Extract values with context prefixes and suffixes')
	extr_dt = pattern_extraction.extract_context_affixed_values(texts, pattern_dt)
	match_types = dict(zip(pattern_dt.pattern_name, pattern_dt.match_type))
	extr_dt["match_type"] = extr_dt["pattern_name"].map(match_types)
	extr_dt["text_id"] =  [text_ids[i] for i in extr_dt["pos"].values] # attach text ids
	# generate obs_id which is unique for observed component (a row); sequential number for each text
	extr_dt["obs_id"] = extr_dt.groupby("text_id").cumcount()+1 # make unique id for a text
	extr_dt["obs_id"] += extr_dt["text_id"].apply(lambda x : x * 1000) # make unique id in general
	#extr_dt = extr_dt.rename_axis('temp_index').sort_values(by=['pos','obs_id','temp_index'], ignore_index=True) # order by pos and obs_id
	parsed_dt = parse_gleason_value_string_elements(list(extr_dt["value"].map(str)), list(extr_dt["match_type"]))
	text_dict = dict(zip(extr_dt.index, extr_dt.text_id))
	parsed_dt["text_id"] = parsed_dt["pos"].map(text_dict)
	obs_dict = dict(zip(extr_dt.index, extr_dt.obs_id))
	parsed_dt["obs_id"] = parsed_dt["pos"].map(obs_dict)
	for value_col_nm in ["a", "b", "t", "c"]:
		if not value_col_nm in parsed_dt.columns:
			parsed_dt[value_col_nm] = np.nan
	parsed_dt = parsed_dt[~((~parsed_dt.a.isin([2,3,4,5,np.nan])) | (~parsed_dt.b.isin([2,3,4,5,np.nan])) | (~parsed_dt.t.isin([2,3,4,5,np.nan])) | (~parsed_dt.c.isin([4,5,6,7,8,9,10,np.nan])))]
	if (format == "standard"):
		parsed_dt["orig_obs_id"] = parsed_dt.obs_id
		parsed_dt["obs_id"] = parsed_dt.groupby("text_id").cumcount()+1 # make unique id for a text
		parsed_dt["obs_id"] += parsed_dt["text_id"].apply(lambda x : x * 1000) # make unique id in general
		id_dt = dict(zip(parsed_dt.obs_id, parsed_dt.orig_obs_id))
		parsed_dt = utils.typed_format_dt_to_standard_format_dt(parsed_dt)
	parsed_dt = parsed_dt.where(parsed_dt.notnull(), None)

	# add warning flag: gleasonscore does not match primary and secondary values
	if 'warning' not in parsed_dt:
		parsed_dt['warning'] = None
	score_mask = (~parsed_dt[['a','b','c']].isna().any(axis=1) & (parsed_dt.a + parsed_dt.b != parsed_dt.c)) #(parsed_dt.a.notnull() and parsed_dt.b.notnull() and parsed_dt.c.notnull() and
	score_warning = "a + b != c"
	parsed_dt.loc[score_mask, 'warning'] = parsed_dt.loc[score_mask, 'warning'].apply(lambda x: x + '||' + score_warning if x else score_warning)
	
	return parsed_dt
