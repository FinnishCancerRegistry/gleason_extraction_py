import re

# utility functions for tests

def compare_dts(dt1, dt2, cols):
	"""This is a helper function for testing
	It compares dt:s by columns

	Args:
		dt1 (DataFrame): expected
		dt2 (DataFrame): produced
		cols (list(str)): columns to compare against

	Returns:
		DataFrame: Empty Dataframe if dt:s equal by tested columns 
	"""
	comparison_dt = dt1.merge(dt2, indicator=True, how='outer', on = cols)
	diff = comparison_dt[comparison_dt["_merge"] != "both"]
	return diff


def is_substituted(regex, value, substitute = "", remainder = ""):
	"""Check if regex finds and substitues input as expected

	Args:
		regex (str): regex
		value (str): input text
		substitute (str, optional): Defaults to "".
		remainder (str, optional): Defaults to "".

	Returns:
		bool: True if remainder equals string after substitution
	"""
	return re.sub(regex, substitute, value) == remainder
	

def is_found(regex, value, result="don't need this if there is no match"):
	"""Check if regex finds expected string from input

	Args:
		regex (str): regex
		value (str): input text
		result (str, optional): string that regex finds from input text

	Returns:
		bool: False if there in no match
	"""
	res = re.search(regex, value)
	if res:
		return res.group(0) == result
	else:
		return bool(res)
