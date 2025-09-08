import regex as re
import numpy as np

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
	diff = comparison_dt.loc[comparison_dt["_merge"] != "both", :]
	diff = diff.copy()
	diff["_merge"] = diff["_merge"].astype("str")
	diff.loc[diff["_merge"] == "left_only", "_merge"] = "exp_only"
	diff.loc[diff["_merge"] == "right_only", "_merge"] = "obs_only"
	sort_col_nms = np.intersect1d(["text_id", "grp", "_merge"], diff.columns)
	if diff.shape[0] > 0 and len(sort_col_nms) > 0:
		diff.sort_values(by=sort_col_nms, inplace=True)
	diff.rename(columns={"_merge": "source"}, inplace=True)
	diff.reset_index(inplace=True, drop = True)
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
