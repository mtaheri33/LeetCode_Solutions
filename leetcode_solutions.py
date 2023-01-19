# 1. Two Sum
def twoSum(nums, target):
    """
    This takes in a list of integers, nums, and a single integer,
    target.  It returns a list of two integers that are the indices of
    values in nums which sum to target.  The two indices in the list
    can be in any order.  The same index cannot be used twice.
    """
    # The format of the dictionary is number: index.
    mydict = {}
    # This iterates through nums.  For each value, it calculates the
    # difference that is target - value.  It checks if that difference
    # was a previous value in nums.  If it was, then the solution is
    # the indices of the current number and the previous value.  If it
    # was not, then the current value is added to the dictionary and
    # the loop continues.
    for i in range(len(nums)):
        current_number = nums[i]
        difference = target - current_number
        if difference in mydict:
            # The difference was a previous value in nums.  This
            # returns the current index and the index of the previous
            # value. 
            return [i, mydict[difference]]
        else:
            # The difference was not a previous value in nums.  This
            # adds the number and its index to the dictionary.
            mydict[current_number] = i
