def twoSum(nums, target):
    """
    1. Two Sum
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


def maxProfit(prices):
    """
    121. Best Time to Buy and Sell Stock
    This takes in a list of prices that represent stock prices each
    day.  It returns an integer of the maximum profit that can be
    achieved if someone buys a stock one day and sells it on a
    different day in the future.  If no profit is possible, it returns
    0.
    """
    # The lowest price
    lowest_price = prices[0]
    max_profit = 0

    # This iterates through each price, and as it does so it keeps
    # track of the maximum profit as well as lowest price up to that
    # point.
    # For the price each day, it subtracts the past lowest price in
    # order to calculate the profit of selling on that day.  If this
    # value is greater than a previously calculated profit amount, a
    # new maximum has been found.  In addition, it checks if the price
    # that day is a new lowest price.
    for price in prices:
        profit = price - lowest_price
        if profit > max_profit:
            # Selling on this day is a new maximum profit.
            max_profit = profit
        if price < lowest_price:
            # The price on this day is a new lowest price.
            lowest_price = price

    # If no profit was possible because the profit each day was 0 or
    # lower, the variable max_profit was never changed from its initial
    # value of 0.  In this case, 0 is returned.  Otherwise, the maximum
    # profit is returned.
    return max_profit


def containsDuplicate(nums):
    """
    217. Contains Duplicate
    This takes in a list of integers, nums.  It returns True if any
    value is present more than once.  Otherwise, it returns False.
    """
    # The format of the dictionary is number: 1.
    mydict = {}

    # This iterates through each number.  It utilizes a dictoinary that
    # stores numbers from the list.  If the current number is a key in
    # the dictionary, it was present in the list the previously.  So,
    # true is returned.  If it is not a key, then it is a new number
    # and it is added to the dictionary.
    for num in nums:
        if num in mydict:
            return True
        else:
            mydict[num] = 1

    # The iteration completed and never exited prematurely by returning
    # True.  This means each value was only present once.  So, false is
    # returned.
    return False


def productExceptSelf(nums):
    """
    238. Product of Array Except Self
    This takes in a list of integers, nums.  It returns a list of
    integers, answer, where answer[i] is equal to the product of every
    element in nums except nums[i].
    """
    # This creates a list, left_to_right_products, where
    # left_to_right_products[i] is equal to the product of every
    # element in nums up to and including nums[i].
    left_to_right_products = [nums[0]]
    for i in range(1, len(nums)):
        left_to_right_products.append(left_to_right_products[i - 1] * nums[i])

    # This creates a list, right_to_left_products, where
    # right_to_left_products[i] is equal to the product of every
    # element in nums after and including nums[i].
    nums_reverse = nums[::]
    nums_reverse.reverse()
    right_to_left_products = [nums_reverse[0]]
    for i in range(1, len(nums_reverse)):
        right_to_left_products.append(right_to_left_products[i - 1] * nums_reverse[i])
    right_to_left_products.reverse()

    # This creates the result list, answer.  It calculates the value
    # at index i by multiplying left_to_right_products[i-1] with
    # right_to_left_products[i+1].  This is equivalent to multiplying
    # the product of all elements before nums[i] with the product of
    # all elements after nums[i].
    # The first element in answer is the product of all elements after
    # nums[0].
    answer = [right_to_left_products[1]]
    for i in range(1, len(nums)-1):
        answer.append(left_to_right_products[i - 1] * right_to_left_products[i + 1])
    # The last element in answer is the product of all elements before
    # the last element of nums.
    answer.append(left_to_right_products[-2])

    return answer
