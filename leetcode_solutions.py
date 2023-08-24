# TOC:
# Arrays
# Bits
# Dynamic Programming
# Strings
# Linked Lists
# Math
# Trees
# Algorithms
# Graphs

import collections


# Arrays
def missingNumber(nums):
    """
    268. Missing Number
    This takes in an array of distinct integers in the range [0, length
    of nums] (both inclusive).  It returns the integer in the range
    that is not included in nums.
    """
    # The sum of the range [0, n] minus the sum of nums will be equal
    # to the value that is missing.
    return sum(range(len(nums)+1)) - sum(nums)


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
    # the dictionary, it was present in the list previously.  So, true
    # is returned.  If it is not a key, then it is a new number and it
    # is added to the dictionary.
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


def maxSubArray(nums):
    """
    53. Maximum Subarray
    This takes in a list of integers, nums.  It returns an integer of
    the sum of the subarray with the largest sum.
    """
    # max_sum is updated to store the largest sum as nums is iterated
    # through.
    max_sum = nums[0]
    # current_sum stores the sum of the current subarray as nums is
    # iterated through.
    current_sum = 0

    # This iterates through nums.  For each element, it adds it to the
    # sum of the current subarray.  That sum is compared to max_sum to
    # check if there is a new largest sum.  If the current sum is ever
    # negative, the current subarray is cleared, because adding those
    # elements to a future subarray will decrease its sum.
    for num in nums:
        current_sum += num
        if current_sum > max_sum:
            max_sum = current_sum
        if current_sum < 0:
            current_sum = 0

    return max_sum


def maxProduct(nums):
    """
    152. Maximum Product Subarray
    This takes in a list of integers, nums.  It returns an integer of
    the product of the subarray with the largest product.
    """
    # max_product is updated to store the largest product as nums is
    # iterated through.
    max_product = nums[0]
    # current_max and current_min store the product of the current
    # maximum and minimum subarrays as nums is iterated through.
    current_max = 1
    current_min = 1

    # This iterates through nums and keeps track of the product of two
    # current subarrays, one for the maximum and one for the minimum.
    # For each iteration, there are three possible values:
    # current_max * the element, current_min * the element, and the
    # element itself.  The current max and min are updated each time
    # based on these values.
    # The current max is needed to check if there is a new max product.
    # The current min is needed in case the next element in the
    # iteration is negative, which results in the value becoming
    # positive and possibly greater than the max product.
    # If the element is ever 0, the current subarrays are cleared and
    # max and min are reset to 1, because the product of a subarray
    # containing 0 will always be 0.
    for num in nums:
        current_product = current_max * num
        current_max = max(current_product, current_min*num, num)
        current_min = min(current_product, current_min*num, num)
        if current_max > max_product:
            max_product = current_max
        if num == 0:
            current_max = 1
            current_min = 1

    return max_product


def findMin(nums):
    """
    153. Find Minimum in Rotated Sorted Array
    This takes in a rotated sorted array of integers with unique values
    It returns an integer of the minimum value in the array. It runs in
    O(log n) time.
    """
    # This stores the current subarray.
    current_nums = nums[::]

    # This continues to loop until the minimum value is found.  Each
    # iteration, it reduces the size of the current subarray by half.
    while True:
        # This checks the base cases where the current subarray only
        # has 1 or 2 elements.
        if len(current_nums) == 1:
            return current_nums[0]
        if len(current_nums) == 2:
            if current_nums[0] < current_nums[1]:
                return current_nums[0]
            return current_nums[1]
        # This checks if the current subarray is sorted, so the minimum
        # value is the first element.
        if current_nums[0] < current_nums[-1]:
            return current_nums[0]
        # The current subarray has 3 or more elements and is not
        # sorted.  So, the min has to be somewhere after the first
        # element.  This divides the current subarray in half,
        # resulting in a left subarray and right subarray.
        # The min is in the left subarray if its first element is
        # greater than its last element, because that means the values
        # are increasing and then drop to a lower value once it reaches
        # the min element.  On the other hand, the min is in the right
        # subarray if the left subarray's first element is less than
        # its last element, because that means the left subarray is
        # always increasing and never drops to a lower value by
        # reaching the min element.
        middle_index = int(len(current_nums) / 2)
        if current_nums[0] > current_nums[middle_index]:
            # The min is in the left subarray.
            current_nums = current_nums[:middle_index+1]
        else:
            # The min is in the right subarray.
            current_nums = current_nums[middle_index+1:]


def search(nums, target):
    """
    33. Search in Rotated Sorted Array
    This takes in a rotated sorted array of integers with unique
    values, nums, and a single integer, target.  If the target is in
    nums, it returns its index.  Otherwise, it returns -1.  It runs in
    O(log n) time.
    """
    # This stores the indices of the current subarray.
    start_index = 0
    end_index = len(nums) - 1

    # This continues to loop until the target value is found or a base
    # case is reached.  Each iteration, it reduces the size of the
    # current subarray by half.
    while True:
        # This checks the base cases where the current subarray only
        # has 1 or 2 elements.
        if start_index == end_index:
            if nums[start_index] == target:
                return start_index
            return -1
        if start_index == (end_index - 1):
            if nums[start_index] == target:
                return start_index
            if nums[end_index] == target:
                return end_index
            return -1
        # The current subarray has 3 or more elements.  This divides
        # the current subarray in half, resulting in left and right
        # subarrays.  Only one subarray will be sorted, where the first
        # element is less than the last element.  So, it checks if the
        # target value can be within the side that is sorted.  If it
        # can be, the current subarray is updated.  Otherwise, the
        # current subarray is updated to the other side.
        middle_index = int((start_index + end_index) / 2)
        # This checks if the middle element is the target.
        if nums[middle_index] == target:
            return middle_index
        if nums[start_index] < nums[middle_index]:
            # The left subarray is sorted.
            if nums[start_index] <= target < nums[middle_index]:
                end_index = middle_index - 1
            else:
                start_index = middle_index + 1
        else:
            # The right subarray is sorted.
            if nums[middle_index] < target <= nums[end_index]:
                start_index = middle_index + 1
            else:
                end_index = middle_index - 1


def twoSum(numbers, target):
    """
    167. Two Sum II - Input Array Is Sorted
    This takes in an sorted ascending array of integers, numbers, and a
    single integer, target.  It returns a list of two integers that are
    the indices plus 1 of values in numbers which sum to target.  The
    lesser index is the first element, and the greater index is the
    second element.  The same index cannot be used twice.
    """
    left_index = 0
    right_index = len(numbers) - 1

    # This continues to iterate until the target is found.  It adds the
    # left and right elements of the array.  It checks if the value
    # equals target, and if so the indices plus 1 are returned.  If the
    # sum is less than the target, the value needs to increase.  So,
    # the left element is updated to the element to its right.  If the
    # sum is greater than the target, the value needs to decrease.  So,
    # the right element is updated to the element to its left.
    while True:
        sum = numbers[left_index] + numbers[right_index]
        if sum == target:
            return [left_index+1, right_index+1]
        if sum < target:
            left_index += 1
        else:
            right_index -= 1


def threeSum(nums):
    """
    15. 3Sum
    This takes in a list of integers, nums.  It returns a list where
    each element is a list of three different elements from nums that
    sum to 0.  The result list does not contain duplicate triplets.
    """
    result = []
    nums.sort()

    # For each element of nums, this uses a subarray of every element
    # after the current element.  It finds all pairs of elements that
    # when combined with the current element sum to 0.  This triplet is
    # then added to the result list.
    for i in range(len(nums)-2):
        left_index = i + 1
        # If the new current element equals the previous current
        # element, it will result in duplicate triplets.  So, this
        # continues to iterate until a different current element is
        # found.
        if i > 0 and nums[i] == nums[i - 1]:
            continue
        right_index = len(nums) - 1
        target = -1 * nums[i]
        # This finds all pairs that sum to the opposite of the current
        # element.  It uses the method from 167. Two Sum II - Input
        # Array Is Sorted.  However, since it needs to find all pairs,
        # when a solution is found it continues to iterate.
        while left_index != right_index:
            sum = nums[left_index] + nums[right_index]
            if sum == target:
                result.append([nums[i], nums[left_index], nums[right_index]])
                left_index += 1
                # If the new left element equals the previous left
                # element, it will result in a duplicate pair.  So,
                # this continues to iterate until a different left
                # element is found.
                while (
                    nums[left_index] == nums[left_index - 1]
                    and left_index != right_index
                ):
                    left_index += 1
            elif sum < target:
                # The sum of the pair is too small.  So, to increase
                # the value this moves one end to a larger value.
                left_index += 1
            else:
                # The sum of the pair is too large.  So, to decrease
                # the value this moves one end to a smaller value.
                right_index -= 1

    return result


def maxArea(height):
    """
    11. Container With Most Water
    This takes in a list of integers.  The values represent heights of
    vertical lines.  This returns an integer that is the maximum area
    which can be created by connecting two heights with a horizontal
    line.
    """
    left_index = 0
    right_index = len(height) - 1
    max_area = 0

    # This starts with the lines at each end.  It calculates the area
    # to determine if there is a new max.  It then moves inward by
    # going to the next line left or right of the smaller line and
    # continues to iterate until the ends meet.
    while left_index != right_index:
        # This calculates the current area and compares it to the past
        # max.
        width = right_index - left_index
        smaller_height = min(height[left_index], height[right_index])
        area = width * smaller_height
        if area > max_area:
            max_area = area
        # This moves one end inward.
        if height[right_index] < height[left_index]:
            right_index -= 1
        else:
            left_index += 1

    return max_area


def countFairPairs(nums, lower, upper):
    """
    2563. Count the Number of Fair Pairs
    This takes in a list of integers, nums, and two integers, lower and
    upper.  It returns an integer that is the number of pairs of
    different elements whose sum is in the range [lower, upper] both
    inclusive.
    """
    # This takes in a sorted list of integers, sorted_list, and an
    # integer, value.  It returns an integer that is the number of
    # pairs of different elements whose sum is lower than value.
    def countPairsLessThan(sorted_list, value):
        count = 0
        left_index = 0
        right_index = len(sorted_list) - 1
        # This starts at the ends of the list.  It moves inward and
        # adds the difference between the indices whose elements sum to
        # less than value to count.
        while left_index < right_index:
            # This keeps shifting the right pointer to the left until
            # the left plus right elements are less than value.
            while (
                    left_index < right_index
                    and sorted_list[left_index] + sorted_list[right_index] >= value
            ):
                right_index -= 1
            # This adds the right minus left indices to count, because
            # if you keep the left element constant then you can make
            # pairs whose sum is less than value with every element
            # from the right element (inclusive) to the left element
            # (not inclusive since you can't use the same element in a
            # pair).
            count += right_index - left_index
            # This shifts the left pointer to the right by one element
            # and repeats the process.
            left_index += 1
        return count

    # nums can be sorted, because the count of pairs will be the same.
    nums.sort()
    # This finds the number of pairs whose sum is less than upper plus 1.
    pairs_less_than_or_equal_to_upper = countPairsLessThan(nums, upper+1)
    # This finds the number of pairs whose sum is less than lower.
    pairs_less_than_lower = countPairsLessThan(nums, lower)
    # The pairs less than or equal to upper include the correct pairs
    # as well as incorrect pairs that sum to less than lower.  So, this
    # subtracts the number of incorrect pairs to get the result.
    return pairs_less_than_or_equal_to_upper - pairs_less_than_lower


def runningSum(nums):
    """
    1480. Running Sum of 1d Array
    This takes in a list of integers.  It returns a list of integers
    where each element is the sum of the corresponding element and all
    elements before it in nums.
    """
    result = [nums[0]]

    # This iterates through the elements of nums.  It adds the current
    # element and the previous sum then stores the result.
    for i in range(1, len(nums)):
        result.append(result[i-1]+nums[i])

    return result


def pivotIndex(nums):
    """
    724. Find Pivot Index
    This takes in a list of integers.  It returns an integer that is
    the leftmost index where the sum of all the elements to the left is
    equal to the sum of all the elements to the right.  For the first
    and last indices, the left and right sums are 0 respectively.  If
    there is no index that satisfies the conditions, it returns -1.
    """
    # This creates a dictionary where dictionary[i] is the sum of
    # nums[i] and every element to its right.
    dictionary = dict()
    # The right sum for the last index is 0.
    dictionary[len(nums)] = 0
    # The value for dictionary[0] does not need to be calculated,
    # because the leftmost index that can be returned is 0, and the
    # right sum for this index starts at index 1.
    for i in range(len(nums)-1, 0, -1):
        dictionary[i] = nums[i] + dictionary[i + 1]

    # This iterates through nums and keeps track of the left sum.  If
    # the left sum is equal to the right sum of an index, it returns
    # that index.
    left_sum = 0
    for i in range(len(nums)):
        if left_sum == dictionary[i + 1]:
            return i
        left_sum += nums[i]

    # There is no index that satisfies the conditions.
    return -1


# Bits
def getSum(a, b):
    """
    371. Sum of Two Integers
    This takes in two integers, a and b.  It returns an integer that is
    the sum of a and b without using + or -.

    This calculates the exclusive or of a and b to handle bit addition
    1 + 0 = 1 and 0 + 1 = 1.
    It also calculates the and of a and b then shifts the bits to the
    left by 1 bit to handle bit addition 1 + 1 = 0 carry over a 1.
    No calculation is needed for bit addition 0 + 0 = 0 since the
    result is 0 in both calculations.
    The sum of these two integers is equal to the sum of a and b.
    However, since + cannot be used, this continues to perform the
    calculations on the two new integers until the and value is 0.
    The solution code is written in Java because of how Python
    represents the bits of negative integers:
    while (b != 0) {
        int xor = a ^ b;
        b = (a & b) << 1;
        a = xor;
    }

    return a;
    """
    pass


def hammingWeight(n):
    """
    191. Number of 1 Bits
    This takes in the binary representation of an integer.  It returns
    the number of 1 bits.
    """
    count = 0

    # This iterates through the bits by shifting it to the right by 1
    # bit each loop.  It calculates whether n is even or odd.  If it is
    # odd, the rightmost bit (current bit element) is 1.
    while n > 0:
        if n % 2 == 1:
            count += 1
        n = n >> 1

    return count


def countBits(n):
    """
    338. Counting Bits
    This takes in an integer.  It returns an array of integers of
    length n + 1.  The elements of the array are the number of 1 bits
    in the integers from 0 to n.
    """
    result = []

    # offset is used to index the result array by subtracting it from
    # the current element.
    offset = 0
    # The offset value needs to change whenever the current element is
    # the result of 2 to the power of a number.
    next_offset = 1
    # This iterates through the integers from 0 to n (including n).  It
    # uses dynamic programming to find the number of 1 bits in each
    # element.
    for num in range(n+1):
        # This is the base case.
        if num == 0:
            result.append(0)
            continue
        # This runs whenever the current offset reaches the new offset
        # and needs to be updated.
        if num == next_offset:
            offset = next_offset
            next_offset = offset * 2
        # Whenever the current element is the result of 2 to the power
        # of a number (1, 2, 4, 8, 16, ...), the leftmost bit (without
        # padding) is 1 and all of the remaining bits are 0 (1, 10,
        # 100, 1000, 10000, ...).  All of these elements have a single
        # 1 bit.  They are the offset values, because all of the
        # elements between them can be made by adding the offset value
        # bits with the bits of the previous elements starting from 0
        # (4 100 = 4 100 + 0 0, 5 101 = 4 100 + 1 1, 6 110 = 4 100 + 2
        # 10, 7 111 = 4 100 + 3 11, 8 1000 = 8 1000 + 0 0, 9 1001 = 8
        # 1000 + 1 1, 10 1010 = 8 1000 + 2 10, 11 1011 = 8 1000 + 3 11,
        # ...).  This means the number of 1 bits for an integer = 1 +
        # number of 1 bits in the integer current element - current
        # offset.  These were calculated in previous elements, so it is
        # the same as 1 + result[current element - current offset].
        # The current offset is updated in the if block above whenever
        # the current element is the result of 2 to the power of a
        # number (offset 2 for elements 2-3; offset 4 for elements 4-7,
        # offset 8 for elements 8-15, ...).
        result.append(1+result[num-offset])

    return result


def reverseBits(n):
    """
    190. Reverse Bits
    This reverses the bits in a 32 bit unsigned integer.
    """
    result = 0

    # This finds the bits of n from right to left by shifting n to the
    # right by the current loop number (i) and then calculating and (&)
    # 1.  To put each bit in the correct spot of the result, it is
    # shifted to the left by 31 - i and then added to the result.
    for i in range(32):
        bit = (n >> i) & 1
        bit = bit << (31 - i)
        result += bit

    return result


# Dynamic Programming
def climbStairs(n):
    """
    70. Climbing Stairs
    There is a staircase with n steps.  You can move up 1 or 2 steps.
    This calculates how many different ways there are to climb the
    staircase and returns the integer.
    """
    # This is a base case.
    if n == 1:
        return 1

    # This uses the bottom up approach and tabulation.
    different_ways_list = [0] * (n+1)
    # For 1 step, there is 1 way to climb to the top.
    different_ways_list[1] = 1
    # For 2 steps, there are 2 ways to climb to the top.
    different_ways_list[2] = 2
    # To reach the i step, there are only two options.  You take a one
    # step from the i-1 step or you take a two step from the i-2 step.
    # So, the ways to reach the i step is the sum of the ways to reach
    # the i-1 and i-2 steps.
    for i in range(3, n+1):
        different_ways_list[i] = different_ways_list[i-1] + different_ways_list[i-2]

    return different_ways_list[n]


def coinChange(coins, amount):
    """
    322. Coin Change
    This takes in a list of integers, coins.  It also takes in a single
    integer, amount.  It calculates and returns the integer that is the
    least number of coins to make up the amount.  If the amount cannot
    be made from the coins, it returns -1.
    """
    # This uses the bottom up approach and tabulation.
    least_coins_list = [amount + 1] * (amount+1)

    # For 0 amount, it takes 0 coins.
    least_coins_list[0] = 0
    # Each iteration, it uses the coins less than or equal to the
    # current amount.  For each coin, it calculates 1 (the coin) plus
    # the number of coins to make the current amount minus the coin.
    # If this is a new minimum, it is put in the list.
    # If the current amount cannot be formed from the coins, its value
    # in the list stays as the initialized value, amount + 1.
    for current_amount in range(1, amount+1):
        for coin in coins:
            if current_amount - coin >= 0:
                least_coins_list[current_amount] = min(
                    least_coins_list[current_amount],
                    1 + least_coins_list[current_amount - coin]
                )

    if least_coins_list[amount] == (amount + 1):
        # The amount cannot be formed from the coins.
        return -1

    return least_coins_list[amount]


def lengthOfLIS(nums):
    """
    300. Longest Increasing Subsequence
    This takes in a list of integers.  It returns an integer that is
    the length of the longest increasing subsequence (an array that can
    be derived from another array by deleting some or no elements
    without changing the order of the remaining elements).
    """
    # This uses the bottom up approach and tabulation.
    longest_lengths = [0] * len(nums)

    # For the last element in nums, the length of the longest possible
    # subsequence is just the element itself, so 1.
    longest_lengths[-1] = 1
    # This iterates through the elements of nums in reverse order.  For
    # each element, it calculates the length of the longest subsequence
    # that can be made starting with the element.
    for i in range(len(nums)-2, -1, -1):
        # The longest subsequence length may be just the element itself.
        max_length = 1
        # The second element of the subsequence can be any of the
        # elements after the current element, as long as it is greater.
        # The subsequence length is the current element (1) plus the
        # longest length starting with the second element.  So, the
        # longest length for the current element is the max of all the
        # subsequence lengths.
        for j in range(i+1, len(nums)):
            if nums[j] > nums[i] and (1 + longest_lengths[j]) > max_length:
                max_length = 1 + longest_lengths[j]
        longest_lengths[i] = max_length

    # Out of all the longest subsequence lengths, the result is the
    # highest one.
    return max(longest_lengths)


def longestCommonSubsequence(text1, text2):
    """
    1143. Longest Common Subsequence
    This takes in two strings of all lowercase letters.  It returns an
    integer of the length of the longest subsequence that can be formed
    out of both strings.  If there is no common subsequence, it returns
    0.
    """
    # This uses the bottom up approach and tabulation with a 2D matrix
    # (each element of the list is an inner list that is one of the
    # rows).  The rows of the matrix represent 0 (empty string) and
    # then the letters of the first string, text1.  The columns
    # represent 0 and then the letters of the other string, text2.
    matrix = []
    for _ in range(len(text1)+1):
        matrix.append([None] * (len(text2)+1))

    # This iterates through the first string, text1.  For each of these
    # iterations, it then iterates through the second string, text2.
    # For both iterations, the substring is the current character and
    # everything before it.  If the current characters equal each
    # other, the length of the common subsequence is 1 + the length of
    # the common subsequence for the substrings before the current
    # characters.  This is 1 + matrix[row-1][col-1].  If the characters
    # do not equal each other, there are two options.  It can be the
    # length of the common subsequence between the substring before
    # text1's current character and the current text2 substring.  Or,
    # it can be the length of the common subsequence between the
    # current text1 substring and the substring before text2's current
    # character.  This is the max of matrix[row-1][col] and
    # matrix[row][col-1].
    for row in range(len(matrix)):
        for col in range(len(matrix[0])):
            if row == 0 or col == 0:
                # Any string and an empty string has no common subsequence.
                matrix[row][col] = 0
            elif text1[row - 1] == text2[col - 1]:
                matrix[row][col] = 1 + matrix[row - 1][col - 1]
            else:
                matrix[row][col] = max(matrix[row - 1][col], matrix[row][col - 1])

    return matrix[-1][-1]


def wordBreak(s, wordDict):
    """
    139. Word Break
    This takes in a string, s, and a list of strings, wordDict.  It
    returns true if the string can be made by joining together elements
    from wordDict.  Otherwise, it returns false.
    """
    # This uses the bottom up approach and tabulation.
    result = [False] * (len(s)+1)
    result[len(s)] = True

    # This iterates through the characters of s in reverse order.  For
    # each character, it then iterates through the strings of wordDict.
    # If it does not cause an index out of bounds error, it checks if
    # the string equals the same length of characters starting at the
    # current character.  If they are equal, those characters in s can
    # be made from an element in wordDict.  In addition, if the
    # characters after those characters can also be made
    # (result[current index + len(word)] is True), then it is still
    # possible for the string to be made and the value for that spot
    # needs to be set to True for later iterations
    # (result[current index] is set to True).
    for i in range(len(s)-1, -1, -1):
        for word in wordDict:
            if (i + len(word)) <= len(s) and word == s[i: i + len(word)]:
                if result[i + len(word)]:
                    result[i] = True

    return result[0]


def combinationSum4(nums, target):
    """
    377. Combination Sum IV
    This takes in a list of unique integers, nums, and an integer,
    target.  It  returns an integer that is the number of combinations
    made up of elements from nums that sum to target.
    """
    # This uses the bottom up approach and tabulation.
    result = [0] * (target+1)
    result[0] = 1

    # This iterates from 1 to the target (inclusive).  For each value,
    # it then iterates through nums.  If the num is less than or equal
    # to the value, new combinations will be the num added to all of
    # the combinations that sum to the value minus the num.  This is
    # result[value - num].  This applies to every element of nums, so
    # all of those combinations are added together to get the total
    # number of combinations for the current value.
    for i in range(1, target+1):
        counter = 0
        for num in nums:
            if num <= i:
                counter += result[i - num]
        result[i] = counter

    return result[-1]


def rob(nums):
    """
    198. House Robber
    This takes in a list of integers, nums, that represent the amount
    of money in houses on a street.  Two adjacent houses cannot be
    robbed.  This returns an integer of the max amount of money that
    can be stolen from the street.
    """
    # This uses the bottom up approach and tabulation.
    result = [0] * (len(nums)+1)
    result[0] = 0
    result[1] = nums[0]

    # This iterates through nums.  For each current num, there are two
    # options.  One option is to rob the house, so the thief gets the
    # money from the house plus the max amount of money from robbing
    # houses up to the previous house (not inclusive).  This is
    # result[current index - 2].  The other option is to not rob the
    # house, so the thief gets the max amount of money from robbing
    # houses up to the previous house (inclusive).  This is
    # result[current index - 1].  The larger amount of money is the max
    # amount of money from robbing houses up to that point.
    for i in range(2, len(nums)+1):
        money = max(nums[i-1]+result[i-2], result[i-1])
        result[i] = money

    return result[-1]


def rob(nums):
    """
    213. House Robber II
    This takes in a list of integers, nums, that represent the amount
    of money in houses on a circular street.  Two adjacent houses
    cannot be robbed, and the last house in nums is adjacent with the
    first house.  This returns an integer of the max amount of money
    that can be stolen from the street.
    """
    def inner_rob(nums):
        # This uses the bottom up approach and tabulation.
        result = [0] * (len(nums)+1)
        result[0] = 0
        result[1] = nums[0]

        # This iterates through nums.  For each current num, there are two
        # options.  One option is to rob the house, so the thief gets the
        # money from the house plus the max amount of money from robbing
        # houses up to the previous house (not inclusive).  This is
        # result[current index - 2].  The other option is to not rob the
        # house, so the thief gets the max amount of money from robbing
        # houses up to the previous house (inclusive).  This is
        # result[current index - 1].  The larger amount of money is the max
        # amount of money from robbing houses up to that point.
        for i in range(2, len(nums)+1):
            money = max(nums[i-1]+result[i-2], result[i-1])
            result[i] = money

        return result[-1]

    # This is the base case.
    if len(nums) == 1:
        return nums[0]

    # To ensure the first and last house are not both robbed, this runs
    # the house robber function on two subarrays.  The first subarray
    # is all of the elements of nums except the last.  The second
    # subarray is all of the elements except the first.  The max amount
    # that can be robbed is the larger value.
    return max(inner_rob(nums[:len(nums)-1]), inner_rob(nums[1:]))


def numDecodings(s):
    """
    91. Decode Ways
    This takes in a string, s, of digits.  Using the coding 1=A, 2=B,
    and so on, it returns an integer that is the number of different
    ways to decode the string.
    """
    # These are the base cases.
    if s[0] == '0':
        return 0
    if len(s) == 1:
        return 1

    # This uses the bottom up approach and tabulation.  results[i]
    # represents the number of ways to decode the substring of s
    # starting at s[i] and going to the end of the string.
    results = [0] * len(s)
    # This sets the number of ways for the last index.
    if s[-1] == '0':
        results[-1] = 0
    else:
        results[-1] = 1
    # This sets the number of ways for the second to last index.
    if s[-2] == '0':
        results[-2] = 0
    elif int(s[-2]+s[-1]) <= 26:
        # The last two characters are numbers from 10-26 inclusive.
        # If it is 10 or 20, the number of ways is 1 + 0 = 0.  For the
        # other numbers, it is 1 + 1 = 2.
        results[-2] = 1 + results[-1]
    else:
        # The last two characters is a number 27 or greater.  For
        # multiples of 10, the number of ways is 0 since it is greater
        # than 26 and the two characters cannot be decoded
        # individually.  Otherwise, the number of ways is 1.
        results[-2] = results[-1]

    # To start to decode a string, you can start with either the first
    # character if it is not 0 or the first two characters if it is
    # less than 27 since either option may be the first letter.  So,
    # the number of ways to decode a string is the number of ways to
    # decode the substring coming after the first character plus the
    # number of ways to decode the substring coming after the first
    # two characters.  This is results[i + 1] + results[i + 2].
    for i in range(len(s)-3, -1, -1):
        if s[i] == '0':
            results[i] = 0
        else:
            # This gets the number of ways to decode the substring
            # coming after the first character.
            result = results[i + 1]
            # This adds the number of ways to decode the substring
            # coming after the first two characters if it is less than
            # 27.
            if int(s[i]+s[i+1]) <= 26:
                result += results[i + 2]
            results[i] = result

    return results[0]


# Strings
def reverseString(s):
    """
    344. Reverse String
    This takes in a string and reverses it in-place.  It returns None.
    """
    left_index = 0
    right_index = len(s) - 1

    # This starts at the ends of the string.  It swaps the elements and
    # then moves inward.  It repeats this process until all of the
    # elements have been swapped (length of s is even) or it reaches
    # the middle character (length of s is odd).
    while left_index < right_index:
        # This swaps the elements.
        left_element = s[left_index]
        s[left_index] = s[right_index]
        s[right_index] = left_element
        # This moves the indices inward.
        left_index += 1
        right_index -= 1

    return None


def lengthOfLongestSubstring(s):
    """
    3. Longest Substring Without Repeating Characters
    This takes in a string.  It returns an integer that is the length
    of the longest substring that does not contain duplicate
    characters.
    """
    max_length = 0
    left_index = 0
    char_set = set()

    # This uses a window of a substring that does not contain
    # duplicates.  It also uses a set of the characters in the window.
    # It then iterates through the characters of s.  If the character
    # is already in the window, it slides the left side of the window
    # until the duplicate character is gone.  Based on the new window,
    # it checks if there is a new longest length.
    for right_index in range(len(s)):
        while s[right_index] in char_set:
            char_set.remove(s[left_index])
            left_index += 1
        char_set.add(s[right_index])
        max_length = max(max_length, right_index - left_index + 1)

    return max_length


def longestPalindrome(s):
    """
    5. Longest Palindromic Substring
    This takes in a string.  It returns a string of the longest
    substring of s that is a palindrome.
    """
    result = ''

    # This iterates through each character in the string.  It starts
    # with the character and expands outward by one character in both
    # directions.  If the substring is a palindrome, it checks if it is
    # a new longest substring.  It then continues to expand until the
    # substring is not a palindrome.
    # This loop handles cases when the longest substring is an odd
    # length.
    for i in range(len(s)):
        left_index = i
        right_index = i
        while (
                left_index >= 0
                and right_index < len(s)
                and s[left_index] == s[right_index]
        ):
            if len(s[left_index:right_index+1]) > len(result):
                result = s[left_index: right_index + 1]
            left_index -= 1
            right_index += 1

    # This loop handles cases when the longest substring is an even
    # length.  It uses the same process as above, except it starts with
    # both the character and the character to the right.
    for i in range(len(s)):
        left_index = i
        right_index = i + 1
        while (
                left_index >= 0
                and right_index < len(s)
                and s[left_index] == s[right_index]
        ):
            if len(s[left_index:right_index+1]) > len(result):
                result = s[left_index: right_index + 1]
            left_index -= 1
            right_index += 1

    return result


def isIsomorphic(s, t):
    """
    205. Isomorphic Strings
    This takes in two strings.  It returns True if the characters in s
    can be replaced to form t.  All occurrences of the character must
    be replaced with the same character.  Also, multiple characters
    cannot be replaced with the same character.  If t cannot be formed,
    it returns False.
    """
    mapping = dict()
    replacement_characters = dict()

    # This iterates through the characters of s.  It either creates or
    # checks the mapping to the corresponding character in t.
    for i in range(len(s)):
        if s[i] not in mapping:
            # It is a new character.
            if t[i] in replacement_characters:
                # The character is new, but the replacement character has
                # already been used.
                return False
            # The replacement character has not been used, so this
            # creates the mapping between s and t.  It also stores the
            # character from t in order for future iterations to check
            # if multiple characters are being replaced by the same
            # character.
            mapping[s[i]] = t[i]
            replacement_characters[t[i]] = 1
        elif mapping[s[i]] != t[i]:
            # It is not a new character, so the character from s must be
            # replaced by the mapped character from t.
            return False

    return True


def isSubsequence(s, t):
    """
    392. Is Subsequence
    This takes in two strings.  It returns True if s is a subsequence
    (a string that can be derived from another string by deleting some
    or no elements without changing the order of the remaining
    elements) of t.  Otherwise, it returns False.
    """
    s_index = 0
    t_index = 0

    # This continues to iterate through t and sometimes s until it
    # reaches the end of either string.
    while s_index < len(s) and t_index < len(t):
        # The current character of s has been found in t.  So, this
        # iterates to the next character of s.
        if s[s_index] == t[t_index]:
            s_index += 1
        t_index += 1

    if s_index == len(s):
        # All of the characters of s were found in t.
        return True
    return False


def longestPalindrome(s):
    """
    409. Longest Palindrome
    This takes in a string.  It returns an integer that is the length
    of the longest palindrome that can be formed using the characters
    (case sensitive) in s.
    """
    dictionary = dict()
    # This creates a dictionary out of the characters in s in the
    # format char: count of char.
    for char in s:
        if char in dictionary:
            dictionary[char] += 1
        else:
            dictionary[char] = 1

    result = 0
    # This is a flag that is set if there is at least one count that is
    # odd.
    contains_odd_count = False
    # This iterates through the character dictionary.  If the count is
    # even, it is added to the result length.  If the count is odd, the
    # count minus 1 is added to the result.
    for char in dictionary:
        if dictionary[char] % 2 == 0:
            result += dictionary[char]
        else:
            result += dictionary[char] - 1
            contains_odd_count = True

    if contains_odd_count:
        # There is at least one character with an odd count.  So, one more
        # character can be added in the middle of the palindrome.
        return result + 1
    return result


def mergeAlternately(word1: str, word2: str) -> str:
    """
    1768. Merge Strings Alternately
    This merges the two strings by putting letters in alternating
    order, starting with word1.  If the lengths are unequal, the
    remaining characters from the longer string are added to the end.
    """
    result = ''
    word1_index = 0
    word2_index = 0

    # This iterates through the strings and adds the next character.
    while word1_index < len(word1) or word2_index < len(word2):
        # These add the remaining characters if one string is longer
        # than the other.
        if word1_index >= len(word1):
            result += word2[word2_index]
            word2_index += 1
            continue
        if word2_index >= len(word2):
            result += word1[word1_index]
            word1_index += 1
            continue
        # This adds the next two characters from the strings.
        result += word1[word1_index]
        result += word2[word2_index]
        word1_index += 1
        word2_index += 1

    return result


def gcdOfStrings(self, str1: str, str2: str) -> str:
    """
    1071. Greatest Common Divisor of Strings
    This returns the largest string that can be concatenated with
    itself zero or more times to equal both of the given strings.  If
    there is no possible string, it returns an empty string.
    """
    # This finds the smaller and larger of the two given strings.
    if len(str1) < len(str2):
        smaller_string = str1
        larger_string = str2
    else:
        smaller_string = str2
        larger_string = str1

    # The current string starts with the entire smaller string.  This
    # determines if the current string can be concatenated with itself
    # to equal both of the given strings.  Each iteration, the current
    # string loses its last character.
    for offset in range(len(smaller_string)):
        current_string = smaller_string[:len(smaller_string)-offset]
        # This checks if the length of the current string is a multiple
        # of both of the lengths of the given strings.
        if (
            len(smaller_string) % len(current_string) != 0
            or len(larger_string) % len(current_string) != 0
        ):
            continue
        # This checks if the current string concatenated to the lengths
        # of the given strings equals them.
        if (
            current_string * int(
                len(smaller_string) / len(current_string)
            ) == smaller_string
            and current_string * int(
                len(larger_string) / len(current_string)
            ) == larger_string
        ):
            return current_string

    # There was no possible largest string.
    return ''


# Linked Lists
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next


def mergeTwoLists(list1, list2):
    """
    21. Merge Two Sorted Lists
    This takes in the head nodes of two sorted linked lists.  It merges
    them into a single sorted linked list and returns the head node.
    """
    # These are the base cases.
    if list1 is None:
        return list2
    if list2 is None:
        return list1

    head_node = None
    current_node = None
    list1_current_node = list1
    list2_current_node = list2

    # This iterates through the two linked lists.  Each iteration, it
    # adds the smaller value to the result linked list.
    while True:
        if list1_current_node is None and list2_current_node is None:
            # The loop has iterated through both of the linked lists,
            # so the merging is finished.
            return head_node

        if list1_current_node is None:
            # The loop has iterated through list1, but list2 still has
            # remaining values.
            new_node = ListNode(val=list2_current_node.val)
            list2_current_node = list2_current_node.next
        elif list2_current_node is None:
            # The loop has iterated through list2, but list1 still has
            # remaining values.
            new_node = ListNode(val=list1_current_node.val)
            list1_current_node = list1_current_node.next
        elif list1_current_node.val < list2_current_node.val:
            # The list1 value is less than the list2 value, so it
            # should be added to the result linked list.
            new_node = ListNode(val=list1_current_node.val)
            list1_current_node = list1_current_node.next
        else:
            # The list2 value is less than or equal to the list1 value,
            # so it should be added to the result linked list.
            new_node = ListNode(val=list2_current_node.val)
            list2_current_node = list2_current_node.next

        if head_node is None:
            # This runs for the first element added to the result
            # linked list.
            head_node = new_node
            current_node = head_node
        else:
            # This runs every time after the first element has been
            # added to the result linked list.
            current_node.next = new_node
            current_node = current_node.next


def addTwoNumbers(l1, l2):
    """
    2. Add Two Numbers
    This takes in two non-empty linked lists.  Each list represents the
    digits of a non-negative integer in reverse order.  This adds the
    integers together and stores the digits in reverse order in a new
    linked list.  It returns the head of the new list.
    """
    head_node = None
    current_node = None
    l1_current_node = l1
    l2_current_node = l2
    remainder = 0

    # This iterates through the digits of the integers in reverse
    # order.  It performs addition, carries over any remainder, and
    # adds the sum digit to the result linked list.
    while True:
        # This sets the l1 and 2 digits.
        if l1_current_node is None:
            l1_digit = 0
        else:
            l1_digit = l1_current_node.val
            l1_current_node = l1_current_node.next
        if l2_current_node is None:
            l2_digit = 0
        else:
            l2_digit = l2_current_node.val
            l2_current_node = l2_current_node.next

        # This performs the addition to get the sum digit and carries
        # over any remainder.
        current_digit = l1_digit + l2_digit + remainder
        if current_digit >= 10:
            new_node = ListNode(val=current_digit-10)
            remainder = 1
        else:
            new_node = ListNode(val=current_digit)
            remainder = 0

        # This sets the digit as a new node in the result linked list.
        if head_node is None:
            head_node = new_node
            current_node = head_node
        else:
            current_node.next = new_node
            current_node = current_node.next

        # The loop has iterated through both l1 and l2, and there is no
        # remainder to add as the last digit.
        if (
                l1_current_node is None
                and l2_current_node is None
                and remainder == 0
        ):
            return head_node


def reverseList(head):
    """
    206. Reverse Linked List
    This takes in the head node of a linked list.  It creates a new
    linked list with the elements in reverse order and returns the head
    node of this list.
    """
    current_node = head
    previous_node = None
    if head is not None:
        next_node = head.next
    # This iterates through the list.  It keeps track of the current,
    # previous, and next nodes.  Each iteration, it sets the previous
    # node as the element after the current node.
    while current_node is not None:
        # This saves a reference to the next element in the iteration.
        next_node = current_node.next
        # This sets the previous node as the element after the current
        # node and updates the node variables for the next iteration.
        current_node.next = previous_node
        previous_node = current_node
        current_node = next_node
        if current_node is not None:
            next_node = current_node.next

    return previous_node


def middleNode(head):
    """
    876. Middle of the Linked List
    This takes in the head node of a linked list.  It returns the
    middle node of the list.  If the list has an even number of
    elements, it returns the second middle node.
    """
    # This determines the index of the middle element.
    count = 1
    current_node = head
    # This counts how many elements are in the list.
    while current_node.next is not None:
        current_node = current_node.next
        count += 1
    middle_index = int(count/2)

    # This iterates until it reaches the middle element.
    current_node = head
    for _ in range(middle_index):
        current_node = current_node.next

    return current_node


def detectCycle(head):
    """
    142. Linked List Cycle II
    This takes in the head node of a linked list.  If there is a cycle
    in the list, where the tail node is connected to a previous node in
    the list, it returns the start node of the cycle.  Otherwise, it
    returns None.
    """
    # This is the base case.
    if head is None:
        return None

    dictionary = dict()
    current_node = head
    # This iterates until it determines there is a cycle or it reaches
    # the end of the list.
    while current_node.next is not None:
        if current_node in dictionary:
            # The start of the cycle has been found, since the current
            # node has already been iterated over.
            return current_node
        # The current node has not been iterated over, so it is stored.
        dictionary[current_node] = 1
        current_node = current_node.next

    # The iteration reached the tail of the list and there is no next
    # node, which means there is no cycle.
    return None


# Math
def reverse(x):
    """
    7. Reverse Integer
    This takes in an integer in the range [-2^31, 2^31 - 1] both
    inclusive.  It reverses the digits and returns the new integer if
    it is also within the range, otherwise it returns 0.
    """
    number = str(abs(x))
    result = 0
    multiplier = 1

    # This iterates through the digits of x from left to right.  Each
    # iteration, it adds the digit times its place in the new integer
    # (ones, tens, hundreds, ...) as long as the sum does not go over
    # the max range.
    for digit in number:
        remaining_space = (2**31 - 1) - result
        amount_to_add = int(digit) * multiplier
        if amount_to_add > remaining_space:
            # Reversing the digits results in an integer greater than
            # then max range.
            return 0
        result += amount_to_add
        multiplier *= 10

    # x is negative, so the new integer also needs to be negative.
    if x < 0:
        return result * -1
    # x is positive.
    return result


# Trees
class Node:
    def __init__(self, val=None, children=None):
        self.val = val
        self.children = children


class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


def preorder(root):
    """
    589. N-ary Tree Preorder Traversal
    This takes in the root node of a tree.  It returns a list of
    integers that are the values of the nodes in the tree.  The order
    of the elements is based on pre-order traversal (perform the
    operation on the node, then traverse to the left subtree, then to
    the right subtree).
    """
    def pre_order_traversal(node, result):
        """
        This takes in the root node of a tree, node, and an empty list,
        result.  It traverses through the tree using pre-order
        traversal, and for each node it adds the value to the end of
        the list.
        """
        if node is not None:
            result.append(node.val)
            for child in node.children:
                pre_order_traversal(child, result)

    # This performs the pre-order traversal on the root node.
    result = []
    pre_order_traversal(root, result)

    return result


def levelOrder(root):
    """
    102. Binary Tree Level Order Traversal
    This takes in the root node of a binary tree.  It returns a 2D list
    of integers that are the values of the nodes in the tree.  The
    order of the elements is based on level-order traversal (start at
    the root and perform the operation on it, then go to the next level
    and perform the operation on each node from left to right).  Each
    nested list contains the values from each level in the tree.
    """
    def level_order_traversal(tree_node, result):
        """
        This takes in the root node of a binary tree, tree_node, and an
        empty list, result.  It traverses through the tree using
        level-order traversal, and for each node in a level it adds it
        to a temporary list.  Once the end of a level is reached, it
        adds the tempoary list to the end of result.
        """
        queue = collections.deque()
        queue.append(tree_node)
        # This uses a queue and iterates through entire levels of the
        # tree.
        while len(queue) != 0:
            nested_list = []
            level_length = len(queue)
            # level_length is the number of nodes in a level.  This
            # iterates through each node in the level by dequeueing
            # level_length number of times.  For each node, it adds the
            # value to the tempoary list, nested_list, and then
            # enqueues the left and right nodes for the next level
            # iteration.
            for _ in range(level_length):
                current_tree_node = queue.popleft()
                if current_tree_node is not None:
                    nested_list.append(current_tree_node.val)
                    queue.append(current_tree_node.left)
                    queue.append(current_tree_node.right)
            # The entire level has been traversed.
            if len(nested_list) > 0:
                result.append(nested_list)

    # This performs the level-order traversal on the root node.
    result = []
    level_order_traversal(root, result)

    return result


def isValidBST(root):
    """
    98. Validate Binary Search Tree
    This takes in the root node of a binary tree.  It returns True if
    the tree is a binary search tree, where every value in a node's
    left subtree is less than the node's value, every value in a
    node's right subtree is greater than the node's value, and the
    subtrees are also binary search trees.  Otherwise, it returns
    False.
    """
    def in_order_traversal(node, values):
        """
        This takes in the root node of a binary tree, node, and an
        empty list, values.  It traverses through the tree using
        in-order traversal (traverse the left subtree, then perform the
        operation on the node, then traverse to the right subtree), and
        for each node it adds the value to the end of the list.
        """
        if node is not None:
            in_order_traversal(node.left, values)
            values.append(node.val)
            in_order_traversal(node.right, values)

    # This performs in-order traversal on the root node.
    values = []
    in_order_traversal(root, values)

    # In-order traversal on a binary search tree results in all of the
    # values being in sorted ascending order.  So, this checks if the
    # list, values, is sorted.
    previous_value = values[0]
    # This iterates through the elements of values.  It checks if the
    # current element is not greater than the previous element.
    for i in range(1, len(values)):
        if values[i] <= previous_value:
            # The list, values, is not in sorted ascending order.  So,
            # it is not a binary search tree.
            return False
        previous_value = values[i]

    # The list, values, is in sorted ascending order.  So, it is a
    # binary search tree.
    return True


def lowestCommonAncestor(root, p, q):
    """
    235. Lowest Common Ancestor of a Binary Search Tree
    This takes in the root node of a binary search tree, root, and two
    nodes in the tree, p and q.  It returns the lowest node in the tree
    that has both p and q as descendants (a node can be a descendant of
    itself).
    """
    current_node = root
    # This iterates through each level of the tree.  If the values for
    # p and q are greater than the current node, it moves to the right
    # child.  If the values are less, it moves to the left child.
    # Otherwise, p and q are in different subtrees of the current node.
    # Or, the current node is p or q.  For either scenario, the current
    # node is the lowest node.
    while True:
        if p.val > current_node.val and q.val > current_node.val:
            current_node = current_node.right
        elif p.val < current_node.val and q.val < current_node.val:
            current_node = current_node.left
        else:
            return current_node


# Algorithms
def search(nums, target):
    """
    704. Binary Search
    This takes in a sorted ascending list of integers, nums, and an
    integer, target.  If target is in nums, it returns an integer that
    is the index, otherwise it returns -1.
    """
    first_index = 0
    last_index = len(nums) - 1
    # This uses binary search.  It keeps track of a subarray of nums.
    # Each iteration, it compares the target to the middle element of
    # the subarray.  If the target is not found and it is less than the
    # middle element, the subarray is updated to the left half,
    # otherwise it is updated to the right half.
    while first_index <= last_index:
        middle_index = int((first_index+last_index)/2)
        if target == nums[middle_index]:
            return middle_index
        if target < nums[middle_index]:
            last_index = middle_index - 1
        else:
            first_index = middle_index + 1

    return -1


def isBadVersion(version):
    """
    This is defined to not cause errors in the function
    firstBadVersion.
    """
    pass


def firstBadVersion(n):
    """
    278. First Bad Version
    This takes in an integer representing a list of versions in order
    from 1 to n.  isBadVersion(int) returns True or False if the int
    representing the version is good or bad.  All versions after a bad
    version are also bad.  This returns an integer of the first bad
    version.
    """
    first_version = 1
    last_version = n
    # This keeps track of a subarray of n.  Each iteration, it checks
    # if the middle version of the subarray is good or bad.  If it is
    # good and the version after it is bad, it returns the bad version.
    # Otherwise, the subarray is updated to the right half.  If it is
    # bad and the first version or the version before it is good, it
    # returns the bad version.  Otherwise, the subarray is updated to
    # the left half.
    while first_version <= last_version:
        middle_version = int((first_version+last_version)/2)
        if not isBadVersion(middle_version):
            # The middle version is good.
            if isBadVersion(middle_version+1):
                return middle_version + 1
            first_version = middle_version + 1
        else:
            # The middle version is bad.
            if middle_version == 1:
                return 1
            if not isBadVersion(middle_version-1):
                return middle_version
            last_version = middle_version - 1


# Graphs
def floodFill(image: list[list[int]], sr: int, sc: int,
              color: int) -> list[list[int]]:
    """
    733. Flood Fill
    This takes in a 2-D list of integers, image, and integers sr, sc,
    and color.  sr and sc represent the starting row and column
    indices.  This updates the starting spot value to the value of
    color.  If any spot up, down, left, or right of the starting spot
    has the same value as the original value of the starting spot, it
    is also updated to color.  This pattern continues for any of those
    spots that are updated.  Once all possible spots are updated, the
    new image is returned.
    """
    def flood_fill_recursion(image: list[list[int]], row: int, col: int,
                             starting_color: int, new_color: int) -> None:
        """
        This takes in a 2-D list of integers, image, and integers row,
        col, starting_color, and new_color.  row and col represent the
        row and column indices of the current spot.  If the current
        spot value matches starting_color, it is updated to the value
        of new_color.  It then repeats this process up, down, left, and
        right of the updated spot.
        """
        # This checks if the spot does not exist.
        if row < 0 or row >= len(image) or col < 0 or col >= len(image[0]):
            return None

        # The spot exists, so if the value is the same as
        # starting_color, it is updated to new_color and the process
        # repeats from the updated spot.
        if image[row][col] == starting_color:
            image[row][col] = new_color
            flood_fill_recursion(image, row-1, col, starting_color, new_color)
            flood_fill_recursion(image, row+1, col, starting_color, new_color)
            flood_fill_recursion(image, row, col-1, starting_color, new_color)
            flood_fill_recursion(image, row, col+1, starting_color, new_color)

    # This is the base case, when the starting spot value already
    # equals color.
    if image[sr][sc] == color:
        return image

    # This updates the starting spot and all connected spots.
    flood_fill_recursion(image, sr, sc, image[sr][sc], color)

    return image


def numIslands(grid: list[list[str]]) -> int:
    """
    200. Number of Islands
    This takes in a 2-D list of strings of either '1' for land or '0'
    for water.  Islands are formed from land connected vertically or
    horizontally and surrounded on each side by water or a spot that
    does not exist.  This returns an integer of the number of islands.
    """
    def find_island(grid: list[list[str]], row: int, col: int,
                    visited: set((int, int))) -> None:
        """
        This takes in a 2-D list of strings of either '1' or '0', grid,
        integers, row and col, and a set, visited.  row and col
        represent the indices of the current spot in grid.  visited
        contains tuples of integers row, col representing spots already
        checked in grid.  This takes the current spot and finds all of
        connected spots with values of '1'.  It adds them as tuples to
        visited.
        """
        # This checks if the spot does not exist or has already been
        # visited.
        if (
                row < 0
                or row >= len(grid)
                or col < 0
                or col >= len(grid[0])
                or (row, col) in visited
        ):
            return None

        # The spot exists, so if the value is '1' it is added to
        # visited.  The process repeats for the spots up, down, left,
        # and right.
        if grid[row][col] == '1':
            visited.add((row, col))
            find_island(grid, row-1, col, visited)
            find_island(grid, row+1, col, visited)
            find_island(grid, row, col-1, visited)
            find_island(grid, row, col+1, visited)

    result = 0
    visited = set()
    # This iterates through each spot.  If it has not already been
    # visited and is not water, it is a new island.
    for row in range(len(grid)):
        for col in range(len(grid[row])):
            if (row, col) not in visited and grid[row][col] != '0':
                result += 1
                # This adds all of the spots of the island to visited.
                find_island(grid, row, col, visited)

    return result
