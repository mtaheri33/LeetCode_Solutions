import collections
import heapq
import math
import random
import string
from typing import Union


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


def twoSum(nums: list[int], target: int) -> list[int]:
    """
    1. Two Sum
    This returns a list of two indices from nums where the elements sum
    to target.  The same index is not used twice.
    """
    # The key is an element from nums, and the value is the index of
    # that element.
    past_nums = {}
    # This iterates through each element of nums.  It checks if there
    # has been a previous element that when summed with the current
    # element equals target.  If so it returns the two indices.
    # Otherwise, it stores the current element and its index in the
    # dictionary.
    for i in range(len(nums)):
        num = nums[i]
        difference = target - num
        if difference in past_nums:
            return [i, past_nums[difference]]
        else:
            past_nums[num] = i


def maxProfit(prices: list[int]) -> int:
    """
    121. Best Time to Buy and Sell Stock
    This takes in a list of prices that represent stock prices each
    day.  It calculates and returns the maximum profit that can be
    achieved if someone buys a stock one day and sells it on a
    different day in the future.  If no profit is possible, it returns
    0.
    """
    max_profit = 0
    min_price = prices[0]

    # This iterates from the second to the last price.  For the current
    # price, it subtracts the past lowest price in order to calculate
    # the profit of selling on that day.  It then checks if this is a
    # new max profit.  It also checks if the current price is a new
    # lowest price.
    for i in range(1, len(prices)):
        max_profit = max(max_profit, prices[i] - min_price)
        min_price = min(min_price, prices[i])

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


def productExceptSelf(nums: list[int]) -> list[int]:
    """
    238. Product of Array Except Self
    This calculates and returns a list where list[i] is the product of
    every element in nums except nums[i].
    """
    # This creates two lists.  Each element i of the first list is the
    # product of all the elements in nums from the start to i both
    # inclusive.  Each element i of the second list is the product of
    # all of the elements in nums from the end to i both inclusive.
    left_to_right = [nums[0]]
    for i in range(1, len(nums)):
        left_to_right.append(nums[i] * left_to_right[-1])
    right_to_left = [nums[-1]]
    for i in range(len(nums)-2, -1, -1):
        right_to_left.append(nums[i] * right_to_left[-1])
    right_to_left.reverse()

    # For each element of nums, this multiplies the product of all the
    # elements before it with the product of all the elements after it.
    result = [right_to_left[1]]
    for i in range(1, len(nums)-1):
        result.append(left_to_right[i-1] * right_to_left[i+1])
    result.append(left_to_right[-2])

    return result


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


def twoSum(numbers: list[int], target: int) -> list[int]:
    """
    167. Two Sum II - Input Array Is Sorted
    numbers needs to be sorted ascending.  This finds two distinct
    elements in numbers that sum to target.  It returns the indices + 1
    of the elements in a list.
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
        total = numbers[left_index] + numbers[right_index]
        if total == target:
            return [left_index+1, right_index+1]
        if total < target:
            left_index += 1
        else:
            right_index -= 1


def threeSum(nums: list[int]) -> list[list[int]]:
    """
    15. 3Sum
    This returns a list where each element is a list of three different
    elements from nums that sum to 0.  The result list does not contain
    duplicate triplets.
    """
    result = []
    nums.sort()

    # This is a base case when there are only negative elements, so any
    # triplet will sum to a negative number.
    if nums[-1] < 0:
        return result

    # This iterates through each element up to and including the third
    # to last one, or to the first positive element, whichever comes
    # first.  It finds all pairs after the current element that when
    # combined with the current element sum to 0.
    for i in range(len(nums)-2):
        # This stops at the first positive element, because any triplet
        # will sum to a positive number.
        if nums[i] > 0:
            return result
        # This skips over elements that are the same as the previous
        # element, because it will result in duplicate triplets.
        if i > 0 and nums[i] == nums[i-1]:
            continue
        left_index = i + 1
        right_index = len(nums) - 1
        # This starts at the ends of the subsection coming after the
        # current element.  It moves inward until the ends meet and
        # searches for pairs that when combined with the current
        # element sum to 0.  If the pair sum is too large, then it
        # needs to decrease, so the right end moves inward to the left,
        # which is a smaller value.  If the pair sum is too small, then
        # it needs to increase, so the left end moves inward to the
        # right, which is a larger value.  If the pair sum is the
        # target amount, the left end moves inward in order to continue
        # searching for more pairs.
        while left_index < right_index:
            if nums[i] + nums[left_index] + nums[right_index] == 0:
                # This checks to make sure duplicate triplets are not
                # added again.
                if (
                        len(result) == 0
                        or [nums[i], nums[left_index],
                            nums[right_index]] != result[-1]
                ):
                    result.append(
                        [nums[i], nums[left_index], nums[right_index]]
                    )
            if nums[i] + nums[left_index] + nums[right_index] > 0:
                right_index -= 1
            else:
                left_index += 1

    return result


def maxArea(height: list[int]) -> int:
    """
    11. Container With Most Water
    The elements of height represent heights of vertical lines at x
    positions equal to the indices + 1 of the elements.  This returns
    the maximum area that can be created by connecting two heights with
    a horizontal line.
    """
    max_area = 0
    left_index = 0
    right_index = len(height) - 1
    # This starts with the lines at each end and moves inward by going
    # to the next line left or right of the smaller line, until the
    # ends meet.  Each iteration, it checks if there is a new max area.
    while left_index < right_index:
        max_area = max(
            max_area,
            (right_index-left_index) * min(height[left_index],
                                           height[right_index])
        )
        if height[left_index] <= height[right_index]:
            left_index += 1
        else:
            right_index -= 1

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


def pivotIndex(nums: list[int]) -> int:
    """
    724. Find Pivot Index
    This finds the leftmost index of nums where the sum of all the
    elements to the left equals the sum of all the elements to the
    right.  If there are no elements to the left or right, the sum is
    0.  It then returns the index.  If there is no possible index, it
    returns -1.
    """
    # This stores the right sum of each element.
    right_sums = {len(nums)-1: 0}
    for i in range(len(nums)-2, -1, -1):
        right_sums[i] = right_sums[i+1] + nums[i+1]

    left_sum = 0
    # This iterates through each element and finds the left sum.  It
    # then compares that to the right sum to check if they are equal.
    for i in range(len(nums)):
        if left_sum == right_sums[i]:
            return i
        left_sum += nums[i]

    return -1


def kidsWithCandies(candies: list[int], extraCandies: int) -> list[bool]:
    """
    1431. Kids With the Greatest Number of Candies
    Each element of candies represents the amount of candy a kid has.
    This creates and then returns a list of booleans.  The ith element
    is True if the ith element of candies plus extraCandies is greater
    than or equal to the max number in candies.  Otherwise, it is
    False.
    """
    max_candies = max(candies)
    result = []

    for i in range(len(candies)):
        result.append(candies[i] + extraCandies >= max_candies)

    return result


def canPlaceFlowers(flowerbed: list[int], n: int) -> bool:
    """
    605. Can Place Flowers
    flowerbed represents plots where a flower is either planted (1) or
    it is not (0).  Two flowers cannot be planted next to each other.
    This returns True if n new flowers can be planted in the flowerbed.
    Otherwise, it returns False.
    """
    # This handles the base case.
    if len(flowerbed) == 1:
        if flowerbed[0] == 0:
            # Only 1 or 0 flowers can be planted.
            return n <= 1
        # There is already a flower in the plot, so only 0 flowers can
        # be planted.
        return n == 0

    # This handles the first element.
    if flowerbed[0] == 0 and flowerbed[1] == 0:
        flowerbed[0] = 1
        n -= 1

    # This iterates through the flowerbed between the first and last
    # plots.  A new flower can be planted if the plot is empty and the
    # plots before and after it are empty.
    for i in range(1, len(flowerbed)-1):
        if flowerbed[i] == 0 and flowerbed[i-1] == 0 and flowerbed[i+1] == 0:
            flowerbed[i] = 1
            n -= 1

    # This handles the last element.
    if flowerbed[len(flowerbed)-1] == 0 and flowerbed[len(flowerbed)-2] == 0:
        n -= 1

    return n <= 0


def increasingTriplet(nums: list[int]) -> bool:
    """
    334. Increasing Triplet Subsequence
    This returns True if there are three indicies i, j, and k where
    i < j < k and nums[i] < nums[j] < nums[k].  Otherwise, it returns False.
    """
    # This is a base case when there aren't enough elements for a
    # triplet subsequence.
    if len(nums) <= 2:
        return False

    # The value of s1 will be less than or equal to s2, and most of the
    # time the index of s1 will be before the index of s2.
    s1 = nums[0]
    s2 = 2 ** 31

    # This iterates through all of the elements and updates the values
    # for s1 and s2.
    for num in nums:
        # This checks if there is an element after s2 with a greater
        # value than s2.
        if num > s2:
            return True
        # The element is less than or equal to s2.  If it is greater
        # than s1, then s2 is set to the element, because now there is
        # a smaller value that is still greater than s1 and comes after
        # s1.
        if num > s1:
            s2 = num
        # The element is less than or equal to s1.  So, set s1 to the
        # element, because it is a smaller value.  This can cause some
        # iterations where the index of s1 comes after the index of s2,
        # until s2 is set to a new value.  That is okay, because the
        # old s1 value is still less than the s2 value and comes before
        # s2.
        else:
            s1 = num

    # No increasing triplet subsequence exists.
    return False


def moveZeroes(self, nums: list[int]) -> None:
    """
    283. Move Zeroes
    This moves the zeroes in nums to the end of the list.
    """
    count = nums.count(0)

    # This removes all of the zeroes from nums.
    for _ in range(count):
        nums.remove(0)

    # This adds the zeroes that were removed to the end of the list.
    for _ in range(count):
        nums.append(0)

    return None


def maxOperations(nums: list[int], k: int) -> int:
    """
    1679. Max Number of K-Sum Pairs
    This finds the most pairs of elements in nums that sum to k.  Each
    element can only be used in one pair.
    """
    counts = {}
    # This creates a dictionary of the elements in nums and their
    # counts.
    for num in nums:
        if num in counts:
            counts[num] += 1
        else:
            counts[num] = 1

    operations = 0
    used_numbers = set()
    # This iterates through each element of nums to find pairs that sum
    # to k.
    for num in nums:
        # This checks if the element has already been iterated over.
        if num in used_numbers:
            continue

        difference = k - num
        # This checks if the element and its pair are the same.  In
        # this case, the count should not be added to the number of
        # operations, because it has to be used twice to sum to k.  So,
        # half of the count rounded down is added to the number of
        # operations.
        if difference == num:
            operations += counts[num] // 2
        # This checks if there is a pair element in nums that sums to
        # k.  If so, it adds the lower count of the element or the pair
        # element to the number of operations.
        elif difference in counts:
            operations += min(counts[num], counts[difference])
            used_numbers.add(difference)
        used_numbers.add(num)

    return operations


def findMaxAverage(nums: list[int], k: int) -> float:
    """
    643. Maximum Average Subarray I
    This finds a contiguous subarray of nums with length k that has the
    highest average.  It then returns that average.
    """
    # This calculates the average of the subarray starting with the
    # first element.
    start_index = 0
    end_index = start_index + k - 1
    subarray_sum = sum(nums[start_index:end_index+1])
    max_avg = subarray_sum / k

    # This uses a sliding window of a k length subarray.  It calculates
    # the average and checks if it is a new max.
    while end_index < len(nums) - 1:
        subarray_sum -= nums[start_index]
        start_index += 1
        end_index += 1
        subarray_sum += nums[end_index]
        max_avg = max(max_avg, subarray_sum/k)

    return max_avg


def longestOnes(nums: list[int], k: int) -> int:
    """
    1004. Max Consecutive Ones III
    nums is made up of 0s and 1s.  This finds the largest contiguous
    subarray of 1s if at most k number of 0s can be considered 1s.  It
    returns the length of the subarray.
    """
    max_ones = 0
    start_index = 0
    end_index = 0

    # This uses start and end pointers to represent the current
    # subarray as it iterates through nums.  Each time, it checks if
    # there is a new largest contiguous subarray.
    while end_index < len(nums):
        # This checks if the new element in the subarray is a 0, so k
        # is decremented since the element is considered a 1.
        if nums[end_index] == 0:
            k -= 1

        # This runs when more than k number of 0s in the subarray are
        # considered 1s.  It increments the start index until k number
        # of 0s are in it.
        while k < 0:
            if nums[start_index] == 0:
                k += 1
            start_index += 1

        max_ones = max(max_ones, end_index - start_index + 1)

        end_index += 1

    return max_ones


def longestSubarray(nums: list[int]) -> int:
    """
    1493. Longest Subarray of 1's After Deleting One Element
    nums is made up of 0s and 1s.  This finds the largest contiguous
    subarray of 1s after one element is deleted.  It returns the length
    of the subarray.
    """
    max_ones = 0
    start_index = 0
    end_index = 0
    # This stores how many 0s can be deleted from the subarray.
    k = 1

    # This uses start and end pointers to represent the current
    # subarray as it iterates through nums.  Each time, it checks if
    # there is a new largest contiguous subarray.
    while end_index < len(nums):
        # This checks if the new element in the subarray is a 0, so k
        # is decremented since the element is deleted.
        if nums[end_index] == 0:
            k -= 1

        # This runs when more than one 0 in the subarray has been
        # deleted.  It increments the start index until only one 0 has
        # been deleted.
        while k < 0:
            if nums[start_index] == 0:
                k += 1
            start_index += 1

        max_ones = max(max_ones, end_index - start_index + 1)

        end_index += 1

    # The one element is not actually deleted from nums, so it is
    # included in the length of the largest subarray.  Therefore, it
    # needs to be subtracted out.
    return max_ones - 1


def largestAltitude(gain: list[int]) -> int:
    """
    1732. Find the Highest Altitude
    A person starts a trip at altitude 0 with multiple points.  Each
    element in gain is the change in altitude from the previous point
    (the previous point of the first element is the start with altitude
    of 0).  This returns the highest altitude of the points.
    """
    altitudes = [0]

    # This calculates the altitudes of the points.
    for net_gain in gain:
        altitudes.append(altitudes[-1] + net_gain)

    return max(altitudes)


def findDifference(nums1: list[int], nums2: list[int]) -> list[list[int]]:
    """
    2215. Find the Difference of Two Arrays
    This returns a list with 2 elements.  The first is a list of the
    distinct integers in nums1 that are not in nums2.  The second is a
    list of the distinct integers in nums2 that are not in nums1.
    """
    return [
        list(
            set(nums1).difference(set(nums2))
        ),
        list(
            set(nums2).difference(set(nums1))
        )
    ]


def uniqueOccurrences(arr: list[int]) -> bool:
    """
    1207. Unique Number of Occurrences
    This returns True if each of the number of occurrences of the
    values in arr is unique.  Otherwise, it returns False.
    """
    occurrences = collections.Counter(arr)

    # This compares the numbers of occurrences with a set of the values
    # to check if any duplicates were removed.  So, the lengths will
    # not be equal.
    return len(list(occurrences.values())) == len(set(occurrences.values()))


def equalPairs(grid: list[list[int]]) -> int:
    """
    2352. Equal Row and Column Pairs
    This takes in a grid.  It counts the number of pairs of a row and
    column that equal each other by having the same elements in the
    same order.  It then returns the count.
    """
    # This creates a list of tuples for the columns in the grid.
    columns = []
    for i in range(len(grid)):
        column = []
        for j in range(len(grid)):
            column.append(grid[j][i])
        columns.append(tuple(column))

    # This creates a dictionary of the unique columns in the grid.  The
    # key is a tuple of the elements in the column, and the value is
    # how many times that column is in the grid.
    column_counts = collections.Counter(columns)

    # This iterates through each row.  It checks if there is an equal
    # column, and if so it adds the number of times that column is in
    # the grid to the count.
    pairs = 0
    for i in range(len(grid)):
        if tuple(grid[i]) in column_counts:
            pairs += column_counts[tuple(grid[i])]

    return pairs


def asteroidCollision(asteroids: list[int]) -> list[int]:
    """
    735. Asteroid Collision
    asteroids contains elements representing asteroids all moving along
    the same space at the same speed.  The sign gives the direction of
    the asteroid, where positive is right and negative is left.  The
    absolute value of the element is its size.  Eventually, some
    asteroids collide.  The smaller size asteroid is destroyed.  If
    they are the same size, both asteroids are destroyed.  This returns
    a list of asteroids after all of the collisions have occured.
    """
    stack = []

    i = 0
    # This iterates through the asteroids in order to add the ones that
    # never collide and remove the ones that will be destroyed in a
    # collision.
    while i < len(asteroids):
        # This checks if the asteroids will collide.  The latest
        # asteroid has to be moving right, and the current asteroid has
        # to be moving left.
        if (
                len(stack) > 0
                and stack[-1] > 0
                and asteroids[i] < 0
        ):
            # The asteroids will collide.  This determines which one
            # will get destroyed.
            if abs(asteroids[i]) < abs(stack[-1]):
                # The current asteroid will be destroyed.
                i += 1
            elif abs(asteroids[i]) == abs(stack[-1]):
                # Both asteroids will be destroyed.
                stack.pop()
                i += 1
            else:
                # The latest asteroid will be destroyed, and the
                # current asteroid will continue to move left.
                stack.pop()

        else:
            # The asteroids will never collide.  So, the current
            # asteroid is added to the stack and becomes the latest
            # asteroid.
            stack.append(asteroids[i])
            i += 1

    return stack


def findKthLargest(nums: list[int], k: int) -> int:
    """
    215. Kth Largest Element in an Array
    This finds and returns the kth largest element in nums.
    """
    # This splits the current subsection of nums into three groups: one
    # with elements less than the partition value, one with elements
    # equal, and one with elements greater.  The partition value is the
    # value of the last element in the current subsection.  It then
    # determines which group the kth largest element is in and updates
    # the current subsection to that group.
    while True:
        partition_value = nums[-1]
        less_group = [element for element in nums
                      if element < partition_value]
        equal_group = [element for element in nums
                       if element == partition_value]
        greater_group = [element for element in nums
                         if element > partition_value]

        # This runs when the kth largest element is in the greater
        # group.  k does not need to be updated, because removing the
        # elements in the equal and less group from the current
        # subsection has no impact on the kth largest element in the
        # greater group.
        if len(greater_group) >= k:
            nums = greater_group
            continue
        # This runs when the kth largest element is in the equal group.
        # Since all of the elements are the same, the first one can be
        # returned.
        if len(equal_group) >= k - len(greater_group):
            return equal_group[0]
        # This runs when the kth largest element is in the less group.
        # k needs to be updated, because the kth largest element in the
        # current subsection comes after the equal and greater groups
        # elements.  Those elements will be removed, so k needs to be
        # reduced in order for it to still point to the same kth
        # largest element.
        nums = less_group
        k = k - len(greater_group) - len(equal_group)


def guess(x):
    """
    This is defined to not cause errors in the function guessNumber.
    """
    pass


def guessNumber(n: int) -> int:
    """
    374. Guess Number Higher or Lower
    This finds and returns a picked number from 1 to n (both
    inclusive).
    """
    start = 1
    end = n
    # This continues to run until it finds the picked number.  It keeps
    # track of a subsection of the possible numbers.  Each iteration,
    # the subsection is reduced in length by half.
    while True:
        current_guess = int((start + end) / 2)
        result = guess(current_guess)
        if result == 0:
            return current_guess
        if result == 1:
            start = current_guess + 1
        else:
            end = current_guess - 1


def successfulPairs(spells: list[int], potions: list[int],
                    success: int) -> list[int]:
    """
    2300. Successful Pairs of Spells and Potions
    spells represents strengths of spells and potions represents
    strengths of potions.  This returns a list where list[i] is equal
    to the number of potions where its strength can be multiplied with
    spells[i] to get a value greater than or equal to success.
    """
    # This creates a new spells list where each element is (original
    # index, spells element).  It is sorted desc by spells element
    # value.
    sorted_spells = [spell for spell in enumerate(spells)]
    sorted_spells.sort(key=lambda spell: spell[1], reverse=True)
    potions.sort()

    pairs = [-1] * len(spells)
    potions_index = 0
    # This iterates through each spell.  It calculates the minimum
    # potion strength needed to have the product be equal to or greater
    # than success.  It then iterates through potions until it finds
    # that strength value.  The number of potions that can pair with
    # the spell is the length of potions minus the index of the current
    # potion, since potions is sorted ascending.  Spell strength must
    # iterate in descending order, because this will result in minimum
    # potion strength targets in ascending order, which matches the
    # order of elements in potions.
    for original_index, spell_strength in sorted_spells:
        # This checks if there are no possible potion strengths that
        # can be multiplied with the spell strength to be >= success.
        if potions_index >= len(potions):
            pairs[original_index] = 0
            continue
        target = math.ceil(success / spell_strength)
        while (
                potions_index < len(potions)
                and potions[potions_index] < target
        ):
            potions_index += 1
        pairs[original_index] = len(potions) - potions_index

    return pairs


def findPeakElement(nums: list[int]) -> int:
    """
    162. Find Peak Element
    This finds and returns an index position where the element is
    greater than the element before and after it.  The first element is
    by default greater than the imaginary element before it, and the
    last element is by default greater than the imaginary element after
    it.  An element cannot be the same as the element after it.
    """
    # These are the base cases when there are only 1 or 2 elements.
    if len(nums) == 1 or nums[0] > nums[1]:
        return 0
    if nums[-1] > nums[-2]:
        return len(nums) - 1

    start_index = 0
    end_index = len(nums) - 1
    # This keeps track of a subsection of nums.  It checks if the
    # middle element in the subsection is a peak.  If not, the
    # subsection moves to the side before or after the middle element
    # depending on which neighbor is greater than it.  If the neighbor
    # is less than the middle element, it is possible that all elements
    # after it are descending.  This means there is no peak on that
    # side.  However, if the neighbor is greater, then the element
    # after it may be less, which means the neighbor is the peak.  Or,
    # every element after the neighbor is ascending, which means the
    # peak is the last element on that side.  Or, the elements after
    # the neighbor are ascending until one of them drops.  In any of
    # the cases, the peak is on that side.
    while start_index <= end_index:
        middle_index = int((start_index + end_index) / 2)
        if (
                nums[middle_index] > nums[middle_index - 1]
                and nums[middle_index] > nums[middle_index + 1]

        ):
            return middle_index
        if nums[middle_index - 1] > nums[middle_index]:
            end_index = middle_index - 1
        else:
            start_index = middle_index + 1


def minEatingSpeed(piles: list[int], h: int) -> int:
    """
    875. Koko Eating Bananas
    piles contains the amounts of bananas in piles.  You can eat
    bananas at a constant rate in hours.  If your rate is higher than
    the amount of bananas left in a pile, then after finishing the pile
    you stop eating for that hour.  This calculates and returns the
    slowest rate you can eat at in order to finish every banana within
    h hours.
    """
    start = 1
    end = max(piles)

    min_speed = end
    # The minimum rate you can eat at is 1, and the maximum rate is the
    # largest value in piles.  So, the result is somewhere in this
    # range.  This keeps track of a subsection within the range.  It
    # takes the middle speed and calculates how many hours it will take
    # to eat every banana.  If this rate is higher than h, you need to
    # eat faster, so the subsection moves to the right of the middle
    # speed.  If the rate is lower than or equal to h, there may be an
    # even slower speed you can eat at, so the subsection moves to the
    # left of the middle speed.
    while start <= end:
        speed = int((start + end) / 2)
        hours = 0
        for pile in piles:
            hours += math.ceil(pile / speed)
        if hours > h:
            start = speed + 1
        else:
            end = speed - 1
            min_speed = min(min_speed, speed)

    return min_speed


def combinationSum3(k: int, n: int) -> list[list[int]]:
    """
    216. Combination Sum III
    This finds and returns every combination of k numbers that sum to
    n.  A number has to be between 1-9 (both inclusive) and can only be
    used in a combination once.  Combinations with identical numbers in
    different orders are considered the same.
    """
    def find_combinations(current_numbers: list, k: int,
                          n: int, combinations: list) -> list:
        """
        current_numbers takes in a list with one starting element.  It
        then finds the combinations using the starting element and
        numbers greater than it with a length of k that sum to n.
        """
        # This iterates through each possible number greater than the
        # last element in the current list of numbers.
        for next_number in range(current_numbers[-1]+1, 10):
            new_numbers = current_numbers + [next_number]
            if len(new_numbers) == k:
                if sum(new_numbers) == n:
                    combinations.append(new_numbers)
                    return combinations
                # This stops iterating through subsequent numbers if
                # the sum is already over n.
                if sum(new_numbers) > n:
                    return combinations
            # This uses recursion to continue to add numbers to the
            # current list.
            if len(new_numbers) < k and sum(new_numbers) < n:
                find_combinations(new_numbers, k, n, combinations)

        return combinations

    combinations = []
    for start in range(1, 10):
        if start < n:
            find_combinations([start], k, n, combinations)

    return combinations


def maxProfit(prices: list[int], fee: int) -> int:
    """
    714. Best Time to Buy and Sell Stock with Transaction Fee
    prices represents stock prices each day.  You can buy a stock one
    day and sell it on a different day in the future.  The fee will be
    subtracted from your profit.  You can only hold onto one stock at a
    time, but you can perform as many transactions as you want.  This
    calculates and returns the max profit you can make.  If no profit
    is possible, it returns 0.
    """
    # The ith element of the lists represents the max profit you can
    # achieve from the first day to the ith day (0-indexed).  One list
    # is for when you own stock on the end of the ith day, and the
    # other list is for when you don't own stock on the end of the day.
    do_own = [-prices[0]]
    dont_own = [0]

    # This iterates from the second day to the last and calculates the
    # max profit for each list.
    for i in range(1, len(prices)):
        # In order to own stock at the end of the day, you either owned
        # stock the previous day and didn't sell it, or you didn't own
        # stock the previous day and you bought today's stock.  The max
        # profit of the former is the same as the previous day's do own
        # stock max profit.  The max profit of the latter is the
        # previous day's don't own stock max profit minus the price of
        # today's stock since you bought the stock.
        do_own_value = max(do_own[-1], dont_own[-1] - prices[i])
        # In order to not own stock at the end of the day, you either
        # didn't own stock the previous day and didn't buy anything
        # today, or you did own stock the previous day and you sold it
        # today.  The max profit of the former is the same as the
        # previous day's don't own stock max profit.  The max profit of
        # the latter is the previous day's do own stock max profit plus
        # the price of today's stock since you sold your stock minus
        # the fee.
        dont_own_value = max(dont_own[-1], do_own[-1] + prices[i] - fee)
        do_own.append(do_own_value)
        dont_own.append(dont_own_value)

    return dont_own[-1]


def suggestedProducts(products: list[str],
                      searchWord: str) -> list[list[str]]:
    """
    1268. Search Suggestions System
    This uses each substring of searchWord that starts from the first
    character, up to and including the last character.  For each of
    those substrings, it finds at most 3 strings in products that start
    with the substring as its prefix.  If there are more than 3
    options, it uses the 3 lexicographical minimums.  Those strings are
    put in a list, and that list is then added to the returned list.
    """
    products.sort()

    results = []
    left_index = 0
    right_index = len(products) - 1
    # This iterates through each character of searchWord.  Each time,
    # it finds the 3 or less min strings that have the searchWord
    # substring as a prefix.
    for i in range(len(searchWord)):
        current_letter = searchWord[i]
        # This iterates the left pointer forward until it points at the
        # min string that has the substring as a prefix.
        while (
                left_index <= right_index
                and (
                    # If the substring is longer than the current
                    # string, or the current searchWord character does
                    # not match the current string character in the
                    # same spot, then it does not have the substring as
                    # a prefix.
                    i >= len(products[left_index])
                    or products[left_index][i] != current_letter
                )
        ):
            left_index += 1
        # This iterates the right pointer backwards until it points at
        # the max string that has the substring as a prefix.
        while (
                left_index <= right_index
                and (
                    i >= len(products[right_index])
                    or products[right_index][i] != current_letter
                )
        ):
            right_index -= 1
        result = []
        # This adds the 3 or less min strings to the returned list.
        for j in range(min(3, right_index - left_index + 1)):
            result.append(products[left_index+j])
        results.append(result)

    return results


def eraseOverlapIntervals(intervals: list[list[int]]) -> int:
    """
    435. Non-overlapping Intervals
    Each element in intervals represents an interval.  It starts at the
    value of the first element and ends at the value of the second
    element.  This calculates and returns the minimum number of
    intervals that need to be removed so that no intervals overlap.
    """
    def first_element(interval: list[int]) -> int:
        """
        This returns the first element in an interval.
        """
        return interval[0]
    # This sorts intervals based on the value of the first element in
    # each interval.
    intervals.sort(key=first_element)

    result = 0
    end = intervals[0][1]
    # This iterates through each interval and keeps track of the end of
    # a non-overlapping group of intervals.
    for i in range(1, len(intervals)):
        # This checks if the current interval starts before the end of
        # the group.  If so, it is overlapping.
        if intervals[i][0] < end:
            # The start of the interval in the group containing the end
            # is unknown.  However, the current interval must start at
            # the same spot or after it since they were sorted by the
            # first element.  So, either the interval containing the
            # end or the current interval need to be deleted to remove
            # the overlap.  This picks the one with the smaller end
            # value, because then there is less chance of an overlap in
            # future iterations.
            result += 1
            end = min(end, intervals[i][1])
        else:
            # The current interval does not overlap with the group.
            # This updates the end of the group.
            end = intervals[i][1]

    return result


def findMinArrowShots(points: list[list[int]]) -> int:
    """
    452. Minimum Number of Arrows to Burst Balloons
    Each element of points represents a balloon against a wall.  The
    start is the first value and the end is the second value.  This
    calculates and returns the minimum number of arrows to pop every
    balloon if they are shot upwards through them.  Two balloons next
    to each other can both be shot by 1 arrow at the point of contact.
    """
    def first_element(point):
        """
        This returns the first element of a point.
        """
        return point[0]

    # This sorts the balloons by their starts.
    points.sort(key=first_element)

    arrows = 0
    start = points[0][0]
    end = points[0][1]
    # This iterates through each balloon.  It checks if the current
    # balloon is overlapping with the current balloon interval.  If so,
    # the interval is updated so that an arrow shot anywhere within it
    # can pierce every balloon.  If not, an arrow needs to be shot to
    # pierce the balloons in the current interval, and a new interval
    # starts with the current balloon.
    for i in range(1, len(points)):
        point = points[i]
        if point[0] <= end:
            # The new interval becomes the overlapping part of the
            # current interval and current balloon.  An arrow shot
            # anywhere outside of this will not pierce every balloon.
            start = max(start, point[0])
            end = min(end, point[1])
        else:
            # The current balloon is outside of the current interval.
            arrows += 1
            start = point[0]
            end = point[1]

    # This shoots a final arrow to pierce the balloons in the last
    # interval.
    arrows += 1

    return arrows


def dailyTemperatures(temperatures: list[int]) -> list[int]:
    """
    739. Daily Temperatures
    temperatures contains daily temperatures.  This creates and returns
    a list where list[i] is the number of days it takes from the ith
    day to have a warmer temperature.  If there is never a warmer
    temperature, the number of days is 0.
    """
    results = [0] * len(temperatures)
    # The stack will contain indices of the argument, temperatures.  If
    # you replace the indices with their values from the list, it will
    # be in decreasing order and can have repeat values.
    stack = [0]
    # This iterates through each temperature.  If it is higher than any
    # values at the top of the stack, it calculates the days since that
    # value and stores it.
    for i in range(1, len(temperatures)):
        while len(stack) > 0 and temperatures[i] > temperatures[stack[-1]]:
            past_i = stack.pop()
            results[past_i] = i - past_i
        stack.append(i)

    return results


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


def countBits(n: int) -> list[int]:
    """
    338. Counting Bits
    This returns a list where the elements are the number of 1 bits in
    the integers from 0 to n (both inclusive).
    """
    result = [0]
    offset = 0

    # This iterates through each number from 1 to n (both inclusive).
    # When the current number is the result of 2 to the power of an
    # integer, the leftmost bit (without padding) is 1 and all of the
    # remaining bits are 0.  All of these numbers have a single 1 bit.
    # These are the offset values, because all of the numbers between
    # them can be made by adding the first offset value less than the
    # current number with the difference of the current number minus
    # the offset (ex: 7 = 4 + (7-4) = 4 + 3).  So, the number of 1 bits
    # in the current number is the number of 1 bits in the offset (1)
    # plus the number of 1 bits in the remainder.
    for num in range(1, n+1):
        log_base_2 = math.log2(num)
        # The current number is an offset if its log base 2 is an
        # integer.
        if log_base_2 - int(log_base_2) == 0:
            result.append(1)
            offset = num
        else:
            remainder = num - offset
            result.append(1 + result[remainder])

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


def singleNumber(nums: list[int]) -> int:
    """
    136. Single Number
    nums must have at least 1 element, and each distinct element must
    be present in the list exactly 2 times, except for one value that
    is only present 1 time.  This returns that one value.
    """
    result = 0
    # Exclusive or (^) between bits equals 0 when the bits are the same
    # (1^1, 0^0) and 1 when the bits are different (1^0, 0^1).  So, a
    # number exclusive or itself is 0.  This uses exclusive or between
    # each element of nums.  The duplicates cancel out, and the result
    # will be the nonduplicate number.
    for num in nums:
        result = result ^ num

    return result


def minFlips(a: int, b: int, c: int) -> int:
    """
    1318. Minimum Flips to Make a OR b Equal to c
    a, b, and c must be positive integers.  This calculates and returns
    the minimum number of bits that need to be changed in a and b to
    make a bitwise or b equal to c.
    """
    flips = 0
    # This iterates through each bit in a, b, and c from right to left.
    # Each iteration, it checks if any bits in a or b need to be
    # changed.
    for i in range(30):
        # This checks whether the current bit is 0 or 1.
        a_bit = 1 if (a >> i) % 2 == 1 else 0
        b_bit = 1 if (b >> i) % 2 == 1 else 0
        c_bit = 1 if (c >> i) % 2 == 1 else 0
        # When the bit in c is 0, a or b needs to equal 0.  Otherwise,
        # it needs to equal 1.
        if c_bit == 0:
            # In order for a or b to equal 0, both a and b need to be
            # 0.
            if (
                a_bit == 0 and b_bit == 1
                or a_bit == 1 and b_bit == 0
            ):
                flips += 1
            elif a_bit == 1 and b_bit == 1:
                flips += 2
        else:
            # In order for a or b to equal 1, either a, b, or both need
            # to be 1.
            if a_bit == 0 and b_bit == 0:
                flips += 1

    return flips


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
        different_ways_list[i] = different_ways_list[i-1] + \
            different_ways_list[i-2]

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


def longestCommonSubsequence(text1: str, text2: str) -> int:
    """
    1143. Longest Common Subsequence
    This takes in two strings of all lowercase letters.  It returns an
    integer of the length of the longest subsequence that can be formed
    out of both strings.  If there is no common subsequence, it returns
    0.
    """
    ROWS = len(text1)
    COLS = len(text2)
    # This creates a 2-D matrix.  Each row represents a letter in
    # text1, and each column represents a letter in text2.
    matrix = []
    for _ in range(ROWS):
        matrix.append([None] * COLS)

    # This sets the values for the first row and column.  It is 0 until
    # the current letter of one string is found in the other string.
    current_amount = 0
    for col in range(COLS):
        if current_amount == 0 and text1[0] == text2[col]:
            current_amount = 1
        matrix[0][col] = current_amount
    current_amount = 0
    for row in range(ROWS):
        if current_amount == 0 and text2[0] == text1[row]:
            current_amount = 1
        matrix[row][0] = current_amount

    # This iterates from the second row to the last row.  Within each
    # iteration, it iterates from the second column to the last column.
    # It checks if the current row and col characters equal each other.
    # If so, the longest common subsequence is the lcs of
    # text1[:current row char] and text2[:current col char] plus one,
    # because if you take those strings and add the current row and col
    # chars then the subsequence increases by 1.  If they don't equal
    # each other, the value can be two options.  The first is the lcs
    # of text1 up to and including the current row char and
    # text2[:current col char], since adding the current col char to
    # text2 does nothing.  Or, it can be the lcs of
    # text1[:current row char] and text2 up to and including the
    # current col char, since adding the current row char to text1 does
    # nothing.  This picks the greater of the two options.
    for row in range(1, ROWS):
        for col in range(1, COLS):
            if text1[row] == text2[col]:
                matrix[row][col] = matrix[row-1][col-1] + 1
            else:
                matrix[row][col] = max(
                    matrix[row][col-1],
                    matrix[row-1][col],
                )

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


def rob(nums: list[int]) -> int:
    """
    198. House Robber
    nums contains the amounts of money from robbing houses on a street.
    Two adjacent houses cannot be robbed.  This calculates and returns
    the max amount of money that can be stolen.
    """
    # This is the base case when there is only one house.
    if len(nums) == 1:
        return nums[0]

    # The ith element of this list represents the most money you can
    # steal from the start to the ith house.
    max_money = [0] * len(nums)
    max_money[0] = nums[0]
    max_money[1] = max(nums[0], nums[1])

    # This iterates from the third house to the last.  Each time, it
    # calculates the most money you can steal up to that house.
    for i in range(2, len(nums)):
        # For the current house, there are two options for the most
        # money: you can rob the house and get its money + the most
        # money from the house 2 behind.  Or, you can not rob it and
        # get the most money from the house 1 behind.  This picks the
        # greater of the two options.
        rob_house_max_money = nums[i] + max_money[i-2]
        dont_rob_house_max_money = max_money[i-1]
        max_money[i] = max(rob_house_max_money, dont_rob_house_max_money)

    return max_money[-1]


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


def minCostClimbingStairs(cost: list[int]) -> int:
    """
    746. Min Cost Climbing Stairs
    cost contains costs to travel from a step.  You can travel either 1
    or 2 steps ahead.  This returns the minimum cost to go past the
    last step if you start on either the first or second step.
    """
    # The ith element of this list represents the minimum cost to reach
    # past the last step if you start from the ith step.
    costs = [0] * len(cost)
    costs[-1] = cost[-1]
    costs[-2] = cost[-2]

    # This iterates from the third to last step to the first step.
    # Each time, it calculates the minimum cost to reach past the last
    # step if you start from that step.
    for i in range(len(cost)-3, -1, -1):
        current_cost = cost[i]
        # You can move one or two steps ahead from the current step.
        # So, there are two cost options to reach past the last step:
        # the cost of the current step + the total cost of the step one
        # ahead, or the cost of the current step + the total cost of
        # the step two ahead.  This picks the minimum of those two
        # options.
        costs[i] = min(current_cost+costs[i+1], current_cost+costs[i+2])

    # This picks the lower total cost out of starting from either the
    # first or second step.
    return min(costs[0], costs[1])


def uniquePaths(m: int, n: int) -> int:
    """
    62. Unique Paths
    There is a grid with m rows and n columns.  You start at the top
    left spot.  This finds and returns the number of unique paths you
    can take to the bottom right spot if you only move right or down.
    """
    grid = []
    for row in range(m):
        grid.append([None] * n)
    # The number of unique paths for each spot in the last row and last
    # column is 1, because you can only move right or down every time.
    for col in range(n):
        grid[m-1][col] = 1
    for row in range(m):
        grid[row][n-1] = 1

    # This iterates through each row from the second to last to the
    # first.
    for row in range(m-2, -1, -1):
        # This iterates through each spot in the current row from the
        # second to rightmost to the leftmost.
        for col in range(n-2, -1, -1):
            # The number of unique paths for a spot is the number of
            # unique paths for the spot to the right plus the number of
            # unique paths for the spot below.
            grid[row][col] = grid[row][col+1] + grid[row+1][col]

    return grid[0][0]


def minDistance(word1: str, word2: str) -> int:
    """
    72. Edit Distance
    For one operation, you can insert, delete, or replace a character
    in word1.  This calculates and returns the minimum number of
    operations to make word1 the same as word2.
    """
    # These are the base cases.  If word1 is empty, you just insert
    # every character in word2.  If word2 is empty, you just delete
    # every character in word1.
    if len(word1) == 0:
        return len(word2)
    if len(word2) == 0:
        return len(word1)

    # Each row represents a character in word1.  Each column represents
    # a character in word2.  The intersection is the minimum number of
    # operations to make the characters up to and including the current
    # character of word1 the same as the characters up to and including
    # the current character of word2.
    ROWS = len(word1)
    COLS = len(word2)
    matrix = []
    for _ in range(ROWS):
        matrix.append([None] * COLS)

    # This fills in the values for the first column and row.
    # This checks if the first characters equal each other.  If so, no
    # operations are needed.  Otherwise, 1 replace operation is needed.
    if word1[0] == word2[0]:
        matrix[0][0] = 0
    else:
        matrix[0][0] = 1
    # If the current word1 row character equals the first character of
    # word2, then all of the characters before it need to be deleted to
    # make them the same.  Otherwise, you have to make the word1
    # characters up to and including the current character equal to the
    # first character of word2.  This is the same as the number of
    # operations to make the word1 characters up to but not including
    # the current character equal to the first character of word2, plus
    # one 1 delete operation to remove the current word1 character.
    # This equals the value in the row above and same column + 1.  The
    # same applies for the column characters and the first character of
    # word1, but you use the value in the same row and column to the
    # left + 1.
    for row in range(1, ROWS):
        if word1[row] == word2[0]:
            matrix[row][0] = row
        else:
            matrix[row][0] = matrix[row-1][0] + 1
    for col in range(1, COLS):
        if word1[0] == word2[col]:
            matrix[0][col] = col
        else:
            matrix[0][col] = matrix[0][col-1] + 1

    # This iterates through each row from the second to the last.  For
    # each iteration, it iterates from the second column to the last.
    # If the current row and col characters equal each other, then the
    # value is the same as the value up and to the left.  That
    # represents the min operations for the word1 and 2 substrings up
    # to but not including the current characters.  You then add the
    # same character to both substrings, so no operation is needed to
    # make the new strings the same.  If the current characters do not
    # equal each other, you need to perform one of the operations.  You
    # can delete the current row character from the word1 substring,
    # and now you have the min operations to make the word1 substring
    # up to but not including the current row character the same as the
    # word2 substring up to and including the current col character + 1
    # for the deletion.  This is the value above the current
    # intersection + 1.  You can insert the current col character at
    # the end of the word1 substring, and now you have the min
    # operations to make the word1 substring up to and including the
    # current row character the same as the word2 substring up to but
    # not including the current col character + 1 for the insertion.
    # This is the value left of the current intersection + 1.  Finally,
    # you can replace the current row character from the word1
    # substring and make it the same as the current col character.  Now
    # you have the min operations to make the word1 substring up to but
    # not including the current row character the same as the word2
    # substring up to but not including the current col character + 1
    # for the replacement.  This is the value above and to the left of
    # the current intersection + 1.  This uses the minimum of the three
    # options.
    for row in range(1, ROWS):
        for col in range(1, COLS):
            if word1[row] == word2[col]:
                matrix[row][col] = matrix[row-1][col-1]
            else:
                matrix[row][col] = min(
                    matrix[row-1][col],
                    matrix[row][col-1],
                    matrix[row-1][col-1],
                ) + 1

    return matrix[-1][-1]


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


def lengthOfLongestSubstring(s: str) -> int:
    """
    3. Longest Substring Without Repeating Characters
    This finds and returns the length of the longest substring from s
    that does not contain duplicate characters.
    """
    max_length = 0
    substring_set = set()
    first_char_index = 0
    # This iterates through each character of s.  It continues to
    # remove the first character of the current substring until it does
    # not contain the current character.  It then adds the current
    # character to the end of the substring and checks if there is a
    # new max length.
    for char in s:
        while char in substring_set:
            substring_set.remove(s[first_char_index])
            first_char_index += 1
        substring_set.add(char)
        max_length = max(max_length, len(substring_set))

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


def isIsomorphic(s: str, t: str) -> bool:
    """
    205. Isomorphic Strings
    This returns True if the characters in s can be replaced to form t.
    All occurrences of the character must be replaced with the same
    character.  Also, multiple characters cannot be replaced with the
    same character.  If t cannot be formed, it returns False.
    """
    mapping = {}
    mapped_chars = set()
    i = 0
    # This iterates through the characters of s.  It either creates or
    # checks the mapping to the corresponding character in t.
    while i < len(s):
        # This checks if the character has not already been mapped.
        if s[i] not in mapping:
            # This checks if the mapped character in the new mapping
            # between s and t has already been used.
            if t[i] in mapped_chars:
                return False
            # This creates the mapping.
            mapping[s[i]] = t[i]
            mapped_chars.add(t[i])
        else:
            # This checks if the current character can be replaced with
            # its mapped character to become the corresponding t
            # character.
            if mapping[s[i]] != t[i]:
                return False
        i += 1

    return True


def isSubsequence(s: str, t: str) -> bool:
    """
    392. Is Subsequence
    This returns True if s is a subsequence (a string that can be
    derived from another string by deleting some or no elements without
    changing the order of the remaining elements) of t.  Otherwise, it
    returns False.
    """
    s_index = 0
    t_index = 0
    # This iterates through s and t until it reaches the end of either
    # string.
    while s_index < len(s) and t_index < len(t):
        # This checks if the current character of s has been found in
        # t.
        if s[s_index] == t[t_index]:
            s_index += 1
        t_index += 1

    # This is True when all of the characters of s were found in t.
    return s_index == len(s)


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


def reverseVowels(s: str) -> str:
    """
    345. Reverse Vowels of a String
    This reverses the upper and lower case vowels in s then returns the
    resulting string.
    """
    vowels = ['a', 'e', 'i', 'o', 'u', 'A', 'E', 'I', 'O', 'U']
    left_index = 0
    right_index = len(s) - 1

    # This iterates from both sides of the string towards the middle.
    while left_index < right_index:
        # This moves both pointers until they are on vowels.
        while left_index < len(s) and s[left_index] not in vowels:
            left_index += 1
        while right_index >= 0 and s[right_index] not in vowels:
            right_index -= 1
        # The pointers may cross each other and reach vowels that were
        # already reversed.  In this case, they should not be reversed
        # again.
        if left_index >= right_index:
            continue

        # This reverses the vowels.
        left_char = s[left_index]
        s = (s[:left_index] + s[right_index] + s[left_index+1:right_index]
             + left_char + s[right_index+1:])
        left_index += 1
        right_index -= 1

    return s


def reverseWords(s: str) -> str:
    """
    151. Reverse Words in a String
    This reverses the words (a group of characters separated by one or
    more spaces) in s.  It returns a string of the reversed words
    separated by a single space.
    """
    # This creates a list of the words in reverse order.
    words = s.split()
    words.reverse()

    # This turns the list of reversed words into a string.
    return ' '.join(words)


def compress(chars: list[str]) -> int:
    """
    443. String Compression
    This iterates through the characters in chars.  For every group of
    characters (consecutive repeating characters) it prepends the
    group's character to chars.  If the group's length is greater than
    1, it also prepends characters of the digits of the length.  It
    then returns the number of characters prepended to chars.
    """
    original_length = len(chars)
    current_char = chars[0]
    current_length = 1
    # original_i iterates through the original characters in chars.
    # new_i is used to keep track of where to insert characters in
    # chars.
    original_i = 1
    new_i = 0

    # This iterates through the original characters in chars.
    for _ in range(original_length-1):
        # This checks if the character is part of the current group.
        if chars[original_i] == current_char:
            current_length += 1
            original_i += 1

        # This checks if the character is not part of the current
        # group.  So, the current group needs to be prepended to chars.
        else:
            chars.insert(new_i, current_char)
            new_i += 1
            original_i += 1
            # This adds the characters of the digits of the length if
            # it is over 1.
            if current_length != 1:
                string_length = str(current_length)
                for char in string_length:
                    chars.insert(new_i, char)
                    new_i += 1
                    original_i += 1
            # This prepares for the next iteration.
            current_char = chars[original_i]
            current_length = 1
            original_i += 1

    # This prepends the last group to chars.
    chars.insert(new_i, current_char)
    new_i += 1
    if current_length != 1:
        string_length = str(current_length)
        for char in string_length:
            chars.insert(new_i, char)
            new_i += 1

    return len(chars) - original_length


def maxVowels(s: str, k: int) -> int:
    """
    1456. Maximum Number of Vowels in a Substring of Given Length
    This finds a contiguous substring of s with length k that has the
    most vowels ('a', 'e', 'i', 'o', 'u').  It then returns the number
    of most vowels.
    """
    max_vowels = 0
    # This uses a set for the vowels so they can be accessed in
    # constant time.
    vowels = set()
    vowels.add('a')
    vowels.add('e')
    vowels.add('i')
    vowels.add('o')
    vowels.add('u')

    # This calculates the number of vowels in the substring starting
    # with the first element.
    start_index = 0
    end_index = start_index + k - 1
    for letter in s[:end_index+1]:
        if letter in vowels:
            max_vowels += 1

    current_vowels = max_vowels
    # This uses a sliding window of a k length substring.  It
    # calculates the number of vowels in it and checks if it is a new
    # max.
    while end_index < len(s) - 1:
        if s[start_index] in vowels:
            current_vowels -= 1
        start_index += 1
        end_index += 1
        if s[end_index] in vowels:
            current_vowels += 1
        max_vowels = max(max_vowels, current_vowels)

    return max_vowels


def closeStrings(word1: str, word2: str) -> bool:
    """
    1657. Determine if Two Strings Are Close
    You can perform operation 1 on a string by swapping existing
    characters.  You can perform operation 2 on a string by changing
    every occurrence of one character to another existing character,
    and vice versa.  This returns True if the operations can be used
    any number of times to get word1 and word2 to equal each other.
    Otherwise, it returns False.
    """
    # Based on operation 1, the order of characters doesn't matter as
    # long as they are the same.
    # Based on operation 2, the frequencies of specific characters
    # doesn't matter as long as the frequency values are the same.
    # So, to get the strings to equal each other, they just need to
    # have the same characters and frequency values.

    word1_frequencies = collections.Counter(word1).values()
    word2_frequencies = collections.Counter(word2).values()
    # This uses the character frequencies to create dictionaries where
    # the key is the frequency value and its value is the number of
    # occurrences.  This allows the frequency values to be compared
    # easily, since it just has to check if each dictionary has the
    # same keys and if the values for each key are the same.
    word1_frequency_counts = collections.Counter(word1_frequencies)
    word2_frequency_counts = collections.Counter(word2_frequencies)

    return (
        # This checks if the characters are the same.
        set(word1) == set(word2)
        # This checks if the frequency values are the same.
        and word1_frequency_counts == word2_frequency_counts
    )


def removeStars(s: str) -> str:
    """
    2390. Removing Stars From a String
    s contains the character, *.  The operation that can be performed
    is removing the * and the closest non-* character to the left of
    it.  This continually performs the operation until there are no *
    characters left in s.  It then returns the resulting string.
    """
    stack = []
    # This iterates through the characters in the string.  It adds
    # non-* characters to the result.  If the element is a *, it
    # removes the current last character from the result.
    for char in s:
        if char == '*':
            # This doesn't perform the removal from the result if there
            # is no character to remove.
            if len(stack) > 0:
                stack.pop()
        else:
            stack.append(char)

    return ''.join(stack)


def decodeString(s: str) -> str:
    """
    394. Decode String
    This takes in an encoded string with the encoding k[substring].
    This is decoded by multiplying the substring k times.  It then
    returns the decoded string.
    """
    stack = []
    # This adds each character to the stack, other than ].
    for i in range(len(s)):
        # When a ] is reached, the characters at the top of the stack
        # need to be decoded.  These are the substring characters and k
        # value preceding the ].
        if s[i] == ']':
            # This gets the substring within the brackets.
            current_string = ''
            while stack[-1] != '[':
                current_string = stack[-1] + current_string
                stack.pop()

            # This removes the [ from the stack.
            stack.pop()

            # This gets the k value.
            current_int = ''
            while len(stack) > 0 and stack[-1].isdigit():
                current_int = stack[-1] + current_int
                stack.pop()

            # This creates the string formed by multiplying the
            # substring k times.  It adds it as a single element back
            # onto the stack to be used in future iterations.
            stack.append(int(current_int) * current_string)

        # This adds the character to the stack when it is not ].
        else:
            stack.append(s[i])

    return ''.join(stack)


def predictPartyVictory(senate: str) -> str:
    """
    649. Dota2 Senate
    senate represents senators from either the Radiant or Dire party.
    Each character in the string is either a R or D.  The order of
    characters represents the voting order.  Each round of voting, a
    senator can either take away voting rights forever from another
    senator, or declare his party has won if there are no senators from
    the other party with voting rights.  This determines which party
    will win and returns the name of the party, either Radiant or Dire.
    """
    # This creates a queue for each party of the index positions of the
    # senators.
    R_queue = collections.deque()
    D_queue = collections.deque()
    for i in range(len(senate)):
        if senate[i] == 'R':
            R_queue.append(i)
        else:
            D_queue.append(i)

    # Each iteration, this selects the next senator that still has
    # voting rights.  This senator will take away voting rights from
    # the closest senator of the other party.  This continues to happen
    # until there are only senators from one party remaining.
    while len(R_queue) > 0 and len(D_queue) > 0:
        # This finds which senator is voting next.  It removes the
        # closest senator of the other party and then moves the
        # selected senator from the front to the end of the queue for
        # the next voting round.  The index position is incremented by
        # the number of senators since the senator can only act once
        # per round.  Otherwise, the senator with index 0 would be able
        # to remove the closest senator every iteration.
        if R_queue[0] < D_queue[0]:
            D_queue.popleft()
            R_senator_position = R_queue.popleft()
            R_queue.append(R_senator_position+len(senate))
        else:
            R_queue.popleft()
            D_senator_position = D_queue.popleft()
            D_queue.append(D_senator_position+len(senate))

    # There are only senators from one party remaining.  This checks
    # which party won.
    if len(R_queue) == 0:
        return 'Dire'
    return 'Radiant'


def letterCombinations(digits: str) -> list[str]:
    """
    17. Letter Combinations of a Phone Number
    digits contains a phone number where each digit is between 2-9
    (both inclusive).  This returns a list of all the possible letter
    combinations that the phone number can be.
    """
    def create_letter_combinations(
            current_digits: str,
            mapping: dict,
            current_string: str = '',
            combinations: list = None,
    ) -> list:
        """
        This uses recursion to come up with all possible letter
        combinations for a phone number.  Each iteration, it adds a
        letter to the combination and moves onto the next digit.  It
        then returns a list of all the possibilities.
        """
        if combinations is None:
            combinations = []

        if len(current_digits) == 0:
            combinations.append(current_string)
            return combinations

        for letter in mapping[current_digits[0]]:
            create_letter_combinations(
                current_digits[1:],
                mapping,
                current_string + letter,
                combinations,
            )

        return combinations

    # This is the base case when there is no phone number.
    if len(digits) == 0:
        return []

    mapping = {
        '2': 'abc',
        '3': 'def',
        '4': 'ghi',
        '5': 'jkl',
        '6': 'mno',
        '7': 'pqrs',
        '8': 'tuv',
        '9': 'wxyz',
    }

    return create_letter_combinations(digits, mapping)


class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next


def mergeTwoLists(list1: ListNode, list2: ListNode) -> ListNode:
    """
    21. Merge Two Sorted Lists
    This merges two sorted ascending linked lists and maintains the
    sorted order.  It then returns the head of the new list.
    """
    head = None
    # This continues to iterate until it has gone through all of the
    # nodes in both list1 and list2.  First, it determines which value
    # to use out of the current list1 and list2 nodes.  It then creates
    # the new node with that value and iterates to the next node of the
    # current node that was used.  Finally, it attaches the new node to
    # the end of the new list.
    while list1 is not None or list2 is not None:
        if list1 is None:
            # This runs when all of the nodes in list1 have been
            # iterated through, but there are still nodes remaining in
            # list2.
            new_node = ListNode(val=list2.val)
            list2 = list2.next
        elif list2 is None:
            # This runs when all of the nodes in list2 have been
            # iterated through, but there are still nodes remaining in
            # list1.
            new_node = ListNode(val=list1.val)
            list1 = list1.next
        # This compares the values of the current list1 and list2
        # nodes.
        elif list1.val <= list2.val:
            new_node = ListNode(val=list1.val)
            list1 = list1.next
        else:
            new_node = ListNode(val=list2.val)
            list2 = list2.next
        if head is None:
            # This only runs during the first iteration when the new
            # list is empty.
            head = new_node
            current_node = head
        else:
            # This runs everytime after the first iteration.
            current_node.next = new_node
            current_node = current_node.next

    return head


def addTwoNumbers(l1: ListNode, l2: ListNode) -> ListNode:
    """
    2. Add Two Numbers
    The nodes in each list represent the digits of numbers in reverse
    order.  This sums the numbers and puts the digits in reverse order
    in a new list.  It then returns the new list.
    """
    # This gets the two numbers.
    l1_num = ''
    current_node = l1
    while current_node is not None:
        l1_num += str(current_node.val)
        current_node = current_node.next
    l1_num = l1_num[::-1]
    l2_num = ''
    current_node = l2
    while current_node is not None:
        l2_num += str(current_node.val)
        current_node = current_node.next
    l2_num = l2_num[::-1]

    # This sums the two numbers and stores the digits in reverse order.
    list_sum = str(int(l1_num)+int(l2_num))
    list_sum = list_sum[::-1]

    # This creates the new list.
    head_node = ListNode(val=int(list_sum[0]))
    current_node = head_node
    for i in range(1, len(list_sum)):
        next_node = ListNode(val=int(list_sum[i]))
        current_node.next = next_node
        current_node = current_node.next

    return head_node


def reverseList(head: ListNode) -> ListNode:
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
        # node.
        current_node.next = previous_node

        # This updates the node variables for the next iteration.
        previous_node = current_node
        current_node = next_node

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


def deleteMiddle(head: ListNode) -> ListNode:
    """
    2095. Delete the Middle Node of a Linked List
    This deletes the middle node in a linked list.  If the length of
    the list is even and there are two middle nodes, it deletes the
    right one.  It then returns the head of the modified linked list.
    """
    # This is a base case when there is only one element.
    if head.next is None:
        return None

    # This finds the length of the linked list.
    length = 1
    current_node = head
    while current_node.next is not None:
        length += 1
        current_node = current_node.next

    # This calculates the index of the middle node.
    middle_index = length // 2

    # This iterates through the nodes until it reaches the middle one.
    previous_node = None
    current_node = head
    for _ in range(middle_index):
        previous_node = current_node
        current_node = current_node.next

    # This deletes the middle node.
    previous_node.next = current_node.next

    return head


def oddEvenList(head: ListNode) -> ListNode:
    """
    328. Odd Even Linked List
    This groups the odd numbered nodes together as well as the even
    numbered nodes.  The relative order of the nodes in both groups is
    the same.  It then puts the even group after the odd group.
    Finally, it returns the head of the new linked list.
    """
    # These are the base cases when there are only 0, 1, or 2 nodes.
    if head is None:
        return None
    if head.next is None or head.next.next is None:
        return head

    even_head = head.next
    current_odd_node = head
    current_even_node = head.next
    current_node = head.next.next
    i = 3
    # This iterates through the linked list.  Depending on whether the
    # current node is odd or even, it adds it to the relevant group.
    while current_node is not None:
        if i % 2 != 0:
            current_odd_node.next = current_node
            current_odd_node = current_odd_node.next
        else:
            current_even_node.next = current_node
            current_even_node = current_even_node.next

        current_node = current_node.next
        i += 1

    # This is needed when there are an odd number of nodes.  The next
    # node of the last node in the even group will point to the last
    # node of the linked list, which is odd.  The next node of this odd
    # node will be the head of the even group.  So, a cycle would be
    # created.
    current_even_node.next = None

    # This puts the even group after the odd group.
    current_odd_node.next = even_head

    return head


def pairSum(head: ListNode) -> int:
    """
    2130. Maximum Twin Sum of a Linked List
    This takes in a linked list with an even number of nodes.  Each
    node has a twin with the node on the other half of the list
    mirrored (first and last, second and second to last, ...).  This
    finds and returns the maximum sum of the values out of every pair
    of twin nodes.
    """
    # This calculates the length of the list.
    current_node = head
    length = 1
    while current_node.next is not None:
        current_node = current_node.next
        length += 1

    # This iterates through the first half of the list in order to
    # reverse the order.  It keeps track of the current, previous, and
    # next nodes.  Each iteration, it sets the previous node as the
    # element after the current node.
    previous_node = None
    current_node = head
    for _ in range(int(length/2)):
        next_node = current_node.next
        current_node.next = previous_node
        previous_node = current_node
        current_node = next_node

    current_left_half_node = previous_node
    current_right_half_node = current_node
    max_twin_sum = 0
    # This iterates in both directions outward from the middle of the
    # list.  Each iteration, it checks if there is a new max twin sum.
    while current_right_half_node is not None:
        max_twin_sum = max(
            max_twin_sum,
            current_left_half_node.val + current_right_half_node.val
        )
        current_left_half_node = current_left_half_node.next
        current_right_half_node = current_right_half_node.next

    return max_twin_sum


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


def tribonacci(n: int) -> int:
    """
    1137. N-th Tribonacci Number
    T0 = 0, T1 = 1, and T2 = 1.
    Tn = Tn-3 + Tn-2 + Tn-1 for all n >= 3.
    This returns the value of Tn using the given value for n.
    """
    # These are the base cases.
    if n == 0:
        return 0
    if n == 1 or n == 2:
        return 1

    past_three = collections.deque([0, 1, 1])
    # This keeps track of the past three T values.  Each iteration, it
    # adds the next T value and removes the oldest T value.  It does
    # this until it finds the value for Tn.
    for _ in range(n-2):
        current_sum = sum(past_three)
        past_three.append(current_sum)
        past_three.popleft()

    return past_three[-1]


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


def maxDepth(root: TreeNode) -> int:
    """
    104. Maximum Depth of Binary Tree
    This returns the number of nodes in the path from the root to the
    lowest node.
    """
    # This is a base case when the tree is empty.
    if root is None:
        return 0

    queue = collections.deque()
    level = 0
    queue.append(root)
    # This uses breadth-first traversal on the tree to count the number
    # of levels.
    while len(queue) > 0:
        current_level_length = len(queue)
        # This iterates through each node in the level.  It adds its
        # children to the end of the queue.
        for _ in range(current_level_length):
            node = queue.popleft()
            if node.left is not None:
                queue.append(node.left)
            if node.right is not None:
                queue.append(node.right)
        level += 1

    return level


def leafSimilar(root1: TreeNode, root2: TreeNode) -> bool:
    """
    872. Leaf-Similar Trees
    The leaf value sequence of a tree is the values of its leaves from
    left to right.  This compares the sequences of two trees.  It
    returns True if they are the same.  Otherwise, it returns False.
    """
    def pre_order_traversal_leaf_sequence(
        node: TreeNode,
        tree_leaf_sequence: list = None
    ) -> list:
        """
        This takes in the root node of a tree.  It performs pre-order
        traversal on the tree to find the leaf value sequence.  It then
        returns the sequence.
        """
        if tree_leaf_sequence is None:
            tree_leaf_sequence = []

        # This performs the operation on the node, then traverses to
        # the left subtree, and then to the right subtree.
        if node is not None:
            # This adds the node's value to the sequence when it is
            # leaf, meaning it has no children.
            if node.left is None and node.right is None:
                tree_leaf_sequence.append(node.val)
            pre_order_traversal_leaf_sequence(node.left, tree_leaf_sequence)
            pre_order_traversal_leaf_sequence(node.right, tree_leaf_sequence)

        return tree_leaf_sequence

    tree1_leaf_sequence = pre_order_traversal_leaf_sequence(root1)
    tree2_leaf_sequence = pre_order_traversal_leaf_sequence(root2)

    return tree1_leaf_sequence == tree2_leaf_sequence


def goodNodes(root: TreeNode) -> int:
    """
    1448. Count Good Nodes in Binary Tree
    A node is good if there are no values in the path from the node to
    the root that are greater than the node's value.  This calculates
    and returns the number of good nodes in a tree.
    """
    def pre_order_traversal_good_nodes(node: TreeNode, path_max: int) -> int:
        """
        This takes in a node of a tree and the greatest value in the
        path from the node to the root.  It treats the node as a root
        and performs pre-order traversal to find the number of good
        nodes in the subtree.  It then returns the amount.
        """
        if node is None:
            return 0

        # This performs the operation on the node, then traverses to
        # the left subtree, and then to the right subtree.
        # The node is not good if the greatest value in the path from
        # the current node to the root is greater than the node's
        # value.  Otherwise, it is good, because the node's value is
        # greater than or equal to the path's max.  For either option,
        # there is no value in the path greater than the node's value.
        current_node_good = 0 if path_max > node.val else 1
        # This checks if there is a new max node value for the path.
        path_max = max(path_max, node.val)
        good_nodes = (
            current_node_good
            + pre_order_traversal_good_nodes(node.left, path_max)
            + pre_order_traversal_good_nodes(node.right, path_max)
        )

        return good_nodes

    return pre_order_traversal_good_nodes(root, root.val)


def hasPathSum(root: TreeNode, targetSum: int) -> bool:
    """
    112. Path Sum
    This returns True if there is a path from the root to a leaf where
    the sum of the node values equals targetSum.  Otherwise, it returns
    False.
    """
    def pre_order_traversal_path_sum(
            node: TreeNode,
            target: int,
            current_path_sum: int = 0
    ) -> bool:
        """
        This takes in the root node of a tree and a target value.  It
        performs pre-order traversal on the tree to check if there is a
        path from the root to a leaf where the sum of the node values
        equals target.  If so, it returns True.  Otherwise, it returns
        False.
        """
        # This performs the operation on the node, then traverses to
        # the left subtree, and then to the right subtree.
        if node is not None:
            # This checks if the node is a leaf, meaning it has no
            # children.  If so, a comparison can be done between the
            # path sum and the target.
            if (
                    node.left is None
                    and node.right is None
                    and current_path_sum + node.val == target
            ):
                return True
            # These check if there is a path from the current node
            # where the path sum equals the target.  If not, they both
            # evaluate to False.
            if pre_order_traversal_path_sum(node.left,
                                            target,
                                            current_path_sum+node.val):
                return True
            if pre_order_traversal_path_sum(node.right,
                                            target,
                                            current_path_sum+node.val):
                return True

        # There was no path where the path sum equals the target.
        return False

    return pre_order_traversal_path_sum(root, targetSum)


def longestZigZag(root: TreeNode) -> int:
    """
    1372. Longest ZigZag Path in a Binary Tree
    A zig zag path in the tree starts at any node and moves to either
    the left or right child.  It then moves to the next child in the
    opposite way of the previous direction.  The zig zag length is the
    number of nodes in the path minus 1.  This finds and returns the
    longest zig zag length in the tree.
    """
    def zig_zag(
            node: TreeNode,
            current_direction: str,
            current_depth: int = 0
    ) -> int:
        """
        This takes in either the left or right child of the root of a
        tree and the direction of 'left' or 'right'.  It finds and
        returns the longest zig zag length in the subtree.
        """
        if node is None:
            return current_depth

        # This uses the current direction and depth to continue zig
        # zagging.  It also uses the other child to start a new zig
        # zag.
        if current_direction == 'left':
            zig_zag_depth = zig_zag(node.right, 'right', current_depth+1)
            non_zig_zag_depth = zig_zag(node.left, 'left')
        else:
            zig_zag_depth = zig_zag(node.left, 'left', current_depth+1)
            non_zig_zag_depth = zig_zag(node.right, 'right')

        # This compares the depth reached from zig zagging or starting
        # a new zig zag.
        return max(zig_zag_depth, non_zig_zag_depth)

    return max(zig_zag(root.left, 'left'), zig_zag(root.right, 'right'))


def lowestCommonAncestor(
        root: TreeNode,
        p: TreeNode,
        q: TreeNode
) -> TreeNode:
    """
    236. Lowest Common Ancestor of a Binary Tree
    The lowest common ancestor of two nodes p and q is the lowest node
    in the tree that has both the nodes as children.  p or q can be the
    lca if it has the other as its child.  This finds and returns the
    lca of the two given nodes.
    """
    def pre_order_traversal_lca(
            node: TreeNode,
            p: TreeNode,
            q: TreeNode,
    ) -> TreeNode:
        """
        This takes in the root of a tree and two nodes p and q.  It
        uses pre-order traversal to search through the tree until it
        finds the lca of the nodes.
        """
        if node is not None:
            if node == p or node == q:
                return node
            found_left = pre_order_traversal_lca(node.left, p, q)
            found_right = pre_order_traversal_lca(node.right, p, q)

            # This checks if the node is the lca, because it has p or q
            # as its left child and the other as its right child.
            if found_left and found_right:
                return node

            # Only one of p or q was found.  That node is the lca,
            # because the other node was not found in the other side of
            # the tree.  So, p and q are on the same side of the tree.
            # This means the other node is a child further down the
            # tree of the one that was found.
            if found_left:
                return found_left
            if found_right:
                return found_right

        return None

    return pre_order_traversal_lca(root, p, q)


def rightSideView(root: TreeNode) -> list[int]:
    """
    199. Binary Tree Right Side View
    This returns the node values from top to bottom if you are facing a
    tree from its right side.
    """
    # This is the base case when the tree is empty.
    if root is None:
        return []

    node_values = []
    queue = collections.deque()
    queue.append(root)
    # This uses breadth-first traversal on the tree to find the right
    # most node in each level.
    while len(queue) > 0:
        node_values.append(queue[-1].val)
        current_level_length = len(queue)
        # This iterates through each node in the level.  It adds its
        # children to the end of the queue.
        for _ in range(current_level_length):
            node = queue.popleft()
            if node.left is not None:
                queue.append(node.left)
            if node.right is not None:
                queue.append(node.right)

    return node_values


def maxLevelSum(root: TreeNode) -> int:
    """
    1161. Maximum Level Sum of a Binary Tree
    This finds and returns the level with the greatest sum of node
    values.  The level of the root is 1.
    """
    max_level = 1
    max_sum = root.val
    current_level = 1
    queue = collections.deque()
    queue.append(root)
    # This uses breadth-first traversal on the tree to calculate the sum
    # of the nodes in a level.
    while len(queue) > 0:
        current_level_length = len(queue)
        current_level_sum = 0
        for _ in range(current_level_length):
            node = queue.popleft()
            current_level_sum += node.val
            if node.left is not None:
                queue.append(node.left)
            if node.right is not None:
                queue.append(node.right)
        # This checks if the current level is a new max.
        if current_level_sum > max_sum:
            max_sum = current_level_sum
            max_level = current_level
        current_level += 1

    return max_level


def searchBST(root: TreeNode, val: int) -> Union[TreeNode, None]:
    """
    700. Search in a Binary Search Tree
    This finds and returns the node with a value equal to the given val
    in a binary search tree.  If the node does not exist, this returns
    None.
    """
    current_node = root
    # This traverses through the tree until it finds the node or
    # reaches the end of the path.
    while True:
        if current_node.val == val:
            return current_node
        if val < current_node.val:
            current_node = current_node.left
        else:
            current_node = current_node.right
        # This checks if the end of the path has been reached, so the
        # node does not exist.
        if current_node is None:
            return None


def deleteNode(root: TreeNode, key: int) -> Union[TreeNode, None]:
    """
    450. Delete Node in a BST
    This finds and deletes the node with a value equal to the given key
    in a binary search tree.  It then returns the root of the tree.  If
    the node does not exist, it returns None.
    """
    # This is the base case when the tree is empty.
    if root is None:
        return None

    parent_node = None
    current_node = root
    # This traverses through the tree until it finds the node or
    # reaches the end of the path.
    while True:
        # This runs when the node has been found.  It deletes the node
        # and handles any children.
        if key == current_node.val:
            # This runs when the node has both a left and right child.
            if (
                    current_node.left is not None
                    and current_node.right is not None
            ):
                # This saves a reference to the left child.
                left_child = current_node.left
                # This replaces the node with its right child.
                if parent_node is None:
                    root = current_node.right
                else:
                    if current_node is parent_node.left:
                        parent_node.left = current_node.right
                    else:
                        parent_node.right = current_node.right
                # This adds the left child somewhere as a descendant of
                # the right child.
                parent_node = None
                current_node = current_node.right
                # This traverses through the right child subtree until
                # it finds the appropriate place for the left child.
                while current_node is not None:
                    parent_node = current_node
                    if left_child.val < current_node.val:
                        current_node = current_node.left
                    else:
                        current_node = current_node.right
                if left_child.val < parent_node.val:
                    parent_node.left = left_child
                else:
                    parent_node.right = left_child
            # This runs when the node only has a left child.
            elif current_node.left is not None and current_node.right is None:
                # This replaces the node with its left child.
                if parent_node is None:
                    root = current_node.left
                else:
                    if current_node is parent_node.left:
                        parent_node.left = current_node.left
                    else:
                        parent_node.right = current_node.left
            # This runs when the node only has a right child.
            elif current_node.left is None and current_node.right is not None:
                # This replaces the node with its right child.
                if parent_node is None:
                    root = current_node.right
                else:
                    if current_node is parent_node.left:
                        parent_node.left = current_node.right
                    else:
                        parent_node.right = current_node.right
            # This runs when the node has no children.
            else:
                # This deletes the node.
                if parent_node is None:
                    return None
                if current_node is parent_node.left:
                    parent_node.left = None
                else:
                    parent_node.right = None
            return root

        parent_node = current_node
        if key < current_node.val:
            current_node = current_node.left
        else:
            current_node = current_node.right
        # This checks if the end of the path has been reached, so the
        # node does not exist.
        if current_node is None:
            return root


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
                    visited: set[(int, int)]) -> None:
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


def canVisitAllRooms(rooms: list[list[int]]) -> bool:
    """
    841. Keys and Rooms
    rooms represents rooms numbered 0 to the length of rooms - 1.  All
    of the rooms are locked except for room 0.  Each room contains keys
    to other rooms.  You start at room 0.  This returns True if you can
    visit every room.  Otherwise, it returns False.
    """
    def visit_room(
            rooms: list[list[int]],
            room: int,
            visited: set = None
    ) -> set:
        """
        This adds the given room to the set of visited rooms.  It then
        iterates through the keys and visits rooms that have not been
        visited before.
        """
        if visited is None:
            visited = set()

        visited.add(room)
        for key in rooms[room]:
            if key not in visited:
                visit_room(rooms, key, visited)

        return visited

    # This visits every possible room and checks if all rooms have been
    # visited.
    return len(visit_room(rooms, 0)) == len(rooms)


def findCircleNum(isConnected: list[list[int]]) -> int:
    """
    547. Number of Provinces
    Each element of isConnected represents connections that a city has.
    Each element in a city's list of connections represents another
    city.  The element is 1 if the two cities are connected, otherwise
    it is 0.  A province is a group of cities that are either directly
    or indirectly connected to each other.  This finds the number of
    provinces and returns it.
    """
    def dfs_cities(
            connections: list[list[int]],
            city: int,
            visited: set
    ) -> None:
        """
        This performs a depth-first search for a province.  It visits
        every city within the province.
        """
        for i in range(len(connections[city])):
            if connections[city][i] == 1 and i not in visited:
                visited.add(i)
                dfs_cities(connections, i, visited)

        return None

    visited = set()
    provinces = 0
    # This iterates through each city.  For each city, it visits every
    # other city in that province.  As the iteration goes on, if the
    # city has already been visited then it is skipped.  If the city
    # has not already been visited then it is a new province.
    for i in range(len(isConnected)):
        if i not in visited:
            provinces += 1
            visited.add(i)
            dfs_cities(isConnected, i, visited)

    return provinces


def minReorder(n: int, connections: list[list[int]]) -> int:
    """
    1466. Reorder Routes to Make All Paths Lead to the City Zero
    Each list in connections represents a one direction road from the
    city that is the value of the first element to the second element
    value city.  This finds and returns the number of roads that need
    to have the direction flipped in order for every city to be able to
    reach city 0.
    """
    def dfs(
            roads: set[(int, int)],
            neighbors: dict[int, list[int]],
            visited: set[int],
            changes: list[int],
            city: int
    ) -> None:
        """
        This visits every city using depth-first search.  It checks if
        the city's neighbors can reach it.  If not, the road between
        the cities needs to be flipped.  
        """
        for connected_city in neighbors[city]:
            if connected_city in visited:
                continue
            if (connected_city, city) not in roads:
                changes[0] += 1
            visited.add(connected_city)
            dfs(roads, neighbors, visited, changes, connected_city)

        return None

    # This creates a set of the roads to be able to access them in
    # constant time.
    roads = set()
    for connection in connections:
        roads.add(tuple(connection))

    # This creates a dictionary of a city's neighbors.  The key is the
    # city int, and the value is a list of the neighbor ints.
    neighbors = {}
    for i in range(n):
        neighbors[i] = []
    for a, b in connections:
        neighbors[a].append(b)
        neighbors[b].append(a)

    changes = [0]
    visited = set([0])
    dfs(roads, neighbors, visited, changes, 0)

    return changes[0]


def calcEquation(
        equations: list[list[str]],
        values: list[float],
        queries: list[list[str]]
) -> list[float]:
    """
    399. Evaluate Division
    Each element in equations has two elements.  Each element is a
    variable, and the first is the numerator while the second is the
    denominator.  values[i] equals the result of division between the
    variables in equations[i].  queries is structured the same as
    equations.  This returns a list where list[i] equals the result of
    division between the variables in queries[i].  If a variable does
    not exist in equations, the result is -1.
    """
    def bfs_equations(
            start: str,
            end: str,
            graph: dict[str, list[tuple[str, float]]]
    ):
        """
        This performs a breadth-first search on a graph connecting the
        variables.  It begins with the numerator (start) and continues
        to add neighbors and the amount to reach that neighbor from the
        start until it finds the denominator (end).
        """
        # This runs when the variable does not exist.
        if start not in graph or end not in graph:
            return -1

        queue = collections.deque()
        visited = set([start])
        queue.append([start, 1])

        while len(queue) > 0:
            variable, amount_to_variable = queue.popleft()
            if variable == end:
                return amount_to_variable
            for neighbor, weight in graph[variable]:
                # This does not add neighbors that have already been
                # visited, otherwise it will cause a cycle.
                if neighbor not in visited:
                    queue.append([neighbor, weight * amount_to_variable])
                    visited.add(neighbor)

        # Both the start and end variables exist, but there is no path
        # between them.
        return -1

    # This creates a graph connecting the variables.  The key is a
    # variable (numerator).  The value is a tuple.  The first element
    # in the tuple is another variable (denominator).  The second
    # element is the result of division between the numerator and
    # denominator variables.
    graph = collections.defaultdict(list)
    for i, equation in enumerate(equations):
        numerator, denominator = equation
        graph[numerator].append((denominator, values[i]))
        graph[denominator].append((numerator, 1 / values[i]))

    return [
        bfs_equations(numerator, denominator, graph)
        for numerator, denominator in queries
    ]


def nearestExit(maze: list[list[str]], entrance: list[int]) -> int:
    """
    1926. Nearest Exit from Entrance in Maze
    maze is a 2-D list where maze[i] represents rows and the elements
    of maze[i] represent columns.  A spot is either open with a value
    of '.' or a wall with a value of '+'.  You start at the row, column
    spot given by entrance.  You can move up, down, left, or right.
    This finds and returns the smallest number of moves to reach a spot
    that is a border of the maze.  If there is no way to the border, it
    returns -1.
    """
    ROWS = len(maze)
    COLS = len(maze[0])

    distances = [[float('inf')] * COLS for _ in range(ROWS)]
    queue = collections.deque()
    distances[entrance[0]][entrance[1]] = 0
    queue.append((entrance[0], entrance[1], 0))
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    # This uses a breadth-first search starting from the entrance.  The
    # spots in a level are one move away from the spots in the previous
    # level.  It skips over invalid spots and spots already in the
    # queue.
    while len(queue) > 0:
        row, col, distance = queue.popleft()
        # This checks if the border has been reached.
        if (
                row == 0
                or row == ROWS - 1
                or col == 0
                or col == COLS - 1
        ):
            # This checks that the spot is not the entrance.
            if not (row == entrance[0] and col == entrance[1]):
                return distance
        for row_move, col_move in directions:
            new_row = row + row_move
            new_col = col + col_move
            # This checks if the spot to move to is invalid.  This can
            # be because it is not in the maze, it is a wall, it has
            # already been visited, or it is already in the queue.
            if (
                0 <= new_row < ROWS
                and 0 <= new_col < COLS
                and maze[new_row][new_col] != '+'
                and distances[new_row][new_col] == float('inf')
            ):
                distances[new_row][new_col] = distance + 1
                queue.append((new_row, new_col, distance+1))

    return -1


def orangesRotting(grid: list[list[int]]) -> int:
    """
    994. Rotting Oranges
    grid is a 2-D list where grid[i] represents rows and the elements
    of grid[i] represent columns.  A spot is either empty with a value
    of 0, a fresh orange with a value of 1, or a rotten orange with a
    value of 2.  Each minute, a rotten orange can turn a fresh orange
    in a spot up, down, left, or right from it to rotten.  This
    calculates and returns the minimum number of minutes it takes to
    have every orange be rotten.  If not every orange can be rotten, it
    returns -1.
    """
    ROWS = len(grid)
    COLS = len(grid[0])

    queue = collections.deque()
    fresh_orange_count = 0
    # This counts the number of fresh oranges and the locations of
    # every rotten orange in the grid.
    for i in range(ROWS):
        for j in range(COLS):
            if grid[i][j] == 1:
                fresh_orange_count += 1
            elif grid[i][j] == 2:
                queue.append((i, j))

    minute = 0
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    # This uses a breadth-first search starting from each rotten
    # orange.  The spots in a level are one direction away from the
    # spots in the previous level.  It skips over invalid spots and
    # spots that do not have a fresh orange.
    while len(queue) > 0 and fresh_orange_count > 0:
        for _ in range(len(queue)):
            row, col = queue.popleft()
            for row_move, col_move in directions:
                new_row = row + row_move
                new_col = col + col_move
                # This checks if the next spot is invalid.  This can be
                # because it is not in the grid or not a fresh orange.
                # If so, it is skipped over.
                if (
                    new_row < 0
                    or new_row >= ROWS
                    or new_col < 0
                    or new_col >= COLS
                    or grid[new_row][new_col] != 1
                ):
                    continue
                # The next spot is in the grid and is a fresh orange.
                grid[new_row][new_col] = 2
                fresh_orange_count -= 1
                queue.append((new_row, new_col))
        minute += 1

    if fresh_orange_count == 0:
        return minute
    return -1


class RecentCounter:
    """
    933. Number of Recent Calls
    This class stores and counts recent requests within a certain time
    frame.
    """

    def __init__(self):
        self.request_times = collections.deque()

    def ping(self, t: int) -> int:
        """
        This takes in a new request at time t in milliseconds and
        stores it.  It then returns the number of requests from time
        [t-3000, t] both inclusive.
        """
        # This stores the new request.
        self.request_times.append(t)

        # This removes the oldest requests until every time is within
        # the range [t-3000, t].
        while self.request_times[0] < t - 3000:
            self.request_times.popleft()

        return len(self.request_times)


class SmallestInfiniteSet:
    """
    2336. Smallest Number in Infinite Set
    This class contains a set of all positive integers.
    """

    def __init__(self):
        self.set = []
        for num in range(1, 1001):
            heapq.heappush(self.set, num)

    def popSmallest(self) -> int:
        """
        This removes and returns the smallest value in the set.
        """
        return heapq.heappop(self.set)

    def addBack(self, num: int) -> None:
        """
        This adds the given value to the set if it is not already in
        it.
        """
        if num not in self.set:
            heapq.heappush(self.set, num)
        return None


class TrieNode:
    """
    These are nodes used for the class Trie.
    """

    def __init__(self):
        self.children = {}
        self.end_of_word = False


class Trie:
    """
    208. Implement Trie (Prefix Tree)
    This class stores and retrieves keys.
    """

    def __init__(self):
        """
        This uses a tree where each node is a character.
        """
        self.root = TrieNode()

    def insert(self, word: str) -> None:
        """
        This adds a new key to the tree.
        """
        i = 0
        node = self.root
        # This checks if part of word is already in the tree.
        while i < len(word) and word[i] in node.children:
            node = node.children[word[i]]
            i += 1
        # This adds the remaining characters of word to the tree.
        for j in range(i, len(word)):
            new_node = TrieNode()
            node.children[word[j]] = new_node
            node = new_node
        # This marks the node that is the last character in word.
        node.end_of_word = True
        return None

    def search(self, word: str) -> bool:
        """
        This checks if word is a key that has been added to the tree.
        """
        i = 0
        node = self.root
        # This iterates through each character in word and the tree.
        # For each character, it checks if it is one of the next nodes
        # in the tree.
        while i < len(word):
            if word[i] not in node.children:
                return False
            node = node.children[word[i]]
            i += 1
        # All of the characters in word are in the tree.  However, word
        # can be a substring of one of the keys, in which case it has
        # not been added.  So, the node for the last character in word
        # has to be marked as the end of a key.
        return node.end_of_word

    def startsWith(self, prefix: str) -> bool:
        """
        This checks if prefix is a substring of a key in the tree
        starting at index 0.
        """
        i = 0
        node = self.root
        # This iterates through each character in prefix and the tree.
        # For each character, it checks if it is one of the next nodes
        # in the tree.
        while i < len(prefix):
            if prefix[i] not in node.children:
                return False
            node = node.children[prefix[i]]
            i += 1
        # All of the characters in prefix are in the tree.
        return True


class StockSpanner:
    """
    901. Online Stock Span
    This class stores daily stock prices.
    """

    def __init__(self):
        self.stack = []

    def next(self, price: int) -> int:
        """
        This takes in the price of a stock on a day.  It returns the
        number of consecutive days the previous stock prices were less
        than or equal to the current price, plus one for the current
        day.
        """
        span = 1
        # This stores previous stock prices and spans in a stack.  It
        # continues to pop off the stack while the current price is
        # greater than or equal to the top of the stack.  Each pop
        # contains the span for the stock from that day.  This previous
        # span is then added to the current price's span.  The current
        # price is greater than or equal to the previous price.  So,
        # the span for the current stock will include all of the days
        # that were part of the span of the previous stock.
        while len(self.stack) > 0 and price >= self.stack[-1][0]:
            _, previous_span = self.stack.pop()
            span += previous_span

        # This runs once all of the past lower stocks have been popped
        # off the stack, the stack is empty, or the current price is
        # less than the previous price.  It adds the current price and
        # its span to the top of the stack.
        self.stack.append((price, span))

        return span


def maxScore(nums1: list[int], nums2: list[int], k: int) -> int:
    """
    2542. Maximum Subsequence Score
    You must choose a subsequence of indices of k length.  These
    indices are used to get elements from nums1 and nums2.  The nums1
    elements are summed together.  The minimum value of the nums2
    elements is found.  The result is the product of the sum and the
    min value.  This finds and returns the maximum possible result.
    """
    def second_element(pair: tuple) -> int:
        """
        This returns the second element in the pair.
        """
        return pair[1]

    # This creates a list of tuple pairs.  The first element is from
    # nums1.  The second element is from nums2 in the same index
    # position.
    pairs = [pair for pair in zip(nums1, nums2)]
    # This sorts the pairs in descending order based on the nums2
    # value.
    pairs.sort(key=second_element, reverse=True)

    nums1_sum = 0
    min_heap = []
    # This starts from the first element in pairs and iterates through
    # it in order.  It adds pairs to the current subsequence until it
    # contains k elements.
    for i in range(k):
        nums1_sum += pairs[i][0]
        heapq.heappush(min_heap, pairs[i][0])
    # The min of the nums2 values is from the kth element (0-indexed)
    # in pairs, because it is sorted in descending order.
    max_score = nums1_sum * pairs[k-1][1]

    # This iterates through each pair after the first k pairs to
    # determine if there is a new max score.
    for i in range(k, len(pairs)):
        # A new pair will be added to the current subsequence.  So, a
        # pair currently in it needs to be removed.  This removes the
        # pair with the smallest nums1 value.
        smallest_nums1 = heapq.heappop(min_heap)
        nums1_sum -= smallest_nums1
        # This adds the new pair to the current subsequence.
        heapq.heappush(min_heap, pairs[i][0])
        nums1_sum += pairs[i][0]
        # Since pairs is sorted in descending order based on nums2
        # values, the nums2 value for the new pair is the min.  This
        # checks if there is a new max score.
        max_score = max(max_score, nums1_sum * pairs[i][1])

    return max_score


def totalCost(costs: list[int], k: int, candidates: int) -> int:
    """
    2462. Total Cost to Hire K Workers
    Each element of costs represents a worker and the cost to hire
    them.  The workers have an order based on their placement in costs.
    You hire the cheapest worker k times.  Each time, you can pick from
    the first or last candidates amount of workers.  Once a worker is
    picked, they are removed and not available in the next rounds.
    This returns the total cost you spent.
    """
    min_heap = []

    # This adds the workers available to be picked in the first round
    # to a heap.
    left_index = 0
    for _ in range(candidates):
        heapq.heappush(min_heap, (costs[left_index], 'left'))
        left_index += 1
    right_index = len(costs) - 1
    for _ in range(candidates):
        if right_index >= left_index:
            heapq.heappush(min_heap, (costs[right_index], 'right'))
            right_index -= 1

    total_cost = 0
    # This picks the cheapest worker k times.  Depending on whether the
    # worker was from the first or last side, it adds the next worker
    # to the heap.
    for _ in range(k):
        cost, side = heapq.heappop(min_heap)
        total_cost += cost
        if left_index <= right_index:
            if side == 'left':
                heapq.heappush(min_heap, (costs[left_index], 'left'))
                left_index += 1
            else:
                heapq.heappush(min_heap, (costs[right_index], 'right'))
                right_index -= 1

    return total_cost


def merge(nums1: list[int], m: int, nums2: list[int], n: int) -> None:
    """
    88. Merge Sorted Array
    This takes in two sorted ascending lists.  It modifies nums1 to
    contain all of the elements from both lists in ascending order.
    """
    nums1_index = m - 1
    nums2_index = n - 1
    insert_index = m + n - 1

    # nums1 contains its elements and then nums2 length 0s.  This
    # iterates from the last 0 in nums1 to the start.  It uses pointers
    # that move from the last elements in nums1 and nums2 to the starts
    # of the lists.  Each iteration, it moves the larger pointer
    # element to the current iteration index.
    while nums1_index >= 0 and nums2_index >= 0:
        if nums1[nums1_index] >= nums2[nums2_index]:
            nums1[insert_index] = nums1[nums1_index]
            nums1_index -= 1
        else:
            nums1[insert_index] = nums2[nums2_index]
            nums2_index -= 1
        insert_index -= 1

    # This moves any remaining elements from nums2 to nums1.  If the
    # opposite occurs and there are no remaining elements in nums2 but
    # there are some remaining in nums1, they do not need to be moved
    # since they are already in their correct spots.
    while nums2_index >= 0:
        nums1[insert_index] = nums2[nums2_index]
        nums2_index -= 1
        insert_index -= 1

    return None


def removeDuplicates(nums: list[int]) -> int:
    """
    80. Remove Duplicates from Sorted Array II
    This takes in a sorted ascending list.  It alters the list in-place
    so that the first i elements are the unique elements still in
    ascending order.  If the unique element was present two or more
    times in the list, it is present only twice in the first i
    elements.  If it was present only once, it is still present only
    once.  It then returns i.
    """
    # This is the base case when there is only one element.
    if len(nums) == 1:
        return 1

    left_index = 0
    right_index = 1
    modifier_index = 0
    # This iterates through the list and modifies it in-place.
    while left_index < len(nums):
        current_num = nums[left_index]
        # This uses two pointers that are next to each other.  If the
        # pointer elements are not the same, the current element (left
        # pointer) is present in the list only once.  So, the current
        # spot to be modified is changed to the current element.
        if right_index >= len(nums) or nums[left_index] != nums[right_index]:
            nums[modifier_index] = current_num
            modifier_index += 1
        # If the pointer elements are the same, the current element is
        # present in the list two or more times.  So, the current spot
        # to be modified and the spot after it are both changed to the
        # current element.
        else:

            nums[modifier_index] = current_num
            nums[modifier_index + 1] = current_num
            modifier_index += 2
        # This increments the two pointers until the left pointer is on
        # a new element.
        while left_index < len(nums) and nums[left_index] == current_num:
            left_index += 1
            right_index += 1

    return modifier_index


def majorityElement(nums: list[int]) -> int:
    """
    169. Majority Element
    The majority element is the element that is in the list more than
    the length / 2 times.  This takes in a list that does contain a
    majority element.  It finds and returns that element.
    """
    result = nums[0]
    count = 1
    # This iterates through the second element to the last.  It keeps
    # track of a stored element and compares it to the current element
    # each iteration.  If they are the same, the element's count is
    # incremented.  If they are different, the element's count is
    # decremented.  If the count ever reaches 0, the stored element is
    # updated to the current element.  Since there has to be a majority
    # element, once it becomes the stored element its count may always
    # stay above 0.  Or, it may reach a count of 0 and be set later
    # again as the stored element multiple times.  In this case, all of
    # the other elements that are stored will reach a count of 0.
    # Eventually, the majority element will be stored and the count
    # will stay above 0, or it will be the last element in the list and
    # set in the last iteration.
    for i in range(1, len(nums)):
        if nums[i] == result:
            count += 1
        else:
            count -= 1
        if count == 0:
            result = nums[i]
            count = 1

    return result


def rotate(nums: list[int], k: int) -> None:
    """
    189. Rotate Array
    This rotates every element in the list to the right k times.  It
    modifies the list in-place and does not return anything.
    """
    # k may be greater than the length of the list.  In this case, the
    # rotation would just start to repeat.  So, this changes k to a
    # value less than the length that results in the same final
    # rotation.
    k = k % len(nums)

    # In the original list, the last k elements will be the starting
    # elements of the new rotated list.  So, by reversing the original
    # list, those last k elements are now the starting elements.
    # However, they are in reverse order.  After them are the remaining
    # elements that are also in reverse order.
    nums.reverse()

    # This reverses the first k elements to put them in their proper
    # order.
    left_index = 0
    right_index = k - 1
    while left_index < right_index:
        left_element = nums[left_index]
        nums[left_index] = nums[right_index]
        nums[right_index] = left_element
        left_index += 1
        right_index -= 1

    # This reverses the elements after the first k elements to put them
    # in their proper order.
    left_index = k
    right_index = len(nums) - 1
    while left_index < right_index:
        left_element = nums[left_index]
        nums[left_index] = nums[right_index]
        nums[right_index] = left_element
        left_index += 1
        right_index -= 1

    return None


def maxProfit(prices: list[int]) -> int:
    """
    122. Best Time to Buy and Sell Stock II
    This takes in a list of stock prices each day.  You can only hold
    one stock at a time.  You can buy and sell as many times as you
    want.  This calculates and returns the maximum profit you can
    achieve.
    """
    max_profit = 0

    # This iterates from the second to the last price.  If the price is
    # greater than the previous day, you can make a profit, because you
    # buy on the previous day and sell on the current day.  If the
    # price is less than or equal to the previous day, you do nothing.
    for i in range(1, len(prices)):
        if prices[i] > prices[i - 1]:
            max_profit += prices[i] - prices[i - 1]

    return max_profit


def canJump(nums: list[int]) -> bool:
    """
    55. Jump Game
    You start at the first element of nums.  The value for each element
    is the maximum number of elements you can jump ahead to.  This
    returns True if you can reach the last element, otherwise it
    returns False.
    """
    target_index = len(nums) - 1

    # This iterates from the second to last element to the first.  The
    # initial target is the last element.  If you can reach the target
    # from the current element, then the current element becomes the
    # new target, because if you can reach it then you can reach the
    # last element.
    for i in range(len(nums)-2, -1, -1):
        if i + nums[i] >= target_index:
            target_index = i

    if target_index == 0:
        return True
    return False


def jump(nums: list[int]) -> int:
    """
    45. Jump Game II
    You start at the first element of nums.  The value for each element
    is the maximum number of elements you can jump ahead to.  It is guaranteed that
    you can reach the last element.  This calculates and returns the minimum number of jumps
    to reach the last element.
    """
    jumps = 0
    left_index = 0
    right_index = 0

    # This uses 2 indices to track a subsection of the list.  Each iteration, the subsection
    # makes a jump.  The left end moves to the element after the right end of the current subsection.  The right end
    # moves to the furthest element that can be reached based on the possible jumps from
    # the current subsection.
    while right_index < len(nums) - 1:
        new_right_index = 0
        for i in range(left_index, right_index+1):
            new_right_index = max(new_right_index, i + nums[i])
        left_index = right_index + 1
        right_index = new_right_index
        jumps += 1

    return jumps


def hIndex(citations: list[int]) -> int:
    """
    274. H-Index
    This takes in a list that represents papers.  The ith paper has
    citations[i] number of citations.  The h-index is the maximum
    number of papers, h, that all have at least h citations.  This
    calculates and returns the h-index.
    """
    # This creates a count of citations ranging from 0 to the number of
    # papers, n, both inclusive.  If a paper has more citations than n,
    # it is counted in the n citations group.
    citation_counts = {}
    for i in range(len(citations)+1):
        citation_counts[i] = 0
    for citation in citations:
        if citation >= len(citations):
            citation_counts[len(citations)] += 1
        else:
            citation_counts[citation] += 1

    # This iterates from the n citations group to the 0 group.  It
    # keeps track of the sum of paper counts.  If the sum ever equals
    # or is greater than the number of citations for the group, then
    # the h-index is the group number.  This is because there are
    # h-index or more papers that all have at least h-index citations.
    count = 0
    for i in range(len(citations), -1, -1):
        count += citation_counts[i]
        if count >= i:
            return i


class RandomizedSet:
    """
    380. Insert Delete GetRandom O(1)
    This class stores elements.  It can add, remove, and return a
    random element in constant time.
    """

    def __init__(self):
        self.dict = {}
        self.list = []

    def insert(self, val: int) -> bool:
        """
        If the value is not already present, it adds it to the
        dictionary and list.  Then it returns True.  Otherwise, it
        returns False.
        """
        if val in self.dict:
            return False
        # This adds the element to the dictionary and to the end of the
        # list.
        self.dict[val] = len(self.list)
        self.list.append(val)
        return True

    def remove(self, val: int) -> bool:
        """
        If the value is already present, it removes it from the
        dictionary and list.  Then it returns True.  Otherwise, it
        returns False.
        """
        if val not in self.dict:
            return False
        # This removes the element from the dictionary and from the
        # list by replacing it with the last element.
        remove_index = self.dict[val]
        last_element = self.list[-1]
        self.list[remove_index] = last_element
        self.dict[last_element] = remove_index
        del (self.dict[val])
        self.list.pop()
        return True

    def getRandom(self) -> int:
        """
        This returns a randon element.
        """
        # For a list of length n, the random number r will be
        # 0 <= r <= n-1.
        return self.list[random.randint(0, len(self.list)-1)]


def canCompleteCircuit(gas: list[int], cost: list[int]) -> int:
    """
    134. Gas Station
    There is a circular path with gas stations on it.  gas[i]
    represents how much gas you can fill up at the ith station.
    cost[i] represents how much gas you use traveling from the ith
    station to the i+1th station.  If you can make a full loop, there
    is only one possible gas station, g, that you can start from.
    Then, this finds and returns the index of g.  If you cannot make a
    full loop from any station, this returns -1.
    """
    # This checks if there's enough gas to make a full loop from any
    # station.
    if sum(cost) > sum(gas):
        return -1

    start = 0
    net_gas = 0
    # There is one single station you can leave from to make a full
    # loop.  This iterates through each station from the start.  It
    # tracks the net amount of gas you have, which is the amount of gas
    # you had when you arrived at the station, plus the gas you can
    # fill up, minus the gas you use to travel to the next station.  If
    # this amount ever drops below 0, you cannot start at the current
    # gas station or any of the ones before it, because eventually you
    # will run out of gas and not be able to continue traveling
    # forward.  In this case, the start moves to the next station and
    # the net gas amount is reset.  Since there has to be a result, it
    # is the station you started from once the iteration is finished.
    # Call this station g.  It cannot be any station before it, because
    # those resulted in running out of gas.  It cannot be any station
    # after it, h, because that means you traveled from h > g > h.
    # Since the net gas never dropped below 0, you are able to travel
    # from g > h.  If you reach h, you can then travel from h > g.
    # This means you can also complete a full loop starting at g.
    # However, this is a contradiction, because there can only be one
    # solution.
    for i in range(len(gas)):
        net_gas += (gas[i] - cost[i])
        if net_gas < 0:
            net_gas = 0
            start = i + 1

    return start


def candy(ratings: list[int]) -> int:
    """
    135. Candy
    There are children standing in a line.  ratings[i] represents the
    rating of the ith child.  Each child must get at least one candy.
    In addition, children with a higher rating than their neighbors
    must get more candy than their neighbor(s) with the lower rating.
    """
    # This ensures that each child gets at least one candy.
    candies = [1] * len(ratings)
    # This compares each child to their left neighbor to ensure they
    # get more candy if they have a higher rating.
    for i in range(1, len(ratings)):
        if ratings[i] > ratings[i-1]:
            candies[i] = candies[i-1] + 1
    # This compares each child to their right neighbor to ensure they
    # get more candy if they have a higher rating.  Changing an amount
    # does not affect the condition that higher rated children have
    # more candy than their left neighbor, because the left neighbor's
    # candy will only change if their rating is higher (than the right
    # neighbor).
    for i in range(len(ratings)-2, -1, -1):
        if ratings[i] > ratings[i+1]:
            # The child may already have a higher candy amount based on
            # their left neighbor.
            candies[i] = max(candies[i], candies[i+1] + 1)

    return sum(candies)


def trap(height: list[int]) -> int:
    """
    42. Trapping Rain Water
    height[i] is the height of a bar at the ith position.  Water can be
    trapped between bars.  This calculates and returns the amount of
    water that can be trapped.
    """
    # This is the base case when there is only one bar, so no water can
    # be trapped.
    if len(height) == 1:
        return 0

    # This creates two lists.  For the first list, the ith element is
    # the max bar height out of the bars to the left of it (not
    # inclusive).  For the second list, the ith element is the max bar
    # height out of the bars to the right of it (not inclusive).
    max_to_the_left = [0, height[0]]
    for i in range(1, len(height)-1):
        max_to_the_left.append(max(max_to_the_left[-1], height[i]))
    max_to_the_right = [0, height[-1]]
    for i in range(len(height)-2, 0, -1):
        max_to_the_right.append(max(max_to_the_right[-1], height[i]))
    max_to_the_right.reverse()

    result = 0
    # This iterates through each element.  The amount of water that can
    # be trapped is the lower of the max bar to the left and max bar to
    # the right, minus the height of the bar at the current spot.  If
    # the bar at the current spot is equal to or greater than that
    # lower amount, then no water can be trapped.
    for i in range(len(height)):
        water = min(max_to_the_left[i], max_to_the_right[i]) - height[i]
        result += water if water > 0 else 0

    return result


def romanToInt(s: str) -> int:
    """
    13. Roman to Integer
    This takes in a roman numeral up to 3,999 (inclusive).  It converts
    it to an integer and returns the value.
    """
    result = 0
    i = 0
    # This iterates through each numeral and adds its value to the
    # result.  If the numeral and the one after it are a pair, it adds
    # the pair value and then iterates through both numerals in the
    # pair.
    while i < len(s):
        match s[i]:
            case 'V':
                result += 5
            case 'L':
                result += 50
            case 'D':
                result += 500
            case 'M':
                result += 1000
            case 'I':
                if i+1 < len(s) and s[i+1] == 'V':
                    result += 4
                    i += 1
                elif i+1 < len(s) and s[i+1] == 'X':
                    result += 9
                    i += 1
                else:
                    result += 1
            case 'X':
                if i+1 < len(s) and s[i+1] == 'L':
                    result += 40
                    i += 1
                elif i+1 < len(s) and s[i+1] == 'C':
                    result += 90
                    i += 1
                else:
                    result += 10
            case 'C':
                if i+1 < len(s) and s[i+1] == 'D':
                    result += 400
                    i += 1
                elif i+1 < len(s) and s[i+1] == 'M':
                    result += 900
                    i += 1
                else:
                    result += 100
        i += 1

    return result


def intToRoman(num: int) -> str:
    """
    12. Integer to Roman
    This takes in an integer from 1 to 3,999 (both inclusive).  It
    returns the roman numeral version of the integer.
    """
    roman = ''
    multiplier = 1
    current_num = 0
    # This iterates through each digit from back to front.  It then
    # multiplies it by its place value to get the number that needs to
    # be added to the roman numeral.
    for i in range(len(str(num))-1, -1, -1):
        current_num = int(str(num)[i]) * multiplier
        # This determines the roman numeral(s) for the current number.
        # 1-3
        if current_num <= 3:
            current_roman = 'I' * current_num
        # 4
        elif current_num == 4:
            current_roman = 'IV'
        # 5-8
        elif current_num <= 8:
            current_roman = 'V'
            current_roman += 'I' * (current_num-5)
        # 9
        elif current_num == 9:
            current_roman = 'IX'
        # 10-30
        elif current_num <= 30:
            current_roman = 'X' * int(current_num/10)
        # 40
        elif current_num == 40:
            current_roman = 'XL'
        # 50-80
        elif current_num <= 80:
            current_roman = 'L'
            current_roman += 'X' * int((current_num-50) / 10)
        # 90
        elif current_num == 90:
            current_roman = 'XC'
        # 100-300
        elif current_num <= 300:
            current_roman = 'C' * int(current_num/100)
        # 400
        elif current_num == 400:
            current_roman = 'CD'
        # 500-800
        elif current_num <= 800:
            current_roman = 'D'
            current_roman += 'C' * int((current_num-500) / 100)
        # 900
        elif current_num == 900:
            current_roman = 'CM'
        # 1000-3000
        elif current_num <= 3000:
            current_roman = 'M' * int(current_num/1000)
        roman = current_roman + roman
        multiplier *= 10

    return roman


def lengthOfLastWord(s: str) -> int:
    """
    58. Length of Last Word
    This takes in a string that contains letters and spaces.  A word is
    a substring of letters with no spaces.  This finds and returns the
    length of the last word.
    """
    # The index starts at the last character in the string.
    i = len(s) - 1

    # This iterates backwards through all of the spaces at the end of
    # the string.
    while s[i] == ' ':
        i -= 1

    length = 0
    # This starts at the last character of the last word in the string.
    # It iterates backwards to count its length.
    while i >= 0 and s[i] != ' ':
        length += 1
        i -= 1

    return length


def longestCommonPrefix(strs: list[str]) -> str:
    """
    14. Longest Common Prefix
    This finds and returns the longest prefix common to every string in the list.
    If there is no common prefix, it returns an empty string.
    """
    common_prefix = ''
    i = 0
    # This starts at index position 0 and moves to the next one each
    # iteration.  It checks the characters for each string at the
    # current index position.  If the position is outside of the string
    # length, or the characters are not all the same, it returns the
    # longest common prefix found up to that point.
    while True:
        for string in strs:
            if i >= len(string) or string[i] != strs[0][i]:
                return common_prefix
        common_prefix += strs[0][i]
        i += 1


def convert(s: str, numRows: int) -> str:
    """
    6. Zigzag Conversion
    This separates s into a zigzag pattern with numRows amount of rows.
    It then moves through each row from top to bottom and gets the
    row's characters.  Each row's characters are then combined together
    and returned.
    """
    # This is the base case when there is no zigzag pattern.
    if numRows == 1:
        return s

    result = ''

    # This determines the characters in the first row.  Starting from
    # the first character in s, it is every 2 * (numRows-1) character,
    # since it takes numRows-1 characters to reach the bottom, and the
    # same number of characters to then reach the top again.
    i = 0
    while i < len(s):
        result += s[i]
        i += 2 * (numRows - 1)

    # This determines the characters in the rows between the first and
    # last.  Starting from the current row character in s, it is every
    # 2 * (numRows-row-1) character and then every 2 * row character,
    # since it takes numRows-row-1 characters to reach the bottom, and
    # the same number of characters to then reach the row again.  After
    # that, it takes row number of characters to reach the top, and the
    # same number to then reach the row again.
    for row in range(1, numRows-1):
        i = row
        top_to_bottom = True
        while i < len(s):
            result += s[i]
            if top_to_bottom:
                i += 2 * (numRows - row - 1)
                top_to_bottom = False
            else:
                i += 2 * row
                top_to_bottom = True

    # This determines the characters in the last row.  Starting from
    # the last row character in s, it is every 2 * (numRows-1)
    # character, since it takes numRows-1 characters to reach the top,
    # and the same number of characters to then reach the bottom again.
    i = numRows - 1
    while i < len(s):
        result += s[i]
        i += 2 * (numRows - 1)

    return result


def strStr(haystack: str, needle: str) -> int:
    """
    28. Find the Index of the First Occurrence in a String
    This returns the index of the first occurence of needle in
    haystack.  If it is not in haystack, it returns -1.
    """
    # This iterates through each character in haystack.  It checks if
    # the substring starting at the current character is equal to
    # needle.
    for i in range(len(haystack)):
        if haystack[i: i+len(needle)] == needle:
            return i

    # needle is not in haystack.
    return -1


def fullJustify(words: list[str], maxWidth: int) -> list[str]:
    """
    68. Text Justification
    This combines the strings from words into left and right justified
    lines.  Each line has a length equal to maxWidth.  To reach this
    length, each string in a line is separated by at least 1 space.
    Extra spaces are added from left to right in order to reach the
    required length.  The last line is only left justified with 1 space
    between words.  The resulting lines are then returned in a list.
    """
    result = []
    current_words = []
    current_words_length = 0
    i = 0
    # This iterates through each word.  If there is space, it adds the
    # word to a list of the current words for a line.  Otherwise, it
    # uses the current words to create the line.
    while i < len(words):
        # This checks if there is space for the next word.  The length
        # including the next word is the lengths of the current words
        # + 1 space between each of the current words + 1 space before
        # the next word + the length of the next word.  This is the
        # lengths of the current words
        # + (the number of current words - 1) + 1 + the length of the
        # next word.
        if (
                (current_words_length
                 + len(current_words) + len(words[i])) <= maxWidth
        ):
            # There is space for the next word, so it is added to the
            # current words.
            current_words.append(words[i])
            current_words_length += len(words[i])
            i += 1
        else:
            # There is no space for the next word, so the current words
            # need to be turned into a line.
            extra_spaces = 0
            current_string = ''
            # This iterates through each of the current words.  It adds
            # the word and then the required number of spaces to the
            # line.
            for j in range(len(current_words)):
                current_string += current_words[j]
                # This determines how many spaces to add.
                if len(current_words) == 1:
                    # There is only one word in the line, so the rest
                    # of it needs to be spaces.
                    current_string += ' ' * (maxWidth - current_words_length)
                else:
                    # There are multiple words in the line.  So, there
                    # needs to be remaining characters
                    # / (number of words - 1) spaces between them.  If
                    # there is a remainder, one extra space needs to be
                    # added until the remainder is 0.
                    spaces = (
                        (maxWidth - current_words_length - extra_spaces)
                        // (len(current_words) - 1)
                    )
                    if (
                            (maxWidth - current_words_length - extra_spaces)
                            % (len(current_words) - 1) != 0
                    ):
                        spaces += 1
                        extra_spaces += 1
                # This checks that spaces are not added after the last
                # word.
                if j != len(current_words) - 1:
                    current_string += ' ' * spaces
            result.append(current_string)
            current_words = []
            current_words_length = 0

    # This turns the remaining current words into the last line.
    current_string = ''
    for j in range(len(current_words)):
        current_string += current_words[j]
        if j != len(current_words) - 1:
            current_string += ' '
    while len(current_string) < maxWidth:
        current_string += ' '
    result.append(current_string)

    return result


def isPalindrome(s: str) -> bool:
    """
    125. Valid Palindrome
    s can contain any character.  This returns True if s is the same
    left to right as right to left, ignoring the case of letters as
    well as characters that are neither letters or numbers.  Otherwise,
    it returns False.
    """
    s = s.lower()
    # This creates a set of the allowed characters, which are all
    # lowercase letters and digits.
    allowed_characters = set(string.ascii_lowercase).union(set(string.digits))

    left_index = 0
    right_index = len(s) - 1
    # This iterates from the ends of s inwards.  It checks if each of
    # the characters are the same.  If not, then s is not a palindrome.
    while left_index <= right_index:
        # This checks if the current left and right characters are
        # valid.  If not, it skips over it.
        if s[left_index] not in allowed_characters:
            left_index += 1
            continue
        if s[right_index] not in allowed_characters:
            right_index -= 1
            continue
        if s[left_index] != s[right_index]:
            return False
        left_index += 1
        right_index -= 1

    return True


def minSubArrayLen(target: int, nums: list[int]) -> int:
    """
    209. Minimum Size Subarray Sum
    This finds and returns the length of the smallest subarray from
    nums with a sum greater than or equal to target.  If there is no
    subarray that meets the criteria, it returns 0 instead.
    """
    min_length = float('inf')
    subarray_sum = 0
    subarray_length = 0
    first_element_index = 0
    # This iterates through each element of nums.  It adds the current
    # element to the current subarray and checks if the subarray sum is
    # greater than or equal to the target in order to determine if
    # there is a new minimum length.  Then, it continues to remove the
    # first element in the subarray until the sum is less than target.
    # Each time, it again checks if there is a new minimum length.
    for num in nums:
        subarray_sum += num
        subarray_length += 1
        if subarray_sum >= target:
            min_length = min(min_length, subarray_length)
        while subarray_sum >= target:
            subarray_sum -= nums[first_element_index]
            subarray_length -= 1
            first_element_index += 1
            if subarray_sum >= target:
                min_length = min(min_length, subarray_length)

    # This checks if there was no subarray with a sum greater than or
    # equal to target.
    if min_length == float('inf'):
        return 0
    return min_length


def findSubstring(s: str, words: list[str]) -> list[int]:
    """
    30. Substring with Concatenation of All Words
    This finds all of the substrings of s that are permutations of
    words combined into a string.  It returns a list containing the
    start indices of the substrings.  If no substrings are found, it
    returns an empty list instead.
    """
    result = []
    words_counts = collections.Counter(words)

    i = 0
    # This iterates through each character in s until the remaining
    # characters are less than the total length of the permutation
    # string.
    while i <= (len(s) - len(words)*len(words[0])):
        s_counts = {}
        j = i
        # This starts at the current character.  It continues to get
        # the words and their counts until it has iterated through the
        # same number of words as the length of the argument words.
        for _ in range(len(words)):
            current_word = s[j:j+len(words[0])]
            s_counts[current_word] = s_counts.get(current_word, 0) + 1
            j += len(words[0])
        # This compares the substring's words and counts to the
        # argument words' words and counts.  If they are all the same,
        # then the current character is the start of a substring that
        # is a permutation of words.
        if s_counts == words_counts:
            result.append(i)
        i += 1

    return result


def minWindow(s: str, t: str) -> str:
    """
    76. Minimum Window Substring
    This finds and returns the smallest substring of s that contains
    every letter of t, including duplicates.  If there is no applicable
    substring, it returns an empty string instead.
    """
    result = None
    # This creates two dictionaries with the same keys, which are the
    # distinct characters of t.  The values for the first dict are the
    # counts of the keys in t, while the values for the second dict are
    # all 0.
    t_counts = collections.Counter(t)
    s_counts = {}
    for char in set(t):
        s_counts[char] = 0

    found_t_chars = 0
    left_index = 0
    right_index = 0
    # This iterates through s and keeps track of a substring using two
    # pointers.  It continues to iterate until the right pointer is out
    # of bounds.
    while right_index < len(s):
        # This increments the count of the current character in s and
        # checks if the new count is now equal to the count in t.  If
        # so, all instances of that character have now been found.  It
        # ignores characters not in t.
        current_char = s[right_index]
        if current_char in s_counts:
            s_counts[current_char] += 1
            if s_counts[current_char] == t_counts[current_char]:
                # This should only increment the found t chars when the
                # s count goes from less than to equal the t count,
                # because after this the same character may continue to
                # be found in s, so its count can continue to increase.
                found_t_chars += 1
        right_index += 1
        # This continues while all instances of every character in t
        # have been found in the current substring.  It checks if the
        # current substring is a new minimum and then removes the first
        # character.
        while found_t_chars == len(t_counts):
            if result is None or right_index - left_index < len(result):
                result = s[left_index:right_index]
            current_char = s[left_index]
            if current_char in s_counts:
                s_counts[current_char] -= 1
                if s_counts[current_char] < t_counts[current_char]:
                    found_t_chars -= 1
            left_index += 1

    if result is None:
        return ''
    return result


def isValidSudoku(board: list[list[str]]) -> bool:
    """
    36. Valid Sudoku
    This checks if a completed or not filled in sudoku board is valid.
    """
    row_digits = set()
    col_digits = set()
    # This checks if each of the rows and columns is valid.
    for i in range(9):
        for j in range(9):
            current_row_element = board[i][j]
            if current_row_element != '.':
                if current_row_element in row_digits:
                    return False
                row_digits.add(current_row_element)
            current_col_element = board[j][i]
            if current_col_element != '.':
                if current_col_element in col_digits:
                    return False
                col_digits.add(current_col_element)
        row_digits.clear()
        col_digits.clear()

    # This checks if each 3x3 square is valid.
    square_digits = set()
    starting_row = 0
    # This iterates through each row of squares.
    for _ in range(3):
        starting_col = 0
        # This iterates through each square in a row of squares.
        for _ in range(3):
            current_row = starting_row
            # This iterates through each row in a square.
            for _ in range(3):
                current_col = starting_col
                # This iterates through each spot in a row.
                for _ in range(3):
                    current_element = board[current_row][current_col]
                    if current_element != '.':
                        if current_element in square_digits:
                            return False
                        square_digits.add(current_element)
                    current_col += 1
                current_row += 1
            square_digits.clear()
            starting_col += 3
        starting_row += 3

    return True


def spiralOrder(matrix: list[list[int]]) -> list[int]:
    """
    54. Spiral Matrix
    This returns a list of the elements in order traversing through
    matrix in spiral order.
    """
    LAST_ROW = len(matrix) - 1
    LAST_COL = len(matrix[0]) - 1
    current_row = 0
    current_col = 0
    visited = set()
    direction = 'right'
    result = []
    # This iterates through matrix in spiral order until it has visited
    # every element.
    while len(result) < len(matrix)*len(matrix[0]):
        result.append(matrix[current_row][current_col])
        visited.add((current_row, current_col))
        # This comes up with the next position to visit based on the
        # current direction.
        match direction:
            case 'right':
                next_row = current_row
                next_col = current_col + 1
            case 'down':
                next_row = current_row + 1
                next_col = current_col
            case 'left':
                next_row = current_row
                next_col = current_col - 1
            case 'up':
                next_row = current_row - 1
                next_col = current_col
        # This checks if the next position is out of bounds or has
        # already been visited.  If so, the direction needs to change.
        if (
                next_col > LAST_COL
                or next_row > LAST_ROW
                or next_col < 0
                or (next_row, next_col) in visited
        ):
            match direction:
                case 'right':
                    direction = 'down'
                    current_row += 1
                case 'down':
                    direction = 'left'
                    current_col -= 1
                case 'left':
                    direction = 'up'
                    current_row -= 1
                case 'up':
                    direction = 'right'
                    current_col += 1
        else:
            # The next position is valid and the current direction
            # should stay the same.
            match direction:
                case 'right':
                    current_col += 1
                case 'down':
                    current_row += 1
                case 'left':
                    current_col -= 1
                case 'up':
                    current_row -= 1

    return result


def rotate(matrix: list[list[int]]) -> None:
    """
    48. Rotate Image
    This rotates the 2D list 90 degrees clockwise in place.
    """
    left_col_index = 0
    right_col_index = len(matrix) - 1
    top_row_index = 0
    bottom_row_index = len(matrix) - 1
    # This starts with the elements in the outer ring and rotates them.
    # It then moves the ring to the next level inward and performs the
    # rotation on those elements.  It continues to do this for each
    # level.
    while left_col_index < right_col_index:
        # Every row/column of the current ring contains n number of
        # elements.  Since the end element in a row/col is the start of
        # the next row/col, there are n-1 elements in a row/col that
        # need to be rotated.
        rotations = right_col_index - left_col_index
        # This starts with the first element in each of the top row,
        # right col, bottom row, and left col of the current ring.  It
        # rotates each one to its new spot.  It then moves onto the
        # next element and performs the rotation again.  It continues
        # to do this for the needed number of rotations.
        for i in range(rotations):
            top_row_value = matrix[top_row_index][left_col_index+i]
            right_col_value = matrix[top_row_index+i][right_col_index]
            bottom_row_value = matrix[bottom_row_index][right_col_index-i]
            left_col_value = matrix[bottom_row_index-i][left_col_index]
            # The right column spot becomes the top row value.
            matrix[top_row_index+i][right_col_index] = top_row_value
            # The bottom row spot becomes the right column value.
            matrix[bottom_row_index][right_col_index-i] = right_col_value
            # The left column spot becomes the bottom row value.
            matrix[bottom_row_index-i][left_col_index] = bottom_row_value
            # The top row spot becomes the left column value.
            matrix[top_row_index][left_col_index+i] = left_col_value
        left_col_index += 1
        right_col_index -= 1
        top_row_index += 1
        bottom_row_index -= 1

    return None


def setZeroes(matrix: list[list[int]]) -> None:
    """
    73. Set Matrix Zeroes
    This modifies the 2D list in place.  If an element is 0, all of the
    elements in that row and column are set to 0.
    """
    ROWS = len(matrix)
    COLS = len(matrix[0])
    first_row = matrix[0][0]
    # This iterates through each spot.  If an element is 0, the spot in
    # the element's row and the left column is set to 0.  Also, the
    # spot in the top row and the element's column is set to 0.
    for row in range(ROWS):
        for col in range(COLS):
            if matrix[row][col] == 0:
                # This uses a separate variable to represent the first
                # row, because the spot at matrix[0][0] also represents
                # the first column.
                if row == 0:
                    first_row = 0
                else:
                    matrix[row][0] = 0
                matrix[0][col] = 0

    # This iterates through each spot except for ones the first row and
    # first column.  If the value at the element's row and left column
    # is 0, then the spot is set to 0, because that means the entire
    # row needs to be 0.  Or, if the value at the top row and element's
    # column is 0, then the spot is set to 0, because that means the
    # entire column needs to be 0.
    for row in range(1, ROWS):
        for col in range(1, COLS):
            if matrix[row][0] == 0 or matrix[0][col] == 0:
                matrix[row][col] = 0

    # This checks if the first column needs to be all 0.
    if matrix[0][0] == 0:
        for row in range(ROWS):
            matrix[row][0] = 0
    # This checks if the first row needs to be all 0.
    if first_row == 0:
        for col in range(COLS):
            matrix[0][col] = 0

    return None


def gameOfLife(board: list[list[int]]) -> None:
    """
    289. Game of Life
    This goes through every spot and modifies the argument board in
    place based on if it is alive or dead and its neighbors (the 8
    adjacent spots).  If it is alive (value of 1) and has exactly 2 or
    3 alive neighbors, it stays alive.  Otherwise, it dies.  If it is
    dead (value of 0) and has exactly 3 alive neighbors, it becomes
    alive.  Otherwise, it stays dead. 
    """
    ROWS = len(board)
    COLS = len(board[0])

    # This iterates through each spot in the board.  It modifies the
    # value based on the following table:
    # original    new    modified value
    # 0           0      0
    # 1           0      1
    # 0           1      2
    # 1           1      3
    for row in range(ROWS):
        for col in range(COLS):
            sum = 0
            modified_values = True
            # This iterates through the current spot and the eight
            # adjacent spots from it.
            for row_offset in range(-1, 2):
                for col_offset in range(-1, 2):
                    neighbor_row = row + row_offset
                    neighbor_col = col + col_offset
                    # This checks if the adjacent spot is out of
                    # bounds.
                    if (
                        neighbor_row < 0
                        or neighbor_row >= ROWS
                        or neighbor_col < 0
                        or neighbor_col >= COLS
                    ):
                        continue
                    # This checks if it has reached the current spot.
                    # Before this, the adjacent spots had modified
                    # values.  Now, the remaining adjacent spots have
                    # their original values.
                    if neighbor_row == row and neighbor_col == col:
                        modified_values = False
                        continue
                    if modified_values:
                        # This checks if the original adjacent spot
                        # value was a 1.
                        if (
                            board[neighbor_row][neighbor_col] == 1
                            or board[neighbor_row][neighbor_col] == 3
                        ):
                            sum += 1
                    else:
                        # The adjacent spot value is its original
                        # value, either a 0 or 1.
                        sum += board[neighbor_row][neighbor_col]
            modified_values = True
            # This updates the current spot to its modified value based
            # on the rules and table.
            if board[row][col] == 0:
                if sum == 3:
                    board[row][col] = 2
            else:
                if sum == 2 or sum == 3:
                    board[row][col] = 3
                else:
                    board[row][col] = 1

    # This iterates through each spot in the board, which contains the
    # modified values.  It changes the value to the new value that it
    # should be based on the table.
    for row in range(ROWS):
        for col in range(COLS):
            if board[row][col] == 1:
                board[row][col] = 0
            elif board[row][col] == 2 or board[row][col] == 3:
                board[row][col] = 1

    return None


def canConstruct(ransomNote: str, magazine: str) -> bool:
    """
    383. Ransom Note
    This returns True if the argument ransomNote can be made using the
    characters of the argument magazine.  Each character can only be
    used once.  Otherwise, it returns False.
    """
    # This creates a dictionary where the keys are the characters in
    # magazine and the values are the count of the character.
    magazine_char_counts = collections.Counter(magazine)
    # This iterates through the characters in ransomNote.  It checks if
    # the character is in magazine and has not been used already.
    for char in ransomNote:
        if char in magazine_char_counts:
            if magazine_char_counts[char] == 0:
                return False
            magazine_char_counts[char] -= 1
        else:
            return False

    return True


def wordPattern(pattern: str, s: str) -> bool:
    """
    290. Word Pattern
    This checks if the words in s separated by spaces follow the same
    pattern as the characters of the argument pattern.
    """
    words = s.split()
    # This checks if there are too many characters in pattern or too
    # many words in s.
    if len(pattern) != len(words):
        return False

    mapping = {}
    mapped_words = set()
    # This iterates through each pattern character and its associated
    # word in s.  It checks if the mapping is the same, or if a new
    # mapping needs to be created.
    for i in range(len(pattern)):
        pattern_char = pattern[i]
        word = words[i]
        # This checks if the char has already been mapped.
        if pattern_char in mapping:
            # This checks if the word is the same as the mapped word
            # for the char.
            if mapping[pattern_char] != word:
                return False
        else:
            # This checks if the word has already been mapped to a
            # different char.
            if word in mapped_words:
                return False
            mapping[pattern_char] = word
            mapped_words.add(word)

    return True


def isAnagram(s: str, t: str) -> bool:
    """
    242. Valid Anagram
    This returns True if it contains the same characters the same
    number of times as s in any order.  Otherwise, it returns False.
    """
    s_counts = collections.Counter(s)
    t_counts = collections.Counter(t)

    return s_counts == t_counts


def groupAnagrams(strs: list[str]) -> list[list[str]]:
    """
    49. Group Anagrams
    This returns a 2D list.  Each inner list contains a group of
    anagrams from the strings in strs.  An anagram contains the same
    characters the same number of times.
    """
    # This creates a mapping of the letters a-z to the numbers 0-25.
    mapping = {}
    i = 0
    for char in string.ascii_lowercase:
        mapping[char] = i
        i += 1

    anagrams = {}
    # This iterates through each string.  It creates a list of 26
    # elements all initially 0.  For each character in the string, it
    # uses the mapping to index the list and increment the value.  That
    # list then becomes the key (as a tuple) representing the anagram.
    # If the anagram has already been found, it adds the string to the
    # anagram's group.  Otherwise, it starts a new group with the
    # string.
    for word in strs:
        counts = [0] * 26
        for char in word:
            counts[mapping[char]] += 1
        counts = tuple(counts)
        if counts in anagrams:
            anagrams[counts].append(word)
        else:
            anagrams[counts] = [word]

    # This returns the groups of each anagram.
    return list(anagrams.values())


def isHappy(n: int) -> bool:
    """
    202. Happy Number
    This takes each digit of n and squares it.  It then sums those
    numbers.  If the sum is 1, it returns True.  Otherwise, it replaces
    n with the sum and repeats the process.  If the sums are a cycle,
    it returns False.
    """
    sums = set()
    # This calculates the sum.  It checks if it is 1 or has been
    # encountered before, meaning there is a cycle.  If not, it repeats
    # the process.
    while True:
        str_n = str(n)
        sum = 0
        for digit in str_n:
            sum += int(digit)**2
        if sum == 1:
            return True
        if sum in sums:
            return False
        sums.add(sum)
        n = sum


def containsNearbyDuplicate(nums: list[int], k: int) -> bool:
    """
    219. Contains Duplicate II
    This returns True if for any element there is a duplicate value
    within k indices (inclusive).  Otherwise, it returns False.
    """
    window = set()
    # This iterates through each element.  It uses a sliding window of
    # nums of length k.  If the current element is already in the
    # window, then there is a duplicate.  Otherwise, it adds the
    # current element and removes the leftmost element in the window.
    for i in range(len(nums)):
        if nums[i] in window:
            return True
        window.add(nums[i])
        # This does not remove the leftmost element for the first k
        # iterations, because the window is not length k yet.
        if i >= k:
            window.remove(nums[i-k])

    return False


def longestConsecutive(nums: list[int]) -> int:
    """
    128. Longest Consecutive Sequence
    This returns the length of the largest group made up of elements of
    nums that are consecutive integers.
    """
    elements = set(nums)
    max_length = 0
    # This iterates through each element.  It checks if it is the start
    # of a new group when the element minus one is not in the set.  If
    # so, it continues to check if the next integer that should be in
    # the group is in the set.  Once it cannot find the next value, it
    # checks if the length of the current group is a new max.
    for num in nums:
        if num - 1 not in elements:
            current_length = 1
            current_num = num
            while current_num + 1 in elements:
                current_length += 1
                current_num += 1
            max_length = max(max_length, current_length)

    return max_length


def summaryRanges(nums: list[int]) -> list[str]:
    """
    228. Summary Ranges
    nums should be in sorted ascending order without duplicates.  This
    returns all of the ranges that cover each element.
    """
    # This is the base case when there are no elements in nums.
    if len(nums) == 0:
        return []

    ranges = []
    starting_num = nums[0]
    last_num = nums[0]
    # This iterates through each element in nums.  It checks if the
    # current range is ongoing.  If not, it adds the range to the
    # output and starts a new range with the current element.
    for i in range(1, len(nums)):
        # This checks if the range is ongoing, which is when the
        # current element is one greater than the previous element.
        if nums[i] == last_num + 1:
            last_num = nums[i]
            continue
        # The current element is the start of a new range.  So, this
        # saves the previous range.  If the previous range is only one
        # number, it just saves the single number.  If it is longer, it
        # saves both the start and end of the range.
        if starting_num == last_num:
            ranges.append(f'{starting_num}')
        else:
            ranges.append(f'{starting_num}->{last_num}')
        # This resets the range with the current element.
        starting_num = nums[i]
        last_num = nums[i]

    # This saves the last range after nums has been iterated through.
    if starting_num == last_num:
        ranges.append(f'{starting_num}')
    else:
        ranges.append(f'{starting_num}->{last_num}')

    return ranges


def merge(intervals: list[list[int]]) -> list[list[int]]:
    """
    56. Merge Intervals
    This combines all of the overlapping intervals and returns every
    interval.
    """
    def first_element(interval):
        """
        This returns the first element in an interval.
        """
        return interval[0]

    # This sorts ascending the intervals based on the start.
    intervals.sort(key=first_element)

    result = []
    interval_start = intervals[0][0]
    interval_end = intervals[0][1]
    # This iterates through each interval.  If it is overlapping with
    # the current interval, then the current interval is updated.
    # Otherwise, it saves the current interval and starts a new one.
    for i in range(1, len(intervals)):
        interval = intervals[i]
        # This checks if the interval is overlapping.
        if interval[0] <= interval_end:
            # The interval end may be less than the current interval's
            # end.
            interval_end = max(interval_end, interval[1])
        # The interval is not overlapping.
        else:
            result.append([interval_start, interval_end])
            interval_start = interval[0]
            interval_end = interval[1]

    # This saves the last interval after intervals has been iterated
    # through.
    result.append([interval_start, interval_end])

    return result


def insert(intervals: list[list[int]],
           newInterval: list[int]) -> list[list[int]]:
    """
    57. Insert Interval
    This adds newInterval to intervals.  It then combines all of the
    overlapping intervals and returns every interval.
    """
    # This is the base case when there is only the new interval.
    if len(intervals) == 0:
        return [newInterval]

    # This adds the new interval to intervals.  It keeps it sorted
    # ascending by the interval start.
    new_intervals = []
    inserted_new_interval = False
    for interval in intervals:
        if not inserted_new_interval and newInterval[0] <= interval[0]:
            new_intervals.append(newInterval)
            inserted_new_interval = True
        new_intervals.append(interval)
    if not inserted_new_interval:
        new_intervals.append(newInterval)

    result = []
    interval_start = new_intervals[0][0]
    interval_end = new_intervals[0][1]
    # This iterates through each interval.  If it is overlapping with
    # the current interval, then the current interval is updated.
    # Otherwise, it saves the current interval and starts a new one.
    for i in range(1, len(new_intervals)):
        interval = new_intervals[i]
        # This checks if the interval is overlapping.
        if interval[0] <= interval_end:
            # The interval end may be less than the current interval's
            # end.
            interval_end = max(interval_end, interval[1])
        # The interval is not overlapping.
        else:
            result.append([interval_start, interval_end])
            interval_start = interval[0]
            interval_end = interval[1]

    # This saves the last interval after intervals has been iterated
    # through.
    result.append([interval_start, interval_end])

    return result


def isValid(s: str) -> bool:
    """
    20. Valid Parentheses
    This checks if every open bracket in s is properly closed.  If so,
    it returns True, otherwise it returns False.
    """
    open_brackets = set(['(', '[', '{'])
    stack = []
    i = 0
    # This iterates through each character in s.  If it is an open
    # bracket, it adds it to a stack, since it will need to be closed
    # eventually.  If it is a closed bracket, it checks if it is the
    # proper one for the last open bracket, which is the character at
    # the top of the stack.
    while i < len(s):
        if s[i] in open_brackets:
            stack.append(s[i])
        else:
            close_bracket = s[i]
            # This checks if there is a close bracket that was never
            # opened.
            if len(stack) == 0:
                return False
            open_bracket = stack.pop()
            # This checks if the close bracket is not the proper one
            # for the last open bracket.
            if (
                open_bracket == '(' and close_bracket != ')'
                or open_bracket == '[' and close_bracket != ']'
                or open_bracket == '{' and close_bracket != '}'
            ):
                return False
        i += 1

    # There may still be open brackets that were never closed.
    return len(stack) == 0


def simplifyPath(path: str) -> str:
    """
    71. Simplify Path
    This takes an absolute path and returns the simplified canonical
    path.
    """
    stack = []
    current_directory = ''
    # This iterates through each character in path.  If the char is
    # part of a directory, it adds it to the current directory string.
    # If the char is a '/', then the current directory is over.  If the
    # current directory is a name, then it is added to the top of a
    # stack.  If it is '..', then the top of the stack is popped since
    # the path needs to move back one directory.  The path may not end
    # in a '/'.  This adds it so that anything left in the current
    # directory can be handled.
    for char in path + '/':
        if char == '/':
            if current_directory == '..':
                if len(stack) > 0:
                    stack.pop()
            # This checks if the current directory is a file/folder
            # name.
            elif current_directory != '' and current_directory != '.':
                stack.append(current_directory)
            current_directory = ''
        else:
            current_directory += char

    return '/' + '/'.join(stack)


class MinStack:
    """
    155. Min Stack
    This class contains methods for a stack and getting the minimum
    value in the stack.
    """

    def __init__(self):
        self.stack = []
        # This uses a separate stack to keep track of the current
        # minimum, which is always the value at the top of the stack.
        # Previous minimums are below this top value.
        self.min_stack = []

    def push(self, val: int) -> None:
        """
        This adds val to the top of the stack.
        """
        # This checks if there is no current minimum.
        if len(self.min_stack) == 0:
            self.min_stack.append(val)
        else:
            # This checks if there is a new minimum.
            current_min = self.min_stack[-1]
            if val <= current_min:
                self.min_stack.append(val)

        self.stack.append(val)

    def pop(self) -> None:
        """
        This removes the value at the top of the stack.
        """
        top_element = self.stack[-1]
        current_min = self.min_stack[-1]
        # This checks if the element at the top of the stack that is
        # being removed is the current minimum.  If so, it needs to be
        # removed as well from the separate stack.  The new minimum
        # will be the previous one.
        if top_element == current_min:
            self.min_stack.pop()

        self.stack.pop()

    def top(self) -> int:
        """
        This returns the value at the top of the stack.
        """
        return self.stack[-1]

    def getMin(self) -> int:
        """
        This returns the minimum value in the stack.
        """
        # The minimum value is the value at the top of the separate
        # stack.
        return self.min_stack[-1]


def evalRPN(tokens: list[str]) -> int:
    """
    150. Evaluate Reverse Polish Notation
    This returns the result of an expression in reverse polish
    notation.
    """
    stack = []
    operands = set(['+', '-', '*', '/'])
    # This iterates through each character in tokens.  If it is an
    # integer, it adds it to the top of a stack.  If it is an operand,
    # it removes the two top values from the stack.  The top value is
    # int2 and the second to last value is int1.  It then evaluates
    # int1 operand int2 (if the operand is '/' it also removes any
    # decimals).  The result is then added to the top of the stack.
    for token in tokens:
        if token not in operands:
            stack.append(int(token))
        else:
            second_value = stack.pop()
            first_value = stack.pop()
            match token:
                case '+':
                    stack.append(first_value+second_value)
                case '-':
                    stack.append(first_value-second_value)
                case '*':
                    stack.append(first_value*second_value)
                case '/':
                    stack.append(int(first_value/second_value))

    # The stack only contains the final resulting value.
    return stack[-1]


def calculate(s: str) -> int:
    """
    224. Basic Calculator
    This takes in a string of an expression with addition and/or
    subtraction.  It returns the result of the expression.
    """
    result = 0
    current_num = '0'
    sign = '+'
    stack = []
    # This iterates through each character in s.  It handles the
    # possible values: a digit, +, -, (, or ).  It then does one final
    # iteration to combine any leftover current number to the result.
    for char in s + '+':
        # If it is a number (or a multi-digit number made up of
        # multiple characters in a row) then it is stored as the
        # current number.
        if char.isdigit():
            current_num += char
        # If it is an addition or subtraction sign, it adds or
        # subtracts the result so far with the current number, based on
        # the previous sign.  It then resets the current number and
        # sets the sign to the current char.
        elif char == '+' or char == '-':
            if sign == '+':
                result += int(current_num)
            else:
                result -= int(current_num)
            current_num = '0'
            sign = char
        # If it is the start of parenthesis, it stores the current
        # result and sign in a stack to be used when the end
        # parenthesis is reached.  They are then reset.  The current
        # number does not need to be reset, because it already has
        # been.  A parenthesis must come after a +, -, or (.  If it
        # comes after a + or -, then it was just reset in the previous
        # iteration.  If it comes after a (, then it was already reset
        # based on the previous statement and has not changed.
        elif char == '(':
            stack.append(result)
            stack.append(sign)
            result = 0
            sign = '+'
        # If it is the end of parenthesis, it combines the current
        # result and number.  It then resets the current number and
        # sign.  After this, it takes the old result and sign from the
        # stack.  It then evaluates oldresult sign currentresult.  This
        # value becomes the new current result.
        elif char == ')':
            if sign == '+':
                result += int(current_num)
            else:
                result -= int(current_num)
            current_num = '0'
            sign = '+'
            old_sign = stack.pop()
            if old_sign == '-':
                result *= -1
            old_result = stack.pop()
            result += old_result
        # For any other characters, such as ' ', it is skipped over.

    return result


def hasCycle(head: ListNode) -> bool:
    """
    141. Linked List Cycle
    This returns True if there is a cycle in the linked list, otherwise
    it returns False.
    """
    # These are the base cases, when the list is empty or there is only
    # one node without any next node.
    if head is None or head.next is None:
        return False

    slow_node = head
    fast_node = head
    # This continues to iterate through the linked list until it finds
    # a cycle or reaches the end.  One pointer moves by 1 node each
    # iteration.  The other pointer moves by 2 nodes each iteration.
    # If the faster pointer reaches the end of the list, there is no
    # cycle.  However, if the faster pointer reaches the slower
    # pointer, then there is a cycle.  This is because it reached the
    # end of the list, moved through the cycle to a node behind the
    # slow pointer, and then caught up to it.
    while fast_node is not None:
        slow_node = slow_node.next
        for _ in range(2):
            if fast_node is not None:
                fast_node = fast_node.next
        if slow_node == fast_node:
            return True

    return False


def copyRandomList(head):
    """
    138. Copy List with Random Pointer
    This makes a deep copy of the linked list.  It returns the head of
    the new list.
    """
    new_head = None
    previous_new_node = None
    i = 0
    # This stores nodes in the new list.  The key is its index
    # position, and the value is the new node object.
    new_nodes_mapping = {}
    # This stores nodes from the original list.  The key is the node
    # object, and the value is its index position.
    nodes_mapping = {}
    current_node = head
    # This iterates through each node in the original list.  It creates
    # a new node with the same value as the current node and attaches
    # it to the end of the new list.
    while current_node is not None:
        new_node = Node(x=current_node.val)
        if new_head is None:
            new_head = new_node
        else:
            previous_new_node.next = new_node
        previous_new_node = new_node
        # This stores the original node and new node mappings.
        new_nodes_mapping[i] = new_node
        nodes_mapping[current_node] = i
        i += 1
        current_node = current_node.next

    current_new_node = new_head
    current_node = head
    # This iterates through each node in both of the lists.  It gets
    # the index of the random node for the current original node.  It
    # then finds the equivalent new node based on that index and
    # attaches it to the current new node.
    while current_node is not None:
        if current_node.random is not None:
            random_index = nodes_mapping[current_node.random]
            current_new_node.random = new_nodes_mapping[random_index]
        current_new_node = current_new_node.next
        current_node = current_node.next

    return new_head
