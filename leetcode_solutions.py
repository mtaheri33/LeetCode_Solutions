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
    # the left element is updated to the element to the right.  If the
    # sum is greater than the target, the value needs to decrease.  So,
    # the right element is updated to the element to the left.
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
