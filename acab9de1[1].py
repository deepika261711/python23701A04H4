
# LeetCode Problem Solutions
# Author: [Your Name]
# Description: This file contains solutions for various algorithm problems including
# Longest Common Prefix, 4Sum, Merge Two Sorted Lists, Swap Nodes in Pairs,
# Divide Two Integers, Next Permutation, Wildcard Matching, Rotate Image,
# Pow(x, n), Maximum Subarray.

class Solution:

    # 11. Longest Common Prefix
    # Finds the longest common prefix string amongst an array of strings.
    def longestCommonPrefix(self, strs):
        if not strs:
            return ""
        strs_length = len(strs[0])
        if len(strs) == 1:
            return strs[0]
        for i in range(strs_length):
            for_check = strs[0][:(strs_length - i)]
            is_good = all(s.startswith(for_check) for s in strs)
            if is_good:
                return for_check
        return ""

    # 12. 4Sum
    # Finds all unique quadruplets in the array that sum up to the target.
    def fourSum(self, nums, target):
        nums.sort()
        n = len(nums)
        if n < 4:
            return []
        ans = []
        for i in range(n):
            if i > 0 and nums[i] == nums[i - 1]:
                continue
            if i + 3 < n and nums[i] + nums[i + 1] + nums[i + 2] + nums[i + 3] > target:
                break
            if nums[i] + nums[n - 1] + nums[n - 2] + nums[n - 3] < target:
                continue
            for j in range(i + 1, n):
                if j > i + 1 and nums[j] == nums[j - 1]:
                    continue
                if j + 2 < n and nums[i] + nums[j] + nums[j + 1] + nums[j + 2] > target:
                    break
                if nums[i] + nums[j] + nums[n - 1] + nums[n - 2] < target:
                    continue
                k = j + 1
                l = n - 1
                while k < l:
                    sum_ = nums[i] + nums[j] + nums[k] + nums[l]
                    if sum_ == target:
                        ans.append([nums[i], nums[j], nums[k], nums[l]])
                        k += 1
                        l -= 1
                        while k < l and nums[k] == nums[k - 1]:
                            k += 1
                        while k < l and nums[l] == nums[l + 1]:
                            l -= 1
                    elif sum_ < target:
                        k += 1
                    else:
                        l -= 1
        return ans

    # 13. Merge Two Sorted Lists
    # Merges two sorted linked lists and returns it as a new sorted list.
    def mergeTwoLists(self, list1, list2):
        # Definition for singly-linked list.
        class ListNode:
            def __init__(self, val=0, next=None):
                self.val = val
                self.next = next

        dummy = ListNode(0)
        current = dummy
        while list1 and list2:
            if list1.val < list2.val:
                current.next = list1
                list1 = list1.next
            else:
                current.next = list2
                list2 = list2.next
            current = current.next
        current.next = list1 if list1 else list2
        return dummy.next

    # 14. Swap Nodes in Pairs
    # Swaps every two adjacent nodes in a linked list.
    def swapPairs(self, head):
        class ListNode:
            def __init__(self, val=0, next=None):
                self.val = val
                self.next = next

        dummy = ListNode(0)
        dummy.next = head
        prev = dummy
        while prev.next and prev.next.next:
            first = prev.next
            second = prev.next.next
            first.next = second.next
            second.next = first
            prev.next = second
            prev = first
        return dummy.next

    # 15. Divide Two Integers
    # Divides two integers without using multiplication, division and mod operator.
    def divide(self, dividend, divisor):
        if dividend == -2**31 and divisor == -1:
            return 2**31 - 1
        negative = (dividend < 0) != (divisor < 0)
        dividend, divisor = abs(dividend), abs(divisor)
        quotient = 0
        while dividend >= divisor:
            temp, multiple = divisor, 1
            while dividend >= (temp << 1):
                temp <<= 1
                multiple <<= 1
            dividend -= temp
            quotient += multiple
        return -quotient if negative else quotient

    # 16. Next Permutation
    # Rearranges numbers into the lexicographically next greater permutation.
    def nextPermutation(self, nums):
        n = len(nums)
        i = n - 2
        while i >= 0 and nums[i] >= nums[i + 1]:
            i -= 1
        if i >= 0:
            j = n - 1
            while nums[j] <= nums[i]:
                j -= 1
            nums[i], nums[j] = nums[j], nums[i]
        left, right = i + 1, n - 1
        while left < right:
            nums[left], nums[right] = nums[right], nums[left]
            left += 1
            right -= 1

    # 17. Wildcard Matching
    # Implements wildcard pattern matching with support for '?' and '*'.
    def isMatch(self, s, p):
        m, n = len(s), len(p)
        dp = [[False] * (n + 1) for _ in range(m + 1)]
        dp[0][0] = True
        for j in range(1, n + 1):
            if p[j - 1] == '*':
                dp[0][j] = dp[0][j - 1]
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if p[j - 1] == '*':
                    dp[i][j] = dp[i][j - 1] or dp[i - 1][j]
                elif p[j - 1] == '?' or s[i - 1] == p[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1]
        return dp[m][n]

    # 18. Rotate Image
    # Rotates an n x n 2D matrix clockwise by 90 degrees in-place.
    def rotate(self, matrix):
        n = len(matrix)
        # Step 1: Reverse the rows
        for i in range(n // 2):
            matrix[i], matrix[n - 1 - i] = matrix[n - 1 - i], matrix[i]
        # Step 2: Transpose the matrix
        for i in range(n - 1):
            for j in range(i + 1, n):
                matrix[i][j], matrix[j][i] = matrix[j][i], matrix[i][j]

    # 19. Pow(x, n)
    # Calculates x raised to the power n.
    def myPow(self, x, n):
        import math
        return math.pow(x, n)

    # 20. Maximum Subarray
    # Finds the contiguous subarray with the largest sum.
    def maxSubArray(self, nums):
        current_sum = max_sum = nums[0]
        for i in range(1, len(nums)):
            current_sum = max(nums[i], current_sum + nums[i])
            max_sum = max(max_sum, current_sum)
        return max_sum


# Example inputs for functions:
# Solution().longestCommonPrefix(["flower","flow","flight"]) => "fl"
# Solution().fourSum([1,0,-1,0,-2,2], 0) => unique quadruplets summing to 0
# ListNodes for mergeTwoLists and swapPairs need to be created beforehand.
# Solution().divide(10, 3) => 3
# nums = [1,2,3]; Solution().nextPermutation(nums); nums => [1,3,2]
# Solution().isMatch("aa", "a") => False
# matrix = [[1,2,3],[4,5,6],[7,8,9]]; Solution().rotate(matrix); matrix changed in place
# Solution().myPow(2.0, 10) => 1024.0
# Solution().maxSubArray([-2,1,-3,4,-1,2,1,-5,4]) => 6
