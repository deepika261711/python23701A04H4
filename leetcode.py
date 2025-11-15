11. Longest Common Prefix
Statement:
Write a function to find the longest common prefix string amongst an array of strings.
If there is no common prefix, return an empty string "".
Code:
class Solution(object):
    def longestCommonPrefix(self, strs):
        strs_length = len(strs[0])
        if len(strs) == 1:
            return strs[0]
        for i in range(strs_length):
            for_check = strs[0][:(strs_length-i)]
            is_good = all(s.startswith(for_check) for s in strs)
            if is_good:
                return for_check
        return ""
output:
strs =
["flower","flow","flight"]

12. 4Sum
Statement:
Given an array nums of n integers, return an array of all the unique quadruplets [nums[a], nums[b], nums[c], nums[d]] such that:
•	0 <= a, b, c, d < n
•	a, b, c, and d are distinct.
•	nums[a] + nums[b] + nums[c] + nums[d] == target
You may return the answer in any order.
Code:
class Solution(object):
    def fourSum(self, nums, target):
        # Optimal approach with pruning
        nums.sort()
        n = len(nums)
        if n < 4:
            return []
        ans = []
        for i in range(n):
            if i > 0 and nums[i] == nums[i - 1]:
                continue

            if i + 3 < n and nums[i] + nums[i+1] + nums[i+2] + nums[i+3] > target:
                break
            if nums[i] + nums[n-1] + nums[n-2] + nums[n-3] < target:
                continue

            for j in range(i + 1, n):
                if j > i + 1 and nums[j] == nums[j - 1]:
                    continue

                if j + 2 < n and nums[i] + nums[j] + nums[j+1] + nums[j+2] > target:
                    break
                if nums[i] + nums[j] + nums[n-1] + nums[n-2] < target:
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
output:
nums =
[1,0,-1,0,-2,2]
target =
0
13.Merge Two Sorted Lists
Statement:
You are given the heads of two sorted linked lists list1 and list2.
Merge the two lists into one sorted list. The list should be made by splicing together the nodes of the first two lists.
Return the head of the merged linked list.
Code:
class Solution(object):
    def mergeTwoLists(self, list1, list2):
        """
        :type list1: Optional[ListNode]
        :type list2: Optional[ListNode]
        :rtype: Optional[ListNode]
        """
        # Create a dummy node to simplify edge cases
        dummy = ListNode(0)
        current = dummy

        # Iterate while both lists have nodes
        while list1 and list2:
            if list1.val < list2.val:
                current.next = list1
                list1 = list1.next
            else:
                current.next = list2
                list2 = list2.next
            current = current.next

        # Attach the remaining nodes of list1 or list2
        current.next = list1 if list1 else list2

        return dummy.next
output:
list1 =
[1,2,4]
list2 =
[1,3,4]
14.Swap Nodes in Pairs
Statement:
Given a linked list, swap every two adjacent nodes and return its head. You must solve the problem without modifying the values in the list's nodes (i.e., only nodes themselves may be changed.)
Code:
class Solution(object):
    def swapPairs(self, head):
        """
        :type head: Optional[ListNode]
        :rtype: Optional[ListNode]
        """
        dummy = ListNode(0)
        dummy.next = head
        prev = dummy

        while prev.next and prev.next.next:
            first = prev.next
            second = prev.next.next

            # Swapping
            first.next = second.next
            second.next = first
            prev.next = second

            # Move prev pointer two nodes ahead
            prev = first

        return dummy.next
output:
head =
[1,2,3,4]
15.Divide Two Integers
Statement:
Given two integers dividend and divisor, divide two integers without using multiplication, division, and mod operator.
The integer division should truncate toward zero, which means losing its fractional part. For example, 8.345 would be truncated to 8, and -2.7335 would be truncated to -2.
Return the quotient after dividing dividend by divisor.
Note: Assume we are dealing with an environment that could only store integers within the 32-bit signed integer range: [−231, 231 − 1]. For this problem, if the quotient is strictly greater than 231 - 1, then return 231 - 1, and if the quotient is strictly less than -231, then return -231.
Code:
class Solution(object):
    def divide(self, dividend, divisor):
        """
        :type dividend: int
        :type divisor: int
        :rtype: int
        """
        # Handle overflow case
        if dividend == -2**31 and divisor == -1:
            return 2**31 - 1
        
        # Determine sign of result
        negative = (dividend < 0) != (divisor < 0)
        
        # Work with positive values
        dividend, divisor = abs(dividend), abs(divisor)
        quotient = 0
        
        # Subtract divisor repeatedly using bit manipulation for efficiency
        while dividend >= divisor:
            temp, multiple = divisor, 1
            while dividend >= (temp << 1):
                temp <<= 1
                multiple <<= 1
            dividend -= temp
            quotient += multiple
        
        # Apply sign
        return -quotient if negative else quotient
output:
dividend =
10
divisor =
3
16.Next Permutation
Statement:
A permutation of an array of integers is an arrangement of its members into a sequence or linear order.
•	For example, for arr = [1,2,3], the following are all the permutations of arr: [1,2,3], [1,3,2], [2, 1, 3], [2, 3, 1], [3,1,2], [3,2,1].
The next permutation of an array of integers is the next lexicographically greater permutation of its integer. More formally, if all the permutations of the array are sorted in one container according to their lexicographical order, then the next permutation of that array is the permutation that follows it in the sorted container. If such arrangement is not possible, the array must be rearranged as the lowest possible order (i.e., sorted in ascending order).
Code:
class Solution(object):
    def nextPermutation(self, nums):
        """
        :type nums: List[int]
        :rtype: None Do not return anything, modify nums in-place instead.
        """
        n = len(nums)
        i = n - 2
        
        # Step 1: Find the first decreasing element from the right
        while i >= 0 and nums[i] >= nums[i + 1]:
            i -= 1
        
        # Step 2: If such element is found, find the element to swap with
        if i >= 0:
            j = n - 1
            while nums[j] <= nums[i]:
                j -= 1
            nums[i], nums[j] = nums[j], nums[i]
        
        # Step 3: Reverse the elements to the right of i
        left, right = i + 1, n - 1
        while left < right:
            nums[left], nums[right] = nums[right], nums[left]
            left += 1
            right -= 1
output:
nums =
[1,2,3]
17.Wildcard Matching
Statement:
Given an input string (s) and a pattern (p), implement wildcard pattern matching with support for '?' and '*' where:
•	'?' Matches any single character.
•	'*' Matches any sequence of characters (including the empty sequence).
The matching should cover the entire input string (not partial).
Code:
class Solution(object):
    def isMatch(self, s, p):
        """
        :type s: str
        :type p: str
        :rtype: bool
        """
        m, n = len(s), len(p)
        dp = [[False] * (n + 1) for _ in range(m + 1)]
        dp[0][0] = True

        for j in range(1, n + 1):
            if p[j - 1] == '*':
                dp[0][j] = dp[0][j - 1]
output:
s =
"aa"
p =
"a"
18.Rotate Image
Statement:
You are given an n x n 2D matrix representing an image, rotate the image by 90 degrees (clockwise).
You have to rotate the image in-place, which means you have to modify the input 2D matrix directly. DO NOT allocate another 2D matrix and do the rotation.
Code:
class Solution(object):
    def rotate(self, matrix):
        """
        :type matrix: List[List[int]]
        :rtype: None Do not return anything, modify matrix in-place instead.
        """
        n = len(matrix)
        for i in range(n // 2):  # Only need to go halfway
            matrix[i], matrix[n - 1 - i] = matrix[n - 1 - i], matrix[i]

        # Step 2: Transpose the matrix in place
        for i in range(n - 1):  # Stop at n-1 to skip diagonal
            for j in range(i + 1, n):  # Start from i+1 to skip diagonal
                matrix[i][j], matrix[j][i] = matrix[j][i], matrix[i][j]
output:
matrix =
[[1,2,3],[4,5,6],[7,8,9]]
`19.Pow(x, n)
Statement:
Implement pow(x, n), which calculates x raised to the power n (i.e., xn).
import math
code:

class Solution(object):
    def myPow(self, x, n):
        """
        :type x: float
        :type n: int
        :rtype: float
        """
        result = math.pow(x,n)
        return result
       # Hope it helps,do upvote!
Output:
x =
2.00000
n =
10
20.Maximum Subarray
Statement:
Given an integer array nums, find the subarray with the largest sum, and return its sum.
Code:
class Solution(object):

    def maxSubArray(self, nums):
        
        current_sum = max_sum = nums[0]

        for i in range(1, len(nums)):

            current_sum = max(nums[i], current_sum + nums[i])

            max_sum = max(max_sum, current_sum)

        return max_sum
output:
nums =
[-2,1,-3,4,-1,2,1,-5,4]

        

 




        


        

 





 











 





        


        



