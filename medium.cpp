#include "medium.h"

---------------------------------------------------------------------

//2 Add Two Numbers
You are given two non - empty linked lists representing two non - negative integers.The digits are stored in reverse orderand each of their nodes contain a single digit.Add the two numbersand return it as a linked list.

You may assume the two numbers do not contain any leading zero, except the number 0 itself.

Example:
    Input: (2 -> 4 -> 3) + (5 -> 6 -> 4)
    Output : 7 -> 0 -> 8
    Explanation : 342 + 465 = 807.

ListNode* addTwoNumbers(ListNode* l1, ListNode* l2)
{
    ListNode *dummy = new ListNode(-1), *cur = dummy;
    int carry = 0;
    while (l1 || l2) {
        int val1 = l1 ? l1->val : 0;
        int val2 = l2 ? l2->val : 0;
        int sum = val1 + val2 + carry;
        carry = sum / 10;
        cur->next = new ListNode(sum % 10);
        cur = cur->next;
        if (l1) l1 = l1->next;
        if (l2) l2 = l2->next;
    }
    if (carry) cur->next = new ListNode(1);
    return dummy->next;
}

---------------------------------------------------------------------

//3 Longest Substring Without Repeating Characters
Given a string, find the length of the longest substring without repeating characters.

Example 1:
    Input: "abcabcbb"
    Output : 3
    Explanation : The answer is "abc", with the length of 3.

Example 2 :
    Input : "bbbbb"
    Output : 1
    Explanation : The answer is "b", with the length of 1.

Example 3 :
    Input : "pwwkew"
    Output : 3
    Explanation : The answer is "wke", with the length of 3.
    Note that the answer must be a substring, "pwke" is a subsequence and not a substring.

int lengthOfLongestSubstring(string s)
{
    vector<int> m(128, -1);
    int res = 0, left = -1;
    for (int i = 0; i < s.size(); ++i) {
        left = max(left, m[s[i]]);
        m[s[i]] = i;
        res = max(res, i - left);
    }
    return res;
}

---------------------------------------------------------------------

//5 Longest Palindromic Substring
Given a string s, find the longest palindromic substring in s.You may assume that the maximum length of s is 1000.

Example 1:
    Input: "babad"
    Output : "bab"
    Note : "aba" is also a valid answer.

Example 2 :
    Input : "cbbd"
    Output : "bb"

void searchPalindrome(string s, int left, int right, int& start, int& maxLen) 
{
    while (left >= 0 && right < s.size() && s[left] == s[right]) {
        --left; ++right;
    }
    if (maxLen < right - left - 1) {
        start = left + 1;
        maxLen = right - left - 1;
    }
}

string longestPalindrome(string s)
{
    if (s.size() < 2) return s;
    int n = s.size(), maxLen = 0, start = 0;
    for (int i = 0; i < n - 1; ++i) {
        searchPalindrome(s, i, i, start, maxLen);
        searchPalindrome(s, i, i + 1, start, maxLen);
    }
    return s.substr(start, maxLen);
}

string Manacher(string s) 
{
    string t = "$#";
    for (int i = 0; i < s.size(); i++) {
        t += s[i];
        t += "#";
    }
    vector<int> p(t.size(), 0);
    int mx = 0, id = 0, resLen = 0, resCenter = 0;
    for (int i = 1; i < t.size(); i++) {
        p[i] = mx > i ? min(p[2 * id - i], mx - i) : 1;
        while (t[i + p[i]] == t[i - p[i]])
            ++p[i];
        if (mx < i + p[i]) {
            mx = i + p[i];
            id = i;
        }
        if (resLen < p[i]) {
            resLen = p[i];
            resCenter = i;
        }
    }
    return s.substr((resCenter - resLen) / 2, resLen - 1);
}

---------------------------------------------------------------------

//6 ZigZag Conversion
The string "PAYPALISHIRING" is written in a zigzag pattern on a given number of rows like this: (you may want to display this pattern in a fixed font for better legibility)

P   A   H   N
A P L S I I G
Y   I   R
And then read line by line : "PAHNAPLSIIGYIR"

Write the code that will take a stringand make this conversion given a number of rows :

string convert(string s, int numRows);
Example 1:
    Input: s = "PAYPALISHIRING", numRows = 3
    Output : "PAHNAPLSIIGYIR"

Example 2 :
    Input : s = "PAYPALISHIRING", numRows = 4
    Output : "PINALSIGYAHRPI"
    Explanation :

    P     I    N
    A   L S  I G
    Y A   H R
    P     I

string convert(string s, int numRows)
{
    if (numRows <= 1) return s;
    string res;
    int size = 2 * numRows - 2, n = s.size();
    for (int i = 0; i < numRows; ++i) {
        for (int j = i; j < n; j += size) {
            res += s[j];
            int pos = j + size - 2 * i;
            if (i != 0 && i != numRows - 1 && pos < n) res += s[pos];
        }
    }
    return res;
}

---------------------------------------------------------------------

//8 String to Integer (atoi)
Implement atoi which converts a string to an integer.

The function first discards as many whitespace characters as necessary until the first non - whitespace character is found.Then, starting from this character, takes an optional initial plus or minus sign followed by as many numerical digits as possible, and interprets them as a numerical value.

The string can contain additional characters after those that form the integral number, which are ignoredand have no effect on the behavior of this function.

If the first sequence of non - whitespace characters in str is not a valid integral number, or if no such sequence exists because either str is empty or it contains only whitespace characters, no conversion is performed.

If no valid conversion could be performed, a zero value is returned.

Note:

Only the space character ' ' is considered as whitespace character.
Assume we are dealing with an environment which could only store integers within the 32 - bit signed integer range : [−231, 231 − 1] .If the numerical value is out of the range of representable values, INT_MAX(231 − 1) or INT_MIN(−231) is returned.

Example 1 :
    Input : "42"
    Output : 42

Example 2 :
    Input : "   -42"
    Output : -42
    Explanation : The first non - whitespace character is '-', which is the minus sign.
    Then take as many numerical digits as possible, which gets 42.

Example 3 :
    Input : "4193 with words"
    Output : 4193
    Explanation : Conversion stops at digit '3' as the next character is not a numerical digit.

Example 4 :
    Input : "words and 987"
    Output : 0
    Explanation : The first non - whitespace character is 'w', which is not a numerical
    digit or a + / -sign.Therefore no valid conversion could be performed.

Example 5 :
    Input : "-91283472332"
    Output : -2147483648
    Explanation : The number "-91283472332" is out of the range of a 32 - bit signed integer.
    Thefore INT_MIN(−231) is returned.

int myAtoi(string str)
{
    if (str.empty()) return 0;
    int sign = 1, base = 0, i = 0, n = str.size();
    while (i < n && str[i] == ' ') ++i;
    if (i < n && (str[i] == '+' || str[i] == '-')) {
        sign = (str[i++] == '+') ? 1 : -1;
    }
    while (i < n && str[i] >= '0' && str[i] <= '9') {
        if (base > INT_MAX / 10 || (base == INT_MAX / 10 && str[i] - '0' > 7)) {
            return (sign == 1) ? INT_MAX : INT_MIN;
        }
        base = 10 * base + (str[i++] - '0');
    }
    return base * sign;
}

---------------------------------------------------------------------

//11 Container With Most Water
Given n non - negative integers a1, a2, ..., an , where each represents a point at coordinate(i, ai).n vertical lines are drawn such that the two endpoints of line i is at(i, ai) and (i, 0).Find two lines, which together with x - axis forms a container, such that the container contains the most water.

Note: You may not slant the containerand n is at least 2.

The above vertical lines are represented by array[1, 8, 6, 2, 5, 4, 8, 3, 7].In this case, the max area of water(blue section) the container can contain is 49.

Example:
    Input: [1, 8, 6, 2, 5, 4, 8, 3, 7]
    Output : 49

int maxArea(vector<int>& height)
{
    int res = 0, i = 0, j = height.size() - 1;
    while (i < j) {
        int h = min(height[i], height[j]);
        res = max(res, h * (j - i));
        while (i < j && h == height[i]) ++i;
        while (i < j && h == height[j]) --j;
    }
    return res;
}

---------------------------------------------------------------------

//12 Integer to Roman
Roman numerals are represented by seven different symbols : I, V, X, L, C, Dand M.

    Symbol       Value
    I             1
    V             5
    X             10
    L             50
    C             100
    D             500
    M             1000
For example, two is written as II in Roman numeral, just two one's added together. Twelve is written as, XII, which is simply X + II. The number twenty seven is written as XXVII, which is XX + V + II.

Roman numerals are usually written largest to smallest from left to right.However, the numeral for four is not IIII.Instead, the number four is written as IV.Because the one is before the five we subtract it making four.The same principle applies to the number nine, which is written as IX.There are six instances where subtraction is used :

I can be placed before V(5) and X(10) to make 4 and 9.
X can be placed before L(50) and C(100) to make 40 and 90.
C can be placed before D(500) and M(1000) to make 400 and 900.
Given an integer, convert it to a roman numeral.Input is guaranteed to be within the range from 1 to 3999.

Example 1:
    Input: 3
    Output : "III"

Example 2 :
    Input : 4
    Output : "IV"

Example 3 :
    Input : 9
    Output : "IX"

Example 4 :
    Input : 58
    Output : "LVIII"
    Explanation : L = 50, V = 5, III = 3.

Example 5 :
    Input : 1994
    Output : "MCMXCIV"
    Explanation : M = 1000, CM = 900, XC = 90 and IV = 4.

string intToRoman(int num)
{
    string res = "";
    vector<int> val{ 1000, 900, 500, 400, 100, 90, 50, 40, 10, 9, 5, 4, 1 };
    vector<string> str{ "M", "CM", "D", "CD", "C", "XC", "L", "XL", "X", "IX", "V", "IV", "I" };
    for (int i = 0; i < val.size(); ++i) {
        while (num >= val[i]) {
            num -= val[i];
            res += str[i];
        }
    }
    return res;
}

---------------------------------------------------------------------

//15 3Sum
Given an array nums of n integers, are there elements a, b, c in nums such that a + b + c = 0 ? Find all unique triplets in the array which gives the sum of zero.

Note :

The solution set must not contain duplicate triplets.

Example :
    Given array nums = [-1, 0, 1, 2, -1, -4],
    A solution set is :
    [
        [-1, 0, 1],
        [-1, -1, 2]
    ]

vector<vector<int>> threeSum(vector<int>& nums)
{
    vector<vector<int>> res;
    sort(nums.begin(), nums.end());
    if (nums.empty() || nums.back() < 0 || nums.front() > 0) return {};
    for (int k = 0; k < (int)nums.size() - 2; ++k) {
        if (nums[k] > 0) break;
        if (k > 0 && nums[k] == nums[k - 1]) continue;
        int target = 0 - nums[k], i = k + 1, j = (int)nums.size() - 1;
        while (i < j) {
            if (nums[i] + nums[j] == target) {
                res.push_back({ nums[k], nums[i], nums[j] });
                while (i < j && nums[i] == nums[i + 1]) ++i;
                while (i < j && nums[j] == nums[j - 1]) --j;
                ++i; --j;
            }
            else if (nums[i] + nums[j] < target) ++i;
            else --j;
        }
    }
    return res;
}

---------------------------------------------------------------------

//16 3Sum Closest
Given an array nums of n integersand an integer target, find three integers in nums such that the sum is closest to target.Return the sum of the three integers.You may assume that each input would have exactly one solution.

Example:
Given array nums = [-1, 2, 1, -4], and target = 1.
The sum that is closest to the target is 2. (-1 + 2 + 1 = 2).

int threeSumClosest(vector<int>& nums, int target)
{
    int closest = nums[0] + nums[1] + nums[2];
    int diff = abs(closest - target);
    sort(nums.begin(), nums.end());
    for (int i = 0; i < nums.size() - 2; ++i) {
        int left = i + 1, right = nums.size() - 1;
        while (left < right) {
            int sum = nums[i] + nums[left] + nums[right];
            int newDiff = abs(sum - target);
            if (diff > newDiff) {
                diff = newDiff;
                closest = sum;
            }
            if (sum < target) ++left;
            else --right;
        }
    }
    return closest;
}

---------------------------------------------------------------------

//17 Letter Combinations of a Phone Number
Given a string containing digits from 2 - 9 inclusive, return all possible letter combinations that the number could represent.

A mapping of digit to letters(just like on the telephone buttons) is given below.Note that 1 does not map to any letters.

Example:
    Input: "23"
    Output : ["ad", "ae", "af", "bd", "be", "bf", "cd", "ce", "cf"] .
Note :
Although the above answer is in lexicographical order, your answer could be in any order you want.

vector<string> letterCombinations(string digits)
{
    if (digits.empty()) return {};
    vector<string> res{ "" };
    vector<string> dict{ "", "", "abc", "def", "ghi", "jkl", "mno", "pqrs", "tuv", "wxyz" };
    for (int i = 0; i < digits.size(); ++i) {
        vector<string> t;
        string str = dict[digits[i] - '0'];
        for (int j = 0; j < str.size(); ++j) {
            for (string s : res) t.push_back(s + str[j]);
        }
        res = t;
    }
    return res;
}

---------------------------------------------------------------------

//18 4Sum
Given an array nums of n integersand an integer target, are there elements a, b, c, and d in nums such that a + b + c + d = target ? Find all unique quadruplets in the array which gives the sum of target.

Note :
The solution set must not contain duplicate quadruplets.

Example :
    Given array nums = [1, 0, -1, 0, -2, 2], and target = 0.
    A solution set is :
    [
        [-1, 0, 0, 1],
        [-2, -1, 1, 2],
        [-2, 0, 0, 2]
    ]

vector<vector<int>> fourSum(vector<int>& nums, int target)
{
    vector<vector<int>> res;
    int n = nums.size();
    sort(nums.begin(), nums.end());
    for (int i = 0; i < n - 3; ++i) {
        if (i > 0 && nums[i] == nums[i - 1]) continue;
        for (int j = i + 1; j < n - 2; ++j) {
            if (j > i + 1 && nums[j] == nums[j - 1]) continue;
            int left = j + 1, right = n - 1;
            while (left < right) {
                int sum = nums[i] + nums[j] + nums[left] + nums[right];
                if (sum == target) {
                    vector<int> out{ nums[i], nums[j], nums[left], nums[right] };
                    res.push_back(out);
                    while (left < right && nums[left] == nums[left + 1]) ++left;
                    while (left < right && nums[right] == nums[right - 1]) --right;
                    ++left; --right;
                }
                else if (sum < target) ++left;
                else --right;
            }
        }
    }
    return res;
}

---------------------------------------------------------------------

//19 Remove Nth Node From End of List
Given a linked list, remove the n - th node from the end of listand return its head.

Example:
    Given linked list : 1->2->3->4->5, and n = 2.
    After removing the second node from the end, the linked list becomes 1->2->3->5.
Note :
Given n will always be valid.

Follow up :
Could you do this in one pass ?


ListNode * removeNthFromEnd(ListNode* head, int n)
{
    if (head == NULL || n <= 0) {
        return head;
    }
    ListNode *front = head;
    for (; front != NULL && n != 0; n--) {
        front = front->next;
    }
    if (front == NULL && n == 0) {
        return head->next;
    }
    else if (front == NULL && n > 0) {
        return head;
    }
    ListNode *back = head;
    while (front->next != NULL) {
        front = front->next;
        back = back->next;
    }
    back->next = back->next->next;
    return head;
}

---------------------------------------------------------------------

//22 Generate Parentheses
Given n pairs of parentheses, write a function to generate all combinations of well - formed parentheses.

For example, given n = 3, a solution set is :
    [
        "((()))",
        "(()())",
        "(())()",
        "()(())",
        "()()()"
    ]

vector<string> generateParenthesis(int n)
{
    vector<string> res;
    if (n <= 0) {
        return res;
    }

    set<string> resSet;
    resSet.insert("()");
    for (int i = 1; i < n; i++) {
        set<string> temp;
        for (set<string>::iterator it = resSet.begin(); it != resSet.end(); it++) {
            for (int i = 0; i < it->size(); i++) {
                if ((*it)[i] == '(') {
                    temp.insert(it->substr(0, i + 1) + "()" + it->substr(i + 1));
                }
            }
            temp.insert("()" + *it);
        }
        resSet = temp;
    }
    return vector<string>(resSet.begin(), resSet.end());
}

---------------------------------------------------------------------

//24 Swap Nodes in Pairs
Given a linked list, swap every two adjacent nodesand return its head.

You may not modify the values in the list's nodes, only nodes itself may be changed.

Example:
    Given 1->2->3->4, you should return the list as 2->1->4->3.

ListNode * swapPairs(ListNode* head)
{
    ListNode* dummy = new ListNode(-1), * pre = dummy;
    dummy->next = head;
    while (pre->next && pre->next->next) {
        ListNode* t = pre->next->next;
        pre->next->next = t->next;
        t->next = pre->next;
        pre->next = t;
        pre = t->next;
    }
    return dummy->next;
}

---------------------------------------------------------------------

//29 Divide Two Integers
Given two integers dividendand divisor, divide two integers without using multiplication, divisionand mod operator.

Return the quotient after dividing dividend by divisor.

The integer division should truncate toward zero.

Example 1:
    Input: dividend = 10, divisor = 3
    Output : 3

Example 2 :
    Input : dividend = 7, divisor = -3
    Output : -2

Note :
Both dividend and divisor will be 32 - bit signed integers.
The divisor will never be 0.
Assume we are dealing with an environment which could only store integers within the 32 - bit signed integer range : [−231, 231 − 1] .For the purpose of this problem, assume that your function returns 231 − 1 when the division result overflows.

int divide(int dividend, int divisor)
{
    if (divisor == 0 || (dividend == INT_MIN && divisor == -1)) {
        return INT_MAX;
    }
    int sign = ((dividend > 0) ^ (divisor > 0));
    long long m = abs((long long)dividend), n = abs((long long)divisor), res = 0;
    
    while (m >= n) {
        long long t = n, p = 1;
        while (m >= (t << 1)) {
            t <<= 1;
            p <<= 1;
        }
        res += p;
        m -= t;
    }
    return sign == 1 ? 0 - res : res;
}

---------------------------------------------------------------------

//31 Next Permutation
Implement next permutation, which rearranges numbers into the lexicographically next greater permutation of numbers.

If such arrangement is not possible, it must rearrange it as the lowest possible order(ie, sorted in ascending order).

The replacement must be in - place and use only constant extra memory.

Here are some examples.Inputs are in the left - hand columnand its corresponding outputs are in the right - hand column.

1, 2, 3 → 1, 3, 2
3, 2, 1 → 1, 2, 3
1, 1, 5 → 1, 5, 1

如果给定数组是降序，则说明是全排列的最后一种情况，则下一个排列就是最初始情况。我们再来看下面一个例子，有如下的一个数组
1　　2　　7　　4　　3　　1
下一个排列为：
1　　3　　1　　2　　4　　7

那么是如何得到的呢，我们通过观察原数组可以发现，如果从末尾往前看，数字逐渐变大，到了2时才减小的，然后我们再从后往前找第一个比2大的数字，是3，那么我们交换2和3，再把此时3后面的所有数字转置一下即可，步骤如下：
1　　2　　7　　4　　3　　1
1　　2　　7　　4　　3　　1
1　　3　　7　　4　　2　　1
1　　3　　1　　2　　4　　7

void nextPermutation(vector<int>& nums)
{
    int n = nums.size(), i = n - 2, j = n - 1;
    while (i >= 0 && nums[i] >= nums[i + 1]) --i;
    if (i >= 0) {
        while (nums[j] <= nums[i]) --j;
        swap(nums[i], nums[j]);
    }
    reverse(nums.begin() + i + 1, nums.end());
}

---------------------------------------------------------------------

//33 Search in Rotated Sorted Array
Suppose an array sorted in ascending order is rotated at some pivot unknown to you beforehand.
(i.e., [0, 1, 2, 4, 5, 6, 7] might become[4, 5, 6, 7, 0, 1, 2]).
You are given a target value to search.If found in the array return its index, otherwise return -1.
You may assume no duplicate exists in the array.
Your algorithm's runtime complexity must be in the order of O(log n).

Example 1:
    Input: nums = [4, 5, 6, 7, 0, 1, 2], target = 0
    Output : 4

Example 2 :
    Input : nums = [4, 5, 6, 7, 0, 1, 2], target = 3
    Output : -1

int search(vector<int>& nums, int target)
{
    int left = 0, right = nums.size() - 1;
    while (left <= right) {
        int mid = left + (right - left) / 2;
        if (nums[mid] == target) {
            return mid;
        }
        if (nums[mid] >= nums[left]) {
            if (nums[left] <= target && nums[mid] > target) {
                right = mid - 1;
            }
            else {
                left = mid + 1;
            }
        }
        else {
            if (nums[mid] < target && nums[right] >= target) {
                left = mid + 1;
            }
            else {
                right = mid - 1;
            }
        }
    }
    return -1;
}

---------------------------------------------------------------------

//34 Find First and Last Position of Element in Sorted Array
Given an array of integers nums sorted in ascending order, find the startingand ending position of a given target value.

Your algorithm's runtime complexity must be in the order of O(log n).

If the target is not found in the array, return[-1, -1].

Example 1:
    Input: nums = [5, 7, 7, 8, 8, 10], target = 8
    Output : [3, 4]
    
Example 2 :
    Input : nums = [5, 7, 7, 8, 8, 10], target = 6
    Output : [-1, -1]

vector<int> searchRange(vector<int>& nums, int target)
{
    vector<int> res(2, -1);
    int left = 0, right = nums.size();
    while (left < right) {
        int mid = left + (right - left) / 2;
        if (nums[mid] < target) left = mid + 1;
        else right = mid;
    }
    if (right == nums.size() || nums[right] != target) return res;
    res[0] = right;
    right = nums.size();
    while (left < right) {
        int mid = left + (right - left) / 2;
        if (nums[mid] <= target) left = mid + 1;
        else right = mid;
    }
    res[1] = right - 1;
    return res;
}

---------------------------------------------------------------------

//36 Valid Sudoku
Determine if a 9x9 Sudoku board is valid.Only the filled cells need to be validated according to the following rules :

Each row must contain the digits 1 - 9 without repetition.
Each column must contain the digits 1 - 9 without repetition.
Each of the 9 3x3 sub - boxes of the grid must contain the digits 1 - 9 without repetition.

bool isValidSudoku(vector<vector<char>>& board)
{
    vector<vector<bool>> rowFlag(9, vector<bool>(9));
    vector<vector<bool>> colFlag(9, vector<bool>(9));
    vector<vector<bool>> cellFlag(9, vector<bool>(9));
    for (int i = 0; i < 9; ++i) {
        for (int j = 0; j < 9; ++j) {
            if (board[i][j] == '.') continue;
            int c = board[i][j] - '1';
            if (rowFlag[i][c] || colFlag[c][j] || cellFlag[3 * (i / 3) + j / 3][c]) return false;
            rowFlag[i][c] = true;
            colFlag[c][j] = true;
            cellFlag[3 * (i / 3) + j / 3][c] = true;
        }
    }
    return true;
}

static const int SHIFT = 3;
static const int MASK = 0X07;
#define SET_BIT(arr, i) do{arr[(i) >> SHIFT] |= (1 << ((i) & MASK));}while(0)
#define CLEAR_BIT(arr, i) do{arr[(i) >> SHIFT] &= ~(1 << ((i) & MASK));}while(0)
#define IS_IN(arr, i) (arr[(i) >> SHIFT] & (1 << ((i) & MASK)))

bool isValidSudoku2(vector<vector<char>>& board)
{
    vector<vector<unsigned char>> rowFlag(9, vector<unsigned char>(2, 0));
    vector<vector<unsigned char>> colFlag(9, vector<unsigned char>(2, 0));
    vector<vector<unsigned char>> cellFlag(9, vector<unsigned char>(2, 0));
    for (int i = 0; i < 9; ++i) {
        for (int j = 0; j < 9; ++j) {
            if (board[i][j] == '.') continue;
            int c = board[i][j] - '1';
            if (IS_IN(rowFlag[i], c) || IS_IN(colFlag[c], j) || IS_IN(cellFlag[3 * (i / 3) + j / 3], c)) return false;
            SET_BIT(rowFlag[i], c);
            SET_BIT(colFlag[c], j);
            SET_BIT(cellFlag[3 * (i / 3) + j / 3], c);
        }
    }
    return true;
}

---------------------------------------------------------------------
//39 Combination Sum
Given a set of candidate numbers(candidates) (without duplicates) and a target number(target), find all unique combinations in candidates where the candidate numbers sums to target.

The same repeated number may be chosen from candidates unlimited number of times.

Note:
All numbers(including target) will be positive integers.
The solution set must not contain duplicate combinations.

Example 1 :
    Input : candidates = [2, 3, 6, 7], target = 7,
    A solution set is :
    [
        [7],
        [2, 2, 3]
    ]

Example 2 :
    Input : candidates = [2, 3, 5], target = 8,
    A solution set is :
    [
        [2, 2, 2, 2],
        [2, 3, 3],
        [3, 5]
    ]

void combinationSumDFS(vector<int>& candidates, int target, int start, vector<int>& out, vector<vector<int>>& res)
{
    if (target < 0) return;
    for (int i = start; i < candidates.size(); ++i) {
        out.push_back(candidates[i]);
        if (target == candidates[i]) {
            res.push_back(out);
        }
        else {
            combinationSumDFS(candidates, target - candidates[i], i, out, res);
        }        
        out.pop_back();
    }
}

vector<vector<int>> combinationSum(vector<int>& candidates, int target)
{
    vector<vector<int>> res;
    vector<int> out;
    sort(candidates.begin(), candidates.end());
    combinationSumDFS(candidates, target, 0, out, res);
    return res;
}

---------------------------------------------------------------------

//40 Combination Sum II
Given a collection of candidate numbers(candidates) and a target number(target), find all unique combinations in candidates where the candidate numbers sums to target.

Each number in candidates may only be used once in the combination.

Note:

All numbers(including target) will be positive integers.
The solution set must not contain duplicate combinations.

Example 1 :
    Input : candidates = [10, 1, 2, 7, 6, 1, 5], target = 8,
    A solution set is :
    [
        [1, 7],
        [1, 2, 5],
        [2, 6],
        [1, 1, 6]
    ]

Example 2:
    Input: candidates = [2, 5, 2, 1, 2], target = 5,
    A solution set is :
    [
        [1, 2, 2],
        [5]
    ]

void combinationSumDFS(vector<int>& candidates, int target, int start, vector<int>& out, vector<vector<int>>& res)
{
    if (target < 0) return;
    for (int i = start; i < candidates.size(); ++i) {
        if (i > start && candidates[i] == candidates[i - 1]) {
            continue;
        }
        out.push_back(candidates[i]);
        if (target == candidates[i]) {
            res.push_back(out);
        }
        else {
            combinationSumDFS(candidates, target - candidates[i], i + 1, out, res);
        }
        out.pop_back();
    }
}

vector<vector<int>> combinationSum2(vector<int>& candidates, int target)
{
    vector<vector<int>> res;
    vector<int> out;
    sort(candidates.begin(), candidates.end());
    combinationSumDFS(candidates, target, 0, out, res);
    return res;
}

---------------------------------------------------------------------

//43 Multiply Strings
Given two non - negative integers num1and num2 represented as strings, return the product of num1 and num2, also represented as a string.

Example 1:
    Input: num1 = "2", num2 = "3"
    Output : "6"

Example 2 :
    Input : num1 = "123", num2 = "456"
    Output : "56088"

Note :
The length of both num1 and num2 is < 110.
Both num1 and num2 contain only digits 0 - 9.
Both num1 and num2 do not contain any leading zero, except the number 0 itself.
You must not use any built - in BigInteger library or convert the inputs to integer directly.

string multiply(string num1, string num2)
{
    string res = "";
    int m = num1.size(), n = num2.size();
    vector<int> vals(m + n);
    for (int i = m - 1; i >= 0; --i) {
        for (int j = n - 1; j >= 0; --j) {
            int mul = (num1[i] - '0') * (num2[j] - '0');
            int p1 = i + j, p2 = i + j + 1, sum = mul + vals[p2];
            vals[p1] += sum / 10;
            vals[p2] = sum % 10;
        }
    }
    for (int val : vals) {
        if (!res.empty() || val != 0) res.push_back(val + '0');
    }
    return res.empty() ? "0" : res;
}

---------------------------------------------------------------------

//46 Permutations
Given a collection of distinct integers, return all possible permutations.

Example:
    Input: [1, 2, 3]
    Output :
    [
        [1, 2, 3],
        [1, 3, 2],
        [2, 1, 3],
        [2, 3, 1],
        [3, 1, 2],
        [3, 2, 1]
    ]

vector<vector<int>> permute(vector<int>& nums)
{
    if (nums.size() == 0) {
        return { {} };
    }
    vector<vector<int>> res{ {} };
    for (int a : nums) {
        vector<vector<int>> temp;
        for (int k = 0; k < res.size(); k++) {
            for (int i = 0; i <= res[k].size(); ++i) {
                vector<int> one = res[k];
                one.insert(one.begin() + i, a);
                temp.push_back(one);
            }
        }
        res = temp;
    }
    return res;
}

---------------------------------------------------------------------

//47 Permutations II
Given a collection of numbers that might contain duplicates, return all possible unique permutations.

Example:
    Input: [1, 1, 2]
    Output :
    [
        [1, 1, 2],
        [1, 2, 1],
        [2, 1, 1]
    ]

vector<vector<int>> permuteUnique(vector<int>& nums)
{
    if (nums.empty()) return { {} };
    set<vector<int>> res;
    int first = nums[0];
    nums.erase(nums.begin());
    vector<vector<int>> words = permuteUnique(nums);
    for (auto& a : words) {
        for (int i = 0; i <= a.size(); ++i) {
            a.insert(a.begin() + i, first);
            res.insert(a);
            a.erase(a.begin() + i);
        }
    }
    return vector<vector<int>>(res.begin(), res.end());
}

---------------------------------------------------------------------

//48 Rotate Image
You are given an n x n 2D matrix representing an image.

Rotate the image by 90 degrees(clockwise).

Note:
You have to rotate the image in - place, which means you have to modify the input 2D matrix directly.DO NOT allocate another 2D matrix and do the rotation.

Example 1 :
    Given input matrix =
    [
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9]
    ] ,

    rotate the input matrix in - place such that it becomes :
    [
        [7, 4, 1],
        [8, 5, 2],
        [9, 6, 3]
    ]

Example 2:
    Given input matrix =
    [
        [ 5, 1, 9, 11],
        [2, 4, 8, 10],
        [13, 3, 6, 7],
        [15, 14, 12, 16]
    ] ,

    rotate the input matrix in - place such that it becomes :
    [
        [15, 13, 2, 5],
        [14, 3, 4, 1],
        [12, 6, 8, 9],
        [16, 7, 10, 11]
    ]

void rotate(vector<vector<int> >& matrix)
{
    int n = matrix.size();
    for (int i = 0; i < n / 2; ++i) {
        for (int j = i; j < n - 1 - i; ++j) {
            int tmp = matrix[i][j];
            matrix[i][j] = matrix[n - 1 - j][i];
            matrix[n - 1 - j][i] = matrix[n - 1 - i][n - 1 - j];
            matrix[n - 1 - i][n - 1 - j] = matrix[j][n - 1 - i];
            matrix[j][n - 1 - i] = tmp;
        }
    }
}

---------------------------------------------------------------------

//49 Group Anagrams
Given an array of strings, group anagrams together.

Example:
    Input: ["eat", "tea", "tan", "ate", "nat", "bat"] ,
    Output :
    [
        ["ate", "eat", "tea"],
        ["nat", "tan"],
        ["bat"]
    ]

Note :
    All inputs will be in lowercase.
    The order of your output does not matter.

vector<vector<string>> groupAnagrams(vector<string>& strs)
{
    vector<vector<string>> res;
    unordered_map<string, vector<string>> m;
    for (string str : strs) {
        string t = str;
        sort(t.begin(), t.end());
        m[t].push_back(str);
    }
    for (auto a : m) {
        res.push_back(a.second);
    }
    return res;
}

---------------------------------------------------------------------

//50 Pow(x, n)
Implement pow(x, n), which calculates x raised to the power n(xn).

Example 1:
    Input: 2.00000, 10
    Output : 1024.00000

Example 2 :
    Input : 2.10000, 3
    Output : 9.26100

Example 3 :
    Input : 2.00000, -2
    Output : 0.25000
    Explanation : 2 - 2 = 1 / 22 = 1 / 4 = 0.25

Note :
    -100.0 < x < 100.0
    n is a 32 - bit signed integer, within the range [−231, 231 − 1]

double myPowHelp(double x, long long n)
{
    if (n == 0) {
        return 1;
    }
    
    double half = myPowHelp(x, n / 2);
    double res = half * half;
    if (n % 2 == 1) {
        res *= x;
    }
    return res;
}

double myPow(double x, int n)
{
    if (n == 0) {
        return 1;
    }
    return n > 0 ? myPowHelp(x, n) : 1 / myPowHelp(x, 0 - (long long)n);
}

---------------------------------------------------------------------

//54 Spiral Matrix
Given a matrix of m x n elements(m rows, n columns), return all elements of the matrix in spiral order.

Example 1:
    Input:
    [
        [ 1, 2, 3 ],
        [4, 5, 6],
        [7, 8, 9]
    ]
    Output : [1, 2, 3, 6, 9, 8, 7, 4, 5]

Example 2 :
    Input :
    [
        [1, 2, 3, 4],
        [5, 6, 7, 8],
        [9, 10, 11, 12]
    ]
    Output : [1, 2, 3, 4, 8, 12, 11, 10, 9, 5, 6, 7]

vector<int> spiralOrder(vector<vector<int>>& matrix)
{
    if (matrix.empty() || matrix[0].empty()) return {};
    int m = matrix.size(), n = matrix[0].size();
    vector<int> res;
    int up = 0, down = m - 1, left = 0, right = n - 1;
    while (true) {
        for (int j = left; j <= right; ++j) res.push_back(matrix[up][j]);
        if (++up > down) break;
        for (int i = up; i <= down; ++i) res.push_back(matrix[i][right]);
        if (--right < left) break;
        for (int j = right; j >= left; --j) res.push_back(matrix[down][j]);
        if (--down < up) break;
        for (int i = down; i >= up; --i) res.push_back(matrix[i][left]);
        if (++left > right) break;
    }
    return res;
}

---------------------------------------------------------------------

//55 Jump Game
Given an array of non - negative integers, you are initially positioned at the first index of the array.

Each element in the array represents your maximum jump length at that position.

Determine if you are able to reach the last index.

Example 1:
    Input: [2, 3, 1, 1, 4]
    Output : true
    Explanation : Jump 1 step from index 0 to 1, then 3 steps to the last index.

Example 2 :
    Input : [3, 2, 1, 0, 4]
    Output : false
    Explanation : You will always arrive at index 3 no matter what.Its maximum
    jump length is 0, which makes it impossible to reach the last index.

贪心算法

bool canJump(vector<int>& nums)
{
    int reach = 0, n = nums.size();
    for (int i = 0; i < n; i++) {
        if (reach < i || reach >= n - 1) {
            break;
        }
        reach = max(reach, i + nums[i]);
    }
    return reach >= n - 1;
}

---------------------------------------------------------------------

//56 Merge Intervals
Given a collection of intervals, merge all overlapping intervals.

Example 1:
    Input: [[1, 3], [2, 6], [8, 10], [15, 18]]
    Output : [[1, 6], [8, 10], [15, 18]]
    Explanation : Since intervals[1, 3] and [2, 6] overlaps, merge them into[1, 6].

Example 2 :
    Input : [[1, 4], [4, 5]]
    Output : [[1, 5]]
    Explanation : Intervals[1, 4] and [4, 5] are considered overlapping.

NOTE : input types have been changed on April 15, 2019. Please reset to default code definition to get new method signature.

static bool myCompare(const vector<int>& a, const vector<int>& b)
{
    return a[0] < b[0];
}

vector<vector<int>> merge(vector<vector<int>>& intervals) {
    if (intervals.size() == 0 || intervals[0].size() == 0) {
        return {};
    }
    int m = intervals.size();
    int n = intervals[0].size();

    stable_sort(intervals.begin(), intervals.end(), myCompare);

    vector<vector<int>> res;
    res.push_back(intervals[0]);
    for (int i = 1; i < m; i++) {
        if (intervals[i][0] <= res.back()[1]) {
            res.back()[1] = res.back()[1] > intervals[i][1] ? res.back()[1] : intervals[i][1];
        }
        else {
            res.push_back(intervals[i]);
        }
    }
    return res;
}

---------------------------------------------------------------------

//59 Spiral Matrix II
Given a positive integer n, generate a square matrix filled with elements from 1 to n2 in spiral order.

Example:
    Input: 3
    Output :
    [
        [ 1, 2, 3 ],
        [8, 9, 4],
        [7, 6, 5]
    ]

vector<vector<int>> generateMatrix(int n) {
    vector<vector<int>> res(n, vector<int>(n, 0));
    int up = 0, down = n - 1, left = 0, right = n - 1, val = 1;
    while (true) {
        for (int j = left; j <= right; ++j) res[up][j] = val++;
        if (++up > down) break;
        for (int i = up; i <= down; ++i) res[i][right] = val++;
        if (--right < left) break;
        for (int j = right; j >= left; --j) res[down][j] = val++;
        if (--down < up) break;
        for (int i = down; i >= up; --i) res[i][left] = val++;
        if (++left > right) break;
    }
    return res;
}

---------------------------------------------------------------------

//60 Permutation Sequence
The set[1, 2, 3, ..., n] contains a total of n!unique permutations.

By listingand labeling all of the permutations in order, we get the following sequence for n = 3:

"123"
"132"
"213"
"231"
"312"
"321"
Given n and k, return the kth permutation sequence.

Note:

Given n will be between 1 and 9 inclusive.
Given k will be between 1 and n!inclusive.
Example 1 :
    Input : n = 3, k = 3
    Output : "213"

Example 2 :
    Input : n = 4, k = 9
    Output : "2314"



---------------------------------------------------------------------
---------------------------------------------------------------------
---------------------------------------------------------------------
---------------------------------------------------------------------
---------------------------------------------------------------------
---------------------------------------------------------------------
---------------------------------------------------------------------
---------------------------------------------------------------------
---------------------------------------------------------------------
---------------------------------------------------------------------
---------------------------------------------------------------------
---------------------------------------------------------------------
---------------------------------------------------------------------
---------------------------------------------------------------------
---------------------------------------------------------------------
---------------------------------------------------------------------
---------------------------------------------------------------------
---------------------------------------------------------------------
---------------------------------------------------------------------
---------------------------------------------------------------------
---------------------------------------------------------------------
---------------------------------------------------------------------
---------------------------------------------------------------------
---------------------------------------------------------------------
---------------------------------------------------------------------
---------------------------------------------------------------------
---------------------------------------------------------------------
---------------------------------------------------------------------
---------------------------------------------------------------------
---------------------------------------------------------------------
---------------------------------------------------------------------
---------------------------------------------------------------------
---------------------------------------------------------------------
---------------------------------------------------------------------
---------------------------------------------------------------------
---------------------------------------------------------------------
---------------------------------------------------------------------
---------------------------------------------------------------------