#include "medium.h"

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

string convert(string s, int numRows) {
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

int myAtoi(string str) {
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

//11 Container With Most Water
Given n non - negative integers a1, a2, ..., an , where each represents a point at coordinate(i, ai).n vertical lines are drawn such that the two endpoints of line i is at(i, ai) and (i, 0).Find two lines, which together with x - axis forms a container, such that the container contains the most water.

Note: You may not slant the containerand n is at least 2.

The above vertical lines are represented by array[1, 8, 6, 2, 5, 4, 8, 3, 7].In this case, the max area of water(blue section) the container can contain is 49.

Example:
    Input: [1, 8, 6, 2, 5, 4, 8, 3, 7]
    Output : 49

int maxArea(vector<int>& height) {
    int res = 0, i = 0, j = height.size() - 1;
    while (i < j) {
        int h = min(height[i], height[j]);
        res = max(res, h * (j - i));
        while (i < j && h == height[i]) ++i;
        while (i < j && h == height[j]) --j;
    }
    return res;
}

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

string intToRoman(int num) {
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

vector<vector<int>> permuteUnique(vector<int>& nums) {
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

bool canJump(vector<int>& nums) {
    int reach = 0;
    int n = nums.size();
    for (int i = 0; i < n; i++) {
        if (reach < i || reach >= n - 1) {
            break;
        }
        reach = i + nums[i] > reach ? i + nums[i] : reach;
    }
    return reach >= n - 1;
}

//56 Merge Intervals
Given a collection of intervals, merge all overlapping intervals.

Example 1:
    Input: [[1,3],[2,6],[8,10],[15,18]]
    Output: [[1,6],[8,10],[15,18]]
    Explanation: Since intervals [1,3] and [2,6] overlaps, merge them into [1,6].

Example 2:
    Input: [[1,4],[4,5]]
    Output: [[1,5]]
    Explanation: Intervals [1,4] and [4,5] are considered overlapping.

static bool myCompare(const vector<int>& a, const vector<int>& b)
{
    return a[0] < b[0];
}

vector<vector<int>> merge(vector<vector<int>>& intervals) {
    if (intervals.size() == 0 || intervals[0].size() == 0) {
        return { {} };
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
