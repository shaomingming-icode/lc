#include "medium.h"

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

Ì°ÐÄËã·¨

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




