#pragma once
#include <iostream>
#include <vector>
#include <map>
#include <set>
#include <iostream>
#include <algorithm>

using namespace std;

typedef struct ListNode {
	int val;
	struct ListNode* next;
	ListNode(int v) {
		val = v;
	}
}ListNode;

ListNode* swapPairs(ListNode*);