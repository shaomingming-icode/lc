#include "medium.h"

---------------------------------------------------------------------

320. Generalized Abbreviation
Write a function to generate the generalized abbreviations of a word. 
Note: The order of the output does not matter.

Example:
	Input: "word"
	Output:
	["word", "1ord", "w1rd", "wo1d", "wor1", "2rd", "w2d", "wo2", "1o1d", "1or1", "w1r1", "1o2", "2r1", "3d", "w3", "4"]

vector<string> generateAbbreviations(string word) {
    vector<string> res{ word };
    helper(word, 0, res);
    return res;
}
void helper(string word, int pos, vector<string>& res) {
    for (int i = pos; i < word.size(); ++i) {  //从pos开始
        for (int j = 1; i + j <= word.size(); ++j) {  //数字
            string t = word.substr(0, i);
            t += to_string(j) + word.substr(i + j);
            res.push_back(t);
            helper(t, i + 1 + to_string(j).size(), res);
        }
    }
}

---------------------------------------------------------------------

322. Coin Change
You are given coins of different denominations and a total amount of money amount. Write a function to compute the fewest number of coins that you need to make up that amount. If that amount of money cannot be made up by any combination of the coins, return -1.

Example 1:
	Input: coins = [1, 2, 5], amount = 11
	Output: 3 
	Explanation: 11 = 5 + 5 + 1
Example 2:
	Input: coins = [2], amount = 3
	Output: -1
Note:
You may assume that you have an infinite number of each kind of coin.

int coinChange(vector<int>& coins, int amount) {
    vector<int> values(amount + 1, amount + 1);
    values[0] = 0;
    for (int i = 0; i <= amount; i++) {
        for (int j = 0; j < coins.size(); j++) {
            if (i - coins[j] >= 0) {
                values[i] = min(values[i - coins[j]] + 1, values[i]);
            }
        }
    }
    return values.back() > amount ? -1 : values.back();
}

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