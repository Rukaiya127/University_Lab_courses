#include <bits/stdc++.h>
using namespace std;

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    int t;
    cin >> t;
    while (t--) {
        long long n;
        int k;
        cin >> n >> k;

        vector<long long> ans;

        if (k == 1) {
            ans.push_back(n);
        }
        else if (k % 2 == 1) {
            ans.push_back(n);
            for (int i = 1; i < k; i++) ans.push_back(0);
        }
        else {
            long long p = 1;
            while ((p << 1) <= n) p <<= 1;

            long long a = p - 1;
            long long b = n ^ a;

            ans.push_back(a);
            ans.push_back(b);
            for (int i = 2; i < k; i++) ans.push_back(0);
        }

        for (auto x : ans) cout << x << " ";
        cout << "\n";
    }
    return 0;
}
