public class Algorithm {

//    public int recursiveKnapsack(int W, int wt[], int val[], int n)
//    {
//        if (n == 0 || W == 0) {
//            return 0;
//        }
//        if (wt[n - 1] > W) {
//            return recursiveKnapsack(W, wt, val, n - 1);
//        }
//        else {
//            int val1 = val[n - 1] + recursiveKnapsack(W - wt[n - 1], wt, val, n - 1);
//            int val2 = recursiveKnapsack(W, wt, val, n - 1);
//            if(val1 > val2) {
//                return val1;
//            }
//            return val2;
//        }
//    }

//    public int dynamicKnapsack(int W, int wt[], int val[], int n)
//    {
//        int i, w;
//        int K[][] = new int[n + 1][W + 1];
//
//        for (i = 0; i<= n; i++) {
//            for (w = 0; w<= W; w++) {
//                if (i == 0 || w == 0) {
//                    K[i][w] = 0;
//                }
//                else if (wt[i - 1]<= w) {
//                    int val1 = val[i - 1] + K[i - 1][w - wt[i - 1]];
//                    int val2 = K[i - 1][w];
//                    if(val1 > val2) {
//                        K[i][w] = val1;
//                    }
//                    else {
//                        K[i][w] = val2;
//                    }
//                }
//                else {
//                    K[i][w] = K[i - 1][w];
//                }
//            }
//        }
//        return K[n][W];
//    }

    public boolean triangle(int a, int b, int c) {
        if(a + b > c && a + c > b && b + c > a) {
            return true;
        }
        return false;
    }

}