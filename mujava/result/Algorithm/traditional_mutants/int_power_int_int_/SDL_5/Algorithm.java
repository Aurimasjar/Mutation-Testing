// This is a mutant program.
// Author : ysma

package mujava.result.Algorithm.traditional_mutants.int_power_int_int_.SDL_5;


public class Algorithm
{

    public  int power( int x, int y )
    {
        if (x == 1) {
            return 0;
        }
        if (y == 1) {
            return x;
        }
        int P = 1;
        int i = 1;
        while (i <= y) {
            P = P * x;
            i++;
        }
        return P;
    }

    public  boolean prime( int n )
    {
        if (n < 2) {
            return false;
        }
        for (int i = 2; i <= Math.sqrt( n ); i++) {
            if (n % i == 0) {
                return false;
            }
        }
        return true;
    }

}