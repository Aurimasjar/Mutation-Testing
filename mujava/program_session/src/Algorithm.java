/* Example program, used to show the process of mutation testing. */

public class Algorithm {

    public boolean triangle(int a, int b, int c) {
        if(a + b > c && a + c > b && b + c > a) {
            return true;
        }
        return false;
    }

}