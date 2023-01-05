package java;

class JavaTest {
    private static String hello() {
        return "Hello";
    }

    public static void main(String[] args) {
        String hello = "Hello", lo = "lo";
        System.out.print((hello == "Hello") + " ");
        System.out.print((hello == ("Hel"+"lo")) + " ");
        System.out.print((hello == ("Hel"+lo)) + " ");
        System.out.print(hello == hello() + " ");
    }
}