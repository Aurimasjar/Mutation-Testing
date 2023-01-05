// A Java implementation of VendingMachine.
// Original in C++ by Liu Ling.
// Taken from a paper by Mary Jean Harrold.
// Translated to Java by Jeff Offutt, January 2004
// Main added later for testing

import java.util.*;
import java.io.*;

public class VendingMachine
{
    private int credit;
    private LinkedList stock;
    String Choc = "";

    // Maximum size of vendingMachine
    private static final int MAX = 10;

// In the C++ version, Ling used a C++ list<string>
// with operations like push(), pop(), etc to look
// a lot like a queue.
// list<string> stock; ??
// In the Java version I'm emulating a queue on
// top of the LinkedList.

    //************************************************
// Constructor
// VendingMachine starts empty.
//************************************************
    VendingMachine()
    {
        credit = 0;
        stock  = new LinkedList(); // Empty stock.
    }

    //************************************************
// A coin is given to the vendingMachine.
// Must be a dime, quarter or dollar.
// Ignores invalid input
//************************************************
    public void coin (int coin)
    {
        if (coin != 10 && coin != 25 && coin != 100)
            return;
        if (credit >= 90)
            return;
        credit = credit + coin;
        return;
    }

    //************************************************
// User asks for a chocolate.
// Returns the change and the sets the
// parameter StringBuffer variable Choc.
// If not enough money or no chocolates,
// returns 0 and a blank string.
//************************************************
// Necessary because strings are immutable
// C++ version returned both choc and change as parameters
    public int getChoc (StringBuffer choc)
    {
        int change;

        if (credit < 90 || stock.size() <= 0)
        {
            change = 0;
            choc.replace (0, choc.length(), "");
            return (change);
        }
        change = credit - 90;
        credit = 0;

        choc.replace (0, choc.length(), (String) stock.removeFirst());

        return (change);
    }

    //************************************************
// Adds one new piece of chocolate to the machine
// If machine is full, nothing happens
//************************************************
    public void addChoc (String choc)
    {
        if (stock.size() >= MAX)
            return;
        stock.add (choc);
        return;
    }

    // Needed for testing
    public int getCredit ()
    {
        return (credit);
    }


    // Needed for testing
    public LinkedList getStock ()
    {
        return (stock);
    }

} // End class VendingMachine