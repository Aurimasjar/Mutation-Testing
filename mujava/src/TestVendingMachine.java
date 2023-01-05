import org.junit.After;
import org.junit.Before;
import org.junit.Test;
import static org.junit.Assert.*;

public class TestVendingMachine
{
    private VendingMachine vm1;
    @Before
    public void setUp() throws Exception
    {
        vm1 = new VendingMachine();
    }

    @After
    public void tearDown() throws Exception
    {
        vm1 = null;
    }

    @Test
    public void testConstructor()
    {
        assertEquals (0, vm1.getCredit());
        assertEquals (0, vm1.getStock().size());
    }

    @Test
    public void testCoin1()
    {  // 20 is not a valid coin
        vm1.coin (20);
        assertEquals (0, vm1.getCredit());
    }

    @Test
    public void testCoin2()
    {  // 25 is a valid coin
        vm1.coin (25);
        assertEquals (25, vm1.getCredit());
    }

    @Test
    public void testCoin3()
    {  // 25 is a valid coin
        vm1.coin (25);
        vm1.coin (25);
        vm1.coin (25);
        vm1.coin (25);
        assertEquals (100, vm1.getCredit());
        // Cannot add if credit >= 90
        vm1.coin (25);
        assertEquals (100, vm1.getCredit());
    }

//    @Test
//    public void testAddChoc()
//    {
//        StringBuffer choc = new StringBuffer().append ("MM");
//        vm1.addChoc (choc);
//        assertEquals (1, vm1.getStock().size());
//    }
//
//    @Test
//    public void testGetChoc1()
//    {  // Happy path test, get a chocoloate and 10 change
//        StringBuffer choc = new StringBuffer().append ("MM");
//        vm1.addChoc (choc);
//        vm1.coin (25);
//        vm1.coin (25);
//        vm1.coin (25);
//        vm1.coin (25);
//        assertEquals (10, vm1.getChoc());
//        assertEquals (0, vm1.getStock().size());
//    }
//
//    @Test
//    public void testGetChoc2()
//    {  // Not enough money
//        StringBuffer choc = new StringBuffer().append ("MM");
//        vm1.addChoc (choc);
//        vm1.coin (25);
//        assertEquals (0, vm1.getChoc());
//        assertEquals (1, vm1.getStock().size());
//    }
//
//    @Test
//    public void testGetChoc3()
//    {  // No chocolate
//        vm1.coin (25);
//        vm1.coin (25);
//        vm1.coin (25);
//        vm1.coin (25);
//        assertEquals (0, vm1.getChoc());
//        assertEquals (0, vm1.getStock().size());
//    }
}