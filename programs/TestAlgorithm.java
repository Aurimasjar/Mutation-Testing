import org.junit.After;
import org.junit.Before;
import org.junit.Test;
import static org.junit.Assert.*;

public class TestAlgorithm
{
   private Algorithm alg;

   @Before
   public void setUp() throws Exception
   {
      alg = new Algorithm();
   }

   @After
   public void tearDown() throws Exception
   {
      alg = null;
   }

   @Test
   public void testPower()
   {
      assertEquals (0, alg.power(0, 1));
      assertEquals (1, alg.power(1, 0));
   }

   @Test
   public void testPrime()
   {
      assertEquals (false, alg.prime(1));
      assertEquals (true, alg.prime(2));
   }
}