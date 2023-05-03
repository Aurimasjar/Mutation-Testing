import org.junit.After;
import org.junit.Before;
import org.junit.Test;
import static org.junit.Assert.*;

public class TestDatasetRootAlgorithm
{
   private DatasetRootAlgorithm alg;

   @Before
   public void setUp() throws Exception
   {
      alg = new DatasetRootAlgorithm();
   }

   @After
   public void tearDown() throws Exception
   {
      alg = null;
   }

   public boolean compareDoubleArrays(double[] arr1, double[] arr2) {
      if(arr1.length != arr2.length) {
         return false;
      }
      for(int i = 0; i < arr1.length; i++) {
         if(Math.abs(arr1[i] - arr2[i]) >= 1e-09) {
            return false;
         }
      }
      return true;
   }

   public boolean compareGraphs(int[][] graph1, int[][] graph2) {
      int n = graph1.length;
      if(n != graph2.length) {
         return false;
      }
      for(int i = 0; i < n; i++) {
         if(graph1[i].length != n || graph2[i].length != n) {
            return false;
         }
         for(int j = 0; j < n; j++) {
            if(graph1[i][j] != graph2[i][j]) {
               return false;
            }
         }
      }
      return true;
   }

   @Test
   public void testLinearSearch1()
   {
      int[] x0 = {};
      int[] x1 = {5};
      int[] x2 = {16, 14};
      int[] x3 = {11, 11};
      int[] x4 = {7, 4, 1, 3, 5, 7, 2, -1, 11, 7, 7, 1};
      assertEquals (-1, alg.linearSearch1(x0, 0));
      assertEquals (-1, alg.linearSearch1(x1, 4));
      assertEquals (0, alg.linearSearch1(x1, 5));
      assertEquals (0, alg.linearSearch1(x2, 16));
      assertEquals (1, alg.linearSearch1(x2, 14));
      assertEquals (-1, alg.linearSearch1(x2, -1));
      assertEquals (0, alg.linearSearch1(x3, 11));
      assertEquals (0, alg.linearSearch1(x4, 7));
      assertEquals (1, alg.linearSearch1(x4, 4));
      assertEquals (2, alg.linearSearch1(x4, 1));
      assertEquals (8, alg.linearSearch1(x4, 11));
      assertEquals (-1, alg.linearSearch1(x4, 0));
   }

   @Test
   public void testLinearSearch2()
   {
      long[] x0 = {};
      long[] x1 = {5};
      long[] x2 = {16, 14};
      long[] x3 = {11, 11};
      long[] x4 = {7, 4, 1, 3, 5, 7, 2, -1, 11, 7, 7, 1};
      assertEquals (-1, alg.linearSearch2(x0, 0));
      assertEquals (-1, alg.linearSearch2(x1, 4));
      assertEquals (0, alg.linearSearch2(x1, 5));
      assertEquals (0, alg.linearSearch2(x2, 16));
      assertEquals (1, alg.linearSearch2(x2, 14));
      assertEquals (-1, alg.linearSearch2(x2, -1));
      assertEquals (0, alg.linearSearch2(x3, 11));
      assertEquals (0, alg.linearSearch2(x4, 7));
      assertEquals (1, alg.linearSearch2(x4, 4));
      assertEquals (2, alg.linearSearch2(x4, 1));
      assertEquals (8, alg.linearSearch2(x4, 11));
      assertEquals (-1, alg.linearSearch2(x4, 0));
   }

   @Test
   public void testLinearSearch3()
   {
      double[] x0 = {};
      double[] x1 = {5.0};
      double[] x2 = {16.1, 14.333};
      double[] x3 = {11.2, 11.2};
      double[] x4 = {7.0, 4.1, 1.0, 3.0, 5.0, 7.0, 2.0, -1.0, 11.0, 7.0, 7.0, 1.0};
      assertEquals (-1, alg.linearSearch3(x0, 0.0), 1e-9);
      assertEquals (-1, alg.linearSearch3(x1, 4.0), 1e-9);
      assertEquals (0, alg.linearSearch3(x1, 5.0), 1e-9);
      assertEquals (0, alg.linearSearch3(x2, 16.1), 1e-9);
      assertEquals (1, alg.linearSearch3(x2, 14.333), 1e-9);
      assertEquals (-1, alg.linearSearch3(x2, -1.0), 1e-9);
      assertEquals (0, alg.linearSearch3(x3, 11.2), 1e-9);
      assertEquals (0, alg.linearSearch3(x4, 7.0), 1e-9);
      assertEquals (1, alg.linearSearch3(x4, 4.10000000001), 1e-9);
      assertEquals (2, alg.linearSearch3(x4, 1.0), 1e-9);
      assertEquals (8, alg.linearSearch3(x4, 11.0), 1e-9);
      assertEquals (-1, alg.linearSearch3(x4, 0.0), 1e-9);
   }

   @Test
   public void testLinearSearch4()
   {
      boolean[] x0 = {};
      boolean[] x1 = {true};
      boolean[] x2 = {false, true};
      boolean[] x3 = {false, false};
      boolean[] x4 = {false, false, true, false, false};
      assertEquals (-1, alg.linearSearch4(x0, true));
      assertEquals (0, alg.linearSearch4(x1, true));
      assertEquals (-1, alg.linearSearch4(x1, false));
      assertEquals (0, alg.linearSearch4(x2, false));
      assertEquals (1, alg.linearSearch4(x2, true));
      assertEquals (0, alg.linearSearch4(x3, false));
      assertEquals (-1, alg.linearSearch4(x3, true));
      assertEquals (0, alg.linearSearch4(x4, false));
      assertEquals (2, alg.linearSearch4(x4, true));
   }

   @Test
   public void testLinearSearch5()
   {
      char[] x0 = {};
      char[] x1 = {'a'};
      char[] x2 = {'b', 't'};
      char[] x3 = {'R', 'R'};
      char[] x4 = {'7', '4', '1', '3', '5', '7', '2', '-', 'T', '7', '7', '1'};
      assertEquals (-1, alg.linearSearch5(x0, '0'));
      assertEquals (-1, alg.linearSearch5(x1, '4'));
      assertEquals (0, alg.linearSearch5(x1, 'a'));
      assertEquals (0, alg.linearSearch5(x2, 'b'));
      assertEquals (1, alg.linearSearch5(x2, 't'));
      assertEquals (-1, alg.linearSearch5(x2, '-'));
      assertEquals (0, alg.linearSearch5(x3, 'R'));
      assertEquals (0, alg.linearSearch5(x4, '7'));
      assertEquals (1, alg.linearSearch5(x4, '4'));
      assertEquals (2, alg.linearSearch5(x4, '1'));
      assertEquals (8, alg.linearSearch5(x4, 'T'));
      assertEquals (-1, alg.linearSearch5(x4, '0'));
   }

   @Test
   public void testLinearSearch6()
   {
      String[] x0 = {};
      String[] x1 = {"ab"};
      String[] x2 = {"rr", "t"};
      String[] x3 = {"R", "R"};
      String[] x4 = {"7", "4", "1", "3", "5", "7", "2", "-", "T", "7", "7", "1"};
      assertEquals (-1, alg.linearSearch6(x0, "0"));
      assertEquals (-1, alg.linearSearch6(x1, "4"));
      assertEquals (0, alg.linearSearch6(x1, "ab"));
      assertEquals (0, alg.linearSearch6(x2, "rr"));
      assertEquals (1, alg.linearSearch6(x2, "t"));
      assertEquals (-1, alg.linearSearch6(x2, "-1"));
      assertEquals (0, alg.linearSearch6(x3, "R"));
      assertEquals (0, alg.linearSearch6(x4, "7"));
      assertEquals (1, alg.linearSearch6(x4, "4"));
      assertEquals (2, alg.linearSearch6(x4, "1"));
      assertEquals (8, alg.linearSearch6(x4, "T"));
      assertEquals (-1, alg.linearSearch6(x4, "0"));
   }



   @Test
   public void testBinarySearch1()
   {
      int[] x0 = {};
      int[] x1 = {5};
      int[] x2 = {14, 16};
      int[] x3 = {11, 11};
      int[] x4 = {-1, 1, 1, 2, 3, 4, 5, 7, 7, 7, 7, 11};
      assertEquals (-1, alg.binarySearch1(x0, 0, x0.length-1, 0));
      assertEquals (-1, alg.binarySearch1(x1, 0, x1.length-1, 4));
      assertEquals (0, alg.binarySearch1(x1, 0, x1.length-1, 5));
      assertEquals (0, alg.binarySearch1(x2, 0, x2.length-1, 14));
      assertEquals (1, alg.binarySearch1(x2, 0, x2.length-1, 16));
      assertEquals (-1, alg.binarySearch1(x2, 0, x2.length-1, -1));
      assertEquals (0, alg.binarySearch1(x3, 0, x3.length-1, 11));
      assertEquals (0, alg.binarySearch1(x4, 0, x4.length-1, -1));
      assertEquals (1, alg.binarySearch1(x4, 0, x4.length-1, 1));
      assertEquals (5, alg.binarySearch1(x4, 0, x4.length-1, 4));
      assertEquals (7, alg.binarySearch1(x4, 0, x4.length-1, 7));
      assertEquals (-1, alg.binarySearch1(x4, 0, x4.length-1, 8));
   }

   @Test
   public void testBinarySearch2()
   {
      long[] x0 = {};
      long[] x1 = {5};
      long[] x2 = {14, 16};
      long[] x3 = {11, 11};
      long[] x4 = {-1, 1, 1, 2, 3, 4, 5, 7, 7, 7, 7, 11};
      assertEquals (-1, alg.binarySearch2(x0, 0, x0.length-1, 0));
      assertEquals (-1, alg.binarySearch2(x1, 0, x1.length-1, 4));
      assertEquals (0, alg.binarySearch2(x1, 0, x1.length-1, 5));
      assertEquals (0, alg.binarySearch2(x2, 0, x2.length-1, 14));
      assertEquals (1, alg.binarySearch2(x2, 0, x2.length-1, 16));
      assertEquals (-1, alg.binarySearch2(x2, 0, x2.length-1, -1));
      assertEquals (0, alg.binarySearch2(x3, 0, x3.length-1, 11));
      assertEquals (0, alg.binarySearch2(x4, 0, x4.length-1, -1));
      assertEquals (1, alg.binarySearch2(x4, 0, x4.length-1, 1));
      assertEquals (5, alg.binarySearch2(x4, 0, x4.length-1, 4));
      assertEquals (7, alg.binarySearch2(x4, 0, x4.length-1, 7));
      assertEquals (-1, alg.binarySearch2(x4, 0, x4.length-1, 8));
   }

   @Test
   public void testBinarySearch3()
   {
      double[] x0 = {};
      double[] x1 = {5.0};
      double[] x2 = {14.333, 16.1};
      double[] x3 = {11.2, 11.2};
      double[] x4 = {-1.0, 1.0, 1.0, 2.0, 3.0, 4.1, 5.0, 7.0, 7.0, 7.0, 7.0, 11.0};
      assertEquals (-1, alg.binarySearch3(x0, 0, x0.length-1, 0.0), 1e-9);
      assertEquals (-1, alg.binarySearch3(x1, 0, x1.length-1, 4.0), 1e-9);
      assertEquals (0, alg.binarySearch3(x1, 0, x1.length-1, 5.0), 1e-9);
      assertEquals (0, alg.binarySearch3(x2, 0, x2.length-1, 14.333), 1e-9);
      assertEquals (1, alg.binarySearch3(x2, 0, x2.length-1, 16.1), 1e-9);
      assertEquals (-1, alg.binarySearch3(x2, 0, x2.length-1, -1.0), 1e-9);
      assertEquals (0, alg.binarySearch3(x3, 0, x3.length-1, 11.2), 1e-9);
      assertEquals (0, alg.binarySearch3(x4, 0, x4.length-1, -1.0), 1e-9);
      assertEquals (1, alg.binarySearch3(x4, 0, x4.length-1, 1.0), 1e-9);
      assertEquals (5, alg.binarySearch3(x4, 0, x4.length-1, 4.10000000001), 1e-9);
      assertEquals (7, alg.binarySearch3(x4, 0, x4.length-1, 7.0), 1e-9);
      assertEquals (-1, alg.binarySearch3(x4, 0, x4.length-1, 8.0), 1e-9);
   }

   @Test
   public void testBinarySearch4()
   {
      boolean[] x0 = {};
      boolean[] x1 = {true};
      boolean[] x2 = {false, true};
      boolean[] x3 = {false, false};
      boolean[] x4 = {false, false, true, true, true};
      assertEquals (-1, alg.binarySearch4(x0, 0, x0.length-1, true));
      assertEquals (0, alg.binarySearch4(x1, 0, x1.length-1, true));
      assertEquals (-1, alg.binarySearch4(x1, 0, x1.length-1, false));
      assertEquals (0, alg.binarySearch4(x2, 0, x2.length-1, false));
      assertEquals (1, alg.binarySearch4(x2, 0, x2.length-1, true));
      assertEquals (0, alg.binarySearch4(x3, 0, x3.length-1, false));
      assertEquals (-1, alg.binarySearch4(x3, 0, x3.length-1, true));
      assertEquals (0, alg.binarySearch4(x4, 0, x4.length-1, false));
      assertEquals (2, alg.binarySearch4(x4, 0, x4.length-1, true));
   }

   @Test
   public void testBinarySearch5()
   {
      char[] x0 = {};
      char[] x1 = {'a'};
      char[] x2 = {'b', 't'};
      char[] x3 = {'R', 'R'};
      char[] x4 = {'a', 'c', 'c', 'd', 'e', 'f', 'g', 'i', 'i', 'i', 'i', 'T'};
      assertEquals (-1, alg.binarySearch5(x0, 0, x0.length-1, '0'));
      assertEquals (-1, alg.binarySearch5(x1, 0, x1.length-1, '4'));
      assertEquals (0, alg.binarySearch5(x1, 0, x1.length-1, 'a'));
      assertEquals (0, alg.binarySearch5(x2, 0, x2.length-1, 'b'));
      assertEquals (1, alg.binarySearch5(x2, 0, x2.length-1, 't'));
      assertEquals (-1, alg.binarySearch5(x2, 0, x2.length-1, '-'));
      assertEquals (0, alg.binarySearch5(x3, 0, x3.length-1, 'R'));
      assertEquals (0, alg.binarySearch5(x4, 0, x4.length-1, 'a'));
      assertEquals (1, alg.binarySearch5(x4, 0, x4.length-1, 'c'));
      assertEquals (5, alg.binarySearch5(x4, 0, x4.length-1, 'f'));
      assertEquals (7, alg.binarySearch5(x4, 0, x4.length-1, 'i'));
      assertEquals (-1, alg.binarySearch5(x4, 0, x4.length-1, '0'));
   }

   @Test
   public void testBinarySearch6()
   {
      String[] x0 = {};
      String[] x1 = {"ab"};
      String[] x2 = {"rr", "tt"};
      String[] x3 = {"R", "R"};
      String[] x4 = {"a", "c", "c", "d", "e", "f", "g", "i", "i", "i", "i", "T"};
      assertEquals (-1, alg.binarySearch6(x0, 0, x0.length-1, "0"));
      assertEquals (-1, alg.binarySearch6(x1, 0, x1.length-1, "4"));
      assertEquals (0, alg.binarySearch6(x1, 0, x1.length-1, "ab"));
      assertEquals (0, alg.binarySearch6(x2, 0, x2.length-1, "rr"));
      assertEquals (1, alg.binarySearch6(x2, 0, x2.length-1, "tt"));
      assertEquals (-1, alg.binarySearch6(x2, 0, x2.length-1, "-1"));
      assertEquals (0, alg.binarySearch6(x3, 0, x3.length-1, "R"));
      assertEquals (0, alg.binarySearch6(x4, 0, x4.length-1, "a"));
      assertEquals (1, alg.binarySearch6(x4, 0, x4.length-1, "c"));
      assertEquals (5, alg.binarySearch6(x4, 0, x4.length-1, "f"));
      assertEquals (7, alg.binarySearch6(x4, 0, x4.length-1, "i"));
      assertEquals (-1, alg.binarySearch6(x4, 0, x4.length-1, "0"));
   }



   @Test
   public void testMax1()
   {
      int[] x1 = {5};
      int[] x2 = {16, 14};
      int[] x3 = {11, 11};
      int[] x4 = {14, 16};
      int[] x5 = {14, 16, 18};
      int[] x6 = {7, 4, 1, 3, 5, 7, 2, -1, 11, 7, 7, 1};
      assertEquals (5, alg.max1(x1));
      assertEquals (16, alg.max1(x2));
      assertEquals (11, alg.max1(x3));
      assertEquals (16, alg.max1(x4));
      assertEquals (18, alg.max1(x5));
      assertEquals (11, alg.max1(x6));
   }

   @Test
   public void testMax2()
   {
      long[] x1 = {5};
      long[] x2 = {16, 14};
      long[] x3 = {11, 11};
      long[] x4 = {14, 16};
      long[] x5 = {14, 16, 18};
      long[] x6 = {7, 4, 1, 3, 5, 7, 2, -1, 11, 7, 7, 1};
      assertEquals (5, alg.max2(x1));
      assertEquals (16, alg.max2(x2));
      assertEquals (11, alg.max2(x3));
      assertEquals (16, alg.max2(x4));
      assertEquals (18, alg.max2(x5));
      assertEquals (11, alg.max2(x6));
   }

   @Test
   public void testMax3()
   {
      double[] x1 = {5.0};
      double[] x2 = {16.1, 14.333};
      double[] x3 = {11.0, 11.0};
      double[] x4 = {14.333, 16.1};
      double[] x5 = {14.0, 16.0, 18.0};
      double[] x6 = {7.0, 4.0, 1.0, 3.0, 5.0, 7.0, 2.0, -1.0, 11.0, 7.0, 7.0, 1.0};
      assertEquals (5.0, alg.max3(x1), 1e-9);
      assertEquals (16.1, alg.max3(x2), 1e-9);
      assertEquals (11.0, alg.max3(x3), 1e-9);
      assertEquals (16.1, alg.max3(x4), 1e-9);
      assertEquals (18.0, alg.max3(x5), 1e-9);
      assertEquals (11.0, alg.max3(x6), 1e-9);
   }


   @Test
   public void testIsEven1()
   {
      assertEquals (true, alg.isEven1(0));
      assertEquals (false, alg.isEven1(1));
      assertEquals (true, alg.isEven1(2));
      assertEquals (false, alg.isEven1(3));
      assertEquals (true, alg.isEven1(10));
      assertEquals (false, alg.isEven1(11));
   }

   @Test
   public void testIsEven2()
   {
      assertEquals (true, alg.isEven2(0));
      assertEquals (false, alg.isEven2(1));
      assertEquals (true, alg.isEven2(2));
      assertEquals (false, alg.isEven2(3));
      assertEquals (true, alg.isEven2(10));
      assertEquals (false, alg.isEven2(11));
   }


   @Test
   public void testPower1()
   {
      assertEquals (0, alg.power1(0, 1));
      assertEquals (1, alg.power1(1, 0));
      assertEquals (1, alg.power1(5, 0));
      assertEquals (1, alg.power1(1, 3));
      assertEquals (3, alg.power1(3, 1));
      assertEquals (8, alg.power1(2, 3));
      assertEquals (9, alg.power1(3, 2));
      assertEquals (9, alg.power1(-3, 2));
      assertEquals (256, alg.power1(4, 4));
   }

   @Test
   public void testPower2()
   {
      assertEquals (0.0, alg.power2(0.0, 1), 1e-9);
      assertEquals (1.0, alg.power2(1, 0), 1e-9);
      assertEquals (1.0, alg.power2(5, 0), 1e-9);
      assertEquals (1.0, alg.power2(1, 3), 1e-9);
      assertEquals (3.0, alg.power2(3, 1), 1e-9);
      assertEquals (8.0, alg.power1(2, 3), 1e-9);
      assertEquals (6.25, alg.power2(2.5, 2), 1e-9);
      assertEquals (6.25, alg.power2(-2.5, 2), 1e-9);
      assertEquals (256, alg.power2(4, 4), 1e-9);
   }



   @Test
   public void testPrime1()
   {
      assertEquals (false, alg.prime1(1));
      assertEquals (true, alg.prime1(2));
      assertEquals (true, alg.prime1(3));
      assertEquals (false, alg.prime1(4));
      assertEquals (true, alg.prime1(5));
      assertEquals (false, alg.prime1(6));
      assertEquals (true, alg.prime1(7));
      assertEquals (false, alg.prime1(8));
      assertEquals (false, alg.prime1(9));
      assertEquals (true, alg.prime1(11));
      assertEquals (false, alg.prime1(27));
      assertEquals (true, alg.prime1(29));
   }

   @Test
   public void testPrime2()
   {
      assertEquals (false, alg.prime2(1));
      assertEquals (true, alg.prime2(2));
      assertEquals (true, alg.prime2(3));
      assertEquals (false, alg.prime2(4));
      assertEquals (true, alg.prime2(5));
      assertEquals (false, alg.prime2(6));
      assertEquals (true, alg.prime2(7));
      assertEquals (false, alg.prime2(8));
      assertEquals (false, alg.prime2(9));
      assertEquals (true, alg.prime2(11));
      assertEquals (false, alg.prime2(27));
      assertEquals (true, alg.prime2(29));
   }

   @Test
   public void testSolveLinearEq()
   {
      double[][] matrix1 = {{3, -15}};
      double[][] matrix2 = {{4, 1, 24}, {12, -1, 16}};
      double[][] matrix3 = {{4, 1, 7}, {2, 3, -9}};
      double[][] matrix4 = {{2, 1, -1, 8}, {-3, -1, 2, -11}, {-2, 1, 2, -3}};
      double[][] matrix5 = {{3, 3, 1, 3}, {-1, 0, 0, -2}, {0, 5, 3, -9}};
      double[][] matrix6 = {{0, 8, 0, -9.6}, {4, 0, 0, 0}, {0, 0 , 2, 14}};
      double[][] matrix7 = {{2, 4, 3, 5, 39}, {3, 5, 4, 2, 33}, {4, 3, 5, 6, 49}, {5, 2, 6, 4, 43}};
      double[][] matrix8 = {{2, 10, -3, -1, 0, 87}, {0, 0, 0, 1, 0, -1}, {2, 0, 0, 1, 5, 8}, {4, 1, 1, 1, 3, 13}, {0, 0, 0, 0, 5, 5}};
      assertTrue (compareDoubleArrays(new double[] {-5}, alg.solveLinearEq(matrix1)));
      assertTrue (compareDoubleArrays(new double[] {2.5, 14}, alg.solveLinearEq(matrix2)));
      assertTrue (compareDoubleArrays(new double[] {3, -5}, alg.solveLinearEq(matrix3)));
      assertTrue (compareDoubleArrays(new double[] {2, 3, -1}, alg.solveLinearEq(matrix4)));
      assertTrue (compareDoubleArrays(new double[] {2, 0, -3}, alg.solveLinearEq(matrix5)));
      assertTrue (compareDoubleArrays(new double[] {0, -1.2, 7}, alg.solveLinearEq(matrix6)));
      assertTrue (compareDoubleArrays(new double[] {1, 2, 3, 4}, alg.solveLinearEq(matrix7)));
      assertTrue (compareDoubleArrays(new double[] {2, 7, -4, -1, 1}, alg.solveLinearEq(matrix8)));
   }

   @Test
   public void testCharCount()
   {
      assertEquals (0, alg.charCount("", 'a'));
      assertEquals (1, alg.charCount("a", 'a'));
      assertEquals (1, alg.charCount("abdceee", 'a'));
      assertEquals (0, alg.charCount("abdceee", 'x'));
      assertEquals (4, alg.charCount("baaaa", 'a'));
      assertEquals (3, alg.charCount("babcabcabc", 'a'));
   }

   @Test
   public void testWordCount()
   {
      assertEquals (0, alg.wordCount("", "a"));
      assertEquals (1, alg.wordCount("a", "a"));
      assertEquals (0, alg.wordCount("abdceee", "a"));
      assertEquals (1, alg.wordCount("a b d c e e e", "a"));
      assertEquals (0, alg.wordCount("a b d c e e e", "x"));
      assertEquals (0, alg.wordCount("ba aaa aaa", "a"));
      assertEquals (3, alg.wordCount("b abc abc abc", "abc"));
   }

   @Test
   public void testAllWordCount()
   {
      assertEquals (0, alg.allWordCount(""));
      assertEquals (1, alg.allWordCount("a"));
      assertEquals (1, alg.allWordCount("   abdceee   "));
      assertEquals (7, alg.allWordCount("a b d c e e e"));
      assertEquals (3, alg.allWordCount("ba aaa aaa   "));
      assertEquals (4, alg.allWordCount("b abc abc abc"));
   }

   @Test
   public void testPalindrome()
   {
      assertEquals (true, alg.palindrome(""));
      assertEquals (true, alg.palindrome("a"));
      assertEquals (false, alg.palindrome("   abdceee   "));
      assertEquals (true, alg.palindrome("a b d d b a"));
      assertEquals (false, alg.palindrome("aaabbb"));
      assertEquals (true, alg.palindrome("aabbaa"));
      assertEquals (false, alg.palindrome("abc abc abc"));
   }

   @Test
   public void testFirstNonRepeatingChar()
   {
      assertEquals ('\u0000', alg.firstNonRepeatingChar(""));
      assertEquals ('a', alg.firstNonRepeatingChar("a"));
      assertEquals ('a', alg.firstNonRepeatingChar("   abdceee   "));
      assertEquals ('\u0000', alg.firstNonRepeatingChar("a b d d b a"));
      assertEquals ('c', alg.firstNonRepeatingChar("aaabbc"));
      assertEquals ('d', alg.firstNonRepeatingChar("abc d abc e abc"));
   }



   @Test
   public void testBubbleSort1()
   {
      assertArrayEquals (new int[] {}, alg.bubbleSort1(new int[] {}));
      assertArrayEquals (new int[] {5}, alg.bubbleSort1(new int[] {5}));
      assertArrayEquals (new int[] {2, 6}, alg.bubbleSort1(new int[] {6, 2}));
      assertArrayEquals (new int[] {-1, 8, 8}, alg.bubbleSort1(new int[] {-1, 8, 8}));
      assertArrayEquals (new int[] {1, 4, 7, 9}, alg.bubbleSort1(new int[] {9, 7, 4, 1}));
      assertArrayEquals (new int[] {1, 2, 2, 3, 3, 4, 5, 7, 11}, alg.bubbleSort1(new int[] {5, 1, 2, 4, 3, 11, 7, 2, 3}));
   }

   @Test
   public void testBubbleSort2()
   {
      assertArrayEquals (new long[] {}, alg.bubbleSort2(new long[] {}));
      assertArrayEquals (new long[] {5}, alg.bubbleSort2(new long[] {5}));
      assertArrayEquals (new long[] {2, 6}, alg.bubbleSort2(new long[] {6, 2}));
      assertArrayEquals (new long[] {-1, 8, 8}, alg.bubbleSort2(new long[] {-1, 8, 8}));
      assertArrayEquals (new long[] {1, 4, 7, 9}, alg.bubbleSort2(new long[] {9, 7, 4, 1}));
      assertArrayEquals (new long[] {1, 2, 2, 3, 3, 4, 5, 7, 11}, alg.bubbleSort2(new long[] {5, 1, 2, 4, 3, 11, 7, 2, 3}));
   }

   @Test
   public void testBubbleSort3()
   {
      assertTrue (compareDoubleArrays(new double[] {}, alg.bubbleSort3(new double[] {})));
      assertTrue (compareDoubleArrays(new double[] {5}, alg.bubbleSort3(new double[] {5})));
      assertTrue (compareDoubleArrays(new double[] {2, 6}, alg.bubbleSort3(new double[] {6, 2})));
      assertTrue (compareDoubleArrays(new double[] {-1, 8, 8}, alg.bubbleSort3(new double[] {-1, 8, 8})));
      assertTrue (compareDoubleArrays(new double[] {1, 4, 7, 9}, alg.bubbleSort3(new double[] {9, 7, 4, 1})));
      assertTrue (compareDoubleArrays(new double[] {1, 2, 2, 3, 3, 4, 5, 7, 11}, alg.bubbleSort3(new double[] {5, 1, 2, 4, 3, 11, 7, 2, 3})));
   }

   @Test
   public void testBubbleSort4()
   {
      assertArrayEquals (new char[] {}, alg.bubbleSort4(new char[] {}));
      assertArrayEquals (new char[] {5}, alg.bubbleSort4(new char[] {5}));
      assertArrayEquals (new char[] {2, 6}, alg.bubbleSort4(new char[] {6, 2}));
      assertArrayEquals (new char[] {0, 8, 8}, alg.bubbleSort4(new char[] {0, 8, 8}));
      assertArrayEquals (new char[] {1, 4, 7, 9}, alg.bubbleSort4(new char[] {9, 7, 4, 1}));
      assertArrayEquals (new char[] {1, 2, 2, 3, 3, 4, 5, 7, 11}, alg.bubbleSort4(new char[] {5, 1, 2, 4, 3, 11, 7, 2, 3}));
   }

   @Test
   public void testBubbleSort5()
   {
      assertArrayEquals (new String[] {}, alg.bubbleSort5(new String[] {}));
      assertArrayEquals (new String[] {"5"}, alg.bubbleSort5(new String[] {"5"}));
      assertArrayEquals (new String[] {"2", "6"}, alg.bubbleSort5(new String[] {"6", "2"}));
      assertArrayEquals (new String[] {"-1", "8", "8"}, alg.bubbleSort5(new String[] {"-1", "8", "8"}));
      assertArrayEquals (new String[] {"1", "4", "7", "9"}, alg.bubbleSort5(new String[] {"9", "7", "4", "1"}));
      assertArrayEquals (new String[] {"1", "2", "2", "3", "3", "4", "5", "7", "9"}, alg.bubbleSort5(new String[] {"5", "1", "2", "4", "3", "9", "7", "2", "3"}));
   }

   @Test
   public void testInsertionSort1()
   {
      assertArrayEquals (new int[] {}, alg.insertionSort1(new int[] {}));
      assertArrayEquals (new int[] {5}, alg.insertionSort1(new int[] {5}));
      assertArrayEquals (new int[] {2, 6}, alg.insertionSort1(new int[] {6, 2}));
      assertArrayEquals (new int[] {-1, 8, 8}, alg.insertionSort1(new int[] {-1, 8, 8}));
      assertArrayEquals (new int[] {1, 4, 7, 9}, alg.insertionSort1(new int[] {9, 7, 4, 1}));
      assertArrayEquals (new int[] {1, 2, 2, 3, 3, 4, 5, 7, 11}, alg.insertionSort1(new int[] {5, 1, 2, 4, 3, 11, 7, 2, 3}));
   }

   @Test
   public void testInsertionSort2()
   {
      assertArrayEquals (new long[] {}, alg.insertionSort2(new long[] {}));
      assertArrayEquals (new long[] {5}, alg.insertionSort2(new long[] {5}));
      assertArrayEquals (new long[] {2, 6}, alg.insertionSort2(new long[] {6, 2}));
      assertArrayEquals (new long[] {-1, 8, 8}, alg.insertionSort2(new long[] {-1, 8, 8}));
      assertArrayEquals (new long[] {1, 4, 7, 9}, alg.insertionSort2(new long[] {9, 7, 4, 1}));
      assertArrayEquals (new long[] {1, 2, 2, 3, 3, 4, 5, 7, 11}, alg.insertionSort2(new long[] {5, 1, 2, 4, 3, 11, 7, 2, 3}));
   }

   @Test
   public void testInsertionSort3()
   {
      assertTrue (compareDoubleArrays(new double[] {}, alg.insertionSort3(new double[] {})));
      assertTrue (compareDoubleArrays(new double[] {5}, alg.insertionSort3(new double[] {5})));
      assertTrue (compareDoubleArrays(new double[] {2, 6}, alg.insertionSort3(new double[] {6, 2})));
      assertTrue (compareDoubleArrays(new double[] {-1, 8, 8}, alg.insertionSort3(new double[] {-1, 8, 8})));
      assertTrue (compareDoubleArrays(new double[] {1, 4, 7, 9}, alg.insertionSort3(new double[] {9, 7, 4, 1})));
      assertTrue (compareDoubleArrays(new double[] {1, 2, 2, 3, 3, 4, 5, 7, 11}, alg.insertionSort3(new double[] {5, 1, 2, 4, 3, 11, 7, 2, 3})));
   }

   @Test
   public void testInsertionSort4()
   {
      assertArrayEquals (new char[] {}, alg.insertionSort4(new char[] {}));
      assertArrayEquals (new char[] {5}, alg.insertionSort4(new char[] {5}));
      assertArrayEquals (new char[] {2, 6}, alg.insertionSort4(new char[] {6, 2}));
      assertArrayEquals (new char[] {0, 8, 8}, alg.insertionSort4(new char[] {0, 8, 8}));
      assertArrayEquals (new char[] {1, 4, 7, 9}, alg.insertionSort4(new char[] {9, 7, 4, 1}));
      assertArrayEquals (new char[] {1, 2, 2, 3, 3, 4, 5, 7, 11}, alg.insertionSort4(new char[] {5, 1, 2, 4, 3, 11, 7, 2, 3}));
   }

   @Test
   public void testInsertionSort5()
   {
      assertArrayEquals (new String[] {}, alg.insertionSort5(new String[] {}));
      assertArrayEquals (new String[] {"5"}, alg.insertionSort5(new String[] {"5"}));
      assertArrayEquals (new String[] {"2", "6"}, alg.insertionSort5(new String[] {"6", "2"}));
      assertArrayEquals (new String[] {"-1", "8", "8"}, alg.insertionSort5(new String[] {"-1", "8", "8"}));
      assertArrayEquals (new String[] {"1", "4", "7", "9"}, alg.insertionSort5(new String[] {"9", "7", "4", "1"}));
      assertArrayEquals (new String[] {"1", "2", "2", "3", "3", "4", "5", "7", "9"}, alg.insertionSort5(new String[] {"5", "1", "2", "4", "3", "9", "7", "2", "3"}));
   }

   @Test
   public void testSelectionSort1()
   {
      assertArrayEquals (new int[] {}, alg.selectionSort1(new int[] {}));
      assertArrayEquals (new int[] {5}, alg.selectionSort1(new int[] {5}));
      assertArrayEquals (new int[] {2, 6}, alg.selectionSort1(new int[] {6, 2}));
      assertArrayEquals (new int[] {-1, 8, 8}, alg.selectionSort1(new int[] {-1, 8, 8}));
      assertArrayEquals (new int[] {1, 4, 7, 9}, alg.selectionSort1(new int[] {9, 7, 4, 1}));
      assertArrayEquals (new int[] {1, 2, 2, 3, 3, 4, 5, 7, 11}, alg.selectionSort1(new int[] {5, 1, 2, 4, 3, 11, 7, 2, 3}));
   }

   @Test
   public void testSelectionSort2()
   {
      assertArrayEquals (new long[] {}, alg.selectionSort2(new long[] {}));
      assertArrayEquals (new long[] {5}, alg.selectionSort2(new long[] {5}));
      assertArrayEquals (new long[] {2, 6}, alg.selectionSort2(new long[] {6, 2}));
      assertArrayEquals (new long[] {-1, 8, 8}, alg.selectionSort2(new long[] {-1, 8, 8}));
      assertArrayEquals (new long[] {1, 4, 7, 9}, alg.selectionSort2(new long[] {9, 7, 4, 1}));
      assertArrayEquals (new long[] {1, 2, 2, 3, 3, 4, 5, 7, 11}, alg.selectionSort2(new long[] {5, 1, 2, 4, 3, 11, 7, 2, 3}));
   }

   @Test
   public void testSelectionSort3()
   {
      assertTrue (compareDoubleArrays(new double[] {}, alg.selectionSort3(new double[] {})));
      assertTrue (compareDoubleArrays(new double[] {5}, alg.selectionSort3(new double[] {5})));
      assertTrue (compareDoubleArrays(new double[] {2, 6}, alg.selectionSort3(new double[] {6, 2})));
      assertTrue (compareDoubleArrays(new double[] {-1, 8, 8}, alg.selectionSort3(new double[] {-1, 8, 8})));
      assertTrue (compareDoubleArrays(new double[] {1, 4, 7, 9}, alg.selectionSort3(new double[] {9, 7, 4, 1})));
      assertTrue (compareDoubleArrays(new double[] {1, 2, 2, 3, 3, 4, 5, 7, 11}, alg.selectionSort3(new double[] {5, 1, 2, 4, 3, 11, 7, 2, 3})));
   }

   @Test
   public void testSelectionSort4()
   {
      assertArrayEquals (new char[] {}, alg.selectionSort4(new char[] {}));
      assertArrayEquals (new char[] {5}, alg.selectionSort4(new char[] {5}));
      assertArrayEquals (new char[] {2, 6}, alg.selectionSort4(new char[] {6, 2}));
      assertArrayEquals (new char[] {0, 8, 8}, alg.selectionSort4(new char[] {0, 8, 8}));
      assertArrayEquals (new char[] {1, 4, 7, 9}, alg.selectionSort4(new char[] {9, 7, 4, 1}));
      assertArrayEquals (new char[] {1, 2, 2, 3, 3, 4, 5, 7, 11}, alg.selectionSort4(new char[] {5, 1, 2, 4, 3, 11, 7, 2, 3}));
   }

   @Test
   public void testSelectionSort5()
   {
      assertArrayEquals (new String[] {}, alg.selectionSort5(new String[] {}));
      assertArrayEquals (new String[] {"5"}, alg.selectionSort5(new String[] {"5"}));
      assertArrayEquals (new String[] {"2", "6"}, alg.selectionSort5(new String[] {"6", "2"}));
      assertArrayEquals (new String[] {"-1", "8", "8"}, alg.selectionSort5(new String[] {"-1", "8", "8"}));
      assertArrayEquals (new String[] {"1", "4", "7", "9"}, alg.selectionSort5(new String[] {"9", "7", "4", "1"}));
      assertArrayEquals (new String[] {"1", "2", "2", "3", "3", "4", "5", "7", "9"}, alg.selectionSort5(new String[] {"5", "1", "2", "4", "3", "9", "7", "2", "3"}));
   }

   @Test
   public void testMerge1()
   {
      assertArrayEquals (new int[] {5}, alg.merge1(new int[] {5}, 0, 0, 0));
      assertArrayEquals (new int[] {2, 6}, alg.merge1(new int[] {6, 2}, 0, 0, 1));
      assertArrayEquals (new int[] {-1, 8, 8}, alg.merge1(new int[] {-1, 8, 8}, 0, 1, 2));
      assertArrayEquals (new int[] {1, 4, 7, 9}, alg.merge1(new int[] {7, 9, 1, 4}, 0, 1, 3));
      assertArrayEquals (new int[] {7, 9, 4, 1}, alg.merge1(new int[] {9, 7, 4, 1}, 0, 0, 1));
      assertArrayEquals (new int[] {9, 7, 1, 4}, alg.merge1(new int[] {9, 7, 4, 1}, 2, 2, 3));
      assertArrayEquals (new int[] {1, 2, 2, 3, 3, 4, 5, 7, 11}, alg.merge1(new int[] {1, 2, 3, 4, 5, 2, 3, 7, 11}, 0, 4, 8));
   }

   @Test
   public void testMerge2()
   {
      assertArrayEquals (new long[] {5}, alg.merge2(new long[] {5}, 0, 0, 0));
      assertArrayEquals (new long[] {2, 6}, alg.merge2(new long[] {6, 2}, 0, 0, 1));
      assertArrayEquals (new long[] {-1, 8, 8}, alg.merge2(new long[] {-1, 8, 8}, 0, 1, 2));
      assertArrayEquals (new long[] {1, 4, 7, 9}, alg.merge2(new long[] {7, 9, 1, 4}, 0, 1, 3));
      assertArrayEquals (new long[] {7, 9, 4, 1}, alg.merge2(new long[] {9, 7, 4, 1}, 0, 0, 1));
      assertArrayEquals (new long[] {9, 7, 1, 4}, alg.merge2(new long[] {9, 7, 4, 1}, 2, 2, 3));
      assertArrayEquals (new long[] {1, 2, 2, 3, 3, 4, 5, 7, 11}, alg.merge2(new long[] {1, 2, 3, 4, 5, 2, 3, 7, 11}, 0, 4, 8));
   }

   @Test
   public void testMerge3()
   {
      assertTrue (compareDoubleArrays(new double[] {5}, alg.merge3(new double[] {5}, 0, 0, 0)));
      assertTrue (compareDoubleArrays(new double[] {2, 6}, alg.merge3(new double[] {6, 2}, 0, 0, 1)));
      assertTrue (compareDoubleArrays(new double[] {-1, 8, 8}, alg.merge3(new double[] {-1, 8, 8}, 0, 1, 2)));
      assertTrue (compareDoubleArrays(new double[] {1, 4, 7, 9}, alg.merge3(new double[] {7, 9, 1, 4}, 0, 1, 3)));
      assertTrue (compareDoubleArrays(new double[] {7, 9, 4, 1}, alg.merge3(new double[] {9, 7, 4, 1}, 0, 0, 1)));
      assertTrue (compareDoubleArrays(new double[] {9, 7, 1, 4}, alg.merge3(new double[] {9, 7, 4, 1}, 2, 2, 3)));
      assertTrue (compareDoubleArrays(new double[] {1, 2, 2, 3, 3, 4, 5, 7, 11}, alg.merge3(new double[] {1, 2, 3, 4, 5, 2, 3, 7, 11}, 0, 4, 8)));
   }

   @Test
   public void testMerge4()
   {
      assertArrayEquals (new char[] {5}, alg.merge4(new char[] {5}, 0, 0, 0));
      assertArrayEquals (new char[] {2, 6}, alg.merge4(new char[] {6, 2}, 0, 0, 1));
      assertArrayEquals (new char[] {0, 8, 8}, alg.merge4(new char[] {0, 8, 8}, 0, 1, 2));
      assertArrayEquals (new char[] {1, 4, 7, 9}, alg.merge4(new char[] {7, 9, 1, 4}, 0, 1, 3));
      assertArrayEquals (new char[] {7, 9, 4, 1}, alg.merge4(new char[] {9, 7, 4, 1}, 0, 0, 1));
      assertArrayEquals (new char[] {9, 7, 1, 4}, alg.merge4(new char[] {9, 7, 4, 1}, 2, 2, 3));
      assertArrayEquals (new char[] {1, 2, 2, 3, 3, 4, 5, 7, 11}, alg.merge4(new char[] {1, 2, 3, 4, 5, 2, 3, 7, 11}, 0, 4, 8));
   }

   @Test
   public void testMerge5()
   {
      assertArrayEquals (new String[] {"5"}, alg.merge5(new String[] {"5"}, 0, 0, 0));
      assertArrayEquals (new String[] {"2", "6"}, alg.merge5(new String[] {"6", "2"}, 0, 0, 1));
      assertArrayEquals (new String[] {"-1", "8", "8"}, alg.merge5(new String[] {"-1", "8", "8"}, 0, 1, 2));
      assertArrayEquals (new String[] {"1", "4", "7", "9"}, alg.merge5(new String[] {"7", "9", "1", "4"}, 0, 1, 3));
      assertArrayEquals (new String[] {"7", "9", "4", "1"}, alg.merge5(new String[] {"9", "7", "4", "1"}, 0, 0, 1));
      assertArrayEquals (new String[] {"9", "7", "1", "4"}, alg.merge5(new String[] {"9", "7", "4", "1"}, 2, 2, 3));
      assertArrayEquals (new String[] {"1", "2", "2", "3", "3", "4", "5", "7", "9"}, alg.merge5(new String[] {"1", "2", "3", "4", "5", "2", "3", "7", "9"}, 0, 4, 8));
   }

   @Test
   public void testMergeSort1()
   {
      assertArrayEquals (new int[] {}, alg.mergeSort1(new int[] {}, 0, -1));
      assertArrayEquals (new int[] {5}, alg.mergeSort1(new int[] {5}, 0, 0));
      assertArrayEquals (new int[] {2, 6}, alg.mergeSort1(new int[] {6, 2}, 0, 1));
      assertArrayEquals (new int[] {-1, 8, 8}, alg.mergeSort1(new int[] {-1, 8, 8}, 0, 2));
      assertArrayEquals (new int[] {1, 4, 7, 9}, alg.mergeSort1(new int[] {9, 7, 4, 1}, 0, 3));
      assertArrayEquals (new int[] {1, 2, 2, 3, 3, 4, 5, 7, 11}, alg.mergeSort1(new int[] {5, 1, 2, 4, 3, 11, 7, 2, 3}, 0, 8));
   }

   @Test
   public void testMergeSort2()
   {
      assertArrayEquals (new long[] {}, alg.mergeSort2(new long[] {}, 0, -1));
      assertArrayEquals (new long[] {5}, alg.mergeSort2(new long[] {5}, 0, 0));
      assertArrayEquals (new long[] {2, 6}, alg.mergeSort2(new long[] {6, 2}, 0, 1));
      assertArrayEquals (new long[] {-1, 8, 8}, alg.mergeSort2(new long[] {-1, 8, 8}, 0, 2));
      assertArrayEquals (new long[] {1, 4, 7, 9}, alg.mergeSort2(new long[] {9, 7, 4, 1}, 0, 3));
      assertArrayEquals (new long[] {1, 2, 2, 3, 3, 4, 5, 7, 11}, alg.mergeSort2(new long[] {5, 1, 2, 4, 3, 11, 7, 2, 3}, 0, 8));
   }

   @Test
   public void testMergeSort3()
   {
      assertTrue (compareDoubleArrays(new double[] {}, alg.mergeSort3(new double[] {}, 0, -1)));
      assertTrue (compareDoubleArrays(new double[] {5}, alg.mergeSort3(new double[] {5}, 0, 0)));
      assertTrue (compareDoubleArrays(new double[] {2, 6}, alg.mergeSort3(new double[] {6, 2}, 0, 1)));
      assertTrue (compareDoubleArrays(new double[] {-1, 8, 8}, alg.mergeSort3(new double[] {-1, 8, 8}, 0, 2)));
      assertTrue (compareDoubleArrays(new double[] {1, 4, 7, 9}, alg.mergeSort3(new double[] {9, 7, 4, 1}, 0, 3)));
      assertTrue (compareDoubleArrays(new double[] {1, 2, 2, 3, 3, 4, 5, 7, 11}, alg.mergeSort3(new double[] {5, 1, 2, 4, 3, 11, 7, 2, 3}, 0, 8)));
   }

   @Test
   public void testMergeSort4()
   {
      assertArrayEquals (new char[] {}, alg.mergeSort4(new char[] {}, 0, -1));
      assertArrayEquals (new char[] {5}, alg.mergeSort4(new char[] {5}, 0, 0));
      assertArrayEquals (new char[] {2, 6}, alg.mergeSort4(new char[] {6, 2}, 0, 1));
      assertArrayEquals (new char[] {0, 8, 8}, alg.mergeSort4(new char[] {0, 8, 8}, 0, 2));
      assertArrayEquals (new char[] {1, 4, 7, 9}, alg.mergeSort4(new char[] {9, 7, 4, 1}, 0, 3));
      assertArrayEquals (new char[] {1, 2, 2, 3, 3, 4, 5, 7, 11}, alg.mergeSort4(new char[] {5, 1, 2, 4, 3, 11, 7, 2, 3}, 0, 8));
   }
   @Test
   public void testMergeSort5()
   {
      assertArrayEquals (new String[] {}, alg.mergeSort5(new String[] {}, 0, -1));
      assertArrayEquals (new String[] {"5"}, alg.mergeSort5(new String[] {"5"}, 0, 0));
      assertArrayEquals (new String[] {"2", "6"}, alg.mergeSort5(new String[] {"6", "2"}, 0, 1));
      assertArrayEquals (new String[] {"-1", "8", "8"}, alg.mergeSort5(new String[] {"-1", "8", "8"}, 0, 2));
      assertArrayEquals (new String[] {"1", "4", "7", "9"}, alg.mergeSort5(new String[] {"9", "7", "4", "1"}, 0, 3));
      assertArrayEquals (new String[] {"1", "2", "2", "3", "3", "4", "5", "7", "9"}, alg.mergeSort5(new String[] {"5", "1", "2", "4", "3", "9", "7", "2", "3"}, 0, 8));
   }


   @Test
   public void testPartition1()
   {
      int[] arr1 = {5};
      int[] arr2 = {6, 2};
      int[] arr3 = {-1, 8, 8};
      int[] arr4 = {9, 7, 4, 1};
      int[] arr5 = {9, 9, 9, 8, 9};
      int[] arr6 = {5, 1, 2, 4, 3, 11, 7, 2, 3};

      assertEquals (0, alg.partition1(arr1, 0, 0));
      assertEquals (0, alg.partition1(arr2, 0, 1));
      assertEquals (2, alg.partition1(arr3, 0, 2));
      assertEquals (0, alg.partition1(arr4, 0, 3));
      assertEquals (4, alg.partition1(arr5, 0, 4));
      assertEquals (4, alg.partition1(arr6, 0, 8));

      assertArrayEquals (new int[] {5}, arr1);
      assertArrayEquals (new int[] {2, 6}, arr2);
      assertArrayEquals (new int[] {-1, 8, 8}, arr3);
      assertArrayEquals (new int[] {1, 7, 4, 9}, arr4);
      assertArrayEquals (new int[] {9, 9, 9, 8, 9}, arr5);
      assertArrayEquals (new int[] {1, 2, 3, 2, 3, 11, 7, 4, 5}, arr6);
   }

   @Test
   public void testPartition2()
   {
      long[] arr1 = {5};
      long[] arr2 = {6, 2};
      long[] arr3 = {-1, 8, 8};
      long[] arr4 = {9, 7, 4, 1};
      long[] arr5 = {9, 9, 9, 8, 9};
      long[] arr6 = {5, 1, 2, 4, 3, 11, 7, 2, 3};

      assertEquals (0, alg.partition2(arr1, 0, 0));
      assertEquals (0, alg.partition2(arr2, 0, 1));
      assertEquals (2, alg.partition2(arr3, 0, 2));
      assertEquals (0, alg.partition2(arr4, 0, 3));
      assertEquals (4, alg.partition2(arr5, 0, 4));
      assertEquals (4, alg.partition2(arr6, 0, 8));

      assertArrayEquals (new long[] {5}, arr1);
      assertArrayEquals (new long[] {2, 6}, arr2);
      assertArrayEquals (new long[] {-1, 8, 8}, arr3);
      assertArrayEquals (new long[] {1, 7, 4, 9}, arr4);
      assertArrayEquals (new long[] {9, 9, 9, 8, 9}, arr5);
      assertArrayEquals (new long[] {1, 2, 3, 2, 3, 11, 7, 4, 5}, arr6);
   }

   @Test
   public void testPartition3()
   {
      double[] arr1 = {5};
      double[] arr2 = {6, 2};
      double[] arr3 = {-1, 8, 8};
      double[] arr4 = {9, 7, 4, 1};
      double[] arr5 = {9, 9, 9, 8, 9};
      double[] arr6 = {5, 1, 2, 4, 3, 11, 7, 2, 3};

      assertEquals (0, alg.partition3(arr1, 0, 0));
      assertEquals (0, alg.partition3(arr2, 0, 1));
      assertEquals (2, alg.partition3(arr3, 0, 2));
      assertEquals (0, alg.partition3(arr4, 0, 3));
      assertEquals (4, alg.partition3(arr5, 0, 4));
      assertEquals (4, alg.partition3(arr6, 0, 8));

      assertTrue (compareDoubleArrays(new double[] {5}, arr1));
      assertTrue (compareDoubleArrays(new double[] {2, 6}, arr2));
      assertTrue (compareDoubleArrays(new double[] {-1, 8, 8}, arr3));
      assertTrue (compareDoubleArrays(new double[] {1, 7, 4, 9}, arr4));
      assertTrue (compareDoubleArrays(new double[] {9, 9, 9, 8, 9}, arr5));
      assertTrue (compareDoubleArrays(new double[] {1, 2, 3, 2, 3, 11, 7, 4, 5}, arr6));
   }

   @Test
   public void testPartition4()
   {
      char[] arr1 = {5};
      char[] arr2 = {6, 2};
      char[] arr3 = {0, 8, 8};
      char[] arr4 = {9, 7, 4, 1};
      char[] arr5 = {9, 9, 9, 8, 9};
      char[] arr6 = {5, 1, 2, 4, 3, 11, 7, 2, 3};

      assertEquals (0, alg.partition4(arr1, 0, 0));
      assertEquals (0, alg.partition4(arr2, 0, 1));
      assertEquals (2, alg.partition4(arr3, 0, 2));
      assertEquals (0, alg.partition4(arr4, 0, 3));
      assertEquals (4, alg.partition4(arr5, 0, 4));
      assertEquals (4, alg.partition4(arr6, 0, 8));

      assertArrayEquals (new char[] {5}, arr1);
      assertArrayEquals (new char[] {2, 6}, arr2);
      assertArrayEquals (new char[] {0, 8, 8}, arr3);
      assertArrayEquals (new char[] {1, 7, 4, 9}, arr4);
      assertArrayEquals (new char[] {9, 9, 9, 8, 9}, arr5);
      assertArrayEquals (new char[] {1, 2, 3, 2, 3, 11, 7, 4, 5}, arr6);
   }

   @Test
   public void testPartition5()
   {
      String[] arr1 = {"5"};
      String[] arr2 = {"6", "2"};
      String[] arr3 = {"-1", "8", "8"};
      String[] arr4 = {"9", "7", "4", "1"};
      String[] arr5 = {"9", "9", "9", "8", "9"};
      String[] arr6 = {"5", "1", "2", "4", "3", "9", "7", "2", "3"};

      assertEquals (0, alg.partition5(arr1, 0, 0));
      assertEquals (0, alg.partition5(arr2, 0, 1));
      assertEquals (2, alg.partition5(arr3, 0, 2));
      assertEquals (0, alg.partition5(arr4, 0, 3));
      assertEquals (4, alg.partition5(arr5, 0, 4));
      assertEquals (4, alg.partition5(arr6, 0, 8));

      assertArrayEquals (new String[] {"5"}, arr1);
      assertArrayEquals (new String[] {"2", "6"}, arr2);
      assertArrayEquals (new String[] {"-1", "8", "8"}, arr3);
      assertArrayEquals (new String[] {"1", "7", "4", "9"}, arr4);
      assertArrayEquals (new String[] {"9", "9", "9", "8", "9"}, arr5);
      assertArrayEquals (new String[] {"1", "2", "3", "2", "3", "9", "7", "4", "5"}, arr6);
   }

   @Test
   public void testQuickSort1()
   {
      assertArrayEquals (new int[] {}, alg.quickSort1(new int[] {}, 0, -1));
      assertArrayEquals (new int[] {5}, alg.quickSort1(new int[] {5}, 0, 0));
      assertArrayEquals (new int[] {2, 6}, alg.quickSort1(new int[] {6, 2}, 0, 1));
      assertArrayEquals (new int[] {-1, 8, 8}, alg.quickSort1(new int[] {-1, 8, 8}, 0, 2));
      assertArrayEquals (new int[] {1, 4, 7, 9}, alg.quickSort1(new int[] {9, 7, 4, 1}, 0, 3));
      assertArrayEquals (new int[] {1, 2, 2, 3, 3, 4, 5, 7, 11}, alg.quickSort1(new int[] {5, 1, 2, 4, 3, 11, 7, 2, 3}, 0, 8));
   }

   @Test
   public void testQuickSort2()
   {
      assertArrayEquals (new long[] {}, alg.quickSort2(new long[] {}, 0, -1));
      assertArrayEquals (new long[] {5}, alg.quickSort2(new long[] {5}, 0, 0));
      assertArrayEquals (new long[] {2, 6}, alg.quickSort2(new long[] {6, 2}, 0, 1));
      assertArrayEquals (new long[] {-1, 8, 8}, alg.quickSort2(new long[] {-1, 8, 8}, 0, 2));
      assertArrayEquals (new long[] {1, 4, 7, 9}, alg.quickSort2(new long[] {9, 7, 4, 1}, 0, 3));
      assertArrayEquals (new long[] {1, 2, 2, 3, 3, 4, 5, 7, 11}, alg.quickSort2(new long[] {5, 1, 2, 4, 3, 11, 7, 2, 3}, 0, 8));
   }

   @Test
   public void testQuickSort3()
   {
      assertTrue (compareDoubleArrays(new double[] {}, alg.quickSort3(new double[] {}, 0, -1)));
      assertTrue (compareDoubleArrays(new double[] {5}, alg.quickSort3(new double[] {5}, 0, 0)));
      assertTrue (compareDoubleArrays(new double[] {2, 6}, alg.quickSort3(new double[] {6, 2}, 0, 1)));
      assertTrue (compareDoubleArrays(new double[] {-1, 8, 8}, alg.quickSort3(new double[] {-1, 8, 8}, 0, 2)));
      assertTrue (compareDoubleArrays(new double[] {1, 4, 7, 9}, alg.quickSort3(new double[] {9, 7, 4, 1}, 0, 3)));
      assertTrue (compareDoubleArrays(new double[] {1, 2, 2, 3, 3, 4, 5, 7, 11}, alg.quickSort3(new double[] {5, 1, 2, 4, 3, 11, 7, 2, 3}, 0, 8)));
   }

   @Test
   public void testQuickSort4()
   {
      assertArrayEquals (new char[] {}, alg.quickSort4(new char[] {}, 0, -1));
      assertArrayEquals (new char[] {5}, alg.quickSort4(new char[] {5}, 0, 0));
      assertArrayEquals (new char[] {2, 6}, alg.quickSort4(new char[] {6, 2}, 0, 1));
      assertArrayEquals (new char[] {0, 8, 8}, alg.quickSort4(new char[] {0, 8, 8}, 0, 2));
      assertArrayEquals (new char[] {1, 4, 7, 9}, alg.quickSort4(new char[] {9, 7, 4, 1}, 0, 3));
      assertArrayEquals (new char[] {1, 2, 2, 3, 3, 4, 5, 7, 11}, alg.quickSort4(new char[] {5, 1, 2, 4, 3, 11, 7, 2, 3}, 0, 8));
   }

   @Test
   public void testQuickSort5()
   {
      assertArrayEquals (new String[] {}, alg.quickSort5(new String[] {}, 0, -1));
      assertArrayEquals (new String[] {"5"}, alg.quickSort5(new String[] {"5"}, 0, 0));
      assertArrayEquals (new String[] {"2", "6"}, alg.quickSort5(new String[] {"6", "2"}, 0, 1));
      assertArrayEquals (new String[] {"-1", "8", "8"}, alg.quickSort5(new String[] {"-1", "8", "8"}, 0, 2));
      assertArrayEquals (new String[] {"1", "4", "7", "9"}, alg.quickSort5(new String[] {"9", "7", "4", "1"}, 0, 3));
      assertArrayEquals (new String[] {"1", "2", "2", "3", "3", "4", "5", "7", "9"}, alg.quickSort5(new String[] {"5", "1", "2", "4", "3", "9", "7", "2", "3"}, 0, 8));
   }



   @Test
   public void testMinDistance1()
   {
      assertEquals (1, alg.minDistance1(
          new int[] {2147483647, 0},
          new boolean[] {false, false}, 2
      ));
      assertEquals (-1, alg.minDistance1(
          new int[] {2147483647, 2147483647},
          new boolean[] {true, true}, 2
      ));
      assertEquals (0, alg.minDistance1(
          new int[] {0, 2147483647, 2147483647, 2147483647, 2147483647, 2147483647, 2147483647, 2147483647, 2147483647},
          new boolean[] {false, false, false, false, false, false, false, false, false}, 9
      ));
      assertEquals (2, alg.minDistance1(
          new int[] {0, 4, 12, 19, 21, 11, 9, 8, 14},
          new boolean[] {true, true, false, false, false, true, true, true, false}, 9
      ));
      assertEquals (3, alg.minDistance1(
          new int[] {0, 4, 12, 19, 21, 11, 9, 8, 14},
          new boolean[] {true, true, true, false, false, true, true, true, true}, 9
      ));
      assertEquals (8, alg.minDistance1(
          new int[] {0, 4, 12, 19, 21, 11, 9, 8, 11},
          new boolean[] {true, true, true, false, false, true, true, true, false}, 9
      ));
   }

   @Test
   public void testMinDistance2()
   {
      assertEquals (1, alg.minDistance2(
          new long[] {2147483647, 0},
          new boolean[] {false, false}, 2
      ));
      assertEquals (-1, alg.minDistance2(
          new long[] {2147483647, 2147483647},
          new boolean[] {true, true}, 2
      ));
      assertEquals (0, alg.minDistance2(
          new long[] {0, 2147483647, 2147483647, 2147483647, 2147483647, 2147483647, 2147483647, 2147483647, 2147483647},
          new boolean[] {false, false, false, false, false, false, false, false, false}, 9
      ));
      assertEquals (2, alg.minDistance2(
          new long[] {0, 4, 12, 19, 21, 11, 9, 8, 14},
          new boolean[] {true, true, false, false, false, true, true, true, false}, 9
      ));
      assertEquals (3, alg.minDistance2(
          new long[] {0, 4, 12, 19, 21, 11, 9, 8, 14},
          new boolean[] {true, true, true, false, false, true, true, true, true}, 9
      ));
      assertEquals (8, alg.minDistance2(
          new long[] {0, 4, 12, 19, 21, 11, 9, 8, 11},
          new boolean[] {true, true, true, false, false, true, true, true, false}, 9
      ));
   }

   @Test
   public void testMinDistance3()
   {
      assertEquals (1, alg.minDistance3(
          new double[] {2147483647, 0},
          new boolean[] {false, false}, 2
      ));
      assertEquals (-1, alg.minDistance3(
          new double[] {2147483647, 2147483647},
          new boolean[] {true, true}, 2
      ));
      assertEquals (0, alg.minDistance3(
          new double[] {0, 2147483647, 2147483647, 2147483647, 2147483647, 2147483647, 2147483647, 2147483647, 2147483647},
          new boolean[] {false, false, false, false, false, false, false, false, false}, 9
      ));
      assertEquals (2, alg.minDistance3(
          new double[] {0, 4, 12, 19, 21, 11, 9, 8, 14},
          new boolean[] {true, true, false, false, false, true, true, true, false}, 9
      ));
      assertEquals (3, alg.minDistance3(
          new double[] {0, 4, 12, 19, 21, 11, 9, 8, 14},
          new boolean[] {true, true, true, false, false, true, true, true, true}, 9
      ));
      assertEquals (8, alg.minDistance3(
          new double[] {0, 4, 12, 19, 21, 11, 9, 8, 11},
          new boolean[] {true, true, true, false, false, true, true, true, false}, 9
      ));
   }

   @Test
   public void testDijkstra1()
   {
      int[][] graph1 = new int[][] {
          {0, 8},
          {8, 0},
      };
      int[][] graph2 = new int[][] {
          { 0, 4, 0, 0, 0, 0, 0, 8, 0 },
          { 4, 0, 8, 0, 0, 0, 0, 11, 0 },
          { 0, 8, 0, 7, 0, 4, 0, 0, 2 },
          { 0, 0, 7, 0, 9, 14, 0, 0, 0 },
          { 0, 0, 0, 9, 0, 10, 0, 0, 0 },
          { 0, 0, 4, 14, 10, 0, 2, 0, 0 },
          { 0, 0, 0, 0, 0, 2, 0, 1, 6 },
          { 8, 11, 0, 0, 0, 0, 1, 0, 7 },
          { 0, 0, 2, 0, 0, 0, 6, 7, 0 }
      };
      assertArrayEquals (new int[] {0, 8}, alg.dijkstra1(graph1, 0));
      assertArrayEquals (new int[] {8, 0}, alg.dijkstra1(graph1, 1));
      assertArrayEquals (new int[] {0, 4, 12, 19, 21, 11, 9, 8, 14}, alg.dijkstra1(graph2, 0));
      assertArrayEquals (new int[] {4, 0, 8, 15, 22, 12, 12, 11, 10}, alg.dijkstra1(graph2, 1));
      assertArrayEquals (new int[] {14, 10, 2, 9, 16, 6, 6, 7, 0}, alg.dijkstra1(graph2, 8));
   }

   @Test
   public void testDijkstra2()
   {
      long[][] graph1 = new long[][] {
          {0, 8},
          {8, 0},
      };
      long[][] graph2 = new long[][] {
          { 0, 4, 0, 0, 0, 0, 0, 8, 0 },
          { 4, 0, 8, 0, 0, 0, 0, 11, 0 },
          { 0, 8, 0, 7, 0, 4, 0, 0, 2 },
          { 0, 0, 7, 0, 9, 14, 0, 0, 0 },
          { 0, 0, 0, 9, 0, 10, 0, 0, 0 },
          { 0, 0, 4, 14, 10, 0, 2, 0, 0 },
          { 0, 0, 0, 0, 0, 2, 0, 1, 6 },
          { 8, 11, 0, 0, 0, 0, 1, 0, 7 },
          { 0, 0, 2, 0, 0, 0, 6, 7, 0 }
      };
      assertArrayEquals (new long[] {0, 8}, alg.dijkstra2(graph1, 0));
      assertArrayEquals (new long[] {8, 0}, alg.dijkstra2(graph1, 1));
      assertArrayEquals (new long[] {0, 4, 12, 19, 21, 11, 9, 8, 14}, alg.dijkstra2(graph2, 0));
      assertArrayEquals (new long[] {4, 0, 8, 15, 22, 12, 12, 11, 10}, alg.dijkstra2(graph2, 1));
      assertArrayEquals (new long[] {14, 10, 2, 9, 16, 6, 6, 7, 0}, alg.dijkstra2(graph2, 8));
   }

   @Test
   public void testDijkstra3()
   {
      double[][] graph1 = new double[][] {
          {0, 8},
          {8, 0},
      };
      double[][] graph2 = new double[][] {
          { 0, 4, 0, 0, 0, 0, 0, 8, 0 },
          { 4, 0, 8, 0, 0, 0, 0, 11, 0 },
          { 0, 8, 0, 7, 0, 4, 0, 0, 2 },
          { 0, 0, 7, 0, 9, 14, 0, 0, 0 },
          { 0, 0, 0, 9, 0, 10, 0, 0, 0 },
          { 0, 0, 4, 14, 10, 0, 2, 0, 0 },
          { 0, 0, 0, 0, 0, 2, 0, 1, 6 },
          { 8, 11, 0, 0, 0, 0, 1, 0, 7 },
          { 0, 0, 2, 0, 0, 0, 6, 7, 0 }
      };
      compareDoubleArrays(new double[] {0, 8}, alg.dijkstra3(graph1, 0));
      compareDoubleArrays(new double[] {8, 0}, alg.dijkstra3(graph1, 1));
      compareDoubleArrays(new double[] {0, 4, 12, 19, 21, 11, 9, 8, 14}, alg.dijkstra3(graph2, 0));
      compareDoubleArrays(new double[] {4, 0, 8, 15, 22, 12, 12, 11, 10}, alg.dijkstra3(graph2, 1));
      compareDoubleArrays(new double[] {14, 10, 2, 9, 16, 6, 6, 7, 0}, alg.dijkstra3(graph2, 8));
   }

   @Test
   public void testDijkstra4()
   {
      boolean[][] graph1 = new boolean[][] {
          {false, true},
          {true, false},
      };
      boolean[][] graph2 = new boolean[][] {
          {false, true, false, false, false, false, false, true, false},
          {true, false, true, false, false, false, false, true, false},
          {false, true, false, true, false, true, false, false, true},
          {false, false, true, false, true, true, false, false, false},
          {false, false, false, true, false, true, false, false, false},
          {false, false, true, true, true, false, true, false, false},
          {false, false, false, false, false, true, false, true, true},
          {true, true, false, false, false, false, true, false, true},
          {false, false, true, false, false, false, true, true, false},
      };
      assertArrayEquals (new int[] {0, 1}, alg.dijkstra4(graph1, 0));
      assertArrayEquals (new int[] {1, 0}, alg.dijkstra4(graph1, 1));
      assertArrayEquals (new int[] {0, 1, 2, 3, 4, 3, 2, 1, 2}, alg.dijkstra4(graph2, 0));
      assertArrayEquals (new int[] {1, 0, 1, 2, 3, 2, 2, 1, 2}, alg.dijkstra4(graph2, 1));
      assertArrayEquals (new int[] {2, 2, 1, 2, 3, 2, 1, 1, 0}, alg.dijkstra4(graph2, 8));
   }

   @Test
   public void testBfs1()
   {
      int[][] graph1 = new int[][] {
          {0, 8},
          {6, 0},
      };
      int[][] graph2 = new int[][] {
          {0, 8},
          {6, 0},
      };
      int[][] graph3 = new int[][] {
          {0, 16, 0, 0, 0, 0},
          {0, 0, 10, 12, 0, 0},
          {13, 4, 0, 0, 1, 0},
          {0, 0, 9, 0, 0, 20},
          {0, 0, 13, 7, 0, 4},
          {0, 0, 0, 0, 0, 0}
      };
      int[][] graph4 = new int[][] {
          {0, 4, 0, 0, 0, 0},
          {12, 0, 10, 0, 0, 0},
          {13, 4, 0, 0, 1, 0},
          {0, 12, 9, 0, 0, 8},
          {0, 0, 13, 7, 0, 4},
          {0, 0, 0, 12, 0, 0},
      };
      int[][] graph5 = new int[][] {
          {0, 16, 0, 0, 0, 0},
          {0, 0, 10, 12, 0, 0},
          {13, 4, 0, 0, 1, 0},
          {0, 0, 9, 0, 0, 20},
          {0, 0, 13, 7, 0, 4},
          {0, 0, 0, 0, 0, 0}
      };
      int[][] graph6 = new int[][] {
          {0, 16, 0, 0, 0, 0, 0, 0 },
          {0, 0, 10, 12, 0, 0, 0, 6 },
          {13, 4, 0, 0, 1, 0, 0, 0 },
          {0, 0, 9, 0, 0, 0, 0, 1 },
          {0, 0, 13, 7, 0, 4, 2, 0 },
          {0, 0, 0, 0, 0, 0, 0, 0 },
          {0, 0, 0, 0, 0, 0, 0, 3 },
          {0, 0, 0, 0, 0, 0, 4, 0 },
      };
      int[] parent1 = {0, 0};
      int[] parent2 = {0, 0};
      int[] parent3 = {-1, 0, 0, 1, 2, 0};
      int[] parent4 = {-1, 0, 0, 1, 2, 3};
      int[] parent5 = {0, 0, 0, 0, 0, 0};
      int[] parent6 = {0, 0, 0, 0, 0, 0, 0, 0};
      assertEquals (true, alg.bfs1(graph1, 0, 1, parent1));
      assertEquals (true, alg.bfs1(graph2, 1, 0, parent2));
      assertEquals (true, alg.bfs1(graph3, 0, 4, parent3));
      assertEquals (true, alg.bfs1(graph4, 0, 5, parent4));
      assertEquals (false, alg.bfs1(graph5, 5, 0, parent5));
      assertEquals (true, alg.bfs1(graph6, 0, 5, parent6));
      assertArrayEquals (new int[] {-1, 0}, parent1);
      assertArrayEquals (new int[] {1, -1}, parent2);
      assertArrayEquals (new int[] {-1, 0, 1, 1, 2, 0}, parent3);
      assertArrayEquals (new int[] {-1, 0, 1, 4, 2, 4}, parent4);
      assertArrayEquals (new int[] {0, 0, 0, 0, 0, -1}, parent5);
      assertArrayEquals (new int[] {-1, 0, 1, 1, 2, 4, 7, 1}, parent6);
   }

   @Test
   public void testBfs2()
   {
      long[][] graph1 = new long[][] {
          {0, 8},
          {6, 0},
      };
      long[][] graph2 = new long[][] {
          {0, 8},
          {6, 0},
      };
      long[][] graph3 = new long[][] {
          {0, 16, 0, 0, 0, 0},
          {0, 0, 10, 12, 0, 0},
          {13, 4, 0, 0, 1, 0},
          {0, 0, 9, 0, 0, 20},
          {0, 0, 13, 7, 0, 4},
          {0, 0, 0, 0, 0, 0}
      };
      long[][] graph4 = new long[][] {
          {0, 4, 0, 0, 0, 0},
          {12, 0, 10, 0, 0, 0},
          {13, 4, 0, 0, 1, 0},
          {0, 12, 9, 0, 0, 8},
          {0, 0, 13, 7, 0, 4},
          {0, 0, 0, 12, 0, 0},
      };
      long[][] graph5 = new long[][] {
          {0, 16, 0, 0, 0, 0},
          {0, 0, 10, 12, 0, 0},
          {13, 4, 0, 0, 1, 0},
          {0, 0, 9, 0, 0, 20},
          {0, 0, 13, 7, 0, 4},
          {0, 0, 0, 0, 0, 0}
      };
      long[][] graph6 = new long[][] {
          {0, 16, 0, 0, 0, 0, 0, 0 },
          {0, 0, 10, 12, 0, 0, 0, 6 },
          {13, 4, 0, 0, 1, 0, 0, 0 },
          {0, 0, 9, 0, 0, 0, 0, 1 },
          {0, 0, 13, 7, 0, 4, 2, 0 },
          {0, 0, 0, 0, 0, 0, 0, 0 },
          {0, 0, 0, 0, 0, 0, 0, 3 },
          {0, 0, 0, 0, 0, 0, 4, 0 },
      };
      int[] parent1 = {0, 0};
      int[] parent2 = {0, 0};
      int[] parent3 = {-1, 0, 0, 1, 2, 0};
      int[] parent4 = {-1, 0, 0, 1, 2, 3};
      int[] parent5 = {0, 0, 0, 0, 0, 0};
      int[] parent6 = {0, 0, 0, 0, 0, 0, 0, 0};
      assertEquals (true, alg.bfs2(graph1, 0, 1, parent1));
      assertEquals (true, alg.bfs2(graph2, 1, 0, parent2));
      assertEquals (true, alg.bfs2(graph3, 0, 4, parent3));
      assertEquals (true, alg.bfs2(graph4, 0, 5, parent4));
      assertEquals (false, alg.bfs2(graph5, 5, 0, parent5));
      assertEquals (true, alg.bfs2(graph6, 0, 5, parent6));
      assertArrayEquals (new int[] {-1, 0}, parent1);
      assertArrayEquals (new int[] {1, -1}, parent2);
      assertArrayEquals (new int[] {-1, 0, 1, 1, 2, 0}, parent3);
      assertArrayEquals (new int[] {-1, 0, 1, 4, 2, 4}, parent4);
      assertArrayEquals (new int[] {0, 0, 0, 0, 0, -1}, parent5);
      assertArrayEquals (new int[] {-1, 0, 1, 1, 2, 4, 7, 1}, parent6);
   }

   @Test
   public void testBfs3()
   {
      double[][] graph1 = new double[][] {
          {0, 8},
          {6, 0},
      };
      double[][] graph2 = new double[][] {
          {0, 8},
          {6, 0},
      };
      double[][] graph3 = new double[][] {
          {0, 16, 0, 0, 0, 0},
          {0, 0, 10, 12, 0, 0},
          {13, 4, 0, 0, 1, 0},
          {0, 0, 9, 0, 0, 20},
          {0, 0, 13, 7, 0, 4},
          {0, 0, 0, 0, 0, 0}
      };
      double[][] graph4 = new double[][] {
          {0, 4, 0, 0, 0, 0},
          {12, 0, 10, 0, 0, 0},
          {13, 4, 0, 0, 1, 0},
          {0, 12, 9, 0, 0, 8},
          {0, 0, 13, 7, 0, 4},
          {0, 0, 0, 12, 0, 0},
      };
      double[][] graph5 = new double[][] {
          {0, 16, 0, 0, 0, 0},
          {0, 0, 10, 12, 0, 0},
          {13, 4, 0, 0, 1, 0},
          {0, 0, 9, 0, 0, 20},
          {0, 0, 13, 7, 0, 4},
          {0, 0, 0, 0, 0, 0}
      };
      double[][] graph6 = new double[][] {
          {0, 16, 0, 0, 0, 0, 0, 0 },
          {0, 0, 10, 12, 0, 0, 0, 6 },
          {13, 4, 0, 0, 1, 0, 0, 0 },
          {0, 0, 9, 0, 0, 0, 0, 1 },
          {0, 0, 13, 7, 0, 4, 2, 0 },
          {0, 0, 0, 0, 0, 0, 0, 0 },
          {0, 0, 0, 0, 0, 0, 0, 3 },
          {0, 0, 0, 0, 0, 0, 4, 0 },
      };
      int[] parent1 = {0, 0};
      int[] parent2 = {0, 0};
      int[] parent3 = {-1, 0, 0, 1, 2, 0};
      int[] parent4 = {-1, 0, 0, 1, 2, 3};
      int[] parent5 = {0, 0, 0, 0, 0, 0};
      int[] parent6 = {0, 0, 0, 0, 0, 0, 0, 0};
      assertEquals (true, alg.bfs3(graph1, 0, 1, parent1));
      assertEquals (true, alg.bfs3(graph2, 1, 0, parent2));
      assertEquals (true, alg.bfs3(graph3, 0, 4, parent3));
      assertEquals (true, alg.bfs3(graph4, 0, 5, parent4));
      assertEquals (false, alg.bfs3(graph5, 5, 0, parent5));
      assertEquals (true, alg.bfs3(graph6, 0, 5, parent6));
      assertArrayEquals (new int[] {-1, 0}, parent1);
      assertArrayEquals (new int[] {1, -1}, parent2);
      assertArrayEquals (new int[] {-1, 0, 1, 1, 2, 0}, parent3);
      assertArrayEquals (new int[] {-1, 0, 1, 4, 2, 4}, parent4);
      assertArrayEquals (new int[] {0, 0, 0, 0, 0, -1}, parent5);
      assertArrayEquals (new int[] {-1, 0, 1, 1, 2, 4, 7, 1}, parent6);
   }

   @Test
   public void testBfs4()
   {
      boolean[][] graph1 = new boolean[][] {
          {false, true},
          {true, false},
      };
      boolean[][] graph2 = new boolean[][] {
          {false, true},
          {true, false},
      };
      boolean[][] graph3 = new boolean[][] {
          {false, true, false, false, false, false},
          {false, false, true, true, false, false},
          {true, true, false, false, true, false},
          {false, false, true, false, false, true},
          {false, false, true, true, false, true},
          {false, false, false, false, false, false},
      };
      boolean[][] graph4 = new boolean[][] {
          {false, true, false, false, false, false},
          {true, false, true, false, false, false},
          {true, true, false, false, true, false},
          {false, true, true, false, false, true},
          {false, false, true, true, false, true},
          {false, false, false, true, false, false},
      };
      boolean[][] graph5 = new boolean[][] {
          {false, true, false, false, false, false},
          {false, false, true, true, false, false},
          {true, true, false, false, true, false},
          {false, false, true, false, false, true},
          {false, false, true, true, false, true},
          {false, false, false, false, false, false},
      };
      boolean[][] graph6 = new boolean[][] {
          {false, true, false, false, false, false, false, false},
          {false, false, true, true, false, false, false, true},
          {true, true, false, false, true, false, false, false},
          {false, false, true, false, false, false, false, true},
          {false, false, true, true, false, true, true, false},
          {false, false, false, false, false, false, false, false},
          {false, false, false, false, false, false, false, true},
          {false, false, false, false, false, false, true, false},
      };
      int[] parent1 = {0, 0};
      int[] parent2 = {0, 0};
      int[] parent3 = {-1, 0, 0, 1, 2, 0};
      int[] parent4 = {-1, 0, 0, 1, 2, 3};
      int[] parent5 = {0, 0, 0, 0, 0, 0};
      int[] parent6 = {0, 0, 0, 0, 0, 0, 0, 0};
      assertEquals (true, alg.bfs4(graph1, 0, 1, parent1));
      assertEquals (true, alg.bfs4(graph2, 1, 0, parent2));
      assertEquals (true, alg.bfs4(graph3, 0, 4, parent3));
      assertEquals (true, alg.bfs4(graph4, 0, 5, parent4));
      assertEquals (false, alg.bfs4(graph5, 5, 0, parent5));
      assertEquals (true, alg.bfs4(graph6, 0, 5, parent6));
      assertArrayEquals (new int[] {-1, 0}, parent1);
      assertArrayEquals (new int[] {1, -1}, parent2);
      assertArrayEquals (new int[] {-1, 0, 1, 1, 2, 0}, parent3);
      assertArrayEquals (new int[] {-1, 0, 1, 4, 2, 4}, parent4);
      assertArrayEquals (new int[] {0, 0, 0, 0, 0, -1}, parent5);
      assertArrayEquals (new int[] {-1, 0, 1, 1, 2, 4, 7, 1}, parent6);
   }

   @Test
   public void testFordFulkerson1()
   {
      int[][] graph1 = new int[][] {
          {0, 8},
          {6, 0},
      };
      int[][] graph2 = new int[][] {
          { 0, 16, 13, 0, 0, 0 },
          { 0, 0, 10, 12, 0, 0 },
          { 0, 4, 0, 0, 14, 0 },
          { 0, 0, 9, 0, 0, 20 },
          { 0, 0, 0, 7, 0, 4 },
          { 0, 0, 0, 0, 0, 0 }
      };

      assertEquals (8, alg.fordFulkerson1(graph1, 0, 1));
      assertEquals (6, alg.fordFulkerson1(graph1, 1, 0));
      assertEquals (14, alg.fordFulkerson1(graph2, 0, 4));
      assertEquals (23, alg.fordFulkerson1(graph2, 0, 5));
      assertEquals (0, alg.fordFulkerson1(graph2, 5, 0));
   }

   @Test
   public void testFordFulkerson2()
   {
      long[][] graph1 = new long[][] {
          {0, 8},
          {6, 0},
      };
      long[][] graph2 = new long[][] {
          { 0, 16, 13, 0, 0, 0 },
          { 0, 0, 10, 12, 0, 0 },
          { 0, 4, 0, 0, 14, 0 },
          { 0, 0, 9, 0, 0, 20 },
          { 0, 0, 0, 7, 0, 4 },
          { 0, 0, 0, 0, 0, 0 }
      };

      assertEquals (8, alg.fordFulkerson2(graph1, 0, 1));
      assertEquals (6, alg.fordFulkerson2(graph1, 1, 0));
      assertEquals (14, alg.fordFulkerson2(graph2, 0, 4));
      assertEquals (23, alg.fordFulkerson2(graph2, 0, 5));
      assertEquals (0, alg.fordFulkerson2(graph2, 5, 0));
   }

   @Test
   public void testFordFulkerson3()
   {
      double[][] graph1 = new double[][] {
          {0, 8},
          {6, 0},
      };
      double[][] graph2 = new double[][] {
          { 0, 16, 13, 0, 0, 0 },
          { 0, 0, 10, 12, 0, 0 },
          { 0, 4, 0, 0, 14, 0 },
          { 0, 0, 9, 0, 0, 20 },
          { 0, 0, 0, 7, 0, 4 },
          { 0, 0, 0, 0, 0, 0 }
      };

      assertEquals (8, alg.fordFulkerson3(graph1, 0, 1), 1e-09);
      assertEquals (6, alg.fordFulkerson3(graph1, 1, 0), 1e-09);
      assertEquals (14, alg.fordFulkerson3(graph2, 0, 4), 1e-09);
      assertEquals (23, alg.fordFulkerson3(graph2, 0, 5), 1e-09);
      assertEquals (0, alg.fordFulkerson3(graph2, 5, 0), 1e-09);
   }

   @Test
   public void testFordFulkerson4()
   {
      boolean[][] graph1 = new boolean[][] {
          {false, true},
          {true, false},
      };
      boolean[][] graph2 = new boolean[][] {
          {false, true, true, false, false, false},
          {false, false, true, true, false, false},
          {false, true, false, false, true, false},
          {false, false, true, false, false, true},
          {false, false, false, true, false, true},
          {false, false, false, false, false, false},
      };

      assertEquals (1, alg.fordFulkerson4(graph1, 0, 1));
      assertEquals (1, alg.fordFulkerson4(graph1, 1, 0));
      assertEquals (1, alg.fordFulkerson4(graph2, 0, 4));
      assertEquals (2, alg.fordFulkerson4(graph2, 0, 5));
      assertEquals (0, alg.fordFulkerson4(graph2, 5, 0));
   }
}