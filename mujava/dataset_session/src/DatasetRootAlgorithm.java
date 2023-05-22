/* Collection of classic java programs.
Dataset consists of 77 original programs.
All methods are mutated and presented in the DatasetRootAlgorithm/traditional_mutants folder */

public class DatasetRootAlgorithm {

    // linearSearch method
    // input: unsorted array arr, searched value value
    // output: position of first element occurence

    public int linearSearch1(int[] arr, int value) {
        for(int i = 0; i < arr.length; i++) {
            if(arr[i] == value) {
                return i;
            }
        }
        return -1;
    }

    public int linearSearch2(long[] arr, long value) {
        for(int i = 0; i < arr.length; i++) {
            if(arr[i] == value) {
                return i;
            }
        }
        return -1;
    }

    public int linearSearch3(double[] arr, double value) {
        for(int i = 0; i < arr.length; i++) {
            if(Math.abs(arr[i] - value) < 1e-09) {
                return i;
            }
        }
        return -1;
    }

    public int linearSearch4(boolean[] arr, boolean value) {
        for(int i = 0; i < arr.length; i++) {
            if(arr[i] == value) {
                return i;
            }
        }
        return -1;
    }

    public int linearSearch5(char[] arr, char value) {
        for(int i = 0; i < arr.length; i++) {
            if(arr[i] == value) {
                return i;
            }
        }
        return -1;
    }

    public int linearSearch6(String[] arr, String value) {
        for(int i = 0; i < arr.length; i++) {
            if(arr[i].equals(value)) {
                return i;
            }
        }
        return -1;
    }



    // binarySearch method
    // input: sorted array arr, start of array index (0), end of array index (arr.length), searched value x
    // output: position of first element occurrence

    public int binarySearch1(int arr[], int l, int r, int value)
    {
        if (r>=l)
        {
            int mid = l + (r - l)/2;
            if (arr[mid] == value && (mid == 0 || arr[mid-1] < value))
               return mid;
            if (arr[mid] >= value)
               return binarySearch1(arr, l, mid-1, value);
            return binarySearch1(arr, mid+1, r, value);
        }
        return -1;
    }

    public int binarySearch2(long arr[], int l, int r, long value)
    {
        if (r>=l)
        {
            int mid = l + (r - l)/2;
            if (arr[mid] == value && (mid == 0 || arr[mid-1] < value))
                return mid;
            if (arr[mid] >= value)
                return binarySearch2(arr, l, mid-1, value);
            return binarySearch2(arr, mid+1, r, value);
        }
        return -1;
    }

    public int binarySearch3(double arr[], int l, int r, double value)
    {
        if (r>=l)
        {
            int mid = l + (r - l)/2;
            if (Math.abs(arr[mid] - value) < 1e-09 && (mid == 0 || arr[mid-1] < value))
                return mid;
            if (arr[mid] >= value)
                return binarySearch3(arr, l, mid-1, value);
            return binarySearch3(arr, mid+1, r, value);
        }
        return -1;
    }

    public int binarySearch4(boolean arr[], int l, int r, boolean value)
    {
        if (r>=l)
        {
            int mid = l + (r - l)/2;
            if (arr[mid] == value && (mid == 0 || arr[mid-1] != value))
                return mid;
            if (arr[mid] && !value)
                return binarySearch4(arr, l, mid-1, value);
            return binarySearch4(arr, mid+1, r, value);
        }
        return -1;
    }

    public int binarySearch5(char arr[], int l, int r, char value)
    {
        if (r>=l)
        {
            int mid = l + (r - l)/2;
            if (arr[mid] == value && (mid == 0 || arr[mid-1] < value))
                return mid;
            if (arr[mid] >= value)
                return binarySearch5(arr, l, mid-1, value);
            return binarySearch5(arr, mid+1, r, value);
        }
        return -1;
    }

    public int binarySearch6(String arr[], int l, int r, String value)
    {
        if (r>=l)
        {
            int mid = l + (r - l)/2;
            int ord = arr[mid].compareTo(value);
            if (ord == 0 && (mid == 0 || arr[mid-1].compareTo(value) < 0))
                return mid;
            if (ord >= 0)
                return binarySearch6(arr, l, mid-1, value);
            return binarySearch6(arr, mid+1, r, value);
        }
        return -1;
    }



    // max method
    // input: non-empty array arr
    // output: biggest element in the array

    public int max1(int[] arr) {
        int maxN = arr[0];
        for(int i = 1; i < arr.length; i++) {
            if(arr[i] > maxN) {
                maxN = arr[i];
            }
        }
        return maxN;
    }

    public long max2(long[] arr) {
        long maxN = arr[0];
        for(int i = 1; i < arr.length; i++) {
            if(arr[i] > maxN) {
                maxN = arr[i];
            }
        }
        return maxN;
    }

    public double max3(double[] arr) {
        double maxN = arr[0];
        for(int i = 1; i < arr.length; i++) {
            if(arr[i] > maxN) {
                maxN = arr[i];
            }
        }
        return maxN;
    }



    // isEven method
    // input: natural number
    // output: true if number is even, else false


    public boolean isEven1(int x) {
        return x % 2 == 0;
    }

    public boolean isEven2(long x) {
        return x % 2 == 0;
    }



    // power method
    // input: number and degree
    // output: number by power of degree

    public long power1(long x, int y) {
        if(x == 1) {
            return 1;
        }
        if(y == 1) {
            return x;
        }
        long P = 1;
        int i = 1;
        while(i <= y) {
            P = P * x;
            i++;
        }
        return P;
    }

    public double power2(double x, int y) {
        if(x == 1) {
            return 1;
        }
        if(y == 1) {
            return x;
        }
        double P = 1;
        int i = 1;
        while(i <= y) {
            P = P * x;
            i++;
        }
        return P;
    }



    // prime method
    // input: natural number
    // output: true if number is prime, else false

    public boolean prime1(int n) {
        if(n < 2) {
            return false;
        }
        for(int i = 2; i <= Math.sqrt(n); i++) {
            if(n % i == 0) {
                return false;
            }
        }
        return true;
    }

    public boolean prime2(long n) {
        if(n < 2) {
            return false;
        }
        for(int i = 2; i <= Math.sqrt(n); i++) {
            if(n % i == 0) {
                return false;
            }
        }
        return true;
    }



    // solveLinearEq method
    // input: augmented matrix of size [n][n+1], that represents non-empty linearly independent linear equation
    // output: linear equation solution

    public double[] solveLinearEq(double[][] matrix) {
        int n = matrix.length;
        double[] solution = new double[n];

        for (int i = 0; i < n; i++) {
            double pivot = matrix[i][i];
            int pivotRow = i;
            for (int j = i + 1; j < n; j++) {
                if (Math.abs(matrix[j][i]) > Math.abs(pivot)) {
                    pivot = matrix[j][i];
                    pivotRow = j;
                }
            }

            if (pivotRow != i) {
                double[] tempRow = matrix[i];
                matrix[i] = matrix[pivotRow];
                matrix[pivotRow] = tempRow;
            }

            for (int j = i + 1; j < n; j++) {
                double factor = matrix[j][i] / matrix[i][i];
                for (int k = i + 1; k < n + 1; k++) {
                    matrix[j][k] -= factor * matrix[i][k];
                }
                matrix[j][i] = 0;
            }
        }

        for (int i = n-1; i >= 0; i--) {
            double sum = 0;
            for (int j = i+1; j < n; j++) {
                sum += matrix[i][j] * solution[j];
            }
            solution[i] = (matrix[i][n] - sum) / matrix[i][i];
        }

        return solution;
    }



    // charCount method
    // input: string line, checked character
    // output: number how many times character occurs in the string value

    public int charCount(String line, char c) {
        int count = 0;
        for(int i = 0; i < line.length(); i++) {
            if(c == line.charAt(i)) {
                count++;
            }
        }
        return count;
    }



    // charCount method
    // input: string line, checked character
    // output: number how many times character occurs in the string value

    public int wordCount(String line, String word) {
        int count = 0;
        String temp[] = line.split(" ");
        for(int i = 0; i < temp.length; i++) {
            if(word.equals(temp[i])) {
                count++;
            }
        }
        return count;
    }



    // allWordCount method
    // input: string line
    // output: number of words in a string

    public int allWordCount(String line) {
        String trimmedLine = line.trim();
        return trimmedLine.isEmpty() ? 0 : trimmedLine.split("\\s+").length;
    }



    // charCount method
    // input: string line
    // output: true if string value is palindrome, else false

    public boolean palindrome(String line) {
        int i = 0;
        while(i < line.length()/2) {
            if(line.charAt(i) != line.charAt(line.length() - 1 - i)) {
                return false;
            }
            i++;
        }
        return true;
    }



    // countNumberOfWords method
    // input: string line
    // output: first non repeating character in string line

    public char firstNonRepeatingChar(String line) {
        int[] flags = new int[256];

        for (int i = 0; i < line.length(); i++) {
            flags[(int)line.charAt(i)]++ ;
        }

        for (int i = 0; i < line.length(); i++) {
            if(flags[(int)line.charAt(i)] == 1)
                return line.charAt(i);
        }
        return 0;
    }



    // bubbleSort method
    // input: array of elements
    // output: array sorted in ascending order

    public int[] bubbleSort1(int[] arr) {
        for(int i = 0; i < arr.length; i++) {
            for(int j = i+1; j < arr.length; j++) {
                if(arr[i] > arr[j]) {
                    int temp = arr[i];
                    arr[i] = arr[j];
                    arr[j] = temp;
                }
            }
        }
        return arr;
    }

    public long[] bubbleSort2(long[] arr) {
        for(int i = 0; i < arr.length; i++) {
            for(int j = i+1; j < arr.length; j++) {
                if(arr[i] > arr[j]) {
                    long temp = arr[i];
                    arr[i] = arr[j];
                    arr[j] = temp;
                }
            }
        }
        return arr;
    }

    public double[] bubbleSort3(double[] arr) {
        for(int i = 0; i < arr.length; i++) {
            for(int j = i+1; j < arr.length; j++) {
                if(arr[i] > arr[j]) {
                    double temp = arr[i];
                    arr[i] = arr[j];
                    arr[j] = temp;
                }
            }
        }
        return arr;
    }

    public char[] bubbleSort4(char[] arr) {
        for(int i = 0; i < arr.length; i++) {
            for(int j = i+1; j < arr.length; j++) {
                if(arr[i] > arr[j]) {
                    char temp = arr[i];
                    arr[i] = arr[j];
                    arr[j] = temp;
                }
            }
        }
        return arr;
    }

    public String[] bubbleSort5(String[] arr) {
        for(int i = 0; i < arr.length; i++) {
            for(int j = i+1; j < arr.length; j++) {
                if(arr[i].compareTo(arr[j]) > 0) {
                    String temp = arr[i];
                    arr[i] = arr[j];
                    arr[j] = temp;
                }
            }
        }
        return arr;
    }



    // insertionSort method
    // input: array of elements
    // output: array sorted in ascending order

    public int[] insertionSort1(int arr[])
    {
        int n = arr.length;
        for (int i = 1; i < n; ++i) {
            int key = arr[i];
            int j = i - 1;
            while (j >= 0 && arr[j] > key) {
                arr[j + 1] = arr[j];
                j = j - 1;
            }
            arr[j + 1] = key;
        }
        return arr;
    }

    public long[] insertionSort2(long arr[])
    {
        int n = arr.length;
        for (int i = 1; i < n; ++i) {
            long key = arr[i];
            int j = i - 1;
            while (j >= 0 && arr[j] > key) {
                arr[j + 1] = arr[j];
                j = j - 1;
            }
            arr[j + 1] = key;
        }
        return arr;
    }

    public double[] insertionSort3(double arr[])
    {
        int n = arr.length;
        for (int i = 1; i < n; ++i) {
            double key = arr[i];
            int j = i - 1;
            while (j >= 0 && arr[j] > key) {
                arr[j + 1] = arr[j];
                j = j - 1;
            }
            arr[j + 1] = key;
        }
        return arr;
    }

    public char[] insertionSort4(char arr[])
    {
        int n = arr.length;
        for (int i = 1; i < n; ++i) {
            char key = arr[i];
            int j = i - 1;
            while (j >= 0 && arr[j] > key) {
                arr[j + 1] = arr[j];
                j = j - 1;
            }
            arr[j + 1] = key;
        }
        return arr;
    }

    public String[] insertionSort5(String arr[])
    {
        int n = arr.length;
        for (int i = 1; i < n; ++i) {
            String key = arr[i];
            int j = i - 1;
            while (j >= 0 && arr[j].compareTo(key) > 0) {
                arr[j + 1] = arr[j];
                j = j - 1;
            }
            arr[j + 1] = key;
        }
        return arr;
    }



    // selectionSort method
    // input: array of elements
    // output: array sorted in ascending order

    public int[] selectionSort1(int arr[])
    {
        int n = arr.length;
        for (int i = 0; i < n-1; i++)
        {
            int min_idx = i;
            for (int j = i+1; j < n; j++){
                if (arr[j] < arr[min_idx])
                    min_idx = j;
            }
            int temp = arr[min_idx];
            arr[min_idx] = arr[i];
            arr[i] = temp;
        }
        return arr;
    }

    public long[] selectionSort2(long arr[])
    {
        int n = arr.length;
        for (int i = 0; i < n-1; i++)
        {
            int min_idx = i;
            for (int j = i+1; j < n; j++){
                if (arr[j] < arr[min_idx])
                    min_idx = j;
            }
            long temp = arr[min_idx];
            arr[min_idx] = arr[i];
            arr[i] = temp;
        }
        return arr;
    }

    public double[] selectionSort3(double arr[])
    {
        int n = arr.length;
        for (int i = 0; i < n-1; i++)
        {
            int min_idx = i;
            for (int j = i+1; j < n; j++){
                if (arr[j] < arr[min_idx])
                    min_idx = j;
            }
            double temp = arr[min_idx];
            arr[min_idx] = arr[i];
            arr[i] = temp;
        }
        return arr;
    }

    public char[] selectionSort4(char arr[])
    {
        int n = arr.length;
        for (int i = 0; i < n-1; i++)
        {
            int min_idx = i;
            for (int j = i+1; j < n; j++){
                if (arr[j] < arr[min_idx])
                    min_idx = j;
            }
            char temp = arr[min_idx];
            arr[min_idx] = arr[i];
            arr[i] = temp;
        }
        return arr;
    }

    public String[] selectionSort5(String arr[])
    {
        int n = arr.length;
        for (int i = 0; i < n-1; i++)
        {
            int min_idx = i;
            for (int j = i+1; j < n; j++){
                if (arr[j].compareTo(arr[min_idx]) < 0)
                    min_idx = j;
            }
            String temp = arr[min_idx];
            arr[min_idx] = arr[i];
            arr[i] = temp;
        }
        return arr;
    }


    // merge method
    // input: array of elements, start of array index l (0), middle of array index m ((l+r)/2), end of array index r (arr.length)
    // output: array in prescribed interval between l and r sorted in ascending order

    public int[] merge1(int arr[], int l, int m, int r)
    {
        int n1 = m - l + 1;
        int n2 = r - m;

        int[] lArr = new int [n1];
        int[] rArr = new int [n2];

        for (int i=0; i<n1; ++i)
            lArr[i] = arr[l + i];
        for (int j=0; j<n2; ++j)
            rArr[j] = arr[m + 1+ j];

        int i = 0, j = 0;
        int k = l;
        while (i < n1 && j < n2)
        {
            if (lArr[i] <= rArr[j])
            {
                arr[k] = lArr[i];
                i++;
            }
            else
            {
                arr[k] = rArr[j];
                j++;
            }
            k++;
        }
        while (i < n1)
        {
            arr[k] = lArr[i];
            i++;
            k++;
        }
        while (j < n2)
        {
            arr[k] = rArr[j];
            j++;
            k++;
        }
        return arr;
    }

    public long[] merge2(long arr[], int l, int m, int r)
    {
        int n1 = m - l + 1;
        int n2 = r - m;

        long[] lArr = new long [n1];
        long[] rArr = new long [n2];

        for (int i=0; i<n1; ++i)
            lArr[i] = arr[l + i];
        for (int j=0; j<n2; ++j)
            rArr[j] = arr[m + 1+ j];

        int i = 0, j = 0;
        int k = l;
        while (i < n1 && j < n2)
        {
            if (lArr[i] <= rArr[j])
            {
                arr[k] = lArr[i];
                i++;
            }
            else
            {
                arr[k] = rArr[j];
                j++;
            }
            k++;
        }
        while (i < n1)
        {
            arr[k] = lArr[i];
            i++;
            k++;
        }
        while (j < n2)
        {
            arr[k] = rArr[j];
            j++;
            k++;
        }
        return arr;
    }

    public double[] merge3(double arr[], int l, int m, int r)
    {
        int n1 = m - l + 1;
        int n2 = r - m;

        double[] lArr = new double [n1];
        double[] rArr = new double [n2];

        for (int i=0; i<n1; ++i)
            lArr[i] = arr[l + i];
        for (int j=0; j<n2; ++j)
            rArr[j] = arr[m + 1+ j];

        int i = 0, j = 0;
        int k = l;
        while (i < n1 && j < n2)
        {
            if (lArr[i] <= rArr[j])
            {
                arr[k] = lArr[i];
                i++;
            }
            else
            {
                arr[k] = rArr[j];
                j++;
            }
            k++;
        }
        while (i < n1)
        {
            arr[k] = lArr[i];
            i++;
            k++;
        }
        while (j < n2)
        {
            arr[k] = rArr[j];
            j++;
            k++;
        }
        return arr;
    }

    public char[] merge4(char arr[], int l, int m, int r)
    {
        int n1 = m - l + 1;
        int n2 = r - m;

        char[] lArr = new char [n1];
        char[] rArr = new char [n2];

        for (int i=0; i<n1; ++i)
            lArr[i] = arr[l + i];
        for (int j=0; j<n2; ++j)
            rArr[j] = arr[m + 1+ j];

        int i = 0, j = 0;
        int k = l;
        while (i < n1 && j < n2)
        {
            if (lArr[i] <= rArr[j])
            {
                arr[k] = lArr[i];
                i++;
            }
            else
            {
                arr[k] = rArr[j];
                j++;
            }
            k++;
        }
        while (i < n1)
        {
            arr[k] = lArr[i];
            i++;
            k++;
        }
        while (j < n2)
        {
            arr[k] = rArr[j];
            j++;
            k++;
        }
        return arr;
    }

    public String[] merge5(String arr[], int l, int m, int r)
    {
        int n1 = m - l + 1;
        int n2 = r - m;

        String[] lArr = new String [n1];
        String[] rArr = new String [n2];

        for (int i=0; i<n1; ++i)
            lArr[i] = arr[l + i];
        for (int j=0; j<n2; ++j)
            rArr[j] = arr[m + 1+ j];

        int i = 0, j = 0;
        int k = l;
        while (i < n1 && j < n2)
        {
            if (lArr[i].compareTo(rArr[j]) <= 0)
            {
                arr[k] = lArr[i];
                i++;
            }
            else
            {
                arr[k] = rArr[j];
                j++;
            }
            k++;
        }
        while (i < n1)
        {
            arr[k] = lArr[i];
            i++;
            k++;
        }
        while (j < n2)
        {
            arr[k] = rArr[j];
            j++;
            k++;
        }
        return arr;
    }

    // mergeSort method
    // input: array of elements, start of array index l (0), end of array index r (arr.length)
    // output: array sorted in ascending order

    public int[] mergeSort1(int arr[], int l, int r)
    {
        if (l < r)
        {
            int m = (l+r)/2;

            mergeSort1(arr, l, m);
            mergeSort1(arr , m+1, r);

            arr = merge1(arr, l, m, r);
        }
        return arr;
    }

    public long[] mergeSort2(long arr[], int l, int r)
    {
        if (l < r)
        {
            int m = (l+r)/2;

            mergeSort2(arr, l, m);
            mergeSort2(arr , m+1, r);

            arr = merge2(arr, l, m, r);
        }
        return arr;
    }


    public double[] mergeSort3(double arr[], int l, int r)
    {
        if (l < r)
        {
            int m = (l+r)/2;

            mergeSort3(arr, l, m);
            mergeSort3(arr , m+1, r);

            arr = merge3(arr, l, m, r);
        }
        return arr;
    }


    public char[] mergeSort4(char arr[], int l, int r)
    {
        if (l < r)
        {
            int m = (l+r)/2;

            mergeSort4(arr, l, m);
            mergeSort4(arr , m+1, r);

            arr = merge4(arr, l, m, r);
        }
        return arr;
    }

    public String[] mergeSort5(String arr[], int l, int r)
    {
        if (l < r)
        {
            int m = (l+r)/2;

            mergeSort5(arr, l, m);
            mergeSort5(arr , m+1, r);

            arr = merge5(arr, l, m, r);
        }
        return arr;
    }




    // partition method
    // input: array of elements, start of array index low (0), end of array index high (arr.length)
    // output: pivot element index (last one of equal elements) taken from the last element of unsorted array,
    //         array arr grouped to the left side of pivot and right side of pivot

    public int partition1(int arr[], int low, int high)
    {
        int pivot = arr[high];
        int i = (low-1);
        for (int j=low; j<high; j++)
        {
            if (arr[j] <= pivot)
            {
                i++;
                int temp = arr[i];
                arr[i] = arr[j];
                arr[j] = temp;
            }
        }
        int temp = arr[i+1];
        arr[i+1] = arr[high];
        arr[high] = temp;

        return i+1;
    }

    public int partition2(long arr[], int low, int high)
    {
        long pivot = arr[high];
        int i = (low-1);
        for (int j=low; j<high; j++)
        {
            if (arr[j] <= pivot)
            {
                i++;
                long temp = arr[i];
                arr[i] = arr[j];
                arr[j] = temp;
            }
        }
        long temp = arr[i+1];
        arr[i+1] = arr[high];
        arr[high] = temp;

        return i+1;
    }

    public int partition3(double arr[], int low, int high)
    {
        double pivot = arr[high];
        int i = (low-1);
        for (int j=low; j<high; j++)
        {
            if (arr[j] <= pivot)
            {
                i++;
                double temp = arr[i];
                arr[i] = arr[j];
                arr[j] = temp;
            }
        }
        double temp = arr[i+1];
        arr[i+1] = arr[high];
        arr[high] = temp;

        return i+1;
    }

    public int partition4(char arr[], int low, int high)
    {
        char pivot = arr[high];
        int i = (low-1);
        for (int j=low; j<high; j++)
        {
            if (arr[j] <= pivot)
            {
                i++;
                char temp = arr[i];
                arr[i] = arr[j];
                arr[j] = temp;
            }
        }
        char temp = arr[i+1];
        arr[i+1] = arr[high];
        arr[high] = temp;

        return i+1;
    }

    public int partition5(String arr[], int low, int high)
    {
        String pivot = arr[high];
        int i = (low-1);
        for (int j=low; j<high; j++)
        {
            if (arr[j].compareTo(pivot) <= 0)
            {
                i++;
                String temp = arr[i];
                arr[i] = arr[j];
                arr[j] = temp;
            }
        }
        String temp = arr[i+1];
        arr[i+1] = arr[high];
        arr[high] = temp;

        return i+1;
    }

    // quickSort method
    // input: array of elements, start of array index low (0), end of array index high (arr.length)
    // output: array sorted in ascending order

    public int[] quickSort1(int arr[], int low, int high)
    {
        if (low < high)
        {
            int pi = partition1(arr, low, high);

            quickSort1(arr, low, pi-1);
            quickSort1(arr, pi+1, high);
        }
        return arr;
    }

    public long[] quickSort2(long arr[], int low, int high)
    {
        if (low < high)
        {
            int pi = partition2(arr, low, high);

            quickSort2(arr, low, pi-1);
            quickSort2(arr, pi+1, high);
        }
        return arr;
    }

    public double[] quickSort3(double arr[], int low, int high)
    {
        if (low < high)
        {
            int pi = partition3(arr, low, high);

            quickSort3(arr, low, pi-1);
            quickSort3(arr, pi+1, high);
        }
        return arr;
    }

    public char[] quickSort4(char arr[], int low, int high)
    {
        if (low < high)
        {
            int pi = partition4(arr, low, high);

            quickSort4(arr, low, pi-1);
            quickSort4(arr, pi+1, high);
        }
        return arr;
    }

    public String[] quickSort5(String arr[], int low, int high)
    {
        if (low < high)
        {
            int pi = partition5(arr, low, high);

            quickSort5(arr, low, pi-1);
            quickSort5(arr, pi+1, high);
        }
        return arr;
    }


    // minDistance method
    // input: array dist as distances of shortest paths to all vertexes, array sptSet that marks if vertex is included in shortest path tree or shortest distance, vertex count V > 2
    // output: minimum distance vertex from the set of vertices not yet processed, result is always equal to src in first iteration

    public int minDistance1(int[] dist, boolean[] sptSet, int V)
    {
        int min = Integer.MAX_VALUE, min_index = -1;

        for (int v = 0; v < V; v++)
            if (!sptSet[v] && dist[v] <= min) {
                min = dist[v];
                min_index = v;
            }

        return min_index;
    }

    public int minDistance2(long[] dist, boolean[] sptSet, int V)
    {
        long min = Long.MAX_VALUE;
        int min_index = -1;

        for (int v = 0; v < V; v++)
            if (!sptSet[v] && dist[v] <= min) {
                min = dist[v];
                min_index = v;
            }

        return min_index;
    }

    public int minDistance3(double[] dist, boolean[] sptSet, int V)
    {
        double min = Double.MAX_VALUE;
        int min_index = -1;

        for (int v = 0; v < V; v++)
            if (!sptSet[v] && dist[v] <= min) {
                min = dist[v];
                min_index = v;
            }

        return min_index;
    }

    // dijkstra method
    // input: undirectional graph with at least 2 vertexes represented as adjacency matrix, start vertex
    // output: array that represents shortest path from start vertex to each vertex

    public int[] dijkstra1(int[][] graph, int src)
    {
        int V = graph.length;
        int[] dist = new int[V];
        boolean[] sptSet = new boolean[V];
        for (int i = 0; i < V; i++) {
            dist[i] = Integer.MAX_VALUE;
            sptSet[i] = false;
        }
        dist[src] = 0;
        for (int count = 0; count < V - 1; count++) {
            int u = minDistance1(dist, sptSet, V);
            sptSet[u] = true;
            for (int v = 0; v < V; v++)
                if (!sptSet[v] && graph[u][v] != 0 && dist[u] != Integer.MAX_VALUE && dist[u] + graph[u][v] < dist[v])
                    dist[v] = dist[u] + graph[u][v];
        }
        return dist;
    }

    public long[] dijkstra2(long[][] graph, int src)
    {
        int V = graph.length;
        long[] dist = new long[V];
        boolean[] sptSet = new boolean[V];
        for (int i = 0; i < V; i++) {
            dist[i] = Long.MAX_VALUE;
            sptSet[i] = false;
        }
        dist[src] = 0;
        for (int count = 0; count < V - 1; count++) {
            int u = minDistance2(dist, sptSet, V);
            sptSet[u] = true;
            for (int v = 0; v < V; v++)
                if (!sptSet[v] && graph[u][v] != 0 && dist[u] != Long.MAX_VALUE && dist[u] + graph[u][v] < dist[v])
                    dist[v] = dist[u] + graph[u][v];
        }
        return dist;
    }

    public double[] dijkstra3(double[][] graph, int src)
    {
        int V = graph.length;
        double[] dist = new double[V];
        boolean[] sptSet = new boolean[V];
        for (int i = 0; i < V; i++) {
            dist[i] = Double.MAX_VALUE;
            sptSet[i] = false;
        }
        dist[src] = 0;
        for (int count = 0; count < V - 1; count++) {
            int u = minDistance3(dist, sptSet, V);
            sptSet[u] = true;
            for (int v = 0; v < V; v++)
                if (!sptSet[v] && graph[u][v] != 0 && dist[u] != Double.MAX_VALUE && dist[u] + graph[u][v] < dist[v])
                    dist[v] = dist[u] + graph[u][v];
        }
        return dist;
    }

    public int[] dijkstra4(boolean[][] graph, int src)
    {
        int V = graph.length;
        int[] dist = new int[V];
        boolean[] sptSet = new boolean[V];
        for (int i = 0; i < V; i++) {
            dist[i] = Integer.MAX_VALUE;
            sptSet[i] = false;
        }
        dist[src] = 0;
        for (int count = 0; count < V - 1; count++) {
            int u = minDistance1(dist, sptSet, V);
            sptSet[u] = true;
            for (int v = 0; v < V; v++)
                if (!sptSet[v] && graph[u][v] && dist[u] != Integer.MAX_VALUE && dist[u] + 1 < dist[v])
                    dist[v] = dist[u] + 1;
        }
        return dist;
    }

    // bfs method
    // input: graph represented as adjacency matrix, source vertex s, target vertex t, parent array that holds visited vertexes
    // output: true if path was found from source s to target t, else false, path is hold in parent array

    public boolean bfs1(int[][] rGraph, int s, int t, int[] parent)
    {
        int V = rGraph.length;
        boolean[] visited = new boolean[V];
        for (int i = 0; i < V; ++i)
            visited[i] = false;

        int[] queue = new int[V];
        queue[0] = s;
        int queueLength = 1;
        visited[s] = true;
        parent[s] = -1;

        while (queueLength != 0) {
            int u = queue[0];
            for(int i = 1; i < queueLength; i++) {
                queue[i-1] = queue[i];
            }
            queueLength--;
            for (int v = 0; v < V; v++) {
                if (!visited[v] && rGraph[u][v] > 0) {
                    if (v == t) {
                        parent[v] = u;
                        return true;
                    }
                    queue[queueLength] = v;
                    queueLength++;
                    parent[v] = u;
                    visited[v] = true;
                }
            }
        }
        return false;
    }

    public boolean bfs2(long[][] rGraph, int s, int t, int[] parent)
    {
        int V = rGraph.length;
        boolean[] visited = new boolean[V];
        for (int i = 0; i < V; ++i)
            visited[i] = false;

        int[] queue = new int[V];
        queue[0] = s;
        int queueLength = 1;
        visited[s] = true;
        parent[s] = -1;

        while (queueLength != 0) {
            int u = queue[0];
            for(int i = 1; i < queueLength; i++) {
                queue[i-1] = queue[i];
            }
            queueLength--;
            for (int v = 0; v < V; v++) {
                if (!visited[v] && rGraph[u][v] > 0) {
                    if (v == t) {
                        parent[v] = u;
                        return true;
                    }
                    queue[queueLength] = v;
                    queueLength++;
                    parent[v] = u;
                    visited[v] = true;
                }
            }
        }
        return false;
    }

    public boolean bfs3(double[][] rGraph, int s, int t, int[] parent)
    {
        int V = rGraph.length;
        boolean[] visited = new boolean[V];
        for (int i = 0; i < V; ++i)
            visited[i] = false;

        int[] queue = new int[V];
        queue[0] = s;
        int queueLength = 1;
        visited[s] = true;
        parent[s] = -1;

        while (queueLength != 0) {
            int u = queue[0];
            for(int i = 1; i < queueLength; i++) {
                queue[i-1] = queue[i];
            }
            queueLength--;
            for (int v = 0; v < V; v++) {
                if (!visited[v] && rGraph[u][v] > 0) {
                    if (v == t) {
                        parent[v] = u;
                        return true;
                    }
                    queue[queueLength] = v;
                    queueLength++;
                    parent[v] = u;
                    visited[v] = true;
                }
            }
        }
        return false;
    }

    public boolean bfs4(boolean[][] rGraph, int s, int t, int[] parent)
    {
        int V = rGraph.length;
        boolean[] visited = new boolean[V];
        for (int i = 0; i < V; ++i)
            visited[i] = false;

        int[] queue = new int[V];
        queue[0] = s;
        int queueLength = 1;
        visited[s] = true;
        parent[s] = -1;

        while (queueLength != 0) {
            int u = queue[0];
            for(int i = 1; i < queueLength; i++) {
                queue[i-1] = queue[i];
            }
            queueLength--;
            for (int v = 0; v < V; v++) {
                if (!visited[v] && rGraph[u][v]) {
                    if (v == t) {
                        parent[v] = u;
                        return true;
                    }
                    queue[queueLength] = v;
                    queueLength++;
                    parent[v] = u;
                    visited[v] = true;
                }
            }
        }
        return false;
    }

    // fordFulkerson method
    // input: graph represented as adjacency matrix, source vertex s, target vertex t
    // output: maximum flow that can flow through the graph from source s to target t

    public int fordFulkerson1(int[][] graph, int s, int t)
    {
        int V = graph.length;
        int u, v;
        int[][] rGraph = new int[V][V];
        for (u = 0; u < V; u++)
            for (v = 0; v < V; v++)
                rGraph[u][v] = graph[u][v];
        int[] parent = new int[V];
        int max_flow = 0;
        while (bfs1(rGraph, s, t, parent)) {
            int path_flow = Integer.MAX_VALUE;
            for (v = t; v != s; v = parent[v]) {
                u = parent[v];
                path_flow = Math.min(path_flow, rGraph[u][v]);
            }
            for (v = t; v != s; v = parent[v]) {
                u = parent[v];
                rGraph[u][v] -= path_flow;
                rGraph[v][u] += path_flow;
            }
            max_flow += path_flow;
        }
        return max_flow;
    }

    public long fordFulkerson2(long[][] graph, int s, int t)
    {
        int V = graph.length;
        int u, v;
        long[][] rGraph = new long[V][V];
        for (u = 0; u < V; u++)
            for (v = 0; v < V; v++)
                rGraph[u][v] = graph[u][v];
        int[] parent = new int[V];
        int max_flow = 0;
        while (bfs2(rGraph, s, t, parent)) {
            long path_flow = Long.MAX_VALUE;
            for (v = t; v != s; v = parent[v]) {
                u = parent[v];
                path_flow = Math.min(path_flow, rGraph[u][v]);
            }
            for (v = t; v != s; v = parent[v]) {
                u = parent[v];
                rGraph[u][v] -= path_flow;
                rGraph[v][u] += path_flow;
            }
            max_flow += path_flow;
        }
        return max_flow;
    }

    public double fordFulkerson3(double[][] graph, int s, int t)
    {
        int V = graph.length;
        int u, v;
        double[][] rGraph = new double[V][V];
        for (u = 0; u < V; u++)
            for (v = 0; v < V; v++)
                rGraph[u][v] = graph[u][v];
        int[] parent = new int[V];
        int max_flow = 0;
        while (bfs3(rGraph, s, t, parent)) {
            double path_flow = Double.MAX_VALUE;
            for (v = t; v != s; v = parent[v]) {
                u = parent[v];
                path_flow = Math.min(path_flow, rGraph[u][v]);
            }
            for (v = t; v != s; v = parent[v]) {
                u = parent[v];
                rGraph[u][v] -= path_flow;
                rGraph[v][u] += path_flow;
            }
            max_flow += path_flow;
        }
        return max_flow;
    }

    public int fordFulkerson4(boolean[][] graph, int s, int t)
    {
        int V = graph.length;
        int u, v;
        int[][] rGraph = new int[V][V];
        for (u = 0; u < V; u++)
            for (v = 0; v < V; v++)
                rGraph[u][v] = graph[u][v] ? 1 : 0;
        int[] parent = new int[V];
        int max_flow = 0;
        while (bfs1(rGraph, s, t, parent)) {
            int path_flow = Integer.MAX_VALUE;
            for (v = t; v != s; v = parent[v]) {
                u = parent[v];
                path_flow = Math.min(path_flow, rGraph[u][v]);
            }
            for (v = t; v != s; v = parent[v]) {
                u = parent[v];
                rGraph[u][v] -= path_flow;
                rGraph[v][u] += path_flow;
            }
            max_flow += path_flow;
        }
        return max_flow;
    }





//    public static void main(String[] args) {
//        boolean[][] graph1 = new boolean[][] {
//            {false, true},
//            {true, false},
//        };
//        boolean[][] graph2 = new boolean[][] {
//            {false, true, false, false, false, false, false, true, false},
//            {true, false, true, false, false, false, false, true, false},
//            {false, true, false, true, false, true, false, false, true},
//            {false, false, true, false, true, true, false, false, false},
//            {false, false, false, true, false, true, false, false, false},
//            {false, false, true, true, true, false, true, false, false},
//            {false, false, false, false, false, true, false, true, true},
//            {true, true, false, false, false, false, true, false, true},
//            {false, false, true, false, false, false, true, true, false},
//        };
//        printArr(dijkstra4(graph1, 0));
//        printArr(dijkstra4(graph1, 1));
//        printArr(dijkstra4(graph2, 0));
//        printArr(dijkstra4(graph2, 1));
//        printArr(dijkstra4(graph2, 8));
//
//    }
//
//    public static void printArr(int[] arr) {
//        for(int i = 0; i < arr.length; i++) {
//            System.out.print(arr[i] + ", ");
//        }
//        System.out.println();
//    }
//
//    public static void printBooleanArr(boolean[] arr) {
//        for(int i = 0; i < arr.length; i++) {
//            System.out.print(arr[i] + ", ");
//        }
//        System.out.println();
//    }
//
//    public static void printGraph(int[][] graph) {
//        System.out.println("{");
//        for(int i = 0; i < graph.length; i++) {
//            System.out.print("{");
//            for(int j = 0; j < graph[i].length; j++) {
//                System.out.print(graph[i][j] + ", ");
//            }
//            System.out.println("},");
//        }
//        System.out.println("}");
//    }
//
//    public static void printBooleanGraph(int[][] graph) {
//        System.out.println("{");
//        for(int i = 0; i < graph.length; i++) {
//            System.out.print("{");
//            for(int j = 0; j < graph[i].length; j++) {
//                System.out.print((graph[i][j] > 0) + ", ");
//            }
//            System.out.println("},");
//        }
//        System.out.println("}");
//    }

}