public class Sort {

    int N;

    //sorts an array of ints using heapsort
    public void heapSort(int[] to_sort) {

		Heap theHeap = new Heap(to_sort);
		int[] holder = new int[to_sort.length];
		int spot = holder.length-1;

		while(!theHeap.isEmpty()) {

			holder[spot]=theHeap.deleteMax();
			spot--;

		}

		for(int i = 0; i<holder.length; i++){
			to_sort[i] = holder[i];
		}

    }

    //sorts an array of ints using selection sort
    public void selectionSort(int[] to_sort) {

    	for(int divide = 0; divide < N; divide++) {

    		int lowest = to_sort[divide];
    		int lowestSpot = divide;

    		for(int swapper = divide; swapper < N; swapper++) {
    			if(to_sort[swapper] < lowest) {
    				lowestSpot = swapper;
    				lowest = to_sort[lowestSpot];
    			}
    		}

    		to_sort[lowestSpot] = to_sort[divide];
    		to_sort[divide] = lowest;

    	}

    }

    //sorts an array of ints using quicksort
    public void quickSort(int[] to_sort) {

    	int pivot = to_sort[to_sort.length-1];
    	int smallerSpot = -1;

    	for(int position = 0; position < to_sort.length-1; position++) {
    		if(to_sort[position] < pivot) {
    			smallerSpot++;
    			int temp = to_sort[smallerSpot];
    			to_sort[smallerSpot] = to_sort[position];
    			to_sort[position] = temp;
    		}
    	}

    	int swapPivot = to_sort[smallerSpot+1];
    	to_sort[to_sort.length-1] = swapPivot;
    	to_sort[smallerSpot+1] = pivot;

   		quickAgain(to_sort,0,smallerSpot);
   		quickAgain(to_sort,smallerSpot+2,to_sort.length-1);

    }

    public void quickAgain(int[] to_sort, int first, int second) {

    	if(first < second) {

    		int pivot = to_sort[second];
    		int smallerSpot = first-1;

	    	for(int position = first; position < second; position++) {
	    		if(to_sort[position] < pivot) {
	    			smallerSpot++;
	    			int temp = to_sort[smallerSpot];
	    			to_sort[smallerSpot] = to_sort[position];
	    			to_sort[position] = temp;
	    		}
	    	}

	    	int swapPivot = to_sort[smallerSpot+1];
	    	to_sort[second] = swapPivot;
	    	to_sort[smallerSpot+1] = pivot;

	    	quickAgain(to_sort,first,smallerSpot);
	   		quickAgain(to_sort,smallerSpot+2,second);

   		}

    }

    //sorts an array of ints using mergesort
    public void mergeSort(int[] to_sort) {

    	int[] lower = new int[(to_sort.length+1)/2];
    	int[] higher = new int[to_sort.length - lower.length];

    	for (int i = 0; i<to_sort.length; i++) {

    		if(i<lower.length) {
    			lower[i] = to_sort[i];
    		}

    		else higher[i-lower.length] = to_sort[i];

    	}

    	divide(to_sort, 0, (to_sort.length+1)/2-1);
    	divide(to_sort, (to_sort.length+1)/2, to_sort.length-1);
 		merge(to_sort,0,(to_sort.length+1)/2-1,to_sort.length-1);

    }

	public void divide(int[] to_sort, int lowest, int highest) {

    	if (lowest < highest) {
    		int middle = (highest+lowest)/2;
    		divide(to_sort, lowest, middle);
    		divide(to_sort,middle+1,highest);
    		merge(to_sort,lowest,middle,highest);
    	}

	}

	public void merge(int[] to_sort, int lowest, int middle, int highest) {

		int[] lower = new int[middle-lowest+1];
    	int[] higher = new int[highest-middle];

    	for (int i = 0; i < (middle-lowest+1); i++) {
            lower[i] = to_sort[lowest + i];
    	}
        for (int j = 0; j < (highest-middle); j++) {
            higher[j] = to_sort[middle + 1+ j];
        }

	   	int lowerSpot = 0;
    	int higherSpot = 0;
    	int totalSpot = 0;

    	while (lowerSpot < lower.length && higherSpot < higher.length) {
    		if(lower[lowerSpot] < higher[higherSpot]) {
    			to_sort[totalSpot + lowest] = lower[lowerSpot];
    			totalSpot++;
    			lowerSpot++;
    		}
    		else {
    			to_sort[totalSpot + lowest] = higher[higherSpot];
    			totalSpot++;
    			higherSpot++;
    		}
    	}

    	while (lowerSpot < lower.length) {
    		to_sort[totalSpot + lowest] = lower[lowerSpot];
    		totalSpot++;
    		lowerSpot++;
    	}

    	while (higherSpot < higher.length) {
    		to_sort[totalSpot + lowest] = higher[higherSpot];
    		totalSpot++;
    		higherSpot++;
    	}

	}

    //sorts an array of ints using insertion sort
    public void insertionSort(int[] to_sort) {
	
	for(int i = 1; i < N; i++) {

	    int val = to_sort[i];
	    int j = i - 1;

	    while(j >= 0 && val < to_sort[j]) {
		to_sort[j + 1] = to_sort[j];
		j--;
	    }

	    to_sort[j+1] = val;

	}

    }


    public void test() {

	int[] my_array = new int[N];

	long total_time = 0;
	int num_iters = 10;

	//test insertionsort
	//sort multiple times to reduce noise
	for(int i = 0; i < num_iters; i++) { 
	    Generate.randomData(my_array); //fill my_array with unsorted data

	    long start_time = System.nanoTime();
	    heapSort(my_array);
	    //mergeSort(my_array);
	    //quickSort(my_array);
	    //selectionSort(my_array);
	    //insertionSort(my_array);
	    long end_time = System.nanoTime();
	    total_time += end_time - start_time;
	}
	System.out.println((total_time/1000000000.0)/(num_iters * 1.0) + " ");
	System.out.println();
	System.out.println();
    }

    //input: number of elements to sort
    public static void main(String[] args) {

	int num_items = Integer.parseInt(args[0]);

	Sort s = new Sort(num_items);
	s.test();

    }

    public Sort(int num_elts) {
	N = num_elts;
    }

}