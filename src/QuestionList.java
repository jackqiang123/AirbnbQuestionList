import java.util.*;

public class QuestionList {
    public static void main(String[]args) throws Exception {
        //testArrayListQueue();
        //test2DIterator();
        testPreferenceList();
    }

    private static void testPreferenceList() {
        PreferenceList pl = new PreferenceList();
        pl.alienOrder(new String[]{"wrt", "wrf", "er", "ett", "rftt"});
    }

    private static void test2DIterator() {
        List<List<Integer>>test = new ArrayList<>();
        List<Integer> d1 = new LinkedList<>(Arrays.asList(new Integer[]{1,2,3,4,5}));
        List<Integer> d2 = new LinkedList<>(Arrays.asList(new Integer[]{6}));
        List<Integer> d3 = new LinkedList<>();
        List<Integer> d4 = new LinkedList<>(Arrays.asList(new Integer[]{7,8}));
        test.add(d1);
        test.add(d2);
        test.add(d3);
        test.add(d4);

        Flattern2DIterator it = new Flattern2DIterator(test);
        boolean isEven = true;
        while(it.hasNext()) {
            it.next();
            if(isEven) {
                it.remove();
            }

            isEven = !isEven;
        }

        it = new Flattern2DIterator(test);
        while(it.hasNext()) {
            System.out.println(it.next());
        }
    }

    public static void testArrayListQueue() throws Exception {
        ArrayListQueue queue = new ArrayListQueue(5);
        for (int i = 0; i < 100; i++) {
            queue.offer(i);
        }

        while(queue.size > 0) {
            System.out.println(queue.poll());
        }
    }
}

class CollatzConjecture{
    Map<Integer, Integer> cache = new HashMap<>();
    public int findLongestSteps(int number){
        if (number < 1) return 0;

        int res = 0;
        for (int i = 1; i <= number; i++) {
            int curRes = findResult(i);
            cache.put(i, curRes);
            res = Math.max(curRes, res);
        }

        return res;
    }

    private int findResult(int i) {
        if (cache.containsKey(i)) return cache.get(i);

        if (i % 2 == 0) {
            return 1 + findResult(i/2);
        }
        else {
            return 1 + findResult(3*i+1);
        }
    }
}

class ArrayListQueue{
    int arraySize;
    Node head;
    Node tail;
    List<Node> queue;
    int size;

    public ArrayListQueue(int fixSizeArray){
        this.arraySize = fixSizeArray;
        this.queue = new LinkedList<>();
        this.size = 0;
    }

    public void offer(int v){
        if (size == 0) {
            queue.add(new Node(this.arraySize));
            this.head = queue.get(0);
            this.tail = queue.get(0);
        }

        if (this.tail.curIndex == this.arraySize) {
            this.tail = new Node(this.arraySize);
            queue.add(this.tail);
        }

        this.size++;
        this.tail.array[this.tail.curIndex++] = v;
    }

    public int poll() throws Exception {
        if (this.size == 0)
            throw new Exception();

        if (this.head.removeIndex == this.arraySize) {
            this.queue.remove(0);
            this.head = this.queue.get(0);
        }

        this.size--;
        return this.head.array[this.head.removeIndex++];
    }

    private int[] getNewBlock(){
        return new int[arraySize];
    }

    class Node{
        int[]array;
        int curIndex;
        int removeIndex;
        public Node(int len){
            this.curIndex = 0;
            this.removeIndex = 0;
            this.array = new int[len];
        }
    }
}

class Flattern2DIterator implements Iterator<Integer> {
    List<List<Integer>> data;
    List<Iterator<Integer>> iterators;
    int index;

    public Flattern2DIterator(List<List<Integer>> data){
        this.data = data;
        this.index = 0;
        iterators = new ArrayList<>();
        for (List<Integer> ls : data) {
            if (!ls.isEmpty())
                iterators.add(ls.iterator());
        }
    }

    @Override
    public boolean hasNext() {
        if (index >= iterators.size())
            return false;

        if (!iterators.get(index).hasNext()) {
            index++;
        }

         return (index < iterators.size() && iterators.get(index).hasNext());
    }

    @Override
    public Integer next() {
        int value = iterators.get(index).next();
        return value;
    }

    @Override
    public void remove() {
        iterators.get(index).remove();
    }
}

class PreferenceList{
    public String alienOrder(String[] words) {
        Map<Character, Set<Character>> higher = new HashMap<>();
        Map<Character, Set<Character>> lower = new HashMap<>();
        Set<Character> set = new HashSet<>();
        for (String w : words) {
            for (int i = 0; i < w.length() - 1; i++) {
                char c1 = w.charAt(i);
                char c2 = w.charAt(i+1);
                if (c1 == c2)
                    continue;
                if (higher.get(c2) == null) higher.put(c2, new HashSet<>());

                higher.get(c2).add(c1);

                if (lower.get(c1) == null) lower.put(c1, new HashSet<>());

                lower.get(c1).add(c2);

                set.add(c1);
                set.add(c2);
            }
        }

        Queue<Character> queue = new LinkedList<>();
        for (char c : set) {
            if (higher.get(c) == null || higher.size() == 0) {
                queue.add(c);
            }
        }

        StringBuilder sb = new StringBuilder();
        while(!queue.isEmpty()) {
            char c = queue.remove();
            sb.append(c);
            Set<Character> next = lower.get(c);
            if (next != null) {
                for (char cc : next) {
                    higher.get(cc).remove(c);
                    if (higher.get(cc).size() == 0) {
                        queue.add(cc);
                    }
                }
            }
        }

        return sb.length() == set.size() ?  sb.toString() : "";
    }
}

class DisplayPages{
    public List<String> displayPages(List<String> input, int pageSize){
        List<String> res = new ArrayList<>();
        Iterator<String> iter = input.iterator();
        Set<String> idSet = new HashSet<>();
        boolean hasReachEnd = false;
        int counter = 0;
        while(iter.hasNext()) {
            String cur = iter.next();
            String id = cur.split(",")[0];
            if (!idSet.contains(id) || hasReachEnd) {
                res.add(cur);
                idSet.add(id);
                counter++;
                iter.remove();
            }

            if (counter == pageSize) {
                if (!input.isEmpty()) {
                    res.add(" ");

                    idSet.clear();
                    counter=0;
                    hasReachEnd = false;
                    iter = input.iterator();
                }
            }

            if (!iter.hasNext()) {
                hasReachEnd = true;
                iter = input.iterator();
            }
        }

        return res;
    }
}

class PalindromePair{
    public List<List<Integer>> palindromePairs(String[] words) {
        List<List<Integer>> res = new ArrayList<>();
        Map<String, Integer> map = new HashMap<>();
        for (int i = 0; i < words.length; i++) {
            map.put(words[i], i);
        }

        for (int i = 0; i < words.length; i++) {
            String cur = words[i];
            String reverse = new StringBuilder(cur).reverse().toString();
            if (map.containsKey(reverse) && !cur.equals(reverse)) {
                List<Integer> temp = new ArrayList();
                temp.add(i);
                temp.add(map.get(reverse));
                res.add(temp);
            }

            if (cur.length() > 0 && isPalindrome(cur) && map.containsKey("")) {
                int index = map.get("");
                List<Integer> temp = new ArrayList();
                temp.add(i);
                temp.add(index);
                res.add(temp);

                temp = new ArrayList();
                temp.add(index);
                temp.add(i);
                res.add(temp);
            }

            for (int j = 1; j < cur.length(); j++) {
                if (isPalindrome(cur.substring(0, j))) {
                    String head = cur.substring(j);
                    String need = new StringBuilder(head).reverse().toString();
                    if (map.containsKey(need)) {
                        List<Integer> temp = new ArrayList<>();
                        temp.add(map.get(need));
                        temp.add(i);
                        res.add(temp);
                    }
                }

                if (isPalindrome(cur.substring(j))) {
                    String tail = cur.substring(0, j);
                    String tailneed = new StringBuilder(tail).reverse().toString();
                    if (map.containsKey(tailneed)) {
                        List<Integer> temp = new ArrayList<>();
                        temp.add(i);
                        temp.add(map.get(tailneed));
                        res.add(temp);
                    }
                }
            }
        }

        return res;
    }

    private boolean isPalindrome(String s) {
        String other = new StringBuilder(s).reverse().toString();
        return other.equals(s);
    }
}

class WaterDrop{
    public int[] pourWater(int[] heights, int V, int K) {
        for (int v = 0; v < V; v++) {
            int leftDrop = findDirection(heights, K, -1);
            if (leftDrop > -1 && leftDrop != K) {
                heights[leftDrop]++;
                continue;
            }

            int rightDrop = findDirection(heights, K, 1);
            if (rightDrop < heights.length && rightDrop != K) {
                heights[rightDrop]++;
                continue;
            }

            heights[K]++;
        }

        return heights;
    }

    private int findDirection(int[] heights, int startIndex, int direction) {
        int lowestIndex = startIndex;
        for (int i = startIndex + direction; i >= 0 && i < heights.length; i += direction) {
            if (heights[i] > heights[i - direction]) {
                break; // can not flow over a peak
            }

            if (heights[i] < heights[lowestIndex]) {
                lowestIndex = i;
            }
        }

        return lowestIndex;
    }
}

class FileSystem{
    Map<String, Integer> data;
    Map<String, Runnable> callbacks;
    public FileSystem(){
        this.data = new HashMap<>();
        this.callbacks = new HashMap<>();
    }

    public boolean create(String path, int value) {
        if (data.containsKey(path))
            return false;

        int lastIndex = path.lastIndexOf("/");
        if (lastIndex == -1) {
            return false;
        }

        String previousPath = path.substring(0, lastIndex);
        if (previousPath.length() == 0 || this.data.containsKey(previousPath)) {
            this.data.put(path, value);
            return true;
        }
        else {
            return false;
        }
    }

    public Integer get(String path) {
        return this.data.get(path);
    }

    public boolean set(String path, int value) {
        if (this.get(path) == null) return false;
        this.data.put(path, value);

        while(path.length() > 0) {
            if (callbacks.containsKey(path)) {
                callbacks.get(path).run();
            }

            int lastIndex = path.lastIndexOf("/");
            path = path.substring(0, lastIndex);
        }

        return true;
    }

    public void watch(String path, Runnable callback){
        if (this.get(path) != null) {
            this.callbacks.put(path, callback);
        }
    }
}